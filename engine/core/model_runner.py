"""
engine/core/model_runner.py - one forward pass over a whole batch of requests.

The batch is "flat" (no padding): the new tokens of every scheduled request are joined
into one sequence of T tokens, each with its own position. Layers that work token by token
(norms, projections, the MoE experts) process all T tokens together - that's where batching
pays off, since the weights are read once per step for everyone. Only attention has to keep
requests apart; it reads each request's keys/values from the KV cache through slot numbers:

    prefill chunk (a new or recomputed prompt): causal attention over its own tokens
    decode tokens (1 new token per request): batched into ONE attention call, each request
    attending to its own cached context. Two backends (ATTN_BACKEND):
        sdpa    gather every request's keys/values into a padded copy, then PyTorch SDPA with a mask
        triton  kernels/paged_attention: reads keys/values in place through the page tables

This file reuses the transformers model's modules and weights (so quantized layers work
as-is) but replaces its attention and cache handling. It runs any architecture listed in
engine/models.py (OLMoE, Qwen3, Qwen3-MoE, Qwen2, Llama, Mistral): the differences between them
(how q/k are normalized, dense MLP vs MoE) are detected there when the model loads.
"""
import torch
import torch.nn.functional as F

from profiling.nvtx import nvtx_range


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class BatchMetadata:
    """Everything attention needs for this step, built once and reused by every layer."""

    def __init__(self, chunks, cache, device, attn_backend="sdpa"):
        # chunks: list of (request, start_position, end_position) for the tokens fed this step
        ids, positions, write_slots = [], [], []
        self.prefill, self.decode_rows, decode_ctx, decode_entries = [], [], [], []
        self.last_rows = []
        self.context_tokens = 0     # keys/values attention reads this step (for bandwidth estimates)
        self.decode_tables = None   # paged-kernel page tables (ATTN_BACKEND=triton)
        offset = 0
        for req, start, end in chunks:
            n = end - start
            ids.extend(req.all_ids[start:end])
            positions.append(torch.arange(start, end))
            write_slots.append(cache.slots(req, start, end))
            self.context_tokens += end
            if n == 1 and start > 0:                       # decode
                self.decode_rows.append(offset)
                if attn_backend == "triton":
                    decode_entries.append((req, end))      # the kernel reads the page table itself
                else:
                    decode_ctx.append(cache.slots(req, 0, end))
            else:                                          # prefill (or recompute)
                self.prefill.append((offset, n, start, cache.slots(req, 0, end).to(device)))
            offset += n
            self.last_rows.append(offset - 1)
        if decode_entries:      # (table, seq_lens, page, max_len, num_splits=None: chosen per step)
            self.decode_tables = (*cache.decode_tables(decode_entries, device), None)
        self.all_decode = not self.prefill          # every row is a decode token: no row indexing

        self.num_tokens = offset
        self.input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self.positions = torch.cat(positions).to(device)
        self.write_slots = torch.cat(write_slots).to(device)
        self.last_rows = torch.tensor(self.last_rows, device=device)

        self.decode_rows = torch.tensor(self.decode_rows, dtype=torch.long, device=device)
        if decode_ctx:
            lens = torch.tensor([len(c) for c in decode_ctx])
            max_len = int(lens.max())
            idx = torch.zeros(len(decode_ctx), max_len, dtype=torch.long)
            for i, c in enumerate(decode_ctx):
                idx[i, : len(c)] = c
            self.decode_idx = idx.to(device)                                     # [B, max_len]
            self.decode_mask = (torch.arange(max_len)[None, :] < lens[:, None]).to(device)
            self.decode_mask = self.decode_mask[:, None, None, :]                # [B, 1, 1, max_len]


class ModelRunner:
    def __init__(self, model, cache, attn_backend="sdpa", arch=None):
        self.model = model
        self.cache = cache
        if arch is None:
            from engine.models import describe
            arch = describe(model, getattr(model.config, "max_position_embeddings", 4096))
        self.arch = arch
        self.qk_norm = arch["qk_norm"]
        self.mlp_label = "moe" if arch["moe"] else "mlp"     # profiling region name
        if attn_backend not in ("sdpa", "triton"):
            raise ValueError(f"Unknown ATTN_BACKEND '{attn_backend}' (use sdpa or triton)")
        self.attn_backend = attn_backend
        if attn_backend == "triton":
            from kernels.paged_attention.decode_triton import paged_decode_attention
            self._paged_decode = paged_decode_attention
        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", None) or self.num_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.num_heads
        self.clip_qkv = getattr(cfg, "clip_qkv", None)
        self.device = next(model.parameters()).device
        self.graphs = None          # engine/core/cuda_graphs.py, when CUDA_GRAPHS=1
        self.fused = None           # engine/core/fused_linear.py, when FUSED_PROJ=1
        self.fused_head = None
        self.layout_report = {}
        inner = model.model
        self.embed, self.layers, self.norm, self.rotary = (inner.embed_tokens, inner.layers,
                                                           inner.norm, inner.rotary_emb)

    @torch.inference_mode()
    def forward(self, chunks, all_logits=False):
        """Run one step. Returns logits [num_chunks, vocab] for the last token of each chunk,
        or [num_tokens, vocab] for every token when all_logits=True (used for perplexity)."""
        if self.graphs is not None and not all_logits:
            logits = self.graphs.run(chunks)             # pure decode steps replay a CUDA graph
            if logits is not None:
                return logits
        with nvtx_range("prepare_batch"):                # Python bookkeeping before the GPU work
            meta = BatchMetadata(chunks, self.cache, self.device, self.attn_backend)
        # profiling/analyze.py parses this label: tokens, decode rows, prefill chunks, context read
        with nvtx_range(f"forward(T={meta.num_tokens}, decode={len(meta.decode_rows)}, "
                        f"prefill={len(meta.prefill)}, ctx={meta.context_tokens})"):
            return self.forward_meta(meta, all_logits)

    def fuse_projections(self, layout="auto", cache_file=None):
        """FUSED_PROJ=1: one matmul for q/k/v and one for gate/up, each weight in its fastest layout.
        Must run before CUDA graphs are captured (graphs record whichever path is active)."""
        from engine.core.fused_linear import fuse_model
        self.fused, self.fused_head, self.layout_report = fuse_model(self.model, layout, cache_file)
        return self.layout_report

    def forward_meta(self, meta, all_logits=False):
        """The forward pass itself, from prepared metadata. CUDA graphs capture this function with
        metadata that points at fixed buffers (meta.last_rows = None: every row is an output)."""
        x = self.embed(meta.input_ids)                                       # [T, hidden]
        cos, sin = self.rotary(x, meta.positions[None])
        cos, sin = cos[0][:, None, :], sin[0][:, None, :]                    # [T, 1, D]

        for i, layer in enumerate(self.layers):
            x = x + self._attention(i, layer.self_attn, layer.input_layernorm(x), cos, sin, meta)
            h = layer.post_attention_layernorm(x)
            f = self.fused[i] if self.fused is not None else {}
            with nvtx_range(self.mlp_label):
                if "gate_up" in f:                      # fused dense MLP: down(act(gate) * up)
                    gate, up = f["gate_up"].split(f["gate_up"](h))
                    x = x + f["down"](f["act"](gate) * up)
                    continue
                out = layer.mlp(h[None])                # dense MLP or MoE block, [1, T, hidden]
                if isinstance(out, tuple):              # some MoE blocks also return router logits
                    out = out[0]
                x = x + out[0]

        x = self.norm(x if all_logits or meta.last_rows is None else x[meta.last_rows])
        head = self.fused_head if self.fused_head is not None else self.model.lm_head
        return head(x).float()

    def _attention(self, layer_idx, attn, h, cos, sin, meta):
        T, H, KV, D = meta.num_tokens, self.num_heads, self.num_kv_heads, self.head_dim
        f = self.fused[layer_idx] if self.fused is not None else {}
        if "qkv" in f:                                 # one matmul, split into views
            q, k, v = f["qkv"].split(f["qkv"](h))
        else:
            q, k, v = attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)   # biases (Qwen2) come along
        if self.qk_norm == "full":                     # OLMoE: norm over the whole projection
            q, k = attn.q_norm(q), attn.k_norm(k)
        if self.clip_qkv is not None:
            q, k, v = (t.clamp(-self.clip_qkv, self.clip_qkv) for t in (q, k, v))
        q, k, v = q.view(T, H, D), k.view(T, KV, D), v.view(T, KV, D)
        if self.qk_norm == "per_head":                 # Qwen3: norm over each head's head_dim
            q, k = attn.q_norm(q), attn.k_norm(k)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # store this step's keys/values, then attend over each request's full context
        # (read back through the cache, which dequantizes if KV_CACHE_DTYPE is fp8/int8)
        self.cache.write(layer_idx, meta.write_slots, k, v)
        out = torch.empty_like(q)
        gqa = KV != H

        with nvtx_range("attention"):
            for offset, n, start, ctx in meta.prefill:
                qi = q[offset:offset + n].transpose(0, 1)[None]                  # [1, H, n, D]
                kc, vc = self.cache.gather(layer_idx, ctx)                       # [ctx, KV, D]
                ki, vi = kc.transpose(0, 1)[None], vc.transpose(0, 1)[None]      # [1, KV, ctx, D]
                if start == 0:
                    o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True,
                                                       scale=attn.scaling, enable_gqa=gqa)
                else:   # tokens after a cached prefix (prefix caching): position p sees keys 0..p
                    qpos = torch.arange(start, start + n, device=q.device)
                    mask = torch.arange(len(ctx), device=q.device)[None, :] <= qpos[:, None]
                    o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask[None, None],
                                                       scale=attn.scaling, enable_gqa=gqa)
                out[offset:offset + n] = o[0].transpose(0, 1)

            if meta.decode_tables is not None:      # paged decode kernel: reads the cache in place
                table, lens, page, max_len, splits = meta.decode_tables
                c = self.cache
                o = self._paged_decode(
                    q if meta.all_decode else q[meta.decode_rows], c.k[layer_idx], c.v[layer_idx],
                    table, lens, page, attn.scaling, max_len,
                    k_scale=None if c.k_scale is None else c.k_scale[layer_idx],
                    v_scale=None if c.v_scale is None else c.v_scale[layer_idx], num_splits=splits)
                if meta.all_decode:
                    out = o
                else:
                    out[meta.decode_rows] = o
            elif len(meta.decode_rows):
                qd = q[meta.decode_rows][:, :, None, :]                          # [B, H, 1, D]
                kc, vc = self.cache.gather(layer_idx, meta.decode_idx)           # [B, L, KV, D]
                kd, vd = kc.transpose(1, 2), vc.transpose(1, 2)                  # [B, KV, L, D]
                o = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=meta.decode_mask,
                                                   scale=attn.scaling, enable_gqa=gqa)
                out[meta.decode_rows] = o[:, :, 0, :]

        out = out.reshape(T, H * D)
        return f["o"](out) if "o" in f else attn.o_proj(out)