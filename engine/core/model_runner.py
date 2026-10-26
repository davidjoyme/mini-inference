"""
engine/core/model_runner.py - one forward pass over a whole batch of requests.

The batch is "flat" (no padding): the new tokens of every scheduled request are joined
into one sequence of T tokens, each with its own position. Layers that work token by token
(norms, projections, the MoE experts) process all T tokens together - that's where batching
pays off, since the weights are read once per step for everyone. Only attention has to keep
requests apart; it reads each request's keys/values from the KV cache through slot numbers:

    prefill chunk (a new or recomputed prompt): causal attention over its own tokens
    decode tokens (1 new token per request): batched into ONE attention call, each request
    attending to its own cached context (padded to the longest, with a mask)

This file reuses the transformers model's modules and weights (so quantized layers work
as-is) but replaces its attention and cache handling. Written for OLMoE
(q_norm/k_norm on the full projection, rotary embeddings, sparse MoE MLP).
"""
import torch
import torch.nn.functional as F

from profiling.nvtx import nvtx_range


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class BatchMetadata:
    """Everything attention needs for this step, built once and reused by every layer."""

    def __init__(self, chunks, cache, device):
        # chunks: list of (request, start_position, end_position) for the tokens fed this step
        ids, positions, write_slots = [], [], []
        self.prefill, self.decode_rows, decode_ctx = [], [], []
        self.last_rows = []
        self.context_tokens = 0     # keys/values attention reads this step (for bandwidth estimates)
        offset = 0
        for req, start, end in chunks:
            n = end - start
            ids.extend(req.all_ids[start:end])
            positions.append(torch.arange(start, end))
            write_slots.append(cache.slots(req, start, end))
            ctx = cache.slots(req, 0, end)
            self.context_tokens += end
            if n == 1 and start > 0:                       # decode
                self.decode_rows.append(offset)
                decode_ctx.append(ctx)
            else:                                          # prefill (or recompute)
                self.prefill.append((offset, n, start, ctx.to(device)))
            offset += n
            self.last_rows.append(offset - 1)

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
    def __init__(self, model, cache):
        self.model = model
        self.cache = cache
        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", None) or self.num_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.num_heads
        self.clip_qkv = getattr(cfg, "clip_qkv", None)
        self.device = next(model.parameters()).device
        inner = model.model
        self.embed, self.layers, self.norm, self.rotary = (inner.embed_tokens, inner.layers,
                                                           inner.norm, inner.rotary_emb)

    @torch.inference_mode()
    def forward(self, chunks, all_logits=False):
        """Run one step. Returns logits [num_chunks, vocab] for the last token of each chunk,
        or [num_tokens, vocab] for every token when all_logits=True (used for perplexity)."""
        with nvtx_range("prepare_batch"):                # Python bookkeeping before the GPU work
            meta = BatchMetadata(chunks, self.cache, self.device)
        # profiling/analyze.py parses this label: tokens, decode rows, prefill chunks, context read
        with nvtx_range(f"forward(T={meta.num_tokens}, decode={len(meta.decode_rows)}, "
                        f"prefill={len(meta.prefill)}, ctx={meta.context_tokens})"):
            x = self.embed(meta.input_ids)                                       # [T, hidden]
            cos, sin = self.rotary(x, meta.positions[None])
            cos, sin = cos[0][:, None, :], sin[0][:, None, :]                    # [T, 1, D]

            for i, layer in enumerate(self.layers):
                x = x + self._attention(i, layer.self_attn, layer.input_layernorm(x), cos, sin, meta)
                h = layer.post_attention_layernorm(x)
                with nvtx_range("moe"):
                    x = x + layer.mlp(h[None])[0]

            x = self.norm(x if all_logits else x[meta.last_rows])
            return self.model.lm_head(x).float()

    def _attention(self, layer_idx, attn, h, cos, sin, meta):
        T, H, KV, D = meta.num_tokens, self.num_heads, self.num_kv_heads, self.head_dim
        q = attn.q_norm(attn.q_proj(h))
        k = attn.k_norm(attn.k_proj(h))
        v = attn.v_proj(h)
        if self.clip_qkv is not None:
            q, k, v = (t.clamp(-self.clip_qkv, self.clip_qkv) for t in (q, k, v))
        q, k, v = q.view(T, H, D), k.view(T, KV, D), v.view(T, KV, D)
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

            if len(meta.decode_rows):
                qd = q[meta.decode_rows][:, :, None, :]                          # [B, H, 1, D]
                kc, vc = self.cache.gather(layer_idx, meta.decode_idx)           # [B, L, KV, D]
                kd, vd = kc.transpose(1, 2), vc.transpose(1, 2)                  # [B, KV, L, D]
                o = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=meta.decode_mask,
                                                   scale=attn.scaling, enable_gqa=gqa)
                out[meta.decode_rows] = o[:, :, 0, :]

        return attn.o_proj(out.reshape(T, H * D))