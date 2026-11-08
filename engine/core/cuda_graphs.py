"""
engine/core/cuda_graphs.py - replay decode steps as CUDA graphs (CUDA_GRAPHS=1).

Why: a decode step launches hundreds of small kernels (16 layers x attention, MoE, norms...),
and at small batch sizes the CPU takes longer to LAUNCH them than the GPU takes to RUN them, so
the GPU sits idle between kernels. A CUDA graph records the whole step's kernels once; replaying
it is a single launch, with no Python and no per-kernel launch cost.

The catch: a graph replays exactly what was recorded - the same kernels, grid sizes and memory
addresses. So every step must look identical to the GPU:
    fixed buffers   input ids, positions, KV write slots, page tables and context lengths live in
                    buffers allocated once; each step copies its values INTO them
    batch buckets   one graph per batch size in BUCKETS; a batch of 5 runs the graph for 8, the
                    extra rows are padding (token 0 at position 0, context length 1)
    scratch block   padding rows write their keys/values into one KV block reserved for that and
                    never given to a request, so they can't overwrite real data
    fixed shapes    the paged attention kernel gets a split count fixed per bucket and a page-table
                    width covering the model's whole context, so its grid never changes; the fused
                    MoE kernels pick tile sizes from the token count, which is fixed per bucket
Only pure-decode steps use graphs. Steps with prompt tokens (prefill, chunked prefill, recompute)
have shapes that change every time and run normally - they're compute-heavy, so launch overhead
matters little there.

Requirements: CUDA GPU, KV_CACHE=paged, ATTN_BACKEND=triton (the SDPA path builds new index and
mask tensors every step, which a graph can't follow). MOE_KERNEL=triton is recommended: the default
experts path may synchronize with the CPU, which a graph can't capture (we fall back if it fails).

On a CPU (tests), `emulate=True` runs the same padded fixed-buffer forward pass without recording a
graph, so the bookkeeping - buckets, padding, scratch block, stale rows - is checked exactly.
"""
import time
import torch
from profiling.nvtx import nvtx_range

BUCKETS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]


class _StaticMeta:
    """Stands in for BatchMetadata: the same fields, pointing at slices of the fixed buffers."""

    def __init__(self, g, n, num_splits):
        self.num_tokens = n
        self.input_ids = g.input_ids[:n]
        self.positions = g.positions[:n]
        self.write_slots = g.write_slots[:n]
        self.prefill = []
        self.all_decode = True
        self.decode_rows = None
        self.last_rows = None
        self.decode_tables = (g.block_table[:n], g.seq_lens[:n], g.page, None, num_splits)


class DecodeGraphs:
    def __init__(self, runner, max_batch, max_model_len, emulate=False):
        from kernels.paged_attention.decode_triton import choose_splits
        cache = runner.cache
        if cache.mode != "paged":
            raise ValueError("CUDA graphs need KV_CACHE=paged")
        if runner.attn_backend != "triton":
            raise ValueError("CUDA graphs need ATTN_BACKEND=triton")
        self.runner, self.cache, self.emulate = runner, cache, emulate
        self.page = cache.allocator.block_size
        self.buckets = [b for b in BUCKETS if b < max_batch] + [max_batch]
        self.max_batch = max_batch
        self.width = -(-max_model_len // self.page)           # pages covering the longest context
        dev = runner.device

        self.scratch_page = cache.allocator.reserve_scratch_block()
        self.scratch_slot = self.scratch_page * self.page

        # the fixed buffers every graph reads (padding values by default)
        self.input_ids = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.positions = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.write_slots = torch.full((max_batch,), self.scratch_slot, dtype=torch.long, device=dev)
        self.block_table = torch.full((max_batch, self.width), self.scratch_page, dtype=torch.int32, device=dev)
        self.seq_lens = torch.ones(max_batch, dtype=torch.int32, device=dev)
        # pinned CPU staging buffers: one host->device copy per buffer per step, without a sync
        pin = dev.type == "cuda"
        self.h_ids = torch.zeros(max_batch, dtype=torch.long, pin_memory=pin)
        self.h_pos = torch.zeros(max_batch, dtype=torch.long, pin_memory=pin)
        self.h_slots = torch.zeros(max_batch, dtype=torch.long, pin_memory=pin)
        self.h_lens = torch.zeros(max_batch, dtype=torch.int32, pin_memory=pin)
        self.h_table = torch.zeros(max_batch, self.width, dtype=torch.int32, pin_memory=pin)

        self.splits = {b: choose_splits(b, runner.num_heads, max_model_len, 64, dev) for b in self.buckets}
        self.graphs, self.outputs = {}, {}
        self.replays = self.eager_steps = 0
        self.capture_s = 0.0

    # ------------------------------------------------------------------ capture
    def capture(self):
        """Record one graph per bucket, largest first, all sharing one memory pool."""
        if self.emulate:
            return
        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode():
            for b in reversed(self.buckets):
                meta = _StaticMeta(self, b, self.splits[b])
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):              # warm up: compile Triton kernels, allocate
                    for _ in range(2):
                        self.runner.forward_meta(meta)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self.outputs[b] = self.runner.forward_meta(meta)
                self.graphs[b] = graph
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - start

    # ------------------------------------------------------------------ replay
    def run(self, chunks):
        """Logits [len(chunks), vocab] for a pure-decode step, or None to run it normally."""
        n = len(chunks)
        if n > self.max_batch or any(end - start != 1 or start == 0 for _, start, end in chunks):
            self.eager_steps += 1             # prompt tokens in this step: run it normally
            return None
        b = next(x for x in self.buckets if x >= n)
        with nvtx_range("prepare_batch"):
            alloc, page, pad = self.cache.allocator, self.page, b - n
            ids, pos, slots, lens, rows = [], [], [], [], []
            for req, start, end in chunks:
                table = alloc.tables[req.id]
                ids.append(req.all_ids[start])
                pos.append(start)
                slots.append(table[start // page] * page + start % page)
                lens.append(end)
                rows.append(table[: -(-end // page)])
            ctx = sum(lens)
            width = max(len(r) for r in rows)
            # padding rows (n..b) get harmless values: token 0 at position 0, context 1, scratch block
            # (set every step, since an earlier, larger batch may have left real values there)
            ids += [0] * pad
            pos += [0] * pad
            slots += [self.scratch_slot] * pad
            lens += [1] * pad
            rows += [[self.scratch_page]] * pad
            self.h_ids[:b] = torch.tensor(ids)
            self.h_pos[:b] = torch.tensor(pos)
            self.h_slots[:b] = torch.tensor(slots)
            self.h_lens[:b] = torch.tensor(lens, dtype=torch.int32)
            self.h_table[:b, :width] = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.int32)
            # Asynchronous copies into the graph's buffers. Reusing the pinned buffers next step is
            # safe: the engine waits for this step's GPU work (sampling .tolist()) before the next.
            nb = self.runner.device.type == "cuda"
            self.input_ids[:b].copy_(self.h_ids[:b], non_blocking=nb)
            self.positions[:b].copy_(self.h_pos[:b], non_blocking=nb)
            self.write_slots[:b].copy_(self.h_slots[:b], non_blocking=nb)
            self.seq_lens[:b].copy_(self.h_lens[:b], non_blocking=nb)
            self.block_table[:b, :width].copy_(self.h_table[:b, :width], non_blocking=nb)
        # same label format as the normal path, so profiling/analyze.py still counts decode steps
        with nvtx_range(f"forward(T={n}, decode={n}, prefill=0, ctx={ctx}) graph[{b}]"):
            if self.emulate:
                with torch.inference_mode():
                    logits = self.runner.forward_meta(_StaticMeta(self, b, self.splits[b]))
            else:
                self.graphs[b].replay()
                logits = self.outputs[b]
        self.replays += 1
        # a view into the graph's output buffer: valid until the next replay of this bucket
        # (the engine samples from it before running the next step)
        return logits[:n]

    def stats(self):
        return {"cuda_graphs": not self.emulate, "graph_buckets": len(self.buckets),
                "graph_capture_s": round(self.capture_s, 1), "graph_replays": self.replays,
                "graph_eager_steps": self.eager_steps}