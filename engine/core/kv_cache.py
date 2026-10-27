"""
engine/core/kv_cache.py - the KV cache: one memory pool, two ways of handing it out,
optional quantized storage, optional prefix caching.

Storage (shared by both modes):
    k, v: [num_layers, capacity_tokens, num_kv_heads, head_dim]
Every token of every request lives in one "slot" (a row of that pool). The attention code
only ever sees slot numbers, so it's identical for both modes. The difference is how slots
are assigned:

ContiguousAllocator ("contiguous") - the pre-vLLM way
    On admission a request reserves ONE contiguous range big enough for
    prompt + max_new_tokens, whether or not it ends up using it.
    Costs: reserved-but-unused memory (internal waste), and free space that's split into
    gaps too small for the next request (external fragmentation).

PagedAllocator ("paged") - the PagedAttention way
    Memory is split into fixed-size blocks (e.g. 16 tokens). A request gets a block table
    and receives a new block only when its last one is full. Waste is at most one partly
    filled block per request, and any free block can be used by anyone.
    If blocks run out mid-generation, the scheduler preempts a request (frees its blocks,
    recomputes it later).

Prefix caching (paged only, PREFIX_CACHING=1) - like vLLM's automatic prefix caching
    Every FULL block gets a hash of its tokens chained with the hash of the block before it,
    so a hash identifies the whole prefix up to that block. When a request is admitted, its
    leading blocks are looked up; matching blocks are shared (reference counted) instead of
    recomputed, so the prefill skips those tokens. Finished requests' blocks are not erased:
    they stay cached and are evicted least-recently-used first, only when memory is needed.

KV quantization (KV_CACHE_DTYPE = "fp8" or "int8")
    Values are stored in 8 bits with one scale per (token, head), computed when the token is
    written: scale = max|x| / QMAX over the head's values. Attention reads them back
    dequantized to the model's dtype. About half the memory of bf16, so ~2x the tokens.
"""
import math
from collections import OrderedDict, deque

import torch

# ---------------------------------------------------------------- KV storage formats
KV_FORMATS = {
    # name: (storage dtype, largest representable magnitude)
    "fp8": (torch.float8_e4m3fn, torch.finfo(torch.float8_e4m3fn).max),   # 448
    "int8": (torch.int8, 127.0),
}


def resolve_kv_dtype(kv_dtype):
    """'auto'/'bf16'/None -> None (store in the model's dtype); else the format name."""
    if kv_dtype in (None, "", "auto", "bf16", "fp16", "none"):
        return None
    if kv_dtype not in KV_FORMATS:
        raise ValueError(f"Unknown KV_CACHE_DTYPE '{kv_dtype}' (use auto, fp8 or int8)")
    return kv_dtype


def kv_bytes_per_token(num_layers, num_kv_heads, head_dim, model_dtype, kv_dtype=None):
    """Bytes of KV cache per token: keys + values, all layers, plus scales if quantized."""
    kv_dtype = resolve_kv_dtype(kv_dtype)
    scale_bytes = torch.tensor([], dtype=model_dtype).element_size()
    if kv_dtype is None:
        per_head = head_dim * scale_bytes
    else:
        per_head = head_dim * torch.tensor([], dtype=KV_FORMATS[kv_dtype][0]).element_size() + scale_bytes
    return 2 * num_layers * num_kv_heads * per_head


def tokens_for_budget(budget_bytes, num_layers, num_kv_heads, head_dim, dtype, kv_dtype=None):
    return int(budget_bytes // kv_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype, kv_dtype))


# ---------------------------------------------------------------- allocators
class _Allocator:
    def __init__(self, capacity):
        self.capacity = capacity
        self.length = {}          # request id -> tokens stored so far

    def stored_tokens(self):
        return sum(self.length.values())

    def register(self, req):
        """Called after each step; only the paged allocator with prefix caching uses it."""


class ContiguousAllocator(_Allocator):
    name = "contiguous"

    def __init__(self, capacity):
        super().__init__(capacity)
        self.free_ranges = [(0, capacity)]   # sorted list of (start, size)
        self.ranges = {}                     # request id -> (start, size)

    def _find(self, size):
        for i, (start, n) in enumerate(self.free_ranges):   # first fit
            if n >= size:
                return i
        return None

    def can_admit(self, req, max_len):
        return self._find(min(req.max_total_len, max_len)) is not None

    def admit(self, req, max_len, size=None):
        """Returns the number of tokens already cached (always 0 here), or None if it doesn't fit."""
        size = size or min(req.max_total_len, max_len)
        i = self._find(size)
        if i is None:
            return None
        start, n = self.free_ranges[i]
        if n == size:
            self.free_ranges.pop(i)
        else:
            self.free_ranges[i] = (start + size, n - size)
        self.ranges[req.id] = (start, size)
        self.length[req.id] = 0
        return 0

    def reserve(self, req, num_tokens):
        """Contiguous: the whole reservation was made at admission."""
        return num_tokens <= self.ranges[req.id][1]

    def ensure(self, req, num_tokens):
        """Room for num_tokens? Always, up to the reservation made at admission."""
        if num_tokens > self.ranges[req.id][1]:
            return False
        self.length[req.id] = num_tokens
        return True

    def free(self, req):
        if req.id not in self.ranges:
            return
        start, size = self.ranges.pop(req.id)
        self.length.pop(req.id, None)
        self.free_ranges.append((start, size))
        self.free_ranges.sort()
        merged = []
        for s, n in self.free_ranges:                       # merge neighbours
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((s, n))
        self.free_ranges = merged

    def slots(self, req, start, end):
        base = self.ranges[req.id][0]
        return torch.arange(base + start, base + end)

    def stats(self):
        reserved = sum(n for _, n in self.ranges.values())
        free = sum(n for _, n in self.free_ranges)
        largest = max((n for _, n in self.free_ranges), default=0)
        return {
            "kv_reserved_tokens": reserved,
            "kv_free_tokens": free,
            # share of free memory that can't be used for a request needing the largest gap
            "kv_fragmentation_pct": round((1 - largest / free) * 100, 1) if free else 0.0,
        }


class PagedAllocator(_Allocator):
    name = "paged"

    def __init__(self, capacity, block_size, watermark=0.01, prefix_caching=False):
        super().__init__(capacity)
        self.block_size = block_size
        self.num_blocks = capacity // block_size
        # blocks kept free at admission so running requests can grow without preemption
        self.watermark_blocks = max(1, math.ceil(self.num_blocks * watermark))
        self.free_blocks = deque(range(self.num_blocks))
        self.tables = {}                     # request id -> list of block ids (the block table)
        self.ref = [0] * self.num_blocks     # how many requests use each block

        # prefix caching
        self.prefix_caching = prefix_caching
        self.hash_to_block = {}              # prefix hash -> block id holding those tokens
        self.block_hash = {}                 # block id -> its prefix hash (registered blocks)
        self.evictable = OrderedDict()       # cached blocks nobody uses, least recently used first
        self.req_hashes = {}                 # request id -> hashes of its full blocks so far
        self.hit_tokens = 0                  # tokens served from the prefix cache
        self.query_tokens = 0                # tokens looked up at admission
        self.evictions = 0

    def blocks_for(self, num_tokens):
        return math.ceil(num_tokens / self.block_size)

    def _available(self):
        return len(self.free_blocks) + len(self.evictable)

    def _take_block(self):
        """A free block, or else the least recently used cached block (evicting its contents)."""
        if self.free_blocks:
            b = self.free_blocks.popleft()
        elif self.evictable:
            b, _ = self.evictable.popitem(last=False)
            del self.hash_to_block[self.block_hash.pop(b)]
            self.evictions += 1
        else:
            return None
        self.ref[b] = 1
        return b

    # ---------------------------------------------------------------- prefix matching
    def _block_hashes(self, token_ids, num_blocks, parent=None):
        hashes = []
        bs = self.block_size
        for i in range(num_blocks):
            parent = hash((parent, tuple(token_ids[i * bs:(i + 1) * bs])))
            hashes.append(parent)
        return hashes

    def _match(self, req):
        """Longest run of leading full blocks already in the cache -> (hashes, block ids)."""
        if not self.prefix_caching:
            return [], []
        ids = req.all_ids
        # at least one token must be computed this step (its logits give the next token)
        max_blocks = (len(ids) - 1) // self.block_size
        hashes, blocks = [], []
        for h in self._block_hashes(ids, max_blocks):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hashes.append(h)
            blocks.append(b)
        return hashes, blocks

    # ---------------------------------------------------------------- admission and growth
    def can_admit(self, req, max_len):
        _, hit = self._match(req)
        needed = self.blocks_for(req.total_len) - len(hit)
        # hit blocks sitting in the evictable list stop being available once we use them
        reused_idle = sum(1 for b in hit if self.ref[b] == 0)
        return needed + self.watermark_blocks <= self._available() - reused_idle

    def admit(self, req, max_len, size=None):
        """Returns the number of leading tokens already cached, or None if it doesn't fit."""
        if not self.can_admit(req, max_len):
            return None
        hashes, hit = self._match(req)
        for b in hit:
            if self.ref[b] == 0:
                self.evictable.pop(b)
            self.ref[b] += 1
        self.tables[req.id] = list(hit)
        self.req_hashes[req.id] = hashes
        self.length[req.id] = 0
        if self.prefix_caching:
            self.query_tokens += req.total_len
            self.hit_tokens += len(hit) * self.block_size
        return len(hit) * self.block_size

    def reserve(self, req, num_tokens):
        """Take all blocks needed for num_tokens now (static batching's worst-case reservation)."""
        table = self.tables[req.id]
        while len(table) < self.blocks_for(num_tokens):
            b = self._take_block()
            if b is None:
                return False
            table.append(b)
        return True

    def ensure(self, req, num_tokens):
        """Hand out blocks until num_tokens fit. False if the pool is empty."""
        if not self.reserve(req, num_tokens):
            return False
        self.length[req.id] = num_tokens
        return True

    def register(self, req):
        """After a step: publish this request's newly completed blocks to the prefix cache."""
        if not self.prefix_caching or req.id not in self.tables:
            return
        known = self.req_hashes[req.id]
        full = req.num_computed // self.block_size
        if full <= len(known):
            return
        ids = req.all_ids
        parent = known[-1] if known else None
        table = self.tables[req.id]
        for i, h in enumerate(self._block_hashes(ids[len(known) * self.block_size:],
                                                 full - len(known), parent), start=len(known)):
            known.append(h)
            b = table[i]
            if h not in self.hash_to_block and b not in self.block_hash:
                self.hash_to_block[h] = b
                self.block_hash[b] = h

    def free(self, req):
        for b in self.tables.pop(req.id, []):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                if b in self.block_hash:
                    self.evictable[b] = None        # keep its contents: newest at the end
                else:
                    self.free_blocks.append(b)
        self.req_hashes.pop(req.id, None)
        self.length.pop(req.id, None)

    def reserve_scratch_block(self):
        """Take one block out of circulation for good (CUDA graphs' padding rows write here)."""
        block = self.free_blocks.popleft()
        self.ref[block] = 1
        return block

    def slots(self, req, start, end):
        positions = torch.arange(start, end)
        table = torch.tensor(self.tables[req.id], dtype=torch.long)
        return table[positions // self.block_size] * self.block_size + positions % self.block_size

    def stored_tokens(self):
        """Physical tokens in use: a block shared by several requests counts once."""
        fill = {}
        for rid, table in self.tables.items():
            n = self.length.get(rid, 0)
            for i, b in enumerate(table):
                fill[b] = max(fill.get(b, 0), min(self.block_size, max(n - i * self.block_size, 0)))
        return sum(fill.values())

    def stats(self):
        in_use = sum(1 for r in self.ref if r > 0)
        s = {
            "kv_reserved_tokens": in_use * self.block_size,
            "kv_free_tokens": self._available() * self.block_size,
            "kv_fragmentation_pct": 0.0,     # any free block fits anyone
            "kv_blocks_used": in_use,
            "kv_blocks_total": self.num_blocks,
        }
        if self.prefix_caching:
            s.update({
                "prefix_cached_blocks": len(self.evictable),   # cached, not in use right now
                "prefix_shared_blocks": sum(1 for r in self.ref if r > 1),
                "prefix_hit_tokens": self.hit_tokens,
                "prefix_query_tokens": self.query_tokens,
                "prefix_hit_pct": round(self.hit_tokens / self.query_tokens * 100, 1) if self.query_tokens else 0.0,
                "prefix_evictions": self.evictions,
            })
        return s


# ---------------------------------------------------------------- the cache
class KVCache:
    def __init__(self, mode, num_layers, num_kv_heads, head_dim, capacity_tokens, dtype, device,
                 block_size=16, watermark=0.01, kv_dtype=None, prefix_caching=False):
        if mode == "paged":
            capacity_tokens = (capacity_tokens // block_size) * block_size
            self.allocator = PagedAllocator(capacity_tokens, block_size, watermark, prefix_caching)
        elif mode == "contiguous":
            if prefix_caching:
                print("Note: prefix caching needs KV_CACHE=paged; it is off for the contiguous cache.")
            self.allocator = ContiguousAllocator(capacity_tokens)
        else:
            raise ValueError(f"Unknown KV_CACHE '{mode}' (use 'contiguous' or 'paged')")
        self.mode = mode
        self.capacity = capacity_tokens
        self.model_dtype = dtype
        self.kv_dtype = resolve_kv_dtype(kv_dtype)
        self.prefix_caching = mode == "paged" and prefix_caching

        shape = (num_layers, capacity_tokens, num_kv_heads, head_dim)
        if self.kv_dtype is None:
            self.k = torch.zeros(shape, dtype=dtype, device=device)
            self.v = torch.zeros(shape, dtype=dtype, device=device)
            self.k_scale = self.v_scale = None
        else:
            store, self.qmax = KV_FORMATS[self.kv_dtype]
            self.k = torch.zeros(shape, dtype=store, device=device)
            self.v = torch.zeros(shape, dtype=store, device=device)
            self.k_scale = torch.ones(shape[:3], dtype=dtype, device=device)   # one per (token, head)
            self.v_scale = torch.ones(shape[:3], dtype=dtype, device=device)
        self.bytes_per_token = kv_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype, self.kv_dtype)

    # thin wrappers so the scheduler doesn't care which allocator is in use
    def can_admit(self, req, max_len):
        return self.allocator.can_admit(req, max_len)

    def admit(self, req, max_len, size=None):
        return self.allocator.admit(req, max_len, size)

    def reserve(self, req, num_tokens):
        return self.allocator.reserve(req, num_tokens)

    def ensure(self, req, num_tokens):
        return self.allocator.ensure(req, num_tokens)

    def free(self, req):
        self.allocator.free(req)

    def register(self, req):
        self.allocator.register(req)

    def slots(self, req, start, end):
        return self.allocator.slots(req, start, end)

    # ---------------------------------------------------------------- reading and writing
    def _quantize(self, x):
        """x [T, heads, dim] -> (8-bit values, per-(token, head) scale in the model dtype)."""
        x = x.float()
        # round the scale to its storage dtype first, so quantize and dequantize use the same scale
        scale = (x.abs().amax(dim=-1) / self.qmax).clamp(min=1e-8).to(self.model_dtype)
        q = x / scale.float()[..., None]
        if self.kv_dtype == "int8":
            q = q.round().clamp(-127, 127)
        else:
            q = q.clamp(-self.qmax, self.qmax)
        return q.to(self.k.dtype), scale

    @staticmethod
    def _copy_rows(dest, slots, src):
        if dest.dtype == torch.float8_e4m3fn:     # copy fp8 as raw bytes
            dest.view(torch.uint8).index_copy_(0, slots, src.view(torch.uint8))
        else:
            dest.index_copy_(0, slots, src)

    def write(self, layer, slots, k, v):
        if self.kv_dtype is None:
            self.k[layer].index_copy_(0, slots, k)
            self.v[layer].index_copy_(0, slots, v)
            return
        qk, sk = self._quantize(k)
        qv, sv = self._quantize(v)
        self._copy_rows(self.k[layer], slots, qk)
        self._copy_rows(self.v[layer], slots, qv)
        self.k_scale[layer].index_copy_(0, slots, sk)
        self.v_scale[layer].index_copy_(0, slots, sv)

    def gather(self, layer, idx):
        """Keys and values at the given slots (any index shape), in the model dtype."""
        if self.kv_dtype is None:
            return self.k[layer][idx], self.v[layer][idx]
        if self.k.dtype == torch.float8_e4m3fn:
            k = self.k[layer].view(torch.uint8)[idx].view(torch.float8_e4m3fn)
            v = self.v[layer].view(torch.uint8)[idx].view(torch.float8_e4m3fn)
        else:
            k, v = self.k[layer][idx], self.v[layer][idx]
        k = k.to(self.model_dtype) * self.k_scale[layer][idx][..., None]
        v = v.to(self.model_dtype) * self.v_scale[layer][idx][..., None]
        return k, v

    def decode_tables(self, entries, device):
        """Page tables for the paged decode-attention kernel (ATTN_BACKEND=triton).
        entries: list of (request, context_length). Returns (block_table int32 [B, pages],
        seq_lens int32 [B], page_size, max_len). The paged cache hands over its block tables
        as they are (PAGE = block size); a contiguous region is described token by token (PAGE = 1)."""
        lens = [n for _, n in entries]
        max_len = max(lens)
        if self.mode == "paged":
            bs = self.allocator.block_size
            tables = [self.allocator.tables[r.id][: -(-n // bs)] for r, n in entries]
            width = max(len(t) for t in tables)
            table = torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32)
            page = bs
        else:
            bases = torch.tensor([self.allocator.ranges[r.id][0] for r, _ in entries], dtype=torch.int32)
            table = bases[:, None] + torch.arange(max_len, dtype=torch.int32)[None, :]
            page = 1
        return table.to(device), torch.tensor(lens, dtype=torch.int32).to(device), page, max_len

    def memory_mb(self):
        total = sum(t.numel() * t.element_size() for t in (self.k, self.v, self.k_scale, self.v_scale)
                    if t is not None)
        return round(total / 1024 ** 2, 1)

    def stats(self):
        used = self.allocator.stored_tokens()
        s = {"kv_mode": self.mode, "kv_cache_dtype": self.kv_dtype or str(self.model_dtype).replace("torch.", ""),
             "kv_capacity_tokens": self.capacity, "kv_used_tokens": used,
             "kv_cache_mb": self.memory_mb(), **self.allocator.stats()}
        s["kv_used_pct"] = round(used / self.capacity * 100, 1)
        s["kv_reserved_pct"] = round(s["kv_reserved_tokens"] / self.capacity * 100, 1)
        # of the memory handed out, how much holds nothing (reserved but unused)
        s["kv_waste_pct"] = (round(max(1 - used / s["kv_reserved_tokens"], 0) * 100, 1)
                             if s["kv_reserved_tokens"] else 0.0)
        return s