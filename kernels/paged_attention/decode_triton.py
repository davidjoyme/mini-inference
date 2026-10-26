"""
kernels/paged_attention/decode_triton.py - paged decode attention in Triton (split-KV, flash-decoding style).

Decode = one new query token per request, attending to that request's whole cached context.
The engine's default path first GATHERS every request's keys/values into a padded [B, max_len]
copy, then runs SDPA on the copy: the KV cache is read once, written once (the copy), read again,
and padding is computed and thrown away. This kernel reads keys/values IN PLACE from the cache
through each request's page table, so each cached byte is read exactly once.

    page table   block_table[b, i] = the page (cache block) holding tokens i*PAGE .. i*PAGE+PAGE-1
                 of request b; slot of token n = block_table[b, n // PAGE] * PAGE + n % PAGE
                 (the paged cache uses PAGE = KV_BLOCK_SIZE; the contiguous cache passes a
                 token-level table with PAGE = 1, the layout SGLang uses for everything)

    split-KV     one program per (request, query head, context split). With few requests a single
                 program per (request, head) would leave most of the GPU idle (1 x 16 heads = 16
                 programs on a 170-SM GPU), so long contexts are cut into splits that run in
                 parallel, each producing a partial output and its log-sum-exp; a second small kernel
                 merges them (the flash-decoding trick). Splits are chosen to fill the GPU.

    online softmax   each program walks its keys in BLOCK_N tiles, keeping a running max m,
                 running sum l and running output acc, rescaled whenever the max grows: no
                 [heads, context] score matrix is ever materialized.

    8-bit KV     FP8/INT8 caches (KV_CACHE_DTYPE) are dequantized in registers with their
                 per-(token, head) scales, so attention reads half the bytes of bf16.

    GQA          query head h reads kv head h // GROUP (OLMoE has no GQA: GROUP = 1).

Decode is memory-bound (a few FLOPs per byte), so the dot products use plain multiply-and-sum
instead of tensor cores; what matters is reading the cache once, fully coalesced.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_decode_split(
    Q, K, V, KS, VS, BT, LENS, OUT, O_PART, LSE_PART,
    stride_qb, stride_qh,
    stride_kt, stride_kh, stride_vt, stride_vh,
    stride_st, stride_sh,
    stride_bt,
    stride_outb, stride_outh,
    stride_pb, stride_ph, stride_ps,
    stride_lb, stride_lh,
    scale, split_len,
    GROUP: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr, HAS_SCALE: tl.constexpr, DIRECT: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    kvh = h // GROUP

    seq_len = tl.load(LENS + b)
    start = s * split_len
    end = tl.minimum(start + split_len, seq_len)

    d = tl.arange(0, BLOCK_D)
    dmask = d < D
    q = tl.load(Q + b * stride_qb + h * stride_qh + d, mask=dmask, other=0.0).to(tl.float32) * scale

    # running softmax state (0-d tensors so their type stays fixed across loop iterations)
    m = tl.max(tl.full([BLOCK_N], float("-inf"), tl.float32), axis=0)
    l = tl.sum(tl.zeros([BLOCK_N], tl.float32), axis=0)
    acc = tl.zeros([BLOCK_D], tl.float32)

    for n0 in range(start, end, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        nmask = offs < end
        page = tl.load(BT + b * stride_bt + offs // PAGE, mask=nmask, other=0)
        slot = page.to(tl.int64) * PAGE + offs % PAGE
        kvmask = nmask[:, None] & dmask[None, :]

        k = tl.load(K + slot[:, None] * stride_kt + kvh * stride_kh + d[None, :], mask=kvmask, other=0.0)
        k = k.to(tl.float32)
        if HAS_SCALE:
            ks = tl.load(KS + slot * stride_st + kvh * stride_sh, mask=nmask, other=0.0).to(tl.float32)
            k = k * ks[:, None]
        scores = tl.sum(k * q[None, :], axis=1)                       # [BLOCK_N]
        scores = tl.where(nmask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - m_new)                                     # rescale what we had so far
        p = tl.exp(scores - m_new)

        v = tl.load(V + slot[:, None] * stride_vt + kvh * stride_vh + d[None, :], mask=kvmask, other=0.0)
        v = v.to(tl.float32)
        if HAS_SCALE:
            vs = tl.load(VS + slot * stride_st + kvh * stride_sh, mask=nmask, other=0.0).to(tl.float32)
            v = v * vs[:, None]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new

    l_safe = tl.where(l > 0, l, 1.0)                                  # empty split: output 0, lse -inf
    out = acc / l_safe
    if DIRECT:                                                        # one split: no merge needed
        tl.store(OUT + b * stride_outb + h * stride_outh + d, out.to(OUT.dtype.element_ty), mask=dmask)
    else:
        tl.store(O_PART + b * stride_pb + h * stride_ph + s * stride_ps + d, out, mask=dmask)
        tl.store(LSE_PART + b * stride_lb + h * stride_lh + s, m + tl.log(l_safe))


@triton.jit
def _paged_attention_decode_reduce(
    O_PART, LSE_PART, OUT,
    stride_pb, stride_ph, stride_ps,
    stride_lb, stride_lh,
    stride_outb, stride_outh,
    num_splits,
    D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """Merge the splits: out = sum_s exp(lse_s - max) * o_s / sum_s exp(lse_s - max)."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, BLOCK_S)
    smask = s < num_splits
    d = tl.arange(0, BLOCK_D)
    dmask = d < D

    lse = tl.load(LSE_PART + b * stride_lb + h * stride_lh + s, mask=smask, other=float("-inf"))
    w = tl.exp(lse - tl.max(lse, axis=0))                             # empty / padded splits -> 0
    o = tl.load(O_PART + b * stride_pb + h * stride_ph + s[:, None] * stride_ps + d[None, :],
                mask=smask[:, None] & dmask[None, :], other=0.0)
    out = tl.sum(w[:, None] * o, axis=0) / tl.sum(w, axis=0)
    tl.store(OUT + b * stride_outb + h * stride_outh + d, out.to(OUT.dtype.element_ty), mask=dmask)


_SM_COUNT = {}


def _sm_count(device):
    if device.type != "cuda":
        return 4
    if device.index not in _SM_COUNT:
        _SM_COUNT[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT[device.index]


def choose_splits(batch, heads, max_len, block_n, device):
    """Enough (request, head, split) programs to fill the GPU about twice, never more splits than tiles."""
    tiles = triton.cdiv(max_len, block_n)
    want = triton.cdiv(2 * _sm_count(device), batch * heads)
    return max(1, min(tiles, want, 64))


def paged_decode_attention(q, k_cache, v_cache, block_table, seq_lens, page_size, scale, max_len,
                           k_scale=None, v_scale=None, num_splits=None, block_n=64, num_warps=4):
    """
    q           [B, H, D]            one query token per request
    k_cache     [slots, KV, D]       one layer of the cache (bf16, fp32, fp8 or int8)
    v_cache     [slots, KV, D]
    block_table [B, max_pages] int32 page ids per request (PAGE = page_size tokens each)
    seq_lens    [B] int32            context length per request (including the new token)
    max_len     int                  max(seq_lens), passed from the CPU so there's no GPU->CPU sync
    k_scale / v_scale [slots, KV]    per-(token, head) scales for 8-bit caches, else None
    returns     [B, H, D] in q's dtype
    """
    B, H, D = q.shape
    KV = k_cache.shape[1]
    assert H % KV == 0, "query heads must be a multiple of kv heads"
    assert k_cache.stride(2) == 1 and v_cache.stride(2) == 1 and q.stride(2) == 1
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(D)
    if num_splits is None:
        num_splits = choose_splits(B, H, max_len, block_n, q.device)
    split_len = triton.cdiv(triton.cdiv(max_len, num_splits), block_n) * block_n
    num_splits = triton.cdiv(max_len, split_len)
    direct = num_splits == 1

    has_scale = k_scale is not None
    ks = k_scale if has_scale else k_cache          # dummy pointers when unused
    vs = v_scale if has_scale else v_cache
    s_st, s_sh = (k_scale.stride(0), k_scale.stride(1)) if has_scale else (0, 0)
    if has_scale:
        assert v_scale.stride() == k_scale.stride()

    if direct:
        o_part, lse_part = out, out                 # unused
        p_strides, l_strides = (0, 0, 0), (0, 0)
    else:
        o_part = torch.empty(B, H, num_splits, D, dtype=torch.float32, device=q.device)
        lse_part = torch.empty(B, H, num_splits, dtype=torch.float32, device=q.device)
        p_strides, l_strides = o_part.stride()[:3], lse_part.stride()[:2]

    _paged_attention_decode_split[(B, H, num_splits)](
        q, k_cache, v_cache, ks, vs, block_table, seq_lens, out, o_part, lse_part,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1),
        s_st, s_sh,
        block_table.stride(0),
        out.stride(0), out.stride(1),
        *p_strides, *l_strides,
        scale, split_len,
        GROUP=H // KV, PAGE=page_size, D=D, BLOCK_D=block_d, BLOCK_N=block_n,
        HAS_SCALE=has_scale, DIRECT=direct, num_warps=num_warps,
    )
    if not direct:
        _paged_attention_decode_reduce[(B, H)](
            o_part, lse_part, out, *p_strides, *l_strides, out.stride(0), out.stride(1), num_splits,
            D=D, BLOCK_D=block_d, BLOCK_S=triton.next_power_of_2(num_splits), num_warps=4,
        )
    return out