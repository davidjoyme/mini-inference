"""
kernels/fused_moe/fused_moe_triton.py - a Mixture-of-Experts layer in two Triton kernels.

Why: transformers' MoE path on this GPU runs one small matmul per expert, so every layer
launches dozens of tiny kernels and the step is limited by launch overhead, not by the GPU.
Here every expert of a layer is handled by ONE launch per projection.

How (the same idea as vLLM's fused_moe):
  1. moe_align (GPU, no CPU sync): flatten the (token, expert) pairs, sort them by expert, and
     pad each expert's group to a multiple of BLOCK_M. Then every BLOCK_M-row tile belongs to a
     single expert, so a tile is an ordinary matmul with that expert's weights. All sizes are
     fixed upper bounds (not data-dependent), which keeps the step CUDA-graph capturable.
  2. kernel 1: for each tile, gather the tokens' hidden states and multiply by the expert's
     gate and up weights together, apply silu(gate) * up in registers, write the result.
  3. kernel 2: multiply that by the expert's down weights, scale by the routing weight,
     write one row per pair; summing each token's top-k rows gives the layer output.

Tiles past the real number of padded rows exit immediately (the grid is sized for the worst
case so that its shape never depends on the routing).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- 1. group pairs by expert
def moe_align(topk_ids, num_experts, block_m):
    """
    topk_ids [T, k] -> (sorted_ids, expert_ids, num_padded)
      sorted_ids [max_padded]   pair index (token * k + slot) per row, grouped by expert, each
                                expert's group padded to a multiple of block_m with T*k (= "none")
      expert_ids [max_blocks]   which expert each block_m-row tile belongs to
      num_padded [1]            rows actually used (tiles beyond this do nothing)
    Uses only fixed-size GPU ops: no .item()/bincount, so no CPU-GPU sync.
    """
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    num_pairs = flat.numel()
    max_padded = num_pairs + num_experts * (block_m - 1)
    max_blocks = triton.cdiv(max_padded, block_m)

    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block_m - 1) // block_m * block_m
    ends = torch.cumsum(padded, 0)
    starts = ends - padded

    order = torch.argsort(flat, stable=True)                 # pair indices, grouped by expert
    sorted_experts = flat[order]
    first_in_sorted = torch.cumsum(counts, 0) - counts       # where each expert's pairs begin
    rank = torch.arange(num_pairs, device=device) - first_in_sorted[sorted_experts]
    dest = starts[sorted_experts] + rank

    sorted_ids = torch.full((max_padded,), num_pairs, dtype=torch.int32, device=device)
    sorted_ids[dest] = order.to(torch.int32)
    block_starts = torch.arange(max_blocks, device=device) * block_m
    expert_ids = torch.searchsorted(ends, block_starts, right=True).clamp_(max=num_experts - 1)
    return sorted_ids, expert_ids.to(torch.int32), ends[-1:].to(torch.int32)


# ---------------------------------------------------------------- 2./3. the grouped matmul kernel
@triton.jit
def _moe_matmul_kernel(
    a_ptr, w_ptr, c_ptr, topk_w_ptr,
    sorted_ids_ptr, expert_ids_ptr, num_padded_ptr,
    N, K, num_pairs, top_k,
    stride_am, stride_ak, stride_we, stride_wn, stride_wk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    A_ROWS_ARE_TOKENS: tl.constexpr,   # kernel 1: A row = token = pair // top_k; kernel 2: A row = pair
    GATED: tl.constexpr,               # kernel 1: compute gate and up, output silu(gate) * up
    MUL_ROUTING_WEIGHT: tl.constexpr,  # kernel 2: scale each row by its pair's routing weight
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m * BLOCK_M >= tl.load(num_padded_ptr):
        return                                         # tile past the used rows

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ids_ptr + offs_m)
    row_ok = pair < num_pairs                          # padding rows hold num_pairs
    if A_ROWS_ARE_TOKENS:
        a_row = pair // top_k
    else:
        a_row = pair
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + a_row[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
    # weights are stored [expert, out, in]; load a [BLOCK_K, BLOCK_N] tile of W^T
    w_ptrs = w_ptr + expert * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if GATED:
        up_ptrs = w_ptrs + N * stride_wn               # "up" rows follow the N "gate" rows
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ok = offs_k < K - k0
        a = tl.load(a_ptrs, mask=row_ok[:, None] & k_ok[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_ok[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, w)
        if GATED:
            u = tl.load(up_ptrs, mask=k_ok[:, None] & (offs_n[None, :] < N), other=0.0)
            acc_up += tl.dot(a, u)
            up_ptrs += BLOCK_K * stride_wk
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    if GATED:
        acc = acc * tl.sigmoid(acc) * acc_up           # silu(gate) * up
    if MUL_ROUTING_WEIGHT:
        rw = tl.load(topk_w_ptr + pair, mask=row_ok, other=0.0).to(tl.float32)
        acc = acc * rw[:, None]

    c_ptrs = c_ptr + pair[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=row_ok[:, None] & (offs_n[None, :] < N))


def _tile_config(num_pairs):
    """Tile sizes by how many (token, expert) pairs there are. A starting point for tuning:
    small batches (decode) want small BLOCK_M so little of each tile is padding."""
    if num_pairs <= 64:
        return dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
    if num_pairs <= 1024:
        return dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
    return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3)


def fused_moe(hidden, w_gate_up, w_down, topk_weights, topk_ids, config=None):
    """
    hidden [T, H]; w_gate_up [E, 2I, H]; w_down [E, H, I]; topk_weights / topk_ids [T, k]
    -> [T, H], same as moe_reference(...)
    """
    T, H = hidden.shape
    E, two_i, _ = w_gate_up.shape
    inter = two_i // 2
    k = topk_ids.shape[1]
    num_pairs = T * k
    cfg = dict(config or _tile_config(num_pairs))
    bm, bn, bk = cfg.pop("BLOCK_M"), cfg.pop("BLOCK_N"), cfg.pop("BLOCK_K")

    hidden = hidden.contiguous()
    topk_w = topk_weights.reshape(-1).contiguous()
    sorted_ids, expert_ids, num_padded = moe_align(topk_ids, E, bm)
    max_blocks = expert_ids.numel()

    # kernel 1: gate + up projections, silu(gate) * up  ->  [pairs, I]
    inter_out = torch.empty(num_pairs, inter, dtype=hidden.dtype, device=hidden.device)
    _moe_matmul_kernel[(max_blocks, triton.cdiv(inter, bn))](
        hidden, w_gate_up, inter_out, topk_w, sorted_ids, expert_ids, num_padded,
        inter, H, num_pairs, k,
        hidden.stride(0), hidden.stride(1),
        w_gate_up.stride(0), w_gate_up.stride(1), w_gate_up.stride(2),
        inter_out.stride(0), inter_out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        A_ROWS_ARE_TOKENS=True, GATED=True, MUL_ROUTING_WEIGHT=False, **cfg)

    # kernel 2: down projection, scaled by routing weight  ->  [pairs, H]
    pair_out = torch.empty(num_pairs, H, dtype=hidden.dtype, device=hidden.device)
    _moe_matmul_kernel[(max_blocks, triton.cdiv(H, bn))](
        inter_out, w_down, pair_out, topk_w, sorted_ids, expert_ids, num_padded,
        H, inter, num_pairs, k,
        inter_out.stride(0), inter_out.stride(1),
        w_down.stride(0), w_down.stride(1), w_down.stride(2),
        pair_out.stride(0), pair_out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        A_ROWS_ARE_TOKENS=False, GATED=False, MUL_ROUTING_WEIGHT=True, **cfg)

    # each token's k expert outputs -> one row
    return pair_out.view(T, k, H).sum(dim=1, dtype=torch.float32).to(hidden.dtype)