"""kernels/paged_attention/reference.py - decode attention over a paged cache in plain PyTorch (ground truth)."""
import torch


def _rows(t, slots):
    """t[slots], also for FP8 tensors (indexed as raw bytes)."""
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)[slots].view(torch.float8_e4m3fn)
    return t[slots]


def paged_decode_reference(q, k_cache, v_cache, block_table, seq_lens, page_size, scale,
                           k_scale=None, v_scale=None):
    """Same contract as decode_triton.paged_decode_attention, one request at a time, in float32."""
    B, H, D = q.shape
    group = H // k_cache.shape[1]
    out = torch.empty_like(q)
    for b in range(B):
        n = int(seq_lens[b])
        pos = torch.arange(n, device=q.device)
        slots = block_table[b, pos // page_size].long() * page_size + pos % page_size
        k = _rows(k_cache, slots).float()
        v = _rows(v_cache, slots).float()
        if k_scale is not None:
            k = k * k_scale[slots].float()[..., None]
            v = v * v_scale[slots].float()[..., None]
        k = k.repeat_interleave(group, dim=1)                       # [n, H, D]
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[b].float(), k) * scale
        out[b] = torch.einsum("hn,nhd->hd", scores.softmax(dim=-1), v).to(q.dtype)
    return out