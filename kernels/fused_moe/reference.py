"""
kernels/fused_moe/reference.py - the MoE experts computation in plain PyTorch, one expert at a
time. Slow, but obviously correct: tests compare the Triton kernels against it.

Weights use transformers' OLMoE layout:
    w_gate_up [E, 2*I, H]   first I output rows = gate, next I = up
    w_down    [E, H, I]
For each token and each of its top-k experts e (routing weight w):
    out += w * W_down[e] @ (silu(W_gate[e] @ x) * (W_up[e] @ x))
"""
import torch
import torch.nn.functional as F


def moe_reference(hidden, w_gate_up, w_down, topk_weights, topk_ids):
    T, H = hidden.shape
    inter = w_down.shape[2]
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
    for e in topk_ids.unique().tolist():
        token, slot = torch.where(topk_ids == e)
        x = hidden[token].float()
        gate_up = x @ w_gate_up[e].float().t()
        h = F.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        y = h @ w_down[e].float().t()
        out.index_add_(0, token, y * topk_weights[token, slot].float()[:, None])
    return out.to(hidden.dtype)