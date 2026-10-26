"""
kernels/fused_moe/module.py - put the fused MoE kernels into a loaded model.

FusedExperts replaces transformers' experts module (called as experts(hidden, top_k_index,
top_k_weights) after the router has picked each token's experts). It shares the original weight
tensors - nothing is copied - so memory use doesn't change. The router and everything else in
the model stay exactly as they were, which is why both engines (hf and custom) can use it.

Only applied where the kernels match the math: unquantized experts in OLMoE's layout
(gate_up_proj [E, 2I, H], down_proj [E, H, I]) with a SiLU activation. Quantized experts
(QuantExperts) keep their own path.
"""
import os

import torch
from torch import nn

from profiling.nvtx import nvtx_range


def _triton_usable(t):
    return t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


class FusedExperts(nn.Module):
    def __init__(self, experts, align="auto"):
        super().__init__()
        self.align = align                             # how pairs are grouped: auto / cuda / torch
        self.gate_up_proj = experts.gate_up_proj       # shared Parameters, not copies
        self.down_proj = experts.down_proj
        self.num_experts = experts.gate_up_proj.shape[0]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        with nvtx_range("fused_moe"):
            if _triton_usable(hidden_states):
                from kernels.fused_moe.fused_moe_triton import fused_moe
                return fused_moe(hidden_states, self.gate_up_proj, self.down_proj,
                                 top_k_weights, top_k_index, align=self.align)
            from kernels.fused_moe.reference import moe_reference   # CPU without the interpreter
            return moe_reference(hidden_states, self.gate_up_proj, self.down_proj,
                                 top_k_weights, top_k_index)

    def extra_repr(self):
        return f"num_experts={self.num_experts}, kernel=triton fused_moe, align={self.align}"


def _eligible(module):
    gu, dn = getattr(module, "gate_up_proj", None), getattr(module, "down_proj", None)
    if not (isinstance(gu, nn.Parameter) and isinstance(dn, nn.Parameter) and gu.dim() == 3 and dn.dim() == 3):
        return False
    E, two_i, h = gu.shape
    if dn.shape != (E, h, two_i // 2):                 # OLMoE layout only
        return False
    if getattr(module, "has_bias", False) or getattr(module, "gate_up_proj_bias", None) is not None:
        return False
    act = getattr(module, "act_fn", None)
    return act is not None and "silu" in type(act).__name__.lower()


def apply_fused_moe(model, align="auto"):
    """Swap every eligible experts module for FusedExperts. Returns how many were swapped."""
    swapped = 0
    for name, module in list(model.named_modules()):
        if name.endswith(".experts") and _eligible(module):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            parent.experts = FusedExperts(module, align=align)
            swapped += 1
    return swapped


def align_backend(align="auto"):
    """Which moe_align will actually run on this machine: 'cuda' or 'torch'."""
    if align == "torch" or not torch.cuda.is_available():
        return "torch"
    from kernels.fused_moe import moe_align_cuda
    return "cuda" if moe_align_cuda.available() else "torch"