"""
kernels/fused_moe - Mixture-of-Experts layer in two Triton kernels.

    reference.py         plain PyTorch, one expert at a time (the ground truth)
    fused_moe_triton.py  moe_align (group pairs by expert, on the GPU) + the two kernels
    module.py            FusedExperts: drop-in replacement for transformers' experts module

Enable with MOE_KERNEL=triton (config.py).
"""
from kernels.fused_moe.module import FusedExperts, apply_fused_moe  # noqa: F401