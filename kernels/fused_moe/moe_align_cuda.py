import os
import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "moe_align.cu")
_ext = None
_error = None


def _load():
    global _ext, _error
    if _ext is not None or _error is not None:
        return _ext
    try:
        
        # nvcc and the CUDA headers must come from one toolkit that matches torch.version.cuda
        print(f"Compiling the CUDA moe_align kernel (first time only, ~1 minute) with the CUDA "
              f"toolkit at {CUDA_HOME} (PyTorch built for CUDA {torch.version.cuda})...")
        _ext = load(name="mini_inference_moe_align", sources=[_SOURCE],
                    extra_cuda_cflags=["-O3"], verbose=os.environ.get("VERBOSE_BUILD") == "1")
    except Exception as e:   # no nvcc / ninja / compiler error: fall back to the PyTorch version
        _error = e
        print(f"CUDA moe_align unavailable ({type(e).__name__}: {str(e)[:300]}); "
              f"using the PyTorch moe_align instead. Set VERBOSE_BUILD=1 to see the full build log.")
    return _ext


def available():
    return torch.cuda.is_available() and _load() is not None


def moe_align_cuda(topk_ids, num_experts, block_m):
    """Same contract as fused_moe_triton.moe_align, in one CUDA kernel launch."""
    flat = topk_ids.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    num_pairs = flat.numel()
    max_padded = num_pairs + num_experts * (block_m - 1)
    max_blocks = (max_padded + block_m - 1) // block_m
    dev = flat.device
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=dev)
    num_padded = torch.empty(1, dtype=torch.int32, device=dev)
    _load().moe_align(flat, num_experts, block_m, sorted_ids, expert_ids, num_padded)
    return sorted_ids, expert_ids, num_padded