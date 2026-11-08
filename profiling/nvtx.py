"""
profiling/nvtx.py - label code regions so they show up in profiler timelines.

    with nvtx_range("decode"):
        ...

The same label goes to both profilers:
  - Nsight Systems (NVTX ranges), when NVTX is enabled in config (set NVTX=1)
  - the PyTorch profiler (record_function), while profiling/torch_profiler.py is recording;
    profiling/analyze.py uses these labels to split time into attention, MoE, scheduling, ...
Costs nothing when neither is on.
"""
import contextlib
import torch
import config

# set by profiling.torch_profiler while the PyTorch profiler is recording
TORCH_PROFILER_ACTIVE = False


@contextlib.contextmanager
def nvtx_range(name: str):
    use_nvtx = config.NVTX_ENABLED and torch.cuda.is_available()
    if not use_nvtx and not TORCH_PROFILER_ACTIVE:
        yield
        return
    with contextlib.ExitStack() as stack:
        if TORCH_PROFILER_ACTIVE:
            stack.enter_context(torch.profiler.record_function(name))
        if use_nvtx:
            torch.cuda.nvtx.range_push(name)
            stack.callback(torch.cuda.nvtx.range_pop)
        yield