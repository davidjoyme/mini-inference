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