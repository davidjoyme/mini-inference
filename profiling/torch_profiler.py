"""
profiling/torch_profiler.py - run code under the PyTorch profiler.

    with torch_profile("my_run") as result:
        ...code to profile...
    result["trace_path"]   # the saved trace

Prints the top operations by GPU (or CPU) time and saves a trace file you can open at
https://ui.perfetto.dev (or chrome://tracing). profiling/analyze.py turns that trace into a
bottleneck report. Code regions labelled with profiling.nvtx.nvtx_range show up by name.

record_shapes / profile_memory add detail but make traces much bigger; off by default.
"""
import contextlib
import os
import time
import torch
from torch.profiler import ProfilerActivity, profile
import config
import profiling.nvtx as nvtx


@contextlib.contextmanager
def torch_profile(name: str, out_dir: str = config.PROFILE_OUTPUT_DIR, row_limit: int = 15,
                  record_shapes: bool = False, profile_memory: bool = False):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(out_dir, exist_ok=True)
    result = {}
    kwargs = dict(activities=activities, record_shapes=record_shapes, profile_memory=profile_memory)
    # The custom engine does its work in a background thread; by default the profiler only
    # records the thread that started it, so ask it to record every thread.
    try:
        from torch._C._profiler import _ExperimentalConfig
        kwargs["experimental_config"] = _ExperimentalConfig(profile_all_threads=True)
    except (ImportError, TypeError):
        print("Note: this PyTorch can't profile all threads; the engine loop may be missing "
              "from the trace. Upgrade PyTorch (2.6+) for a complete picture.")
    with profile(**kwargs) as prof:
        nvtx.TORCH_PROFILER_ACTIVE = True
        try:
            yield result
        finally:
            nvtx.TORCH_PROFILER_ACTIVE = False
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    trace_path = os.path.join(out_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    prof.export_chrome_trace(trace_path)
    result["trace_path"] = trace_path

    averages = prof.key_averages()
    table = None
    sort_keys = ["device_time_total", "cuda_time_total"] if torch.cuda.is_available() else []
    for key in sort_keys + ["cpu_time_total"]:
        try:
            table = averages.table(sort_by=key, row_limit=row_limit)
            print(f"\n===== Top operations by {key} =====")
            break
        except Exception:
            continue   # different PyTorch versions name the GPU-time column differently
    print(table)
    print(f"\nTrace saved to {trace_path} (open it at https://ui.perfetto.dev)")