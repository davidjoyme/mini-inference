"""
profiling/torch_profiler.py - run code under the PyTorch profiler.

    with torch_profile("my_run"):
        ...code to profile...

Prints the top operations by GPU (or CPU) time and saves a trace file you can open
at https://ui.perfetto.dev (or chrome://tracing).
"""
import contextlib
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

import config


@contextlib.contextmanager
def torch_profile(name: str, out_dir: str = config.PROFILE_OUTPUT_DIR, row_limit: int = 25):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(out_dir, exist_ok=True)
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        yield prof
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    trace_path = os.path.join(out_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    prof.export_chrome_trace(trace_path)

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