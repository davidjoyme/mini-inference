"""
engine/metrics.py - resource measurements for the server process and GPU(s).

memory_snapshot() returns (fields are missing/None when not available):
    process_rss_mb, process_peak_rss_mb   CPU RAM used by the server
    process_cpu_pct                       server CPU use since the previous snapshot
                                          (100 = one full core)
    gpu_allocated_mb, gpu_reserved_mb     PyTorch's view of GPU memory
    gpu_peak_allocated_mb, ..._reserved   peaks since the last reset
    gpu_device_used_mb, ..._total_mb      whole-GPU view, like nvidia-smi
    gpu_util_pct                          % of time the GPU was running kernels    (needs NVML)
    gpu_mem_bw_util_pct                   % of time GPU memory was being read/written (needs NVML)
    gpu_power_w                           power draw in watts                      (needs NVML)

NVML comes from the `nvidia-ml-py` package:  pip install nvidia-ml-py
"""
import os
import threading
import time
import torch

# ---------------- NVML (GPU utilization and power) ----------------
_nvml = None
_nvml_handles = None
_nvml_lock = threading.Lock()


def _nvml_init():
    """Set up NVML once. Returns the list of handles for the GPUs PyTorch uses, or []."""
    global _nvml, _nvml_handles
    with _nvml_lock:
        if _nvml_handles is not None:
            return _nvml_handles
        _nvml_handles = []
        if not torch.cuda.is_available():
            return _nvml_handles
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml = pynvml
        except Exception:
            return _nvml_handles   # package missing or driver doesn't support NVML

        for i in range(torch.cuda.device_count()):
            handle = None
            # Match by UUID so CUDA_VISIBLE_DEVICES reordering doesn't mix up GPUs
            uuid = getattr(torch.cuda.get_device_properties(i), "uuid", None)
            if uuid is not None:
                try:
                    handle = _nvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
                except Exception:
                    handle = None
            if handle is None:
                try:
                    handle = _nvml.nvmlDeviceGetHandleByIndex(i)
                except Exception:
                    continue
            _nvml_handles.append(handle)
        return _nvml_handles


def gpu_utilization() -> dict:
    handles = _nvml_init()
    if not handles:
        return {}
    util = bw = power = 0.0
    try:
        for h in handles:
            rates = _nvml.nvmlDeviceGetUtilizationRates(h)
            util += rates.gpu
            bw += rates.memory
            power += _nvml.nvmlDeviceGetPowerUsage(h) / 1000.0   # milliwatts -> watts
    except Exception:
        return {}
    n = len(handles)
    return {
        "gpu_util_pct": round(util / n, 1),          # averaged over GPUs
        "gpu_mem_bw_util_pct": round(bw / n, 1),
        "gpu_power_w": round(power, 1),             # summed over GPUs
    }


# ---------------- CPU ----------------
_cpu_last = None   # (process cpu seconds, wall time) at the previous snapshot


def process_cpu_pct():
    """CPU used by this process since the previous call, in % of one core."""
    global _cpu_last
    t = os.times()
    cpu, wall = t.user + t.system, time.monotonic()
    previous, _cpu_last = _cpu_last, (cpu, wall)
    if previous is None or wall <= previous[1]:
        return None
    return round((cpu - previous[0]) / (wall - previous[1]) * 100, 1)


def read_proc_status_mb(field: str):
    """Read a memory field (e.g. VmRSS, VmHWM) for this process from /proc (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return round(int(line.split()[1]) / 1024, 1)   # kB -> MB
    except OSError:
        pass
    return None


# ---------------- Snapshot ----------------
def memory_snapshot() -> dict:
    snap = {
        "process_rss_mb": read_proc_status_mb("VmRSS"),        # current RAM used by this process
        "process_peak_rss_mb": read_proc_status_mb("VmHWM"),   # highest RAM since start
        "process_cpu_pct": process_cpu_pct(),
    }
    if torch.cuda.is_available():
        allocated = reserved = peak_allocated = peak_reserved = device_used = device_total = 0
        for i in range(torch.cuda.device_count()):
            allocated += torch.cuda.memory_allocated(i)            # held by tensors right now
            reserved += torch.cuda.memory_reserved(i)              # claimed by PyTorch (incl. cache)
            peak_allocated += torch.cuda.max_memory_allocated(i)   # peak since last reset
            peak_reserved += torch.cuda.max_memory_reserved(i)
            free, total = torch.cuda.mem_get_info(i)               # whole-GPU view, like nvidia-smi
            device_used += total - free
            device_total += total
        mb = 1024 ** 2
        snap.update({
            "gpu_allocated_mb": round(allocated / mb, 1),
            "gpu_reserved_mb": round(reserved / mb, 1),
            "gpu_peak_allocated_mb": round(peak_allocated / mb, 1),
            "gpu_peak_reserved_mb": round(peak_reserved / mb, 1),
            "gpu_device_used_mb": round(device_used / mb, 1),
            "gpu_device_total_mb": round(device_total / mb, 1),
        })
        snap.update(gpu_utilization())
    return snap


def reset_peak_memory() -> dict:
    """Reset the GPU peak counters so the next measurement covers only what follows."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    return memory_snapshot()