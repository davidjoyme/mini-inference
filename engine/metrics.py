"""engine/metrics.py - memory measurements for the server process and GPU(s)."""
import torch


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


def memory_snapshot() -> dict:
    snap = {
        "process_rss_mb": read_proc_status_mb("VmRSS"),        # current RAM used by this process
        "process_peak_rss_mb": read_proc_status_mb("VmHWM"),   # highest RAM since start
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
    return snap


def reset_peak_memory() -> dict:
    """Reset the GPU peak counters so the next measurement covers only what follows."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    return memory_snapshot()