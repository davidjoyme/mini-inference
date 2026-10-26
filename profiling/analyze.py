"""
profiling/analyze.py - turn a PyTorch profiler trace into a bottleneck report.

bench/run_engine.py --profile torch runs this automatically. It can also be run by hand:
    python -m profiling.analyze profiles/run_engine_20261025_193501.json
    python -m profiling.analyze --summary profiles/suite_*/report.json     # compare configurations

What it reports (custom engine; the hf engine only gets the GPU and kernel parts):
  Engine loop   where the engine thread's wall time goes: scheduling, building the batch
                (Python), launching the forward pass, waiting for the GPU, delivering tokens.
                Long "wait for GPU" = the GPU is the bottleneck. Long Python phases with little
                waiting = the CPU side is the bottleneck.
  GPU           how much of the time the GPU was actually running kernels (busy %), and how
                kernel time splits between attention, MoE and the rest of the layers, and
                between kinds of kernels (matmuls, attention kernels, gathers/copies, ...)
  Decode steps  time per step by batch size, tokens/s, and an estimate of the memory
                bandwidth achieved versus the GPU's peak (decode is usually bandwidth-bound)
  Hints         rule-of-thumb readings of the numbers above, pointing at what to optimize
"""
import argparse
import glob
import json
import os
import re
import sys
from bisect import bisect_right
from collections import defaultdict

# Peak memory bandwidth (GB/s) of common GPUs, for the bandwidth estimate.
# Override with PEAK_MEMORY_BANDWIDTH_GBS in config.py for anything not listed.
PEAK_BANDWIDTH_GBS = {
    "RTX 5090": 1792, "RTX 5080": 960, "RTX 4090": 1008, "RTX 4080": 717, "RTX 3090": 936,
    "H200": 4800, "H100 SXM": 3350, "H100 PCIe": 2000, "H100": 3350, "A100-SXM4-80GB": 2039,
    "A100 80GB": 1935, "A100": 1555, "L40S": 864, "L4": 300, "A10G": 600, "B200": 8000,
}

LOOP_PHASES = ["schedule", "prepare_batch", "forward", "sample_and_sync", "deliver"]
PHASE_NAMES = {
    "schedule": "scheduling (Python)",
    "prepare_batch": "building the batch (Python)",
    "forward": "launching the forward pass",
    "sample_and_sync": "waiting for GPU + sampling",
    "deliver": "delivering tokens (Python)",
}
FORWARD_RE = re.compile(r"forward\(T=(\d+), decode=(\d+), prefill=(\d+)(?:, ctx=(\d+))?\)")
KERNEL_KINDS = [  # first match wins
    ("attention kernels", re.compile(r"flash|fmha|attention|sdpa|softmax", re.I)),
    ("matmuls (GEMM)", re.compile(r"gemm|cutlass|cublas|xmma|matmul|_mm_|sm\d+_|grouped_mm|splitk", re.I)),
    ("gather / index / copy", re.compile(r"index|gather|scatter|copy|cat_|memcpy|stack", re.I)),
    ("dtype casts", re.compile(r"cast|convert|_to_|bfloat16_copy", re.I)),
]


# ---------------------------------------------------------------- loading
def load_events(path):
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and "dur" in e]


def union_length(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


# ---------------------------------------------------------------- model facts for bandwidth
def model_profile(engine):
    """Bytes the model reads per step, split into experts / everything else (an estimate)."""
    import torch
    model = engine.model
    cfg = model.config
    expert_bytes = other_bytes = 0
    counted = set()
    for name, module in model.named_modules():
        if type(module).__name__.endswith("Experts"):
            for t in list(module.parameters()) + list(module.buffers()):
                expert_bytes += t.numel() * t.element_size()
                counted.add(id(t))
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if id(t) in counted or "embed_tokens" in name:     # embeddings: only a few rows are read
            continue
        other_bytes += t.numel() * t.element_size()
    info = {
        "expert_weight_bytes": expert_bytes,
        "other_weight_bytes": other_bytes,
        "num_experts": getattr(cfg, "num_experts", None),
        "experts_per_token": getattr(cfg, "num_experts_per_tok", None),
        "kv_bytes_per_token": getattr(getattr(engine, "cache", None), "bytes_per_token", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    info["peak_bandwidth_gbs"] = peak_bandwidth(info["gpu_name"])
    return info


def peak_bandwidth(gpu_name):
    try:
        import config
        if getattr(config, "PEAK_MEMORY_BANDWIDTH_GBS", None):
            return config.PEAK_MEMORY_BANDWIDTH_GBS
    except ImportError:
        pass
    if not gpu_name:
        return None
    for key in sorted(PEAK_BANDWIDTH_GBS, key=len, reverse=True):   # most specific name first
        if key.lower() in gpu_name.lower():
            return PEAK_BANDWIDTH_GBS[key]
    return None


def bytes_for_decode_step(batch, ctx_tokens, model_info):
    """Weights touched (each token uses top-k of E experts; with B tokens an expert is read if
    any token picked it) plus the KV cache attention reads."""
    E, k = model_info.get("num_experts"), model_info.get("experts_per_token")
    frac = 1 - (1 - k / E) ** batch if E and k else 1.0
    kv = (model_info.get("kv_bytes_per_token") or 0) * ctx_tokens
    return model_info["other_weight_bytes"] + frac * model_info["expert_weight_bytes"] + kv


# ---------------------------------------------------------------- analysis
def analyze(trace_path, model_info=None):
    events = load_events(trace_path)
    ann = [e for e in events if e.get("cat") == "user_annotation"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    report = {"trace": trace_path, "gpu_trace": bool(kernels)}

    # -------- engine loop phases (wall time on the engine thread)
    forwards = sorted((e for e in ann if e["name"].startswith("forward(")), key=lambda e: e["ts"])
    loop_tid = forwards[0]["tid"] if forwards else None
    loop_ann = [e for e in ann if e["tid"] == loop_tid] if loop_tid is not None else []
    if forwards:
        phases = defaultdict(float)
        for e in loop_ann:
            base = e["name"].split("(")[0]
            if base in LOOP_PHASES:
                phases[base] += e["dur"]
        tops = [e for e in loop_ann if e["name"].split("(")[0] in LOOP_PHASES]
        start = min(e["ts"] for e in tops)
        end = max(e["ts"] + e["dur"] for e in tops)
        wall = end - start
        report["steps"] = len(forwards)
        report["loop_wall_ms"] = round(wall / 1000, 2)
        report["loop_phases_pct"] = {p: pct(phases[p], wall) for p in LOOP_PHASES}
        report["loop_phases_pct"]["idle / waiting for requests / locks"] = pct(
            wall - sum(phases[p] for p in LOOP_PHASES), wall)
        report["avg_step_ms"] = round(wall / len(forwards) / 1000, 3)

        # CPU time inside the forward pass: attention vs MoE vs the rest
        inside = defaultdict(float)
        fwd_total = sum(e["dur"] for e in forwards)
        for e in loop_ann:
            base = e["name"].split("(")[0]
            if base in ("attention", "moe"):
                inside[base] += e["dur"]
        report["forward_cpu_pct"] = {
            "attention": pct(inside["attention"], fwd_total),
            "moe": pct(inside["moe"], fwd_total),
            "rest of the layers": pct(fwd_total - inside["attention"] - inside["moe"], fwd_total),
        }

    # -------- GPU
    gpu_step_time = {}
    if kernels:
        k_start = min(k["ts"] for k in kernels)
        k_end = max(k["ts"] + k["dur"] for k in kernels)
        busy = union_length([(k["ts"], k["ts"] + k["dur"]) for k in kernels])
        if forwards:   # judge utilization over the time the engine was actually stepping
            win_s = forwards[0]["ts"]
            win_e = max(f["ts"] + f["dur"] for f in forwards)
            win_kernels = [(max(k["ts"], win_s), min(k["ts"] + k["dur"], k_end))
                           for k in kernels if k["ts"] + k["dur"] > win_s]
            window = k_end - win_s
            busy = union_length(win_kernels)
        else:
            window = k_end - k_start
        report["gpu_busy_pct"] = pct(busy, window)
        report["gpu_kernel_count"] = len(kernels)
        report["avg_kernel_us"] = round(sum(k["dur"] for k in kernels) / len(kernels), 2)

        # kernel kinds by name
        kinds = defaultdict(float)
        for k in kernels:
            for kind, rx in KERNEL_KINDS:
                if rx.search(k["name"]):
                    kinds[kind] += k["dur"]
                    break
            else:
                kinds["elementwise / other"] += k["dur"]
        ktotal = sum(kinds.values())
        report["gpu_kernel_kinds_pct"] = {k: pct(v, ktotal) for k, v in
                                          sorted(kinds.items(), key=lambda kv: -kv[1])}

        # attribute each kernel to the labelled region that launched it
        launches = {}
        for e in events:
            if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e:
                corr = e["args"].get("correlation")
                if corr is not None:
                    launches[corr] = (e["tid"], e["ts"])
        by_tid = defaultdict(list)
        for e in ann:
            by_tid[e["tid"]].append(e)
        index = {}
        for tid, lst in by_tid.items():
            lst.sort(key=lambda e: e["ts"])
            index[tid] = ([e["ts"] for e in lst], lst)

        def regions_at(tid, ts):
            if tid not in index:
                return []
            starts, lst = index[tid]
            i = bisect_right(starts, ts)
            found = []
            for e in reversed(lst[max(0, i - 400):i]):        # annotations open at ts
                if e["ts"] + e["dur"] >= ts:
                    found.append(e)
            return found

        region_time = defaultdict(float)
        for k in kernels:
            corr = k.get("args", {}).get("correlation")
            tid, ts = launches.get(corr, (None, None))
            open_regions = regions_at(tid, ts) if tid is not None else []
            names = [r["name"].split("(")[0] for r in open_regions]
            if "attention" in names:
                region = "attention"
            elif "moe" in names or "experts" in names:
                region = "moe"
            elif "forward" in names:
                region = "rest of the layers"
            elif "sample_and_sync" in names:
                region = "sampling"
            else:
                region = "outside the forward pass"
            region_time[region] += k["dur"]
            fwd = next((r for r in open_regions if r["name"].startswith("forward(")), None)
            if fwd is not None:
                key = (fwd["ts"], fwd["name"])
                gpu_step_time[key] = gpu_step_time.get(key, 0.0) + k["dur"]
        rtotal = sum(region_time.values())
        report["gpu_time_by_region_pct"] = {r: pct(v, rtotal) for r, v in
                                            sorted(region_time.items(), key=lambda kv: -kv[1])}

    # -------- decode steps by batch size
    if forwards:
        buckets = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 10 ** 9)]
        rows = []
        starts = [f["ts"] for f in forwards]
        periods = [b - a for a, b in zip(starts, starts[1:])] + [None]
        for lo, hi in buckets:
            sel = []
            for f, period in zip(forwards, periods):
                m = FORWARD_RE.match(f["name"])
                if not m or period is None:
                    continue
                T, dec, pre, ctx = (int(x) if x else 0 for x in m.groups())
                if pre == 0 and lo <= dec <= hi:
                    sel.append((dec, ctx, period, gpu_step_time.get((f["ts"], f["name"]))))
            if not sel:
                continue
            n = len(sel)
            avg_b = sum(s[0] for s in sel) / n
            avg_period = sum(s[2] for s in sel) / n
            row = {"batch": f"{lo}" if lo == hi else f"{lo}-{hi if hi < 10 ** 9 else '+'}",
                   "steps": n, "avg_batch": round(avg_b, 1),
                   "step_ms": round(avg_period / 1000, 3),
                   "tokens_per_s": round(avg_b / (avg_period / 1e6), 1)}
            gpu = [s[3] for s in sel if s[3]]
            if gpu:
                avg_gpu = sum(gpu) / len(gpu)
                row["gpu_ms"] = round(avg_gpu / 1000, 3)
                row["gpu_busy_in_step_pct"] = pct(avg_gpu, avg_period)
                if model_info and model_info.get("expert_weight_bytes") is not None:
                    gbs = [bytes_for_decode_step(s[0], s[1], model_info) / (s[3] / 1e6) / 1e9
                           for s in sel if s[3]]
                    row["est_bandwidth_gbs"] = round(sum(gbs) / len(gbs), 1)
                    if model_info.get("peak_bandwidth_gbs"):
                        row["est_bandwidth_pct_of_peak"] = pct(row["est_bandwidth_gbs"],
                                                               model_info["peak_bandwidth_gbs"])
            rows.append(row)
        report["decode_steps"] = rows
        report["prefill_or_mixed_steps"] = sum(
            1 for f in forwards if (m := FORWARD_RE.match(f["name"])) and int(m.group(3)) > 0)

    if model_info:
        report["model"] = {k: model_info.get(k) for k in ("gpu_name", "peak_bandwidth_gbs",
                                                          "num_experts", "experts_per_token")}
        report["model"]["weights_read_per_step_mb_max"] = round(
            (model_info["other_weight_bytes"] + model_info["expert_weight_bytes"]) / 1024 ** 2, 1)
    report["hints"] = hints(report)
    return report


def hints(r):
    out = []
    busy = r.get("gpu_busy_pct")
    if busy is not None and busy < 60:
        out.append(f"GPU idle {round(100 - busy)}% of the time: CPU-side work (Python, scheduling, "
                   "kernel launches) is holding it back. Look at CUDA graphs and overlapping scheduling "
                   "with GPU work.")
    phases = r.get("loop_phases_pct") or {}
    python = sum(phases.get(p) or 0 for p in ("schedule", "prepare_batch", "deliver"))
    if python > 15:
        out.append(f"{round(python)}% of the engine loop is Python bookkeeping (scheduling, batch "
                   "building, delivery) during which the GPU gets no new work.")
    if (phases.get("forward") or 0) > 2 * (phases.get("sample_and_sync") or 0) and r.get("gpu_trace"):
        out.append("Launching the forward pass takes longer than waiting for the GPU: the step is "
                   "launch-bound (too many small kernels). CUDA graphs or fused kernels help most here.")
    kernels = r.get("avg_kernel_us")
    if kernels is not None and kernels < 20:
        out.append(f"Average kernel runs only {kernels} us: many tiny kernels, so launch overhead "
                   "matters. Fewer, fused kernels (e.g. a fused MoE kernel) would help.")
    regions = r.get("gpu_time_by_region_pct") or r.get("forward_cpu_pct") or {}
    if (regions.get("moe") or 0) > 50:
        out.append(f"MoE layers take {regions['moe']}% of the time: a fused/grouped MoE kernel is the "
                   "biggest single target.")
    if (regions.get("attention") or 0) > 35:
        out.append(f"Attention takes {regions['attention']}%: check the gather of cached keys/values "
                   "(a paged-attention kernel reads blocks in place instead of copying them).")
    kinds = r.get("gpu_kernel_kinds_pct") or {}
    if (kinds.get("gather / index / copy") or 0) > 15:
        out.append(f"{kinds['gather / index / copy']}% of GPU time is gathers/copies, not maths: "
                   "data movement that a better kernel or memory layout could avoid.")
    for row in r.get("decode_steps") or []:
        p = row.get("est_bandwidth_pct_of_peak")
        if p is not None and p < 30 and row["avg_batch"] >= 1:
            out.append(f"Decode at batch {row['batch']} reaches about {p}% of peak memory bandwidth "
                       "(decode should approach the bandwidth limit): time is going to overhead or "
                       "inefficient kernels rather than reading weights.")
            break
    if not r.get("gpu_trace"):
        out.append("No GPU kernels in this trace (CPU run): GPU busy %, kernel and bandwidth "
                   "figures need a CUDA run.")
    return out


# ---------------------------------------------------------------- printing
def print_report(r, title=None):
    print(f"\n===== Bottleneck report{': ' + title if title else ''} =====")
    if r.get("model", {}).get("gpu_name"):
        m = r["model"]
        print(f"GPU: {m['gpu_name']} (peak bandwidth {m.get('peak_bandwidth_gbs') or '?'} GB/s)")
    if "steps" in r:
        print(f"Engine steps: {r['steps']} ({r['prefill_or_mixed_steps']} with prefill), "
              f"average {r['avg_step_ms']} ms per step")
        print("\nEngine loop wall time:")
        for p, v in r["loop_phases_pct"].items():
            print(f"  {PHASE_NAMES.get(p, p):42} {v}%")
        print("\nCPU time inside the forward pass:")
        for p, v in r["forward_cpu_pct"].items():
            print(f"  {p:42} {v}%")
    if r.get("gpu_trace"):
        print(f"\nGPU busy: {r['gpu_busy_pct']}% of the time | {r['gpu_kernel_count']} kernels, "
              f"average {r['avg_kernel_us']} us each")
        print("GPU time by region:")
        for k, v in r["gpu_time_by_region_pct"].items():
            print(f"  {k:42} {v}%")
        print("GPU time by kernel kind:")
        for k, v in r["gpu_kernel_kinds_pct"].items():
            print(f"  {k:42} {v}%")
    if r.get("decode_steps"):
        print("\nDecode-only steps by batch size:")
        cols = ["batch", "steps", "avg_batch", "step_ms", "gpu_ms", "gpu_busy_in_step_pct",
                "tokens_per_s", "est_bandwidth_gbs", "est_bandwidth_pct_of_peak"]
        cols = [c for c in cols if any(c in row for row in r["decode_steps"])]
        print("  " + "  ".join(f"{c:>12}" for c in cols))
        for row in r["decode_steps"]:
            print("  " + "  ".join(f"{str(row.get(c, '-')):>12}" for c in cols))
    if r.get("hints"):
        print("\nHints:")
        for h in r["hints"]:
            print(f"  - {h}")


def print_summary(paths):
    rows = []
    for p in paths:
        r = json.load(open(p))
        name = os.path.basename(os.path.dirname(p)) or p
        best = max(r.get("decode_steps") or [{}], key=lambda x: x.get("avg_batch", 0))
        rows.append({
            "config": name,
            "step_ms": r.get("avg_step_ms"),
            "gpu_busy_%": r.get("gpu_busy_pct"),
            "python_%": (round(sum(r["loop_phases_pct"].get(k) or 0
                                   for k in ("schedule", "prepare_batch", "deliver")), 1)
                         if r.get("loop_phases_pct") else None),
            "gpu_wait_%": (r.get("loop_phases_pct") or {}).get("sample_and_sync"),
            "attn_%": (r.get("gpu_time_by_region_pct") or r.get("forward_cpu_pct") or {}).get("attention"),
            "moe_%": (r.get("gpu_time_by_region_pct") or r.get("forward_cpu_pct") or {}).get("moe"),
            "max_batch": best.get("avg_batch"),
            "tok/s@max": best.get("tokens_per_s"),
            "bw_%peak@max": best.get("est_bandwidth_pct_of_peak"),
        })
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c] if r[c] is not None else "-").ljust(widths[c]) for c in cols))


def main():
    p = argparse.ArgumentParser(description="Bottleneck report from a PyTorch profiler trace")
    p.add_argument("paths", nargs="+", help="trace .json files, or report.json files with --summary")
    p.add_argument("--summary", action="store_true", help="compare saved report.json files")
    p.add_argument("--model-info", default=None, help="model_info.json saved by bench/run_engine.py")
    args = p.parse_args()
    paths = sorted({f for x in args.paths for f in glob.glob(x)})
    if args.summary:
        print_summary(paths)
        return
    info = json.load(open(args.model_info)) if args.model_info else None
    for path in paths:
        r = analyze(path, info)
        print_report(r, os.path.basename(path))


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()