"""
bench/run_engine.py - run the engine directly (no server, no HTTP).

This is the script to profile: fewer moving parts, cleaner traces.
Run from the project root:
    python bench/run_engine.py                          # quick timing run
    python bench/run_engine.py -n 5 --max-new-tokens 50
    python bench/run_engine.py --profile torch          # PyTorch profiler trace
    bash profiling/scripts/nsys.sh                      # Nsight Systems timeline
"""
import argparse
import contextlib
import json
import os
import sys
import time

# Make the project root importable when run as `python bench/run_engine.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from engine import metrics  # noqa: E402
from engine.engine import Engine  # noqa: E402
from profiling.nvtx import nvtx_range  # noqa: E402


def run_one(engine, prompt, max_new_tokens):
    """Stream one generation and measure TTFT / TPOT from the caller's side."""
    start = time.perf_counter()
    first_at = None
    pieces = []
    final = {}
    for event in engine.stream(prompt, max_new_tokens):
        if event.get("done"):
            final = event
            continue
        if first_at is None:
            first_at = time.perf_counter()
        pieces.append(event["text"])
    end = time.perf_counter()

    tokens = final.get("generated_tokens") or 0
    result = {
        "prompt": prompt,
        "output": "".join(pieces),
        "generated_tokens": tokens,
        "latency_s": round(end - start, 3),
        "ttft_s": round(first_at - start, 3) if first_at else None,
        "tpot_ms": round((end - first_at) / (tokens - 1) * 1000, 2) if first_at and tokens > 1 else None,
        "tokens_per_s": round(tokens / (end - start), 2) if end > start else None,
        "prefill_time_s": final.get("prefill_time_s"),
    }
    if "error" in final:
        result["error"] = final["error"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the engine directly")
    parser.add_argument("-n", "--num-prompts", type=int, default=config.ENGINE_BENCH_NUM_PROMPTS)
    parser.add_argument("--warmup", type=int, default=config.ENGINE_BENCH_WARMUP)
    parser.add_argument("--prompt", default=config.TEST_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=config.TEST_MAX_NEW_TOKENS)
    parser.add_argument("--profile", choices=["none", "torch"], default="none",
                        help="'torch' wraps the measured runs in the PyTorch profiler")
    parser.add_argument("--out", default=None, help="optional JSON file for the results")
    args = parser.parse_args()

    engine = Engine()

    if args.warmup:
        print(f"Warmup: {args.warmup} run(s), not measured...")
        with nvtx_range("warmup"):
            for _ in range(args.warmup):
                run_one(engine, args.prompt, min(args.max_new_tokens, 10))

    metrics.reset_peak_memory()
    mem_before = metrics.memory_snapshot()

    if args.profile == "torch":
        from profiling.torch_profiler import torch_profile
        profiler_ctx = torch_profile("run_engine")
    else:
        profiler_ctx = contextlib.nullcontext()

    results = []
    print(f"Running {args.num_prompts} prompt(s), max_new_tokens={args.max_new_tokens}\n")
    with profiler_ctx:
        for i in range(args.num_prompts):
            with nvtx_range(f"request_{i}"):
                r = run_one(engine, f"{args.prompt} (request {i})", args.max_new_tokens)
            results.append(r)
            print(f"[{i}] tokens={r['generated_tokens']} latency={r['latency_s']}s "
                  f"TTFT={r['ttft_s']}s TPOT={r['tpot_ms']}ms tok/s={r['tokens_per_s']}"
                  + (f" ERROR={r['error']}" if "error" in r else ""))

    mem_after = metrics.memory_snapshot()

    def avg(key):
        values = [r[key] for r in results if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    summary = {
        "engine": engine.info(),
        "avg_latency_s": avg("latency_s"),
        "avg_ttft_s": avg("ttft_s"),
        "avg_tpot_ms": avg("tpot_ms"),
        "avg_tokens_per_s": avg("tokens_per_s"),
        "gpu_peak_allocated_mb": mem_after.get("gpu_peak_allocated_mb"),
        "gpu_allocated_before_mb": mem_before.get("gpu_allocated_mb"),
        "process_rss_mb": mem_after.get("process_rss_mb"),
    }
    print("\n===== Summary =====")
    for key, value in summary.items():
        if key != "engine":
            print(f"{key:>24}: {value}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "runs": results}, f, indent=2)
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()