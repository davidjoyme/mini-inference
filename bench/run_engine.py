"""
bench/run_engine.py - run the engine directly (no server, no HTTP).

This is the script to profile: fewer moving parts, cleaner traces. With -c it also works as
an offline throughput benchmark: C requests are in flight at once, so the custom engine can
batch them.

Run from the project root:
    python bench/run_engine.py                              # quick timing run (engine from config.py)
    python bench/run_engine.py -n 5 --max-new-tokens 50
    python bench/run_engine.py --quant int4_wo              # INT4 weights
    python bench/run_engine.py --engine custom --scheduler continuous --kv-cache paged -n 32 -c 16
    python bench/run_engine.py --engine custom --scheduler static -n 32 -c 16
    python bench/run_engine.py --engine custom -n 32 -c 16 --prompt-words 1000 --prefix-caching on
    python bench/run_engine.py --engine custom -n 32 -c 16 --kv-dtype fp8
    python bench/run_engine.py --engine custom -n 32 -c 16 --prompt-words 2000 --long-every 4 \
        --unique-prompts --chunked-prefill on --max-batched-tokens 512
    python bench/run_engine.py --profile torch              # PyTorch profiler trace
    bash profiling/scripts/nsys.sh                          # Nsight Systems timeline
"""
import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Make the project root importable when run as `python bench/run_engine.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from bench.prompts import build_prompt  # noqa: E402
from engine import metrics  # noqa: E402
from engine.factory import create_engine, engine_label  # noqa: E402
from profiling.nvtx import nvtx_range  # noqa: E402


def run_one(engine, prompt, max_new_tokens):
    """One generation with exact per-token timings from the engine."""
    r = engine.generate_timed(prompt, max_new_tokens)
    itl = r.pop("itl_ms")
    r["prompt"] = prompt
    r["itl_ms_max"] = max(itl) if itl else None
    return r


def main():
    parser = argparse.ArgumentParser(description="Run the engine directly")
    parser.add_argument("-n", "--num-prompts", type=int, default=config.ENGINE_BENCH_NUM_PROMPTS)
    parser.add_argument("-c", "--concurrency", type=int, default=1,
                        help="requests in flight at once (>1 lets the custom engine batch them)")
    parser.add_argument("--warmup", type=int, default=config.ENGINE_BENCH_WARMUP)
    parser.add_argument("--prompt", default=config.TEST_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=config.TEST_MAX_NEW_TOKENS)
    parser.add_argument("--prompt-words", type=int, default=0,
                        help="prepend ~this many words of text shared by every request (0 = off)")
    parser.add_argument("--unique-prompts", action="store_true",
                        help="start every prompt differently, so prompts share no prefix")
    parser.add_argument("--long-every", type=int, default=0,
                        help="mixed workload: only every K-th request gets --prompt-words (0 = all do)")
    parser.add_argument("--chunked-prefill", default=None, choices=["on", "off"],
                        help="override CHUNKED_PREFILL (custom engine)")
    parser.add_argument("--max-batched-tokens", type=int, default=None,
                        help="override MAX_BATCHED_TOKENS (per-step token budget)")
    parser.add_argument("--kv-dtype", default=None, choices=["auto", "fp8", "int8"],
                        help="override KV_CACHE_DTYPE (custom engine)")
    parser.add_argument("--prefix-caching", default=None, choices=["on", "off"],
                        help="override PREFIX_CACHING (custom engine, paged)")
    parser.add_argument("--profile", choices=["none", "torch"], default="none",
                        help="'torch' wraps the measured runs in the PyTorch profiler")
    parser.add_argument("--quant", default=None, choices=["none", "int8_wo", "int4_wo", "fp8"],
                        help="override QUANTIZATION from config.py")
    parser.add_argument("--engine", default=None, choices=["hf", "custom"], help="override ENGINE")
    parser.add_argument("--scheduler", default=None, choices=["sequential", "static", "continuous"])
    parser.add_argument("--kv-cache", default=None, choices=["contiguous", "paged"])
    parser.add_argument("--out", default=None, help="optional JSON file for the results")
    args = parser.parse_args()

    for flag, name in ((args.quant, "QUANTIZATION"), (args.engine, "ENGINE"),
                       (args.scheduler, "SCHEDULER"), (args.kv_cache, "KV_CACHE")):
        if flag:
            setattr(config, name, flag)
    if args.kv_dtype:
        config.KV_CACHE_DTYPE = args.kv_dtype
    if args.prefix_caching:
        config.PREFIX_CACHING = args.prefix_caching == "on"
    if args.chunked_prefill:
        config.CHUNKED_PREFILL = args.chunked_prefill == "on"
    if args.max_batched_tokens:
        config.MAX_BATCHED_TOKENS = args.max_batched_tokens
    if config.ENGINE == "hf" and args.concurrency > 1:
        print("Note: the hf engine runs one request at a time, so -c only adds queueing.")
    engine = create_engine()

    if args.warmup:
        print(f"Warmup: {args.warmup} run(s), not measured...")
        with nvtx_range("warmup"):
            for _ in range(args.warmup):
                run_one(engine, args.prompt, min(args.max_new_tokens, 10))

    metrics.reset_peak_memory()
    mem_before = metrics.memory_snapshot()
    stats_before = engine.stats()

    if args.profile == "torch":
        from profiling.torch_profiler import torch_profile
        profiler_ctx = torch_profile("run_engine")
    else:
        profiler_ctx = contextlib.nullcontext()

    print(f"Running {args.num_prompts} prompt(s), {args.concurrency} at a time, "
          f"max_new_tokens={args.max_new_tokens} | engine: {engine_label(engine.info())}\n")

    def task(i):
        with nvtx_range(f"request_{i}"):
            words = args.prompt_words if not args.long_every or i % args.long_every == 0 else 0
            r = run_one(engine, build_prompt(args.prompt, i, words, args.unique_prompts), args.max_new_tokens)
        print(f"[{i}] tokens={r['generated_tokens']} latency={r['latency_s']}s "
              f"TTFT={r['ttft_s']}s TPOT={r['tpot_ms']}ms max gap={r['itl_ms_max']}ms tok/s={r['tokens_per_s']}"
              + (f" ERROR={r['error']}" if "error" in r else ""))
        return r

    start = time.perf_counter()
    with profiler_ctx:
        if args.concurrency <= 1:
            results = [task(i) for i in range(args.num_prompts)]
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(pool.map(task, range(args.num_prompts)))
    wall = time.perf_counter() - start

    mem_after = metrics.memory_snapshot()
    stats_after = engine.stats()

    def avg(key):
        values = [r[key] for r in results if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    info = engine.info()
    total_tokens = sum(r["generated_tokens"] for r in results)
    summary = {
        "engine": info,
        "engine_label": engine_label(info),
        "quantization": info["quantization"].get("method"),
        "concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "weights_total_mb": info["weight_memory"]["weights_total_mb"],
        "wall_time_s": round(wall, 3),
        "throughput_tok_s": round(total_tokens / wall, 2),     # all requests together
        "avg_latency_s": avg("latency_s"),
        "avg_queue_s": avg("queue_time_s"),
        "avg_ttft_s": avg("ttft_s"),
        "avg_tpot_ms": avg("tpot_ms"),
        "max_itl_ms": max((r["itl_ms_max"] for r in results if r.get("itl_ms_max") is not None), default=None),
        "avg_tokens_per_s": avg("tokens_per_s"),                # per request
        "gpu_peak_allocated_mb": mem_after.get("gpu_peak_allocated_mb"),
        "gpu_allocated_before_mb": mem_before.get("gpu_allocated_mb"),
        "process_rss_mb": mem_after.get("process_rss_mb"),
    }
    for key in ("steps", "preemptions", "padding_tokens",     # custom engine counters
                "prefix_hit_tokens", "prefix_query_tokens", "prefill_chunks", "mixed_steps"):
        if key in stats_after:
            summary[key] = stats_after[key] - stats_before.get(key, 0)
    if summary.get("prefix_query_tokens"):
        summary["prefix_hit_pct"] = round(summary["prefix_hit_tokens"] / summary["prefix_query_tokens"] * 100, 1)
    if "steps" in summary and summary["steps"]:
        summary["avg_batch_size"] = round((stats_after["decode_tokens"] - stats_before["decode_tokens"] +
                                           summary.get("padding_tokens", 0)) / summary["steps"], 2)

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