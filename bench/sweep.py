"""
bench/sweep.py - measure resource usage across many configurations in one go.

For each quantization x engine variant in the sweep it:
  1. starts the server (python main.py) with that quantization and engine, on its own port
  2. records memory once the model is loaded and idle
  3. runs one short warmup request (not measured)
  4. runs bench/test.py for every combination of users, output length and prompt length,
     sampling memory / GPU utilization / power / CPU while requests run
  5. stops the server, so the next variant starts with an empty GPU

Engine variants are written as "hf" (transformers, one request at a time) or
"<scheduler>/<kv_cache>[+option...]" for the custom engine, e.g. "static/contiguous",
"continuous/paged", "continuous/paged+prefix", "continuous/paged+kv-fp8+prefix".
Options: +prefix (prefix caching, paged only), +kv-fp8 / +kv-int8 (KV cache quantization),
         +chunked (chunked prefill), +budget-N (MAX_BATCHED_TOKENS = N tokens per step),
         +fused-moe (Triton fused MoE kernels; also works as hf+fused-moe),
         +paged-attn (Triton paged decode-attention kernel), +cuda-graphs (decode as CUDA graphs),
         +fused-proj (fused q/k/v and gate/up matmuls, layouts autotuned).

Everything is saved under sweeps/<timestamp>/:
  summary.csv            one row per run (written as it goes, so a crash keeps earlier rows)
  summary.json           the same, plus the sweep settings
  runs/*.json            full results of each run (per-request numbers, resource timeline)
  logs/server_*.log      server output for each quantization
  logs/test_*.log        test bench output for each run

Run from the project root:
    python bench/sweep.py                                  # settings from config.py
    python bench/sweep.py --quants none int4_wo --users 1 4 16
    python bench/sweep.py --engines hf static/contiguous continuous/contiguous continuous/paged
    python bench/sweep.py --max-new-tokens 32 256 --prompt-words 0 2000
    python bench/sweep.py --engines continuous/contiguous continuous/paged --kv-tokens 8192
                                                           # memory pressure: small KV pool
    python bench/sweep.py --engines continuous/paged continuous/paged+prefix --prompt-words 1000
                                                           # prefix caching on a shared document
    python bench/sweep.py --engines continuous/paged continuous/paged+kv-fp8 continuous/paged+kv-int8 \
        --kv-mb 2048                                       # KV quantization: same memory, more tokens
    python bench/sweep.py --engines continuous/paged continuous/paged+chunked+budget-512 \
        --prompt-words 2000 --long-every 4 --unique-prompts
                                                           # chunked prefill on a mixed workload
    python bench/sweep.py --dry-run                        # list the runs, don't start anything
"""
import argparse
import csv
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402



def context_limit():
    """The model's context limit (capped by MAX_MODEL_LEN), for the too-long-prompt warning."""
    try:
        from transformers import AutoConfig
        own = AutoConfig.from_pretrained(config.MODEL_NAME).max_position_embeddings
        return min(own, config.MAX_MODEL_LEN or own)
    except Exception:
        return 4096

# Columns printed in the terminal table (summary.csv has all of them)
SHOW = ["engine", "quant", "users", "out_tok", "in_words", "peak_gpu_mb", "kv_capacity_tokens",
        "batch_size_avg", "preemptions", "prefix_hit_pct", "gpu_util_avg", "throughput_tok_s",
        "ttft_p50_s", "tpot_p50_ms", "failed"]


# ---------------- small HTTP helpers (standard library only) ----------------
def http_json(url, method="GET", timeout=5):
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def server_answers(port):
    try:
        http_json(f"http://127.0.0.1:{port}/health", timeout=2)
        return True
    except Exception:
        return False


# ---------------- server lifecycle ----------------
def engine_env(spec):
    """'hf[+fused-moe]' or '<scheduler>/<kv_cache>[+option...]' -> environment variables"""
    base, *options = spec.split("+")
    if base == "hf":
        if any(o != "fused-moe" for o in options):
            sys.exit(f"'{spec}': the hf engine only takes +fused-moe")
        return {"ENGINE": "hf", "MOE_KERNEL": "triton" if options else "default"}
    scheduler, _, kv = base.partition("/")
    if scheduler not in ("sequential", "static", "continuous") or kv not in ("contiguous", "paged"):
        sys.exit(f"Bad engine variant '{spec}': use hf or <sequential|static|continuous>/<contiguous|paged>"
                 f"[+prefix][+kv-fp8|+kv-int8]")
    env = {"ENGINE": "custom", "SCHEDULER": scheduler, "KV_CACHE": kv,
           "PREFIX_CACHING": "0", "KV_CACHE_DTYPE": "auto", "CHUNKED_PREFILL": "0",
           "MOE_KERNEL": "default", "ATTN_BACKEND": "sdpa", "CUDA_GRAPHS": "0",
           "FUSED_PROJ": "0"}
    for opt in options:
        if opt == "prefix":
            if kv != "paged":
                sys.exit(f"'{spec}': prefix caching needs the paged KV cache")
            env["PREFIX_CACHING"] = "1"
        elif opt in ("kv-fp8", "kv-int8"):
            env["KV_CACHE_DTYPE"] = opt[3:]
        elif opt == "chunked":
            env["CHUNKED_PREFILL"] = "1"
        elif opt == "fused-moe":
            env["MOE_KERNEL"] = "triton"
        elif opt == "paged-attn":
            env["ATTN_BACKEND"] = "triton"
        elif opt == "cuda-graphs":
            env["CUDA_GRAPHS"] = "1"
        elif opt == "fused-proj":
            env["FUSED_PROJ"] = "1"
        elif opt.startswith("budget-") and opt[7:].isdigit():
            env["MAX_BATCHED_TOKENS"] = opt[7:]
        else:
            sys.exit(f"Unknown option '+{opt}' in '{spec}' "
                     f"(use +prefix, +kv-fp8, +kv-int8, +chunked, +budget-N, +fused-moe, +paged-attn, +cuda-graphs, +fused-proj)")
    return env


def start_server(quant, engine, port, log_path, kv_tokens=None, kv_mb=None):
    env = {**os.environ, **engine_env(engine), "QUANT": quant, "PORT": str(port), "MODEL": config.MODEL}
    if kv_tokens:
        env["KV_CACHE_TOKENS"] = str(kv_tokens)     # shrink the KV pool to force memory pressure
    if kv_mb:
        env["KV_CACHE_MB"] = str(kv_mb)             # same memory for every variant
    if os.environ.get("KV_WATERMARK"):
        env["KV_WATERMARK"] = os.environ["KV_WATERMARK"]
    log = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    proc._log_file = log
    return proc


def wait_for_server(proc, port, timeout, log_path):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            print(f"  server exited with code {proc.returncode}. Last lines of {log_path}:")
            print("    " + "\n    ".join(tail(log_path, 25)))
            return None
        try:
            return http_json(f"http://127.0.0.1:{port}/health", timeout=2)
        except Exception:
            time.sleep(2)
    print(f"  server did not answer within {timeout}s (see {log_path})")
    return None


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    # Ctrl+C first (clean shutdown), then harder
    for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    proc._log_file.close()
    time.sleep(3)   # give the driver a moment to release GPU memory


def tail(path, n):
    try:
        with open(path) as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


# ---------------- one load test ----------------
def run_test(port, users, requests, out_tok, in_words, interval, out_path, log_path, memory=True,
             unique=False, long_every=0):
    cmd = [sys.executable, "bench/test.py", "-n", str(requests), "-c", str(users),
           "--max-new-tokens", str(out_tok), "--prompt-words", str(in_words),
           "--mem-interval", str(interval), "--stream", "--out", out_path]
    if unique:
        cmd.append("--unique-prompts")
    if long_every:
        cmd += ["--long-every", str(long_every)]
    cmd.append("--memory" if memory else "--no-memory")
    env = {**os.environ, "PORT": str(port)}
    with open(log_path, "w") as log:
        code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code != 0 or not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        return json.load(f)


def g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def make_row(engine, quant, users, requests, out_tok, in_words, health, loaded, result, run_file):
    s = (result or {}).get("summary", {})
    mem = s.get("memory", {})
    row = {
        "engine": engine,
        "quant": quant,
        "users": users,
        "requests": requests,
        "out_tok": out_tok,
        "in_words": in_words,
        # memory
        "weights_mb": g(health, "weight_memory", "weights_total_mb"),
        "loaded_gpu_mb": (loaded or {}).get("gpu_allocated_mb"),        # model sitting idle
        "loaded_rss_mb": (loaded or {}).get("process_rss_mb"),
        "peak_gpu_mb": mem.get("gpu_peak_allocated_mb"),                 # highest during the run
        "extra_gpu_mb": mem.get("gpu_allocated_increase_mb"),           # peak minus idle
        "extra_gpu_per_user_mb": None,
        "peak_reserved_mb": mem.get("gpu_peak_reserved_mb"),
        "peak_device_used_mb": mem.get("gpu_peak_device_used_mb"),       # like nvidia-smi
        "device_total_mb": mem.get("gpu_device_total_mb"),
        "peak_rss_mb": mem.get("process_peak_rss_mb_sampled"),
        # utilization
        "gpu_util_avg": mem.get("gpu_util_pct_avg"),
        "gpu_util_max": mem.get("gpu_util_pct_max"),
        "gpu_bw_util_avg": mem.get("gpu_mem_bw_util_pct_avg"),
        "power_avg_w": mem.get("gpu_power_w_avg"),
        "power_max_w": mem.get("gpu_power_w_max"),
        "cpu_avg_pct": mem.get("process_cpu_pct_avg"),
        # engine behaviour (custom engine only)
        "kv_capacity_tokens": g(health, "engine", "kv_capacity_tokens"),
        "kv_cache_dtype": g(health, "engine", "kv_cache_dtype"),
        "prefix_hit_pct": mem.get("prefix_hit_pct"),
        "prefix_evictions": mem.get("prefix_evictions"),
        "batch_size_avg": mem.get("batch_size_avg"),
        "batch_size_max": mem.get("batch_size_max"),
        "waiting_max": mem.get("waiting_max"),
        "kv_used_pct_max": mem.get("kv_used_pct_max"),
        "kv_reserved_pct_max": mem.get("kv_reserved_pct_max"),
        "kv_waste_pct_avg": mem.get("kv_waste_pct_avg"),
        "kv_fragmentation_pct_max": mem.get("kv_fragmentation_pct_max"),
        "preemptions": mem.get("preemptions"),
        "padding_waste_pct": mem.get("padding_waste_pct"),
        "steps": mem.get("steps"),
        # performance
        "throughput_tok_s": s.get("overall_tokens_per_s"),
        "requests_per_s": s.get("requests_per_s"),
        "ttft_p50_s": g(s, "ttft_s", "p50"),
        "ttft_p95_s": g(s, "ttft_s", "p95"),
        "tpot_p50_ms": g(s, "tpot_ms", "p50"),
        "tpot_p95_ms": g(s, "tpot_ms", "p95"),
        "itl_p50_ms": g(s, "itl_ms", "p50"),
        "itl_p95_ms": g(s, "itl_ms", "p95"),
        "itl_p99_ms": g(s, "itl_ms", "p99"),
        "itl_max_ms": g(s, "itl_ms", "max"),
        "max_step_tokens": mem.get("max_step_tokens"),
        "prefill_chunks": mem.get("prefill_chunks"),
        "mixed_steps": mem.get("mixed_steps"),
        "latency_p95_s": g(s, "latency_s", "p95"),
        "queue_p95_s": g(s, "queue_time_s", "p95"),
        "joules_per_token": None,
        "failed": s.get("failed") if result else requests,
        "total_time_s": s.get("total_time_s"),
        "run_file": run_file,
    }
    if row["extra_gpu_mb"] is not None:
        row["extra_gpu_per_user_mb"] = round(row["extra_gpu_mb"] / users, 1)
    if row["power_avg_w"] and row["throughput_tok_s"]:
        row["joules_per_token"] = round(row["power_avg_w"] / row["throughput_tok_s"], 3)
    return row


# ---------------- output ----------------
def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def print_table(rows, cols):
    def fmt(v):
        return "-" if v is None else str(v)
    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser(description="Sweep configurations and record resource usage")
    p.add_argument("--model", default=None,
                   help="model short name from config.MODELS (olmoe, qwen3-4b) or a Hugging Face id")
    p.add_argument("--quants", nargs="+", default=config.SWEEP_QUANTIZATIONS,
                   choices=["none", "int8_wo", "int4_wo", "fp8"])
    p.add_argument("--engines", nargs="+", default=config.SWEEP_ENGINES,
                   help="hf and/or <scheduler>/<kv_cache>, e.g. continuous/paged")
    p.add_argument("--users", nargs="+", type=int, default=config.SWEEP_CONCURRENCY,
                   help="concurrency levels (simultaneous users)")
    p.add_argument("--max-new-tokens", nargs="+", type=int, default=config.SWEEP_MAX_NEW_TOKENS)
    p.add_argument("--prompt-words", nargs="+", type=int, default=config.SWEEP_PROMPT_WORDS)
    p.add_argument("--requests-per-user", type=int, default=config.SWEEP_REQUESTS_PER_USER)
    p.add_argument("--mem-interval", type=float, default=config.SWEEP_MEMORY_INTERVAL_S)
    p.add_argument("--long-every", type=int, default=0,
                   help="mixed workload: only every K-th request gets the long prompt (0 = all)")
    p.add_argument("--unique-prompts", action="store_true",
                   help="every prompt starts differently (no shared prefix): the control for +prefix")
    p.add_argument("--kv-tokens", type=int, default=None,
                   help="custom engine KV pool size in tokens (small = memory pressure experiment)")
    p.add_argument("--kv-mb", type=float, default=None,
                   help="custom engine KV pool size in MB (with +kv-fp8/int8 it holds ~2x the tokens)")
    p.add_argument("--port", type=int, default=config.SWEEP_PORT)
    p.add_argument("--startup-timeout", type=int, default=config.SWEEP_STARTUP_TIMEOUT_S)
    p.add_argument("--tag", default="", help="added to the output folder name")
    p.add_argument("--dry-run", action="store_true", help="only list the runs")
    args = p.parse_args()
    if args.model:
        config.MODEL = args.model
        config.MODEL_NAME = config.MODELS.get(args.model, args.model)

    for e in args.engines:
        engine_env(e)                                   # validate early
    variants = [(q, e) for q in args.quants for e in args.engines]
    combos = list(itertools.product(args.users, args.max_new_tokens, args.prompt_words))
    total = len(variants) * len(combos)
    print(f"Sweep: {len(args.quants)} quantization(s) x {len(args.engines)} engine(s) x "
          f"{len(combos)} load test(s) = {total} runs")
    for quant, engine in variants:
        for users, out_tok, in_words in combos:
            print(f"  {engine:22} {quant:8} users={users:<3} out_tok={out_tok:<5} in_words={in_words:<5} "
                  f"requests={users * args.requests_per_user}")
    limit = context_limit()
    for _, out_tok, in_words in combos:
        est = int(in_words * 1.35) + 60 + out_tok   # rough token estimate
        if est > limit:
            print(f"WARNING: in_words={in_words} + out_tok={out_tok} is ~{est} tokens, more than "
                  f"the model's {limit}-token context. Those runs will likely fail.")
    if args.dry_run:
        return
    if server_answers(args.port):
        sys.exit(f"Something is already answering on port {args.port}. Stop it or use --port.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, config.SWEEP_OUTPUT_DIR, stamp + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(os.path.join(out_dir, "runs"))
    os.makedirs(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")
    settings = {k: v for k, v in vars(args).items() if k != "dry_run"}
    print(f"\nSaving to {out_dir}\n")

    rows, proc, done = [], None, 0
    sweep_start = time.time()
    try:
        for quant, engine in variants:
            tag = f"{engine.replace('/', '-')}_{quant}"
            print(f"===== {engine} | {quant}: starting server =====")
            server_log = os.path.join(out_dir, "logs", f"server_{tag}.log")
            t0 = time.time()
            proc = start_server(quant, engine, args.port, server_log, args.kv_tokens, args.kv_mb)
            health = wait_for_server(proc, args.port, args.startup_timeout, server_log)

            loaded = None
            if health is not None:
                print(f"  ready in {time.time() - t0:.0f}s | weights "
                      f"{g(health, 'weight_memory', 'weights_total_mb')} MB")
                time.sleep(1)
                loaded = http_json(f"http://127.0.0.1:{args.port}/metrics/memory/reset", method="POST")
                print(f"  idle: GPU allocated {loaded.get('gpu_allocated_mb')} MB, "
                      f"process RAM {loaded.get('process_rss_mb')} MB")
                run_test(args.port, 1, 1, 8, 0, args.mem_interval,
                         os.path.join(out_dir, "logs", f"warmup_{tag}.json"),
                         os.path.join(out_dir, "logs", f"warmup_{tag}.log"), memory=False)

            for users, out_tok, in_words in combos:
                done += 1
                requests = users * args.requests_per_user
                name = f"{tag}_u{users}_out{out_tok}_in{in_words}"
                run_file = os.path.join("runs", name + ".json")
                print(f"  [{done}/{total}] users={users} out_tok={out_tok} in_words={in_words} ... ",
                      end="", flush=True)
                result = None
                if health is not None:
                    result = run_test(args.port, users, requests, out_tok, in_words, args.mem_interval,
                                      os.path.join(out_dir, run_file),
                                      os.path.join(out_dir, "logs", f"test_{name}.log"),
                                      unique=args.unique_prompts, long_every=args.long_every)
                row = make_row(engine, quant, users, requests, out_tok, in_words, health, loaded,
                               result, run_file)
                rows.append(row)
                append_csv(csv_path, row)
                if result is None:
                    print("FAILED (see logs)" if health is not None else "skipped (server didn't start)")
                else:
                    print(f"peak GPU {row['peak_gpu_mb']} MB (+{row['extra_gpu_mb']}), "
                          f"batch {row['batch_size_avg']}, {row['throughput_tok_s']} tok/s, "
                          f"TTFT p50 {row['ttft_p50_s']}s, failed {row['failed']}")
                if health is not None and proc.poll() is not None:
                    print("  server crashed during the run (often out of memory); see its log")
                    health = None

            stop_server(proc)
            proc = None
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the server and saving what was measured.")
    finally:
        stop_server(proc)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"settings": settings, "model": config.MODEL_NAME,
                   "duration_s": round(time.time() - sweep_start), "rows": rows}, f, indent=2)

    if rows:
        print("\n===== Sweep summary =====")
        cols = SHOW
        if all(r.get("loaded_gpu_mb") is None for r in rows):   # no GPU: show RAM and CPU instead
            cols = ["engine", "quant", "users", "out_tok", "in_words", "kv_capacity_tokens",
                    "batch_size_avg", "preemptions", "prefix_hit_pct", "throughput_tok_s",
                    "ttft_p50_s", "tpot_p50_ms", "failed"]
        print_table(rows, cols)
    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()