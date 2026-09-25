"""
bench/test.py - server load test: send many requests to the server at the same time.

Defaults come from config.py; any of them can be overridden on the command line:
    python bench/test.py                          # use the settings in config.py
    python bench/test.py -n 1 -c 1                # single request (baseline)
    python bench/test.py -n 10 -c 10              # 10 requests, all at once
    python bench/test.py --no-stream              # use /generate (no TTFT/TPOT)
    python bench/test.py --out results_c8.json    # save to a different file

Streaming mode (default) measures:
    TTFT - time to first token (includes queue time + prefill)
    TPOT - time per output token after the first one
    ITL  - gap between consecutive streamed chunks
"""
import argparse
import asyncio
import json
import time

import httpx

import os
import sys

# Make the project root importable when run as `python bench/test.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


# ---------------- One non-streaming request ----------------
async def send_request(client, sem, args, request_id, test_start):
    payload = {"prompt": f"{args.prompt} (request {request_id})", "max_new_tokens": args.max_new_tokens}

    async with sem:
        sent_at = time.perf_counter()
        result = {"id": request_id, "sent_at_s": round(sent_at - test_start, 3)}
        try:
            resp = await client.post(args.url, json=payload)
            result["status"] = resp.status_code
            body = resp.json()
            result["output"] = str(body.get("output", body))[:200]
            for key in ("generated_tokens", "queue_time_s", "generation_time_s"):
                if key in body:
                    result[key] = body[key]
        except Exception as e:
            result.setdefault("status", None)
            result["error"] = f"{type(e).__name__}: {e}"
        result["latency_s"] = round(time.perf_counter() - sent_at, 3)

    print_request(result)
    return result


# ---------------- One streaming request ----------------
async def send_stream_request(client, sem, args, request_id, test_start):
    payload = {"prompt": f"{args.prompt} (request {request_id})", "max_new_tokens": args.max_new_tokens}

    async with sem:
        sent_at = time.perf_counter()
        result = {"id": request_id, "sent_at_s": round(sent_at - test_start, 3)}
        first_at = last_at = None
        itls = []        # gaps between consecutive chunks, in seconds
        pieces = []
        final = {}
        try:
            async with client.stream("POST", args.url, json=payload) as resp:
                result["status"] = resp.status_code
                if resp.status_code != 200:
                    result["error"] = (await resp.aread()).decode()[:200]
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        if event.get("done"):
                            final = event       # server-side metrics
                            continue
                        now = time.perf_counter()
                        if first_at is None:
                            first_at = now
                        else:
                            itls.append(now - last_at)
                        last_at = now
                        pieces.append(event.get("text", ""))
        except Exception as e:
            result.setdefault("status", None)
            result["error"] = f"{type(e).__name__}: {e}"

        end_at = time.perf_counter()

    result["latency_s"] = round(end_at - sent_at, 3)
    result["output"] = "".join(pieces)[:200]
    if "error" in final:
        result["error"] = final["error"]

    tokens = final.get("generated_tokens")
    result["generated_tokens"] = tokens
    for key in ("queue_time_s", "prefill_time_s", "decode_time_s", "generation_time_s"):
        if key in final:
            result[key] = final[key]

    if first_at is not None:
        result["ttft_s"] = round(first_at - sent_at, 3)
        if tokens and tokens > 1:
            # TPOT: time after the first token, divided by the remaining tokens
            result["tpot_ms"] = round((end_at - first_at) / (tokens - 1) * 1000, 2)
        result["itl_ms"] = [round(x * 1000, 2) for x in itls]

    print_request(result)
    return result


def print_request(r):
    ok = r.get("status") == 200 and "error" not in r
    line = (f"[req {r['id']:>3}] {'OK ' if ok else 'ERR'} status={r.get('status')} "
            f"latency={r['latency_s']:.2f}s tokens={r.get('generated_tokens', '-')}")
    if "ttft_s" in r:
        line += f" TTFT={r['ttft_s']:.2f}s TPOT={r.get('tpot_ms', '-')}ms"
    if "queue_time_s" in r:
        line += f" queue={r['queue_time_s']}s"
    if "error" in r:
        line += f" error={r['error'][:80]}"
    print(line)


# ---------------- Memory sampling ----------------
async def sample_memory(client, base_url, interval, test_start, timeline, stop_event):
    """Poll the server's memory endpoint in the background while requests run."""
    while not stop_event.is_set():
        try:
            snap = (await client.get(f"{base_url}/metrics/memory")).json()
            snap["t_s"] = round(time.perf_counter() - test_start, 3)
            timeline.append(snap)
        except Exception:
            pass   # a missed sample isn't worth failing the test over
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def summarize_memory(baseline, final, timeline):
    """Baseline, peak and increase for each memory metric."""
    def peak(key):
        values = [s[key] for s in timeline + [final] if s.get(key) is not None]
        return max(values) if values else None

    summary = {"baseline": baseline}
    if "gpu_allocated_mb" in baseline:
        summary["gpu_peak_allocated_mb"] = final.get("gpu_peak_allocated_mb")   # exact, from PyTorch
        summary["gpu_peak_reserved_mb"] = final.get("gpu_peak_reserved_mb")
        summary["gpu_peak_device_used_mb"] = peak("gpu_device_used_mb")        # sampled, like nvidia-smi
        summary["gpu_device_total_mb"] = baseline.get("gpu_device_total_mb")
        if summary["gpu_peak_allocated_mb"] is not None:
            summary["gpu_allocated_increase_mb"] = round(
                summary["gpu_peak_allocated_mb"] - baseline["gpu_allocated_mb"], 1)
    summary["process_peak_rss_mb_sampled"] = peak("process_rss_mb")
    if baseline.get("process_rss_mb") is not None and summary["process_peak_rss_mb_sampled"] is not None:
        summary["process_rss_increase_mb"] = round(
            summary["process_peak_rss_mb_sampled"] - baseline["process_rss_mb"], 1)
    summary["samples"] = len(timeline)
    return summary


# ---------------- Statistics ----------------
def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def stats(values):
    """avg / p50 / p95 / p99 / max for a list of numbers, or None if empty."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "avg": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "max": round(max(values), 3),
    }


async def main():
    parser = argparse.ArgumentParser(description="Concurrent test bench for the inference server")
    parser.add_argument("--url", default=None, help="override the endpoint URL")
    parser.add_argument("-n", "--num-requests", type=int, default=config.TEST_NUM_REQUESTS)
    parser.add_argument("-c", "--concurrency", type=int, default=config.TEST_CONCURRENCY)
    parser.add_argument("--prompt", default=config.TEST_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=config.TEST_MAX_NEW_TOKENS)
    parser.add_argument("--timeout", type=float, default=config.TEST_TIMEOUT_S)
    parser.add_argument("--out", default=config.TEST_OUTPUT_FILE)
    parser.add_argument("--stream", dest="stream", action="store_true", default=config.TEST_STREAM)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--memory", dest="memory", action="store_true", default=config.TEST_MEMORY)
    parser.add_argument("--no-memory", dest="memory", action="store_false")
    parser.add_argument("--mem-interval", type=float, default=config.TEST_MEMORY_INTERVAL_S,
                        help="seconds between memory samples")
    args = parser.parse_args()

    if args.url is None:
        args.url = config.TEST_STREAM_URL if args.stream else config.TEST_URL

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        # Ask the server which model and settings it's running, so results are labelled
        health_url = args.url.rsplit("/", 1)[0] + "/health"
        try:
            server_info = (await client.get(health_url)).json()
        except Exception as e:
            print(f"Could not reach {health_url} ({type(e).__name__}). Is the server running?")
            return

        print(f"Server model: {server_info.get('model')} on {server_info.get('device')}")
        print(f"Sending {args.num_requests} requests to {args.url} "
              f"(concurrency={args.concurrency}, max_new_tokens={args.max_new_tokens}, "
              f"stream={args.stream})\n")

        base_url = args.url.rsplit("/", 1)[0]
        mem_baseline = None
        if args.memory:
            try:
                # Reset GPU peak counters so this run's peak isn't mixed with earlier runs
                resp = await client.post(f"{base_url}/metrics/memory/reset")
                resp.raise_for_status()
                mem_baseline = resp.json()
            except Exception as e:
                print(f"Memory metrics unavailable ({type(e).__name__}); continuing without them.")

        sender = send_stream_request if args.stream else send_request
        sem = asyncio.Semaphore(args.concurrency)
        test_start = time.perf_counter()

        timeline, stop_event, sampler = [], asyncio.Event(), None
        if mem_baseline is not None:
            sampler = asyncio.create_task(
                sample_memory(client, base_url, args.mem_interval, test_start, timeline, stop_event))

        tasks = [sender(client, sem, args, i, test_start) for i in range(args.num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - test_start

        mem_summary = None
        if sampler is not None:
            stop_event.set()
            await sampler
            try:
                mem_final = (await client.get(f"{base_url}/metrics/memory")).json()
            except Exception:
                mem_final = timeline[-1] if timeline else mem_baseline
            mem_summary = summarize_memory(mem_baseline, mem_final, timeline)

    ok = [r for r in results if r.get("status") == 200 and "error" not in r]
    total_tokens = sum(r.get("generated_tokens") or 0 for r in ok)

    summary = {
        "server": server_info,
        "test_config": {
            "url": args.url,
            "stream": args.stream,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "max_new_tokens": args.max_new_tokens,
            "prompt": args.prompt,
        },
        "total_time_s": round(total_time, 3),
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "requests_per_s": round(len(ok) / total_time, 3),
        "total_generated_tokens": total_tokens,
        "overall_tokens_per_s": round(total_tokens / total_time, 2),   # throughput across all requests
        "latency_s": stats([r["latency_s"] for r in ok]),
        "queue_time_s": stats([r.get("queue_time_s") for r in ok]),
    }
    if args.stream:
        summary["ttft_s"] = stats([r.get("ttft_s") for r in ok])
        summary["tpot_ms"] = stats([r.get("tpot_ms") for r in ok])
        summary["itl_ms"] = stats([x for r in ok for x in r.get("itl_ms", [])])
        summary["prefill_time_s"] = stats([r.get("prefill_time_s") for r in ok])
    if mem_summary is not None:
        summary["memory"] = mem_summary

    print("\n===== Summary =====")
    for key, value in summary.items():
        if key in ("server", "test_config", "memory"):
            continue
        print(f"{key:>22}: {value}")

    if mem_summary is not None:
        print("\n===== Memory =====")
        for key, value in mem_summary.items():
            if key == "baseline":
                continue
            print(f"{key:>30}: {value}")

    output = {"summary": summary, "requests": sorted(results, key=lambda r: r["id"])}
    if timeline:
        output["memory_timeline"] = timeline
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())