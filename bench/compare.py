"""
bench/compare.py - put results from several runs side by side.

Understands the JSON files written by:
    bench/run_engine.py --out ...     (single-request speed, memory)
    bench/test.py --out ...           (server under load)
    bench/eval_quality.py --out ...   (perplexity, agreement with bf16)

Run from the project root:
    python bench/compare.py engine_*.json
    python bench/compare.py load_*.json
    python bench/compare.py quality_*.json
    python bench/compare.py results/*.json --csv summary.csv
"""
import argparse
import csv
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.factory import engine_label  # noqa: E402



def short(model_id):
    """'allenai/OLMoE-1B-7B-0125-Instruct' -> 'OLMoE-1B-7B-0125-Instruct' (older results: unknown)"""
    return model_id.split("/")[-1] if model_id else None

def get(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def row_for(path):
    data = json.load(open(path))
    s = data.get("summary", {})
    name = os.path.basename(path)

    if "perplexity" in data:                                   # eval_quality.py
        ref = data.get("vs_reference", {})
        return "quality", {
            "file": name,
            "model": short(data.get("model")),
            "engine": data.get("engine_label", "hf"),
            "quant": data.get("quantization"),
            "weights_MB": get(data, "engine", "weight_memory", "weights_total_mb"),
            "perplexity": data.get("perplexity"),
            "ppl_change_%": ref.get("perplexity_change_pct"),
            "exact_match": ref.get("exact_match_rate"),
            "matching_prefix": ref.get("avg_matching_prefix"),
        }
    if "avg_tpot_ms" in s:                                     # run_engine.py
        return "engine", {
            "file": name,
            "model": short(s.get("model") or get(s, "engine", "model")),
            "engine": s.get("engine_label", "hf"),
            "conc": s.get("concurrency", 1),
            "quant": s.get("quantization"),
            "weights_MB": s.get("weights_total_mb"),
            "gpu_peak_MB": s.get("gpu_peak_allocated_mb"),
            "TTFT_s": s.get("avg_ttft_s"),
            "TPOT_ms": s.get("avg_tpot_ms"),
            "per_user_tok/s": s.get("avg_tokens_per_s"),
            "latency_s": s.get("avg_latency_s"),
            "total_tok/s": s.get("throughput_tok_s"),
            "avg_batch": s.get("avg_batch_size"),
            "preempt": s.get("preemptions"),
            "prefix_hit_%": s.get("prefix_hit_pct"),
            "max_gap_ms": s.get("max_itl_ms"),
        }
    if "overall_tokens_per_s" in s:                            # test.py
        eng = get(s, "server", "engine") or {}
        return "load", {
            "file": name,
            "model": short(get(s, "server", "model")),
            "engine": engine_label({"engine": eng}),
            "quant": get(s, "server", "quantization", "method"),
            "conc": get(s, "test_config", "concurrency"),
            "reqs": get(s, "test_config", "num_requests"),
            "out_tok": get(s, "test_config", "max_new_tokens"),
            "in_words": get(s, "test_config", "prompt_words"),
            "failed": s.get("failed"),
            "total_tok/s": s.get("overall_tokens_per_s"),
            "TTFT_p50_s": get(s, "ttft_s", "p50"),
            "TTFT_p95_s": get(s, "ttft_s", "p95"),
            "TPOT_p50_ms": get(s, "tpot_ms", "p50"),
            "ITL_p95_ms": get(s, "itl_ms", "p95"),
            "ITL_max_ms": get(s, "itl_ms", "max"),
            "gpu_peak_MB": get(s, "memory", "gpu_peak_allocated_mb"),
            "gpu_extra_MB": get(s, "memory", "gpu_allocated_increase_mb"),
            "gpu_util_%": get(s, "memory", "gpu_util_pct_avg"),
            "power_W": get(s, "memory", "gpu_power_w_avg"),
            "ram_peak_MB": get(s, "memory", "process_peak_rss_mb_sampled"),
            "batch_avg": get(s, "memory", "batch_size_avg"),
            "kv_waste_%": get(s, "memory", "kv_waste_pct_avg"),
            "preempt": get(s, "memory", "preemptions"),
            "padding_%": get(s, "memory", "padding_waste_pct"),
            "prefix_hit_%": get(s, "memory", "prefix_hit_pct"),
        }
    return None, None


def print_table(rows):
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark result files")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--csv", default=None, help="also save the tables to a CSV file")
    args = parser.parse_args()

    files = sorted({f for pattern in args.files for f in glob.glob(pattern)})
    groups = {}
    for f in files:
        try:
            kind, row = row_for(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"skipping {f}: {e}")
            continue
        if kind:
            groups.setdefault(kind, []).append(row)
        else:
            print(f"skipping {f}: not a recognised results file")

    titles = {"engine": "Direct engine runs, no HTTP (run_engine.py)", "load": "Server under load (test.py)",
              "quality": "Quality (eval_quality.py)"}
    for kind, rows in groups.items():
        print(f"\n===== {titles[kind]} =====")
        print_table(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            for kind, rows in groups.items():
                w = csv.DictWriter(f, fieldnames=["table", *rows[0]])
                w.writeheader()
                for r in rows:
                    w.writerow({"table": kind, **r})
        print(f"\nSaved to {args.csv}")


if __name__ == "__main__":
    main()