#!/usr/bin/env bash
# Profile a few configurations, each chosen to expose a different bottleneck, then print a
# side-by-side summary. Run from the project root (don't run main.py at the same time):
#     bash profiling/scripts/profile_suite.sh
#     bash profiling/scripts/profile_suite.sh custom_bs1 custom_bs64     # only some of them
#
# Each run loads the model, does a warmup request (not profiled), profiles the measured
# requests, and writes into profiles/suite_<time>/<name>/:
#     *.json              the trace (open at https://ui.perfetto.dev)
#     *.report.json       the bottleneck report (also printed)
#     result.json         latency / throughput numbers from bench/run_engine.py
set -e
OUT="profiles/suite_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

declare -A CONFIGS=(
  # one request at a time: per-token cost, overhead with no batching to hide it
  [hf_bs1]="--engine hf -n 3 -c 1 --max-new-tokens 64"
  [custom_bs1]="--engine custom -n 3 -c 1 --max-new-tokens 64"
  # batched decode: does the GPU stay busy, how much of peak bandwidth do we reach?
  [custom_bs16]="--engine custom -n 32 -c 16 --max-new-tokens 64"
  [custom_bs64]="--engine custom -n 64 -c 64 --max-new-tokens 64"
  # quantized weights: cost of rebuilding weights + the per-expert loop
  [custom_bs16_int4]="--engine custom --quant int4_wo -n 16 -c 16 --max-new-tokens 32"
  # long contexts: attention and the KV gather grow with context length
  [custom_long_ctx]="--engine custom -n 16 -c 16 --max-new-tokens 64 --prompt-words 2000 --unique-prompts"
)
ORDER=(hf_bs1 custom_bs1 custom_bs16 custom_bs64 custom_bs16_int4 custom_long_ctx)
if [ "$#" -gt 0 ]; then ORDER=("$@"); fi

for name in "${ORDER[@]}"; do
  args="${CONFIGS[$name]}"
  if [ -z "$args" ]; then echo "Unknown config '$name' (choose from: ${!CONFIGS[*]})"; exit 1; fi
  echo; echo "################ $name: $args"
  mkdir -p "$OUT/$name"
  python bench/run_engine.py $args --warmup 1 --profile torch --profile-dir "$OUT/$name" \
      --out "$OUT/$name/result.json" 2>&1 | tee "$OUT/$name/log.txt" \
      | sed -n '/Bottleneck report/,/Report saved/p'
done

echo; echo "################ Summary"
python -m profiling.analyze --summary "$OUT/*/*.report.json"
echo; echo "Everything is in $OUT"