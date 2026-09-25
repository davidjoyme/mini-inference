#!/usr/bin/env bash
# Nsight Compute: detailed analysis of individual kernels. Slow, so keep runs tiny
# and limit which kernels are captured. Run from the project root:
#     bash profiling/scripts/ncu.sh                     # first 20 kernels after warmup
#     KERNEL=gemm bash profiling/scripts/ncu.sh         # only kernels whose name contains "gemm"
# Open the resulting .ncu-rep file in the Nsight Compute GUI.
set -e
mkdir -p profiles
FILTER=()
if [ -n "$KERNEL" ]; then
    FILTER=(--kernel-name "regex:$KERNEL")
fi
ncu \
    --set full \
    --launch-skip 200 \
    --launch-count 20 \
    "${FILTER[@]}" \
    -o "profiles/ncu_$(date +%Y%m%d_%H%M%S)" \
    python bench/run_engine.py -n 1 --warmup 0 --max-new-tokens 5 "$@"