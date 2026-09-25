#!/usr/bin/env bash
# Nsight Systems: whole-run timeline (CPU threads, CUDA calls, kernels, NVTX labels).
# Run from the project root. Any extra arguments are passed to bench/run_engine.py:
#     bash profiling/scripts/nsys.sh -n 2 --max-new-tokens 20
# Open the resulting .nsys-rep file in the Nsight Systems GUI.
set -e
mkdir -p profiles
NVTX=1 nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o "profiles/nsys_$(date +%Y%m%d_%H%M%S)" \
    python bench/run_engine.py --warmup 1 "$@"