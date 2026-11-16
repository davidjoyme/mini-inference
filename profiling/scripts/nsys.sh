set -e
mkdir -p profiles
NVTX=1 nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o "profiles/nsys_$(date +%Y%m%d_%H%M%S)" \
    python bench/run_engine.py --warmup 1 "$@"