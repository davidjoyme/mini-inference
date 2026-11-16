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