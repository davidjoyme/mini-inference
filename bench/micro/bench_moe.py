"""
bench/micro/bench_moe.py - one MoE layer, transformers' default path vs the fused Triton kernels.

Uses OLMoE's real layer sizes (hidden 2048, 64 experts of intermediate 1024, top-8) with random
weights, for a range of token counts (decode batch sizes and prefill sizes). For each:
    time per layer call     CUDA-event timing, after warmup
    kernels per call        how many GPU kernels one MoE layer launches (from the profiler)
    effective bandwidth     expert weights actually read / time, vs the GPU's peak
    relative error          fused output vs the default output (bf16 rounding is expected)

Run from the project root on the GPU (needs CUDA):
    python bench/micro/bench_moe.py
    python bench/micro/bench_moe.py --tokens 1 8 64 512 --out results/micro_moe.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from transformers import OlmoeConfig, OlmoeForCausalLM  # noqa: E402

from kernels.fused_moe.module import FusedExperts  # noqa: E402
from profiling.analyze import peak_bandwidth  # noqa: E402


def time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def kernels_per_call(fn):
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def main():
    p = argparse.ArgumentParser(description="MoE layer: default vs fused Triton kernels")
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024, 4096])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("Needs a CUDA GPU.")

    # one OLMoE layer at real sizes (tiny vocab so it's small)
    cfg = OlmoeConfig(hidden_size=2048, intermediate_size=1024, num_experts=64, num_experts_per_tok=8,
                      num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=16, vocab_size=1024)
    torch.manual_seed(0)
    model = OlmoeForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    for prm in model.parameters():
        prm.data.normal_(0, 0.02)
    moe = model.model.layers[0].mlp
    default_experts = moe.experts
    fused_experts = FusedExperts(default_experts)
    impl = getattr(model.config, "_experts_implementation", "?")
    H, I, E, k = 2048, 1024, 64, 8
    bytes_per_expert = 3 * H * I * 2                       # gate + up + down, bf16
    peak = peak_bandwidth(torch.cuda.get_device_name(0))
    print(f"GPU: {torch.cuda.get_device_name(0)} | default experts implementation: {impl} | "
          f"peak bandwidth: {peak or '?'} GB/s\n")

    rows = []
    header = (f"{'tokens':>7} {'default ms':>11} {'fused ms':>9} {'speedup':>8} {'kernels':>13} "
              f"{'experts':>8} {'fused GB/s':>11} {'% peak':>7} {'rel err':>8}")
    print(header)
    print("-" * len(header))
    with torch.no_grad():
        for T in args.tokens:
            h = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
            # routing as OLMoE does it: softmax over experts, keep the top-k (weights in bf16)
            probs = torch.softmax(h.float() @ moe.gate.weight.float().t(), dim=-1)
            w, idx = torch.topk(probs, k, dim=-1)
            if getattr(cfg, "norm_topk_prob", False):
                w = w / w.sum(dim=-1, keepdim=True)
            w = w.to(torch.bfloat16)
            run_default = lambda: default_experts(h, idx, w)   # noqa: E731
            run_fused = lambda: fused_experts(h, idx, w)       # noqa: E731

            ref, out = run_default(), run_fused()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            t_def, t_fus = time_ms(run_default), time_ms(run_fused)
            k_def, k_fus = kernels_per_call(run_default), kernels_per_call(run_fused)
            touched = idx.unique().numel()
            gbs = touched * bytes_per_expert / (t_fus / 1e3) / 1e9
            row = {"tokens": T, "default_ms": round(t_def, 4), "fused_ms": round(t_fus, 4),
                   "speedup": round(t_def / t_fus, 2), "kernels_default": k_def, "kernels_fused": k_fus,
                   "experts_touched": touched, "fused_gbs": round(gbs, 1),
                   "fused_pct_of_peak": round(100 * gbs / peak, 1) if peak else None,
                   "rel_error": round(err, 5)}
            rows.append(row)
            print(f"{T:>7} {t_def:>11.3f} {t_fus:>9.3f} {row['speedup']:>7.2f}x {f'{k_def} -> {k_fus}':>13} "
                  f"{touched:>8} {gbs:>11.1f} {str(row['fused_pct_of_peak']):>7} {err:>8.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "default_impl": impl, "rows": rows}, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()