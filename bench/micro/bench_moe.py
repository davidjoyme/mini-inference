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


class _quiet_native_output:
    """Hide output printed by C++ code (e.g. PyTorch 2.14's 'USDT ... profiler_start/stop' lines),
    which Python's own redirection can't catch, by pointing file descriptors 1 and 2 at /dev/null."""
    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.saved = [os.dup(1), os.dup(2)]
        os.dup2(self.devnull, 1)
        os.dup2(self.devnull, 2)

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        for fd in (*self.saved, self.devnull):
            os.close(fd)


def kernels_per_call(fn):
    fn()
    torch.cuda.synchronize()
    with _quiet_native_output():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        events = prof.events()
    return sum(1 for e in events if e.device_type == torch.autograd.DeviceType.CUDA)


def cpu_launch_us(fn, iters=50):
    """Host time to issue one call (kernels are queued, not waited for). When this is larger than
    the GPU time, the GPU idles between kernels - the 'launch-bound' regime of small decode batches."""
    import time
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t
    torch.cuda.synchronize()
    return elapsed / iters * 1e6


def main():
    p = argparse.ArgumentParser(description="MoE layer: default vs fused Triton kernels")
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024, 4096])
    p.add_argument("--align", default="auto", choices=["auto", "cuda", "torch"],
                   help="moe_align used by the 'fused' column (the 'fused/torch' column always uses torch)")
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
    fused_torch = FusedExperts(default_experts, align="torch")     # round 1: ~35-op PyTorch moe_align
    fused_experts = FusedExperts(default_experts, align=args.align)  # round 2a: one CUDA kernel
    from kernels.fused_moe.module import align_backend
    align_name = align_backend(args.align)
    impl = getattr(model.config, "_experts_implementation", "?")
    H, I, E, k = 2048, 1024, 64, 8
    bytes_per_expert = 3 * H * I * 2                       # gate + up + down, bf16
    peak = peak_bandwidth(torch.cuda.get_device_name(0))
    print(f"GPU: {torch.cuda.get_device_name(0)} | default experts implementation: {impl} | "
          f"peak bandwidth: {peak or '?'} GB/s | fused moe_align: {align_name}\n")
    print("default = transformers | fused/torch = Triton matmuls + PyTorch moe_align | "
          f"fused = Triton matmuls + {align_name} moe_align\n"
          "cpu us  = host time to launch one layer call (what CUDA graphs / fewer launches save)\n")

    rows = []
    header = (f"{'tokens':>7} {'default ms':>11} {'fused/torch':>12} {'fused ms':>9} {'speedup':>8} "
              f"{'kernels def/torch/fused':>24} {'cpu us torch->fused':>20} "
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
            run_ftorch = lambda: fused_torch(h, idx, w)        # noqa: E731

            ref, out = run_default(), run_fused()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            t_def, t_ft, t_fus = time_ms(run_default), time_ms(run_ftorch), time_ms(run_fused)
            k_def, k_ft, k_fus = (kernels_per_call(run_default), kernels_per_call(run_ftorch),
                                  kernels_per_call(run_fused))
            c_ft, c_fus = cpu_launch_us(run_ftorch), cpu_launch_us(run_fused)
            touched = idx.unique().numel()
            gbs = touched * bytes_per_expert / (t_fus / 1e3) / 1e9
            row = {"tokens": T, "default_ms": round(t_def, 4), "fused_torch_align_ms": round(t_ft, 4),
                   "fused_ms": round(t_fus, 4), "speedup": round(t_def / t_fus, 2),
                   "kernels_default": k_def, "kernels_fused_torch_align": k_ft, "kernels_fused": k_fus,
                   "cpu_launch_us_torch_align": round(c_ft, 1), "cpu_launch_us_fused": round(c_fus, 1),
                   "experts_touched": touched, "fused_gbs": round(gbs, 1),
                   "fused_pct_of_peak": round(100 * gbs / peak, 1) if peak else None,
                   "rel_error": round(err, 5)}
            rows.append(row)
            print(f"{T:>7} {t_def:>11.3f} {t_ft:>12.3f} {t_fus:>9.3f} {row['speedup']:>7.2f}x "
                  f"{f'{k_def} / {k_ft} / {k_fus}':>24} {f'{c_ft:.0f} -> {c_fus:.0f}':>20} "
                  f"{touched:>8} {gbs:>11.1f} {str(row['fused_pct_of_peak']):>7} {err:>8.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "default_impl": impl,
                       "moe_align": align_name, "rows": rows}, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()