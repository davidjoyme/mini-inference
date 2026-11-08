"""
bench/micro/bench_gemm.py - the model's matmuls at decode batch sizes: how close to peak bandwidth?

In decode, every linear layer multiplies a few rows (one per request) by a big weight matrix, so
its cost should be reading the weights once: time ~ weight bytes / memory bandwidth, nearly the
same for 1 row or 16. This measures each projection of a model (default Qwen3-4B) with PyTorch's
F.linear (cuBLAS), at 1/4/16/64 rows:

    ms, GB/s, % peak    per shape; bytes = weights + inputs + outputs
    kernel              the GPU kernel cuBLAS picked (tensor-core "cutlass"/"sm90"/... or "gemv")
    fused               q+k+v as one matmul and gate+up as one (what vLLM does): fewer, wider matmuls
    W^T                 the same matmul with the weight stored transposed ([in, out] instead of
                        [out, in]): a different memory layout can make cuBLAS choose another kernel

and adds up one decode step's matmuls (all layers + the output layer) for separate vs fused,
against the time reading those weights at peak bandwidth would take.

Weights are rotated through several copies totalling >= 256 MB, so each call reads from GPU memory
rather than the 96 MB L2 cache (which would make the numbers look better than in the real model).

Run from the project root on the GPU:
    python bench/micro/bench_gemm.py                                # Qwen3-4B
    python bench/micro/bench_gemm.py --model olmoe --out results/micro_gemm_olmoe.json
"""
import argparse
import json
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import config  # noqa: E402
from profiling.analyze import peak_bandwidth  # noqa: E402

L2_DEFEAT_BYTES = 256 * 1024 ** 2


class _quiet_native_output:
    """Hide C++ prints (PyTorch 2.14's 'USDT ... profiler_start/stop') while profiling."""
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


def shapes_for(cfg):
    """(name, out_features, in_features, count per decode step) for a decoder's linear layers."""
    H = cfg.num_attention_heads
    KV = getattr(cfg, "num_key_value_heads", None) or H
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    hidden, layers = cfg.hidden_size, cfg.num_hidden_layers
    moe = bool(getattr(cfg, "num_experts", None))
    shapes = [("q_proj", H * D, hidden, layers), ("k_proj", KV * D, hidden, layers),
              ("v_proj", KV * D, hidden, layers), ("o_proj", hidden, H * D, layers)]
    fused = [("qkv (fused)", (H + 2 * KV) * D, hidden, layers)]
    if not moe:   # MoE experts are measured by bench_moe.py
        inter = cfg.intermediate_size
        shapes += [("gate_proj", inter, hidden, layers), ("up_proj", inter, hidden, layers),
                   ("down_proj", hidden, inter, layers)]
        fused += [("gate_up (fused)", 2 * inter, hidden, layers)]
    shapes.append(("lm_head", cfg.vocab_size, hidden, 1))
    return shapes, fused


def time_ms(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def kernel_name(fn):
    fn()
    torch.cuda.synchronize()
    with _quiet_native_output():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not events:
        return "?"
    main = max(events, key=lambda e: e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
    name = main.name
    return (name[:44] + "..") if len(name) > 46 else name


def measure(n, k, m, dtype):
    """One linear layer [m, k] x [n, k]^T, with weights rotated so they come from memory."""
    wbytes = n * k * torch.finfo(dtype).bits // 8
    copies = max(2, math.ceil(L2_DEFEAT_BYTES / wbytes))
    ws = [torch.randn(n, k, device="cuda", dtype=dtype) * 0.02 for _ in range(copies)]
    wts = [w.t().contiguous() for w in ws]                    # [k, n] storage for the W^T variant
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    it = {"i": 0}

    def normal():
        it["i"] = (it["i"] + 1) % copies
        return F.linear(x, ws[it["i"]])

    def transposed():
        it["i"] = (it["i"] + 1) % copies
        return x @ wts[it["i"]]

    t_n, t_t = time_ms(normal), time_ms(transposed)
    total_bytes = wbytes + (m * k + m * n) * torch.finfo(dtype).bits // 8
    out = {"ms": t_n, "ms_wt": t_t, "gbs": total_bytes / (t_n / 1e3) / 1e9,
           "gbs_wt": total_bytes / (t_t / 1e3) / 1e9, "weight_bytes": wbytes,
           "kernel": kernel_name(normal), "kernel_wt": kernel_name(transposed)}
    del ws, wts
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description="A model's linear layers at decode batch sizes")
    p.add_argument("--model", default="qwen3-4b", help="config.MODELS short name or HF id")
    p.add_argument("--rows", nargs="+", type=int, default=[1, 4, 16, 64], help="tokens per step (batch)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("Needs a CUDA GPU.")

    from transformers import AutoConfig
    name = config.MODELS.get(args.model, args.model)
    cfg = AutoConfig.from_pretrained(name)
    dtype = torch.bfloat16
    gpu = torch.cuda.get_device_name(0)
    peak = peak_bandwidth(gpu)
    shapes, fused = shapes_for(cfg)
    print(f"Model: {name} | GPU: {gpu} | peak bandwidth: {peak} GB/s | bf16, cuBLAS via F.linear\n")

    results = {}
    for m in args.rows:
        print(f"--- {m} row(s) per step ---")
        header = (f"{'layer':<16} {'out x in':>13} {'MB':>6} {'ms':>7} {'GB/s':>6} {'%peak':>6} "
                  f"{'W^T ms':>7} {'W^T %':>6}  kernel")
        print(header)
        print("-" * (len(header) + 30))
        rows = {}
        for lname, n, k, count in shapes + fused:
            r = measure(n, k, m, dtype)
            r.update(out=n, inp=k, per_step=count)
            rows[lname] = r
            pct = 100 * r["gbs"] / peak if peak else float("nan")
            pct_wt = 100 * r["gbs_wt"] / peak if peak else float("nan")
            print(f"{lname:<16} {f'{n}x{k}':>13} {r['weight_bytes'] / 1e6:>6.1f} {r['ms']:>7.3f} {r['gbs']:>6.0f} "
                  f"{pct:>5.1f}% {r['ms_wt']:>7.3f} {pct_wt:>5.1f}%  {r['kernel']}")

        # one decode step's matmuls: separate projections vs fused ones
        def step(names):
            ms = sum(rows[x]["ms"] * rows[x]["per_step"] for x in names)
            ms_best = sum(min(rows[x]["ms"], rows[x]["ms_wt"]) * rows[x]["per_step"] for x in names)
            byts = sum(rows[x]["weight_bytes"] * rows[x]["per_step"] for x in names)
            return ms, ms_best, byts
        separate = [s[0] for s in shapes]
        fused_names = [f[0] for f in fused] + [s[0] for s in shapes
                                                if s[0] in ("o_proj", "down_proj", "lm_head")]
        sep_ms, sep_best, byts = step(separate)
        fus_ms, fus_best, _ = step(fused_names)
        ideal = byts / (peak * 1e9) * 1e3 if peak else None
        print(f"\nOne decode step's matmuls ({byts / 1e9:.2f} GB of weights; at peak bandwidth: {ideal:.2f} ms)")
        print(f"  separate projections: {sep_ms:6.2f} ms  ({100 * ideal / sep_ms:.0f}% of peak)"
              f"   best layout per shape: {sep_best:6.2f} ms")
        print(f"  fused qkv / gate_up:  {fus_ms:6.2f} ms  ({100 * ideal / fus_ms:.0f}% of peak)"
              f"   best layout per shape: {fus_best:6.2f} ms\n")
        results[m] = {"shapes": rows, "step_separate_ms": sep_ms, "step_fused_ms": fus_ms,
                      "step_separate_best_layout_ms": sep_best, "step_fused_best_layout_ms": fus_best,
                      "step_weight_gb": byts / 1e9, "step_ideal_ms": ideal}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": name, "gpu": gpu, "peak_gbs": peak, "results": results}, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()