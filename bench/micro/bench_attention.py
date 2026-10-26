"""
bench/micro/bench_attention.py - decode attention for one layer: gather + SDPA vs the paged Triton kernel.

Uses OLMoE's attention shape (16 query heads, 16 kv heads, head size 128) and the engine's own
KVCache, with pages handed out in shuffled order like a busy server's pool. For each batch size
and context length:
    sdpa ms       what ATTN_BACKEND=sdpa does: gather every request's keys/values into a padded
                  copy (dequantizing 8-bit caches), then PyTorch SDPA with a mask
    triton ms     ATTN_BACKEND=triton: kernels/paged_attention reads the cache in place
    kernels       GPU kernels per call for each
    copy MB       the temporary gathered copy the sdpa path writes and reads back
    GB/s, % peak  cache bytes the kernel must read / its time (decode attention is memory-bound,
                  so this is the number that matters), vs the GPU's peak bandwidth

Run from the project root on the GPU:
    python bench/micro/bench_attention.py
    python bench/micro/bench_attention.py --kv-dtype fp8 --ragged --out results/micro_attn_fp8.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.core.kv_cache import KVCache  # noqa: E402
from kernels.paged_attention.decode_triton import paged_decode_attention  # noqa: E402
from profiling.analyze import peak_bandwidth  # noqa: E402

H, KV, D, PAGE = 16, 16, 128, 16


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


def kernels_per_call(fn):
    fn()
    torch.cuda.synchronize()
    with _quiet_native_output():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        events = prof.events()
    return sum(1 for e in events if e.device_type == torch.autograd.DeviceType.CUDA)


def build(batch, ctx, kv_dtype, ragged, seed=0):
    g = torch.Generator().manual_seed(seed)
    if ragged:   # a mix of lengths, as on a real server: the sdpa path pads everyone to the longest
        lens = torch.randint(ctx // 4, ctx + 1, (batch,), generator=g)
        lens[0] = ctx
    else:
        lens = torch.full((batch,), ctx)
    pages = [-(-int(n) // PAGE) for n in lens]
    total = sum(pages)
    cache = KVCache("paged", 1, KV, D, total * PAGE, torch.bfloat16, "cuda", block_size=PAGE,
                    kv_dtype=kv_dtype)
    for s in range(0, total * PAGE, 65536):          # fill in chunks to bound temporary memory
        n = min(65536, total * PAGE - s)
        k = torch.randn(n, KV, D, device="cuda", dtype=torch.bfloat16)
        cache.write(0, torch.arange(s, s + n, device="cuda"), k, torch.randn_like(k))
    perm = torch.randperm(total, generator=g)
    table = torch.zeros(batch, max(pages), dtype=torch.int32)
    i = 0
    for b, n in enumerate(pages):
        table[b, :n] = perm[i:i + n].to(torch.int32)
        i += n
    # the sdpa path's inputs, exactly as BatchMetadata builds them: padded slot index + mask
    pos = torch.arange(int(lens.max()))
    idx = (table[:, pos // PAGE].long() * PAGE + pos % PAGE) * (pos[None, :] < lens[:, None])
    mask = (pos[None, :] < lens[:, None])[:, None, None, :]
    q = torch.randn(batch, H, D, generator=g).to("cuda", torch.bfloat16)
    return (cache, q, table.cuda(), lens.to(torch.int32).cuda(), idx.cuda(), mask.cuda(), int(lens.max()),
            int(lens.sum()))


def main():
    p = argparse.ArgumentParser(description="Decode attention: gather + SDPA vs paged Triton kernel")
    p.add_argument("--batch", nargs="+", type=int, default=[1, 16, 64])
    p.add_argument("--ctx", nargs="+", type=int, default=[256, 1024, 4096])
    p.add_argument("--kv-dtype", default="auto", choices=["auto", "fp8", "int8"])
    p.add_argument("--ragged", action="store_true", help="random context lengths up to --ctx (default: all equal)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("Needs a CUDA GPU.")

    gpu = torch.cuda.get_device_name(0)
    peak = peak_bandwidth(gpu)
    scale = D ** -0.5
    print(f"GPU: {gpu} | peak bandwidth: {peak or '?'} GB/s | KV cache: {args.kv_dtype} | "
          f"{'ragged' if args.ragged else 'equal'} context lengths | heads {H}/{KV}, head size {D}\n")
    header = (f"{'batch':>5} {'ctx':>6} {'sdpa ms':>9} {'triton ms':>10} {'speedup':>8} {'kernels':>9} "
              f"{'copy MB':>8} {'GB/s':>7} {'% peak':>7} {'rel err':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    with torch.no_grad():
        for B in args.batch:
            for L in args.ctx:
                cache, q, table, lens, idx, mask, max_len, tokens = build(B, L, args.kv_dtype, args.ragged)
                ks = None if cache.k_scale is None else cache.k_scale[0]
                vs = None if cache.v_scale is None else cache.v_scale[0]

                def run_sdpa():
                    kc, vc = cache.gather(0, idx)                                   # [B, L, KV, D]
                    return F.scaled_dot_product_attention(q[:, :, None, :], kc.transpose(1, 2),
                                                          vc.transpose(1, 2), attn_mask=mask, scale=scale)[:, :, 0]

                def run_triton():
                    return paged_decode_attention(q, cache.k[0], cache.v[0], table, lens, PAGE, scale, max_len,
                                                  k_scale=ks, v_scale=vs)

                ref, out = run_sdpa(), run_triton()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                t_s, t_t = time_ms(run_sdpa), time_ms(run_triton)
                k_s, k_t = kernels_per_call(run_sdpa), kernels_per_call(run_triton)
                bytes_read = tokens * KV * D * 2 * cache.k.element_size()        # keys + values
                if ks is not None:
                    bytes_read += tokens * KV * 2 * ks.element_size()           # their scales
                copy_mb = B * max_len * KV * D * 2 * 2 / 1e6                    # gathered bf16 copy
                gbs = bytes_read / (t_t / 1e3) / 1e9
                row = {"batch": B, "ctx": L, "tokens": tokens, "sdpa_ms": round(t_s, 4), "triton_ms": round(t_t, 4),
                       "speedup": round(t_s / t_t, 2), "kernels_sdpa": k_s, "kernels_triton": k_t,
                       "copy_mb": round(copy_mb, 1), "triton_gbs": round(gbs, 1),
                       "triton_pct_of_peak": round(100 * gbs / peak, 1) if peak else None,
                       "rel_error": round(err, 5)}
                rows.append(row)
                print(f"{B:>5} {L:>6} {t_s:>9.3f} {t_t:>10.3f} {row['speedup']:>7.2f}x {f'{k_s} -> {k_t}':>9} "
                      f"{copy_mb:>8.1f} {gbs:>7.0f} {str(row['triton_pct_of_peak']):>7} {err:>8.4f}")
                del cache, idx, mask
                torch.cuda.empty_cache()

    print("\nSmall cases (a few MB of cache) partly fit in the GPU's L2 cache, so their % of peak can"
          "\nlook high or low; the large cases show the real memory-bound behaviour.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"gpu": gpu, "kv_dtype": args.kv_dtype, "ragged": args.ragged, "rows": rows}, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()