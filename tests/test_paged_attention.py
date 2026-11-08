"""
tests/test_paged_attention.py - the paged decode-attention Triton kernel against plain PyTorch.

Without a GPU the kernel runs in Triton's interpreter (TRITON_INTERPRET=1) in float32, which checks
the indexing and math exactly; with a GPU it runs for real in bfloat16, including OLMoE's shapes
and an FP8 cache.

Run from the project root:
    python tests/test_paged_attention.py
"""
import os
import sys
import torch
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"        # must be set before triton is imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile  # noqa: E402
import config  # noqa: E402
from engine.core.kv_cache import KVCache  # noqa: E402
from kernels.paged_attention.decode_triton import paged_decode_attention  # noqa: E402
from kernels.paged_attention.reference import paged_decode_reference  # noqa: E402

GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"
DTYPE = torch.bfloat16 if GPU else torch.float32
TOL = 1e-2 if GPU else 1e-5


def rel_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def random_paged(lens, H, KV, D, page, dtype=DTYPE, seed=0):
    """A cache whose pages are handed out in shuffled order (like a real, fragmented pool)."""
    g = torch.Generator().manual_seed(seed)
    pages = [-(-n // page) for n in lens]
    total = sum(pages) + 3
    k = torch.randn(total * page, KV, D, generator=g).to(DEVICE, dtype)
    v = torch.randn(total * page, KV, D, generator=g).to(DEVICE, dtype)
    perm = torch.randperm(total, generator=g).tolist()
    table = torch.zeros(len(lens), max(pages), dtype=torch.int32)
    i = 0
    for b, n in enumerate(pages):
        table[b, :n] = torch.tensor(perm[i:i + n])
        i += n
    q = torch.randn(len(lens), H, D, generator=g).to(DEVICE, dtype)
    return q, k, v, table.to(DEVICE), torch.tensor(lens, dtype=torch.int32, device=DEVICE)


def test_matches_reference():
    cases = [  # (lens, H, KV, D, page)
        ([1, 17, 40], 4, 4, 32, 16),       # length 1, a partial last page, several pages
        ([33, 5], 8, 2, 32, 16),           # GQA: 4 query heads per kv head
        ([9, 30], 4, 4, 32, 1),            # token-level table (how the contiguous cache is passed)
        ([50], 4, 4, 24, 16),              # head size that isn't a power of 2
    ]
    if GPU:
        cases += [([1, 100, 513, 2048], 16, 16, 128, 16),         # OLMoE shapes
                  ([4096] * 2 + [17] * 6, 16, 16, 128, 16)]
    for lens, H, KV, D, page in cases:
        q, k, v, table, sl = random_paged(lens, H, KV, D, page)
        want = paged_decode_reference(q, k, v, table, sl, page, D ** -0.5)
        got = paged_decode_attention(q, k, v, table, sl, page, D ** -0.5, max(lens))
        err = rel_error(got, want)
        print(f"  lens={str(lens)[:28]:<28} H={H:<3} KV={KV:<3} D={D:<4} page={page:<3}: rel error {err:.2e}")
        assert err < TOL


def test_split_counts_agree():
    """1 split (no merge kernel), several, and as many as there are tiles: same answer."""
    lens = [70, 3, 129]
    q, k, v, table, sl = random_paged(lens, 4, 4, 32, 16, seed=1)
    want = paged_decode_reference(q, k, v, table, sl, 16, 32 ** -0.5)
    for splits in (1, 2, 3, 9):
        got = paged_decode_attention(q, k, v, table, sl, 16, 32 ** -0.5, max(lens), num_splits=splits, block_n=16)
        assert rel_error(got, want) < TOL, splits


def test_quantized_cache():
    """8-bit caches written by the engine's KVCache: the kernel dequantizes in registers."""
    kinds = ["int8"] + (["fp8"] if GPU else [])
    H = KV = 4
    D, page = 32, 16
    for kind in kinds:
        cache = KVCache("paged", 1, KV, D, 256, DTYPE, DEVICE, block_size=page, kv_dtype=kind)
        lens = [40, 7, 90]
        q, k, v, table, sl = random_paged(lens, H, KV, D, page, seed=2)
        cache.write(0, torch.arange(k.shape[0], device=DEVICE), k, v)
        args = (q, cache.k[0], cache.v[0], table, sl, page, D ** -0.5)
        scales = dict(k_scale=cache.k_scale[0], v_scale=cache.v_scale[0])
        want = paged_decode_reference(*args, **scales)                 # same stored values
        got = paged_decode_attention(*args, max(lens), **scales)
        err, full = rel_error(got, want), rel_error(got, paged_decode_reference(q, k, v, table, sl, page, D ** -0.5))
        print(f"  {kind}: vs dequantized reference {err:.2e}, vs unquantized cache {full:.2e}")
        assert err < TOL and full < 0.1


def test_engine_outputs_exact():
    """Custom engine with ATTN_BACKEND=triton reproduces model.generate() (float32, both KV caches)."""
    if GPU:
        print("  (runs on CPU only: exact float32 comparison)")
        return
    from tests.test_batching import (LONG_PROMPTS, PROMPTS, build_tiny_model, make_engine,
                                     reference_outputs, run_concurrently)
    from transformers import PreTrainedTokenizerFast
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tok = PreTrainedTokenizerFast.from_pretrained(d)
        saved = config.ATTN_BACKEND
        config.ATTN_BACKEND = "triton"
        try:
            for kv, prompts, kv_tokens in (("paged", PROMPTS, 4096), ("contiguous", PROMPTS, 4096),
                                           ("paged", LONG_PROMPTS, 160)):    # tiny pool: preemption
                refs = reference_outputs(model, tok, prompts)
                engine = make_engine(d, "continuous", kv, kv_tokens=kv_tokens)
                assert engine.runner.attn_backend == "triton"
                outs, _ = run_concurrently(engine, prompts)
                print(f"  continuous/{kv} ({len(prompts)} requests, kv_tokens={kv_tokens}): "
                      f"{'identical' if outs == refs else 'DIFFERENT'}")
                assert outs == refs

            # with prefix caching (decode reads blocks shared between requests) and chunked prefill
            saved_pc, saved_ch, saved_bt = config.PREFIX_CACHING, config.CHUNKED_PREFILL, config.MAX_BATCHED_TOKENS
            config.PREFIX_CACHING, config.CHUNKED_PREFILL, config.MAX_BATCHED_TOKENS = True, True, 8
            try:
                shared = " ".join(f"w{100 + i}" for i in range(40))
                prompts = [(f"{shared} w{i}", 6) for i in range(5)]
                refs = reference_outputs(model, tok, prompts)
                engine = make_engine(d, "continuous", "paged")
                outs, _ = run_concurrently(engine, prompts)
                print(f"  continuous/paged + prefix caching + chunked prefill (budget 8): "
                      f"{'identical' if outs == refs else 'DIFFERENT'} "
                      f"(prefix hit {engine.stats().get('prefix_hit_pct')}%)")
                assert outs == refs
            finally:
                config.PREFIX_CACHING, config.CHUNKED_PREFILL, config.MAX_BATCHED_TOKENS = saved_pc, saved_ch, saved_bt
        finally:
            config.ATTN_BACKEND = saved


if __name__ == "__main__":
    print(f"Running on {DEVICE} ({'real kernels' if GPU else 'Triton interpreter'}), dtype {DTYPE}")
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        print(f"{t.__name__} ...")
        try:
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)