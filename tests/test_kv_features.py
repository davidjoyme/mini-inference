"""
tests/test_kv_features.py - KV cache quantization and prefix caching.

KV quantization changes numbers slightly, so it's checked against tolerances:
  - stored-then-read keys/values are close to the originals
  - the model's logits with a quantized cache stay close to the unquantized ones
  - 8-bit storage fits about twice as many tokens
Prefix caching must not change outputs at all, so it's checked for exact equality with
model.generate(), including when requests are preempted and when blocks get evicted.

Run from the project root:
    python tests/test_kv_features.py
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
import config  # noqa: E402
from engine.core.kv_cache import KVCache, PagedAllocator, tokens_for_budget  # noqa: E402
from engine.core.model_runner import ModelRunner  # noqa: E402
from engine.core.request import Request, SamplingParams  # noqa: E402
from tests.test_batching import build_tiny_model, make_engine, reference_outputs, run_concurrently  # noqa: E402

SHARED = " ".join(f"w{100 + i}" for i in range(40))            # 40-token shared prefix
PREFIX_PROMPTS = [(f"{SHARED} w{i} w{i + 1} w{i + 2}", 15) for i in range(3, 9)]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def rel_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def make(kv_dtype, prefix=False, **kw):
    config.KV_CACHE_DTYPE = kv_dtype
    config.PREFIX_CACHING = prefix
    try:
        return make_engine(**kw)
    finally:
        config.KV_CACHE_DTYPE, config.PREFIX_CACHING = "auto", False


# ---------------------------------------------------------------- KV quantization
def test_kv_quant_roundtrip():
    torch.manual_seed(0)
    k = torch.randn(20, 4, 32) * torch.tensor([0.1, 1.0, 5.0, 30.0])[None, :, None]   # heads of very different scale
    v = torch.randn(20, 4, 32)
    slots = torch.randperm(64)[:20]
    for kv_dtype, limit in (("int8", 0.01), ("fp8", 0.05)):
        cache = KVCache("paged", 2, 4, 32, 64, torch.float32, "cpu", block_size=16, kv_dtype=kv_dtype)
        cache.write(1, slots, k, v)
        k2, v2 = cache.gather(1, slots)
        ek, ev = rel_error(k2, k), rel_error(v2, v)
        print(f"  {kv_dtype}: key error {ek:.4f}, value error {ev:.4f}")
        assert ek < limit and ev < limit
        # 2-D index (the decode path's padded [batch, len] index) gives the same values
        k3, _ = cache.gather(1, slots.view(4, 5))
        assert torch.equal(k3.view(20, 4, 32), k2)


def test_kv_quant_capacity():
    budget = 1024 ** 3
    full = tokens_for_budget(budget, 16, 16, 128, torch.bfloat16, "auto")
    for kv_dtype in ("fp8", "int8"):
        q = tokens_for_budget(budget, 16, 16, 128, torch.bfloat16, kv_dtype)
        print(f"  {kv_dtype}: {q} tokens per GB vs {full} for bf16 ({q / full:.2f}x)")
        assert 1.9 < q / full < 2.0


@torch.no_grad()
def test_kv_quant_logits_stay_close():
    """Same tokens fed through the model with each cache format: prefill + 6 decode steps."""
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        cfg = model.config
        prompt = list(range(10, 40))
        results = {}
        for kv_dtype in ("auto", "int8", "fp8"):
            cache = KVCache("paged", cfg.num_hidden_layers, cfg.num_key_value_heads,
                            cfg.hidden_size // cfg.num_attention_heads, 256, torch.float32, "cpu",
                            kv_dtype=kv_dtype)
            runner = ModelRunner(model, cache)
            req = Request(prompt_ids=list(prompt), params=SamplingParams(max_new_tokens=8))
            cache.admit(req, 256)
            steps = []
            for step in range(7):
                cache.ensure(req, req.total_len)
                logits = runner.forward([(req, req.num_computed, req.total_len)])[0]
                steps.append(logits)
                req.num_computed = req.total_len
                # feed the unquantized run's tokens to every run, so inputs stay identical
                req.output_ids.append(int(results["auto"][step].argmax()) if "auto" in results
                                      else int(logits.argmax()))
            results[kv_dtype] = steps
        for kv_dtype, floor in (("int8", 0.999), ("fp8", 0.99)):
            worst = min(cosine(a, b) for a, b in zip(results[kv_dtype], results["auto"]))
            print(f"  {kv_dtype}: worst logit cosine vs unquantized {worst:.5f}")
            assert worst > floor


def test_kv_quant_engine_serves_requests():
    with tempfile.TemporaryDirectory() as d:
        build_tiny_model(d)
        for kv_dtype in ("int8", "fp8"):
            engine = make(kv_dtype, model_dir=d, scheduler="continuous", kv_cache="paged")
            outs, reqs = run_concurrently(engine)
            assert all(len(o) == r.params.max_new_tokens for o, r in zip(outs, reqs))
            assert engine.stats()["kv_cache_dtype"] == kv_dtype


# ---------------------------------------------------------------- prefix caching
def _req(ids, max_new=4):
    return Request(prompt_ids=list(ids), params=SamplingParams(max_new_tokens=max_new))


def test_prefix_allocator_shares_and_evicts():
    a = PagedAllocator(capacity=32, block_size=4, watermark=0.0, prefix_caching=True)   # 8 blocks
    r1 = _req(range(1, 11))                                      # 10 tokens -> 2 full blocks
    assert a.admit(r1, 100) == 0 and a.ensure(r1, 10)
    r1.num_computed = 10
    a.register(r1)
    assert len(a.hash_to_block) == 2

    r2 = _req(list(range(1, 9)) + [99, 98, 97])                  # same first 8 tokens
    assert a.admit(r2, 100) == 8, "first two blocks should be reused"
    shared = a.tables[r2.id][:2]
    assert shared == a.tables[r1.id][:2] and all(a.ref[b] == 2 for b in shared)

    a.free(r1)                                                   # shared blocks still used by r2
    assert all(a.ref[b] == 1 for b in shared) and not a.evictable
    a.ensure(r2, 11)
    a.free(r2)                                                   # now idle but cached
    assert set(a.evictable) == set(shared)

    r3 = _req(range(1, 9))                                       # identical 8-token prompt:
    assert a.admit(r3, 100) == 4, "one token must stay uncomputed, so only 1 block is reused"
    a.free(r3)

    big = _req(range(500, 532))                                  # needs all 8 blocks -> evicts
    a.watermark_blocks = 0
    assert a.admit(big, 100) == 0 and a.ensure(big, 32)
    assert a.evictions == 2 and not a.hash_to_block


def test_prefix_caching_outputs_exact():
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(d)
        refs = reference_outputs(model, tokenizer, PREFIX_PROMPTS)
        for scheduler in ("continuous", "static"):
            engine = make("auto", prefix=True, model_dir=d, scheduler=scheduler, kv_cache="paged")
            first, _ = run_concurrently(engine, PREFIX_PROMPTS[:1])       # fills the cache
            rest, _ = run_concurrently(engine, PREFIX_PROMPTS[1:])        # should hit it
            st = engine.stats()
            print(f"  {scheduler}: prefix hits {st['prefix_hit_tokens']}/{st['prefix_query_tokens']} "
                  f"tokens ({st['prefix_hit_pct']}%)")
            assert st["prefix_hit_tokens"] >= 5 * 32, "each later request should reuse >= 2 blocks"
            for got, want in zip(first + rest, refs):
                assert got == want, f"{scheduler}: {got} != {want}"


def test_prefix_caching_no_false_hits():
    """Prompts that differ in their first block must share nothing, and outputs stay exact."""
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(d)
        prompts = [(f"w{i} {SHARED}", 15) for i in range(3, 9)]        # differ at token 0
        refs = reference_outputs(model, tokenizer, prompts)
        engine = make("auto", prefix=True, model_dir=d, scheduler="continuous", kv_cache="paged")
        run_concurrently(engine, prompts[:1])
        outs, _ = run_concurrently(engine, prompts[1:])
        st = engine.stats()
        print(f"  hits {st['prefix_hit_tokens']}/{st['prefix_query_tokens']}")
        assert st["prefix_hit_tokens"] == 0
        assert outs == refs[1:]


def test_prefix_caching_with_preemption_and_eviction():
    """Tiny pool: requests get preempted, cached blocks get evicted; outputs still exact."""
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(d)
        prompts = [(p, 30) for p, _ in PREFIX_PROMPTS]
        refs = reference_outputs(model, tokenizer, prompts)
        engine = make("auto", prefix=True, model_dir=d, scheduler="continuous", kv_cache="paged",
                      kv_tokens=192)                                     # 12 blocks
        outs, _ = run_concurrently(engine, prompts)
        st = engine.stats()
        print(f"  preemptions={st['preemptions']}, evictions={st['prefix_evictions']}, "
              f"hits={st['prefix_hit_tokens']}")
        assert st["preemptions"] > 0
        for got, want in zip(outs, refs):
            assert got == want


if __name__ == "__main__":
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