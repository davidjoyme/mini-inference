"""
tests/test_chunked_prefill.py - chunked prefill must split long prompts across steps, keep every
step within the token budget, keep decoding requests running in the same steps, and still give
exactly the same tokens as transformers' model.generate().

Run from the project root:
    python tests/test_chunked_prefill.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import PreTrainedTokenizerFast  # noqa: E402

import config  # noqa: E402
from tests.test_batching import PROMPTS, build_tiny_model, make_engine, reference_outputs, run_concurrently  # noqa: E402
from tests.test_kv_features import PREFIX_PROMPTS  # noqa: E402

BUDGET = 8          # tokens per step: far smaller than most prompts here, so they get chunked
LONG = [(" ".join(f"w{(7 * i + j) % 500 + 3}" for j in range(30 + 5 * i)), 12) for i in range(4)]


def make(chunked, budget=BUDGET, prefix=False, **kw):
    saved = (config.CHUNKED_PREFILL, config.MAX_BATCHED_TOKENS, config.PREFIX_CACHING)
    config.CHUNKED_PREFILL, config.MAX_BATCHED_TOKENS, config.PREFIX_CACHING = chunked, budget, prefix
    try:
        return make_engine(**kw)
    finally:
        config.CHUNKED_PREFILL, config.MAX_BATCHED_TOKENS, config.PREFIX_CACHING = saved


def test_chunked_outputs_exact_all_modes():
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tok = PreTrainedTokenizerFast.from_pretrained(d)
        prompts = PROMPTS + LONG
        refs = reference_outputs(model, tok, prompts)
        for scheduler in ("continuous", "sequential"):
            for kv in ("paged", "contiguous"):
                engine = make(True, model_dir=d, scheduler=scheduler, kv_cache=kv)
                outs, _ = run_concurrently(engine, prompts)
                st = engine.stats()
                print(f"  {scheduler:10} {kv:10} steps={st['steps']:<4} prefill slices={st['prefill_chunks']:<4} "
                      f"mixed steps={st['mixed_steps']:<4} largest step={st['max_step_tokens']} tokens")
                assert st["max_step_tokens"] <= BUDGET, "a step went over the token budget"
                assert st["prefill_chunks"] > len(prompts), "long prompts should be split"
                if scheduler == "continuous":
                    assert st["mixed_steps"] > 0, "decodes and prefill slices should share steps"
                for i, (got, want) in enumerate(zip(outs, refs)):
                    assert got == want, f"{scheduler}/{kv} prompt {i}: {got} != {want}"


def test_without_chunking_long_prompts_make_big_steps():
    """The contrast: without chunking, one step processes a whole long prompt."""
    with tempfile.TemporaryDirectory() as d:
        build_tiny_model(d)
        engine = make(False, budget=4096, model_dir=d, scheduler="continuous", kv_cache="paged")
        run_concurrently(engine, PROMPTS + LONG)
        st = engine.stats()
        print(f"  largest step without chunking: {st['max_step_tokens']} tokens, slices={st['prefill_chunks']}")
        assert st["max_step_tokens"] > 45 and st["prefill_chunks"] == 0


def test_chunked_with_prefix_caching_and_preemption():
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tok = PreTrainedTokenizerFast.from_pretrained(d)
        prompts = [(p, 25) for p, _ in PREFIX_PROMPTS]
        refs = reference_outputs(model, tok, prompts)
        engine = make(True, budget=16, prefix=True, model_dir=d, scheduler="continuous",
                      kv_cache="paged", kv_tokens=192)
        first, _ = run_concurrently(engine, prompts[:1])
        rest, _ = run_concurrently(engine, prompts[1:])
        st = engine.stats()
        print(f"  slices={st['prefill_chunks']}, prefix hits={st['prefix_hit_tokens']}, "
              f"preemptions={st['preemptions']}, largest step={st['max_step_tokens']}")
        assert st["max_step_tokens"] <= 16 and st["prefix_hit_tokens"] > 0
        assert first + rest == refs


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