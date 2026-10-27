"""
tests/test_cuda_graphs.py - decode steps replayed as CUDA graphs must give exactly the same tokens.

Without a GPU, DecodeGraphs runs in emulation: the same fixed buffers, batch buckets, padding rows
and scratch block, without recording a graph (Triton kernels in the interpreter). That checks all of
the bookkeeping exactly against model.generate(). With a GPU, graphs are really captured and
replayed, and outputs must match the same engine without graphs.

Run from the project root:
    python tests/test_cuda_graphs.py
"""
import os
import sys

import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"        # must be set before triton is imported

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile  # noqa: E402

import config  # noqa: E402
from tests.test_batching import (LONG_PROMPTS, PROMPTS, build_tiny_model, make_engine,  # noqa: E402
                                 reference_outputs, run_concurrently)

GPU = torch.cuda.is_available()


class settings:
    """Temporarily set config values."""
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: getattr(config, k) for k in self.values}
        for k, v in self.values.items():
            setattr(config, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(config, k, v)


def graph_engine(d, kv_tokens=4096, **extra):
    with settings(ATTN_BACKEND="triton", CUDA_GRAPHS=True, **extra):
        engine = make_engine(d, "continuous", "paged", kv_tokens=kv_tokens, max_num_seqs=8)
    assert engine.runner.graphs is not None, "graphs were not set up"
    return engine


def check(name, engine, prompts, refs):
    outs, _ = run_concurrently(engine, prompts)
    s = engine.stats()
    print(f"  {name}: {'identical' if outs == refs else 'DIFFERENT'} | buckets {engine.runner.graphs.buckets}, "
          f"graph steps {s['graph_replays']}, normal steps {s['graph_eager_steps']}, "
          f"preemptions {s.get('preemptions')}")
    assert outs == refs
    assert s["graph_replays"] > 0, "no decode step used a graph"


def test_matches_generate():
    """Ragged batches: requests finish at different times, so batch sizes fall through the buckets
    (8 -> 4 -> 2 -> 1) and padding rows hold stale values from bigger batches."""
    if GPU:
        print("  (CPU-only exact comparison; see test_gpu_graphs_match_eager)")
        return
    from transformers import PreTrainedTokenizerFast
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tok = PreTrainedTokenizerFast.from_pretrained(d)
        check("ragged batch", graph_engine(d), PROMPTS, reference_outputs(model, tok, PROMPTS))
        check("tiny pool (preemption + recompute)", graph_engine(d, kv_tokens=160), LONG_PROMPTS,
              reference_outputs(model, tok, LONG_PROMPTS))
        shared = " ".join(f"w{100 + i}" for i in range(40))
        prompts = [(f"{shared} w{i}", 6) for i in range(5)]
        with settings(PREFIX_CACHING=True, CHUNKED_PREFILL=True, MAX_BATCHED_TOKENS=8):
            engine = graph_engine(d)
        check("prefix caching + chunked prefill", engine, prompts, reference_outputs(model, tok, prompts))
        with settings(KV_CACHE_DTYPE="int8"):
            q_engine = graph_engine(d)
        with settings(KV_CACHE_DTYPE="int8", CUDA_GRAPHS=False, ATTN_BACKEND="triton"):
            q_ref = make_engine(d, "continuous", "paged", max_num_seqs=8)
        check("int8 KV cache (vs the same engine without graphs)", q_engine, PROMPTS, run_concurrently(q_ref, PROMPTS)[0])


def test_gpu_graphs_match_eager():
    """On the GPU: real capture and replay, against the same engine running every step normally."""
    if not GPU:
        print("  (needs a GPU: skipped)")
        return
    with tempfile.TemporaryDirectory() as d:
        build_tiny_model(d)
        with settings(MOE_KERNEL="triton", ATTN_BACKEND="triton", CUDA_GRAPHS=False):
            eager = make_engine(d, "continuous", "paged", max_num_seqs=8)
        refs, _ = run_concurrently(eager, PROMPTS)
        with settings(MOE_KERNEL="triton"):
            engine = graph_engine(d)
        check("captured graphs vs no graphs", engine, PROMPTS, refs)
        long_refs, _ = run_concurrently(eager, LONG_PROMPTS)
        check("longer generations", engine, LONG_PROMPTS, long_refs)


if __name__ == "__main__":
    print(f"Running on {'cuda (real graphs)' if GPU else 'cpu (emulated graphs, Triton interpreter)'}")
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