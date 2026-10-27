"""
tests/test_batching.py - the custom engine must give exactly the same tokens as
transformers' model.generate(), whatever the scheduler or KV cache mode, even when many
requests share a batch and the paged cache is so small that requests get preempted.

Uses a tiny randomly initialized OLMoE in float32 (exact comparisons), on CPU, no downloads.
Run from the project root:
    python tests/test_batching.py
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast  # noqa: E402

import config  # noqa: E402
from engine.custom_engine import CustomEngine  # noqa: E402
from engine.core.kv_cache import ContiguousAllocator, PagedAllocator  # noqa: E402
from engine.core.request import Request, SamplingParams  # noqa: E402

VOCAB = 512
PROMPTS = [  # different lengths and output limits, so batches are ragged
    ("w5 w9 w14 w3", 12),
    ("w100 w200 w300 w400 w5 w6 w7 w8 w9 w10 w11 w12 w13", 20),
    ("w42", 7),
    ("w7 w7 w7 w7 w7 w7 w7 w7", 16),
    ("w300 w2 w250 w11 w90", 25),
    ("w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79", 10),
]


def build_tiny_model(path):
    torch.manual_seed(0)
    cfg = OlmoeConfig(hidden_size=128, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, num_experts=8, num_experts_per_tok=2, vocab_size=VOCAB,
                      max_position_embeddings=256, pad_token_id=1, eos_token_id=2, bos_token_id=None)
    model = OlmoeForCausalLM(cfg).float().eval()
    model.save_pretrained(path)
    words = ["<unk>", "<pad>", "<eos>"] + [f"w{i}" for i in range(VOCAB - 3)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
                            eos_token="<eos>").save_pretrained(path)
    return model


# Long generations from short prompts: admitted cheaply, then grow until blocks run out
LONG_PROMPTS = [(f"w{10 + i} w{20 + i} w{30 + i} w{40 + i}", 40) for i in range(6)]


@torch.no_grad()
def reference_outputs(model, tokenizer, prompts=PROMPTS):
    """What model.generate() produces for each prompt (greedy)."""
    refs = []
    for text, n in prompts:
        ids = torch.tensor([tokenizer(text)["input_ids"]])
        out = model.generate(ids, max_new_tokens=n, do_sample=False, eos_token_id=2, pad_token_id=1)
        refs.append(out[0, ids.shape[1]:].tolist())
    return refs


def make_engine(model_dir, scheduler, kv_cache, kv_tokens=4096, max_num_seqs=64, quant="none"):
    config.MODEL_NAME = model_dir
    config.DTYPE = "float32"
    config.QUANTIZATION = quant
    config.IS_CHAT_MODEL = False
    config.DO_SAMPLE = False
    config.ENGINE = "custom"
    config.SCHEDULER = scheduler
    config.KV_CACHE = kv_cache
    config.KV_BLOCK_SIZE = 16
    config.KV_CACHE_TOKENS = kv_tokens
    config.MAX_NUM_SEQS = max_num_seqs
    config.STATIC_BATCH_WAIT_S = 0.2
    return CustomEngine()


def run_concurrently(engine, prompts=PROMPTS):
    """Submit every prompt at once (so they share batches) and collect the generated ids."""
    reqs = [engine.submit(text, n) for text, n in prompts]
    outs = [None] * len(reqs)

    def collect(i, req):
        for _ in engine._events(req):
            pass
        outs[i] = engine._delivered(req)

    threads = [threading.Thread(target=collect, args=(i, r)) for i, r in enumerate(reqs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return outs, reqs


# ---------------------------------------------------------------- allocator unit tests
def test_contiguous_allocator_merges_and_fragments():
    a = ContiguousAllocator(100)
    reqs = [Request(prompt_ids=[0] * 10, params=SamplingParams(max_new_tokens=15)) for _ in range(4)]
    assert all(a.admit(r, 1000) is not None for r in reqs)       # 4 x 25 tokens = full
    assert not a.can_admit(Request(prompt_ids=[0], params=SamplingParams(max_new_tokens=1)), 1000)
    a.free(reqs[0]); a.free(reqs[2])                             # two separate 25-token gaps
    big = Request(prompt_ids=[0] * 30, params=SamplingParams(max_new_tokens=10))
    assert not a.can_admit(big, 1000)                            # 50 free, but no 40-token gap
    assert a.stats()["kv_fragmentation_pct"] == 50.0
    a.free(reqs[1])                                              # gaps merge into 75
    assert a.can_admit(big, 1000)


def test_paged_allocator_blocks_on_demand():
    a = PagedAllocator(64, 16)                                   # 4 blocks
    r = Request(prompt_ids=[0] * 20, params=SamplingParams(max_new_tokens=100))
    assert a.admit(r, 1000) == 0 and a.ensure(r, 20)             # 0 tokens already cached
    assert len(a.tables[r.id]) == 2                              # 20 tokens -> 2 blocks, not 120
    assert a.ensure(r, 33) and len(a.tables[r.id]) == 3
    slots = a.slots(r, 0, 33)
    assert len(set(slots.tolist())) == 33
    a.free(r)
    assert len(a.free_blocks) == 4


# ---------------------------------------------------------------- end-to-end equivalence
def test_all_modes_match_generate():
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(d)
        refs = reference_outputs(model, tokenizer)
        for scheduler in ("sequential", "static", "continuous"):
            for kv in ("contiguous", "paged"):
                engine = make_engine(d, scheduler, kv)
                outs, _ = run_concurrently(engine)
                st = engine.stats()
                print(f"  {scheduler:10} {kv:10} steps={st["steps"]:<4} "
                      f"padding_tokens={st['padding_tokens']:<4} preemptions={st['preemptions']}")
                for i, (got, want) in enumerate(zip(outs, refs)):
                    assert got == want, f"{scheduler}/{kv} prompt {i}: {got} != {want}"
                if scheduler == "static":
                    assert st["padding_tokens"] > 0, "static batching should compute padding"
                assert st["running"] == 0 and st["kv_used_tokens"] == 0, "cache not freed"


def test_paged_preemption_keeps_outputs_exact():
    """A KV pool far too small for all requests at once: the scheduler has to preempt and
    recompute, and the final tokens must still be identical."""
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(d)
        refs = reference_outputs(model, tokenizer, LONG_PROMPTS)
        engine = make_engine(d, "continuous", "paged", kv_tokens=96)   # 6 blocks of 16
        outs, reqs = run_concurrently(engine, LONG_PROMPTS)
        st = engine.stats()
        print(f"  preemptions={st['preemptions']}, steps={st['steps']}")
        assert st["preemptions"] > 0, "expected preemptions with a 96-token cache"
        for i, (got, want) in enumerate(zip(outs, refs)):
            assert got == want, f"prompt {i} after preemption: {got} != {want}"


def test_continuous_batches_requests_together():
    """With continuous batching, concurrent requests share steps: far fewer steps than
    running them one after another."""
    with tempfile.TemporaryDirectory() as d:
        build_tiny_model(d)
        seq = make_engine(d, "sequential", "paged")
        run_concurrently(seq)
        cont = make_engine(d, "continuous", "paged")
        run_concurrently(cont)
        s, c = seq.stats()["steps"], cont.stats()["steps"]
        print(f"  steps: sequential={s}, continuous={c}")
        assert c < s / 2


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