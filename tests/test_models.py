"""
tests/test_models.py - the custom engine on other architectures (engine/models.py).

Builds tiny random versions of Qwen3 (dense, GQA, per-head q/k norm, tied embeddings, head size
!= hidden / heads), Qwen3-MoE (MoE in OLMoE's expert layout), Qwen2 (q/k/v biases) and Llama
(GQA, Llama-3 rope scaling), and checks that the engine reproduces model.generate() token for token
with every feature that should carry over: both KV caches, the paged attention kernel, CUDA graphs
(emulated on CPU), prefix caching + chunked prefill, the fused MoE kernel, and quantized loading
including the on-disk cache. Unsupported models must be rejected with a clear error.

Without a GPU, Triton kernels run in the interpreter (float32, exact comparison).
Run from the project root:
    python tests/test_models.py
"""
import os
import sys

import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"        # must be set before triton is imported

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile  # noqa: E402

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (LlamaConfig, LlamaForCausalLM, Olmo2Config, Olmo2ForCausalLM,  # noqa: E402
                          PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Qwen3Config,
                          Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM)

import config  # noqa: E402
from tests.test_batching import PROMPTS, make_engine, reference_outputs, run_concurrently  # noqa: E402

GPU = torch.cuda.is_available()
VOCAB = 512
COMMON = dict(vocab_size=VOCAB, max_position_embeddings=256, pad_token_id=1, eos_token_id=2,
              bos_token_id=None, num_hidden_layers=2)

ARCHS = {
    "qwen3": (Qwen3ForCausalLM, Qwen3Config(hidden_size=64, intermediate_size=96, num_attention_heads=4,
                                            num_key_value_heads=2, head_dim=32, tie_word_embeddings=True,
                                            **COMMON)),
    "qwen3_moe": (Qwen3MoeForCausalLM, Qwen3MoeConfig(hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
                                                      num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                                                      num_experts=8, num_experts_per_tok=2, **COMMON)),
    "qwen2": (Qwen2ForCausalLM, Qwen2Config(hidden_size=64, intermediate_size=96, num_attention_heads=4,
                                            num_key_value_heads=2, **COMMON)),
    "llama": (LlamaForCausalLM, LlamaConfig(hidden_size=64, intermediate_size=96, num_attention_heads=4,
                                            num_key_value_heads=1, rope_scaling={
                                                "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                "high_freq_factor": 4.0, "original_max_position_embeddings": 64},
                                            **COMMON)),
}


class settings:
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: getattr(config, k) for k in self.values}
        for k, v in self.values.items():
            setattr(config, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(config, k, v)


def build(path, cls, cfg, seed=0):
    torch.manual_seed(seed)
    model = cls(cfg).float().eval()
    with torch.no_grad():                    # bigger random weights than the default init, so
        for p in model.parameters():         # greedy choices aren't near-ties (exact comparisons)
            if p.dim() > 1:
                p.normal_(0, 0.1)
    model.save_pretrained(path)
    words = ["<unk>", "<pad>", "<eos>"] + [f"w{i}" for i in range(VOCAB - 3)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
                            eos_token="<eos>").save_pretrained(path)
    return model, PreTrainedTokenizerFast.from_pretrained(path)


def engine_with_tokenizer(d, kv="paged", quant="none"):
    """make_engine, using the toy word-level tokenizer as saved: transformers maps some model types
    (e.g. qwen2) to their own tokenizer class, which would re-split the toy vocabulary."""
    try:
        engine = make_engine(d, "continuous", kv, max_num_seqs=8, quant=quant)
    finally:
        config.QUANTIZATION = "none"
    engine.tokenizer = PreTrainedTokenizerFast.from_pretrained(d)
    return engine


def run(name, d, refs, prompts=PROMPTS, kv="paged", **cfg):
    with settings(**cfg):
        engine = engine_with_tokenizer(d, kv)
    outs, _ = run_concurrently(engine, prompts)
    extra = ""
    if engine.runner.graphs is not None:
        extra = f", {engine.stats()['graph_replays']} graph steps"
    print(f"    {name:<44} {'identical' if outs == refs else 'DIFFERENT'}{extra}")
    assert outs == refs, f"{name}: {outs} != {refs}"
    return engine


def test_architectures_exact():
    if GPU:
        print("  (exact float32 comparison runs on CPU; on GPU see the other tests)")
        return
    shared = " ".join(f"w{100 + i}" for i in range(40))
    prefix_prompts = [(f"{shared} w{i}", 6) for i in range(4)]
    for arch, (cls, cfg) in ARCHS.items():
        with tempfile.TemporaryDirectory() as d:
            model, tok = build(d, cls, cfg)
            refs = reference_outputs(model, tok, PROMPTS)
            print(f"  {arch}:")
            engine = run("paged, SDPA", d, refs)
            info = engine.arch
            print(f"    -> detected: q/k norm {info['qk_norm']}, {'MoE' if info['moe'] else 'dense'}, "
                  f"heads {info['heads']}/{info['kv_heads']}, head size {info['head_dim']}")
            run("contiguous, SDPA", d, refs, kv="contiguous")
            run("paged attention kernel + CUDA graphs (emulated)", d, refs, ATTN_BACKEND="triton", CUDA_GRAPHS=True)
            run("prefix caching + chunked prefill + kernel", d, reference_outputs(model, tok, prefix_prompts),
                prompts=prefix_prompts, ATTN_BACKEND="triton", PREFIX_CACHING=True, CHUNKED_PREFILL=True,
                MAX_BATCHED_TOKENS=8)
            if info["moe"]:
                e = run("fused MoE kernel + all of the above", d, refs, MOE_KERNEL="triton",
                        ATTN_BACKEND="triton", CUDA_GRAPHS=True)
                assert e.load_info.get("moe_kernel") == "triton", "fused MoE should apply to this MoE model"


def test_detection():
    expected = {"qwen3": ("per_head", False, 32), "qwen3_moe": ("per_head", True, 32),
                "qwen2": ("none", False, 16), "llama": ("none", False, 16)}
    from engine.models import describe
    for arch, (cls, cfg) in ARCHS.items():
        info = describe(cls(cfg), 256)
        assert (info["qk_norm"], info["moe"], info["head_dim"]) == expected[arch], (arch, info)
    from transformers import OlmoeConfig, OlmoeForCausalLM
    info = describe(OlmoeForCausalLM(OlmoeConfig(hidden_size=64, intermediate_size=32, num_attention_heads=4,
                                                 num_hidden_layers=1, num_experts=4, num_experts_per_tok=2,
                                                 vocab_size=64)), 256)
    assert (info["qk_norm"], info["moe"]) == ("full", True), info


def test_unsupported_rejected():
    from engine.models import describe
    cases = {
        "sliding window": Qwen2ForCausalLM(Qwen2Config(hidden_size=64, intermediate_size=96, num_attention_heads=4,
                                                       num_key_value_heads=2, use_sliding_window=True,
                                                       sliding_window=8, max_window_layers=0, **COMMON)),
        "different layer layout (OLMo 2)": Olmo2ForCausalLM(Olmo2Config(hidden_size=64, intermediate_size=96,
                                                                        num_attention_heads=4, **COMMON)),
    }
    for name, model in cases.items():
        try:
            describe(model, 256)
        except ValueError as e:
            print(f"  {name}: rejected ({str(e)[:90]}...)")
            continue
        raise AssertionError(f"{name} should have been rejected")


def test_quantized_tied_weights_and_cache():
    """INT8 weights on a dense model with tied embeddings: a fresh load and a load from the
    on-disk cache must give the same outputs (the cache stores the tied matrix once)."""
    cls, cfg = ARCHS["qwen3"]
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as cache:
        build(d, cls, cfg)
        outs = []
        for attempt in ("fresh", "from cache"):
            with settings(QUANT_CACHE_DIR=cache, QUANT_USE_CACHE=True):
                engine = engine_with_tokenizer(d, quant="int8_wo")
            q = engine.load_info["quantization"]
            assert q.get("load_source") == ("fresh" if attempt == "fresh" else "cache"), q.get("load_source")
            assert q.get("linear", 0) > 0, "dense MLP + attention projections should be quantized"
            lm, emb = engine.model.lm_head.weight, engine.model.model.embed_tokens.weight
            assert lm.data_ptr() == emb.data_ptr(), f"{attempt}: lm_head should stay tied to the embeddings"
            outs.append(run_concurrently(engine, PROMPTS)[0])
            print(f"  {attempt}: {q['linear']} linear layers quantized, embeddings tied, "
                  f"{sum(len(o) for o in outs[-1])} tokens generated")
        assert outs[0] == outs[1], "cached model should behave exactly like the freshly quantized one"



def test_quantized_moe_experts():
    """INT4 on a Qwen3-MoE model: its experts use OLMoE's layout, so QuantExperts applies."""
    cls, cfg = ARCHS["qwen3_moe"]
    with tempfile.TemporaryDirectory() as d:
        build(d, cls, cfg)
        with settings(QUANT_USE_CACHE=False, QUANT_GROUP_SIZE=32):
            engine = engine_with_tokenizer(d, quant="int4_wo")
        q = engine.load_info["quantization"]
        outs, _ = run_concurrently(engine, PROMPTS)
        print(f"  int4: {q.get('experts')} expert blocks + {q.get('linear')} linear layers quantized, "
              f"{sum(len(o) for o in outs)} tokens generated")
        assert q.get("experts", 0) == cfg.num_hidden_layers, q
        assert all(len(o) > 0 for o in outs)



def test_fused_projections_exact():
    """FUSED_PROJ=1: one q/k/v matmul and one gate/up matmul, in both weight layouts, must give the
    same tokens as model.generate(), and the original weights must become views (no extra memory)."""
    if GPU:
        print("  (exact float32 comparison runs on CPU)")
        return
    from tests.test_batching import build_tiny_model
    archs = dict(ARCHS)
    archs["olmoe"] = None                               # q/k norm over the full projection
    for arch, spec in archs.items():
        with tempfile.TemporaryDirectory() as d:
            if spec is None:
                model = build_tiny_model(d)
                from transformers import PreTrainedTokenizerFast as Tok
                tok = Tok.from_pretrained(d)
            else:
                model, tok = build(d, *spec)
            refs = reference_outputs(model, tok, PROMPTS)
            print(f"  {arch}:")
            for layout in ("normal", "transposed"):
                e = run(f"fused, weights [{'out, in' if layout == 'normal' else 'in, out'}]", d, refs,
                        FUSED_PROJ=True, WEIGHT_LAYOUT=layout)
                f = e.runner.fused[0]
                assert "qkv" in f, "q/k/v should be fused"
                assert ("gate_up" in f) == (not e.arch["moe"]), "gate/up fused exactly for dense MLPs"
                q = e.model.model.layers[0].self_attn.q_proj.weight
                fused_storage = f["qkv"].weight_view().untyped_storage().data_ptr()
                assert q.untyped_storage().data_ptr() == fused_storage, "q_proj should be a view, not a copy"
            run("fused + paged kernel + CUDA graphs (emulated)", d, refs, FUSED_PROJ=True,
                WEIGHT_LAYOUT="transposed", ATTN_BACKEND="triton", CUDA_GRAPHS=True)


def test_fused_projections_skip_quantized():
    """Quantized layers aren't fused (they're QuantLinear, not nn.Linear); the model still runs."""
    cls, cfg = ARCHS["qwen3"]
    with tempfile.TemporaryDirectory() as d:
        build(d, cls, cfg)
        with settings(QUANT_USE_CACHE=False, FUSED_PROJ=True, WEIGHT_LAYOUT="normal"):
            engine = engine_with_tokenizer(d, quant="int8_wo")
        assert all("qkv" not in f and "gate_up" not in f for f in engine.runner.fused)
        outs, _ = run_concurrently(engine, PROMPTS)
        assert all(len(o) > 0 for o in outs)
        print(f"  int8 + FUSED_PROJ: quantized layers left alone, {sum(len(o) for o in outs)} tokens generated")


if __name__ == "__main__":
    print(f"Running on {'cuda' if GPU else 'cpu (Triton interpreter)'}")
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