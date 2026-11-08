"""
tests/test_quantization.py - correctness checks for the quantization code.

Uses a tiny randomly initialized OLMoE (same layer types and expert layout as the
real model, much smaller), so it runs on CPU in seconds with no downloads.

Run from the project root:
    python tests/test_quantization.py          # plain Python
    python -m pytest tests/test_quantization.py   # or with pytest, if installed

Re-run these whenever you change a quantization method or plug in a custom kernel:
the kernel must give the same results as the reference `linear`.
"""
import copy
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from transformers import OlmoeConfig, OlmoeForCausalLM  # noqa: E402
from engine.quantization import METHODS, get_method  # noqa: E402
from engine.quantization.loader import load_quantized_model, quantize_model, weight_memory  # noqa: E402
from engine.quantization.modules import QuantExperts  # noqa: E402

DTYPE = torch.bfloat16
# How far each method's rebuilt weights may be from the originals (relative error)
MAX_WEIGHT_ERROR = {"int8_wo": 0.01, "int4_wo": 0.12, "fp8": 0.05}
# Minimum cosine similarity between quantized and bf16 logits on the tiny model
MIN_LOGIT_COSINE = {"int8_wo": 0.995, "int4_wo": 0.95, "fp8": 0.99}


def tiny_config():
    return OlmoeConfig(hidden_size=256, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4, num_experts=8,
                       num_experts_per_tok=2, vocab_size=512, max_position_embeddings=512)


def tiny_model():
    torch.manual_seed(0)
    return OlmoeForCausalLM(tiny_config()).to(DTYPE).eval()


def methods(group_size=64):
    out = {}
    for name in METHODS:
        out[name] = get_method(name, device="cpu", compute_dtype=DTYPE, group_size=group_size,
                               fp8_backend="emulate")
    return out


def rel_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


# ---------------------------------------------------------------- tests
def test_quantize_dequantize_roundtrip():
    """Rebuilt weights are close to the originals, including with a leading expert dimension."""
    torch.manual_seed(1)
    w = (torch.randn(3, 64, 256) * 0.02).to(DTYPE)
    for name, m in methods().items():
        q = m.quantize(w)
        for t in q.values():
            assert t.shape[0] == 3, f"{name}: expert dimension not preserved"
        err = rel_error(m.dequantize(**q), w)
        assert err < MAX_WEIGHT_ERROR[name], f"{name}: weight error {err:.4f}"
        print(f"  {name:8} weight error {err:.4f}")


def test_linear_matches_dequantized_matmul():
    """The hot path `linear` computes the same thing as dequantize + ordinary matmul.
    (FP8 also rounds activations, so it gets a looser tolerance.)"""
    torch.manual_seed(2)
    w = (torch.randn(64, 256) * 0.02).to(DTYPE)
    x = torch.randn(2, 5, 256).to(DTYPE)
    for name, m in methods().items():
        q = m.quantize(w)
        expected = torch.nn.functional.linear(x.float(), m.dequantize(**q).float())
        got = m.linear(x, **q)
        assert got.shape == (2, 5, 64) and got.dtype == DTYPE
        err = rel_error(got, expected)
        assert err < (0.05 if name == "fp8" else 0.01), f"{name}: linear error {err:.4f}"


def test_fp8_native_matches_emulated():
    """Native FP8 matmul (torch._scaled_mm) and emulation should agree closely."""
    torch.manual_seed(3)
    w = (torch.randn(64, 256) * 0.02).to(DTYPE)
    x = torch.randn(7, 256).to(DTYPE)
    emu = get_method("fp8", fp8_backend="emulate", compute_dtype=DTYPE)
    nat = get_method("fp8", fp8_backend="native", compute_dtype=DTYPE)
    q = emu.quantize(w)
    y_emu = emu.linear(x, **q)
    y_nat = nat.linear(x, **q)
    if not nat.native:
        print("  native FP8 not available on this device; skipped")
        return
    assert rel_error(y_nat, y_emu) < 0.02


def test_experts_module_matches_original():
    """QuantExperts routes tokens exactly like the original experts module:
    with the original weights replaced by the dequantized ones, outputs must match."""
    model = tiny_model()
    experts = model.model.layers[0].mlp.experts
    torch.manual_seed(4)
    hidden = torch.randn(6, 256).to(DTYPE)
    top_k_index = torch.randint(0, 8, (6, 2))
    top_k_weights = torch.softmax(torch.randn(6, 2), -1).to(DTYPE)

    for name, m in methods().items():
        qe = QuantExperts(experts, m)
        reference = copy.deepcopy(experts)
        with torch.no_grad():
            reference.gate_up_proj.copy_(m.dequantize(**qe._expert_weights("gate_up_", slice(None))))
            reference.down_proj.copy_(m.dequantize(**qe._expert_weights("down_", slice(None))))
        expected = reference(hidden, top_k_index, top_k_weights)
        got = qe(hidden, top_k_index, top_k_weights)
        err = rel_error(got, expected)
        assert err < (0.05 if name == "fp8" else 0.02), f"{name}: experts error {err:.4f}"


def test_full_model_quality_and_memory():
    """Whole-model logits stay close to bf16, and weight memory shrinks as expected."""
    base = tiny_model()
    ids = torch.randint(3, 512, (1, 24))
    with torch.no_grad():
        ref_logits = base(ids).logits
    base_mb = weight_memory(base)["weights_total_mb"]

    for name, m in methods().items():
        model = copy.deepcopy(base)
        stats = quantize_model(model, m, ["experts", "linear"], "cpu")
        assert stats["experts"] == 2 and stats["linear"] == 8, stats   # 2 layers x (experts, q/k/v/o)
        with torch.no_grad():
            logits = model(ids).logits
        cos = cosine(logits, ref_logits)
        mem = weight_memory(model)
        print(f"  {name:8} logit cosine {cos:.4f} | weights {mem['weights_total_mb']} MB (bf16 {base_mb} MB)")
        assert cos > MIN_LOGIT_COSINE[name], f"{name}: logit cosine {cos:.4f}"
        assert mem["weights_total_mb"] < base_mb


def test_cache_roundtrip():
    """Fresh quantization and loading from the saved file give identical results."""
    base = tiny_model()
    ids = torch.randint(3, 512, (1, 16))
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "tiny")
        base.save_pretrained(model_dir)
        cache_dir = os.path.join(tmp, "cache")
        for name, m in methods().items():
            fresh, s1 = load_quantized_model(model_dir, m, ["experts", "linear"], "cpu", DTYPE,
                                             cache_dir=cache_dir, use_cache=True)
            cached, s2 = load_quantized_model(model_dir, m, ["experts", "linear"], "cpu", DTYPE,
                                              cache_dir=cache_dir, use_cache=True)
            assert s1["load_source"] == "fresh" and s2["load_source"] == "cache"
            with torch.no_grad():
                assert torch.equal(fresh(ids).logits, cached(ids).logits), f"{name}: cache mismatch"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            print(f"{t.__name__} ...")
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)