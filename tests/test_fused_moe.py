"""
tests/test_fused_moe.py - the fused MoE Triton kernels against plain PyTorch.

On a machine without a GPU this runs the kernels through Triton's interpreter (TRITON_INTERPRET=1,
numpy on the CPU) in float32: slow, but it checks the indexing and math exactly. With a GPU it
runs the real kernels in bfloat16 with bf16-sized tolerances, including OLMoE's real shapes.

Run from the project root:
    python tests/test_fused_moe.py
"""
import os
import sys

import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"        # must be set before triton is imported

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile  # noqa: E402

import config  # noqa: E402
from kernels.fused_moe.fused_moe_triton import fused_moe, moe_align  # noqa: E402
from kernels.fused_moe.module import FusedExperts, apply_fused_moe  # noqa: E402
from kernels.fused_moe.reference import moe_reference  # noqa: E402

GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"
DTYPE = torch.bfloat16 if GPU else torch.float32
TOL = 2e-2 if GPU else 1e-4                       # relative error


def rel_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def random_moe(T, H, I, E, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(T, H, generator=g) * 0.5).to(DEVICE, DTYPE)
    w1 = (torch.randn(E, 2 * I, H, generator=g) * H ** -0.5).to(DEVICE, DTYPE)
    w2 = (torch.randn(E, H, I, generator=g) * I ** -0.5).to(DEVICE, DTYPE)
    logits = torch.randn(T, E, generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), k, dim=-1)
    return hidden, w1, w2, w.to(DEVICE, DTYPE), ids.to(DEVICE)


# ---------------------------------------------------------------- moe_align
def test_moe_align_groups_and_pads():
    torch.manual_seed(1)
    T, k, E, bm = 13, 3, 8, 4
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)]).to(DEVICE)
    sorted_ids, expert_ids, num_padded = moe_align(ids, E, bm)
    n = int(num_padded.item())
    rows = sorted_ids[:n].cpu()
    real = rows[rows < T * k]
    assert sorted(real.tolist()) == list(range(T * k)), "every pair exactly once"
    flat = ids.reshape(-1).cpu()
    for b in range(n // bm):                         # each tile holds a single expert's pairs
        tile = rows[b * bm:(b + 1) * bm]
        tile = tile[tile < T * k]
        assert all(flat[p] == expert_ids[b].item() for p in tile.tolist())
    assert n % bm == 0 and n <= T * k + E * (bm - 1)


def _groups(sorted_ids, expert_ids, num_padded, num_pairs, bm):
    """{expert: set of pair indices} and the per-tile expert list - the layout-independent content."""
    n = int(num_padded.item())
    rows, owners = sorted_ids[:n].cpu().tolist(), expert_ids[:n // bm].cpu().tolist()
    groups = {}
    for b, e in enumerate(owners):
        real = [r for r in rows[b * bm:(b + 1) * bm] if r < num_pairs]
        if real:
            groups.setdefault(e, set()).update(real)
    return n, owners, groups


def test_cuda_align_matches_torch_align():
    """The CUDA C++ moe_align must produce the same grouping as the PyTorch one.
    (Pairs inside one expert's tiles can come out in a different order - atomics - which is fine:
    the matmul treats every row independently.)"""
    if not GPU:
        print("  (needs a GPU: skipped)")
        return
    from kernels.fused_moe import moe_align_cuda
    assert moe_align_cuda.available(), f"CUDA moe_align did not compile: {moe_align_cuda._error}"
    torch.manual_seed(2)
    for T, k, E, bm in ((1, 8, 64, 16), (13, 3, 8, 4), (16, 8, 64, 16), (300, 8, 64, 32), (4096, 8, 64, 64)):
        ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)]).to(DEVICE)
        for dtype in (torch.int64, torch.int32):
            want = _groups(*moe_align(ids.to(dtype), E, bm), T * k, bm)
            sorted_ids, expert_ids, num_padded = moe_align_cuda.moe_align_cuda(ids.to(dtype), E, bm)
            got = _groups(sorted_ids, expert_ids, num_padded, T * k, bm)
            assert got == want, f"T={T} k={k} E={E} bm={bm} {dtype}"
            n = got[0]
            pad = sorted_ids[:n][sorted_ids[:n] >= T * k]
            assert (pad == T * k).all(), "padding rows must point past the last pair"
    print("  identical expert grouping, tile owners and padded size for 5 shapes x 2 index dtypes")


def test_fused_with_cuda_align_matches_reference():
    if not GPU:
        print("  (needs a GPU: skipped)")
        return
    from kernels.fused_moe import moe_align_cuda
    assert moe_align_cuda.available(), "CUDA moe_align did not compile (see the error above)"
    for T in (1, 16, 256):
        args = random_moe(T, 2048, 1024, 64, 8, seed=T)
        want = moe_reference(*args)
        err_cuda = rel_error(fused_moe(*args, align="cuda"), want)
        err_torch = rel_error(fused_moe(*args, align="torch"), want)
        print(f"  T={T:<4}: relative error cuda-align {err_cuda:.2e}, torch-align {err_torch:.2e}")
        assert err_cuda < TOL


# ---------------------------------------------------------------- kernels vs reference
def test_fused_matches_reference():
    shapes = [(1, 64, 32, 8, 2), (5, 64, 32, 8, 2), (33, 96, 48, 6, 3)]   # (T, H, I, E, k)
    if GPU:
        shapes += [(1, 2048, 1024, 64, 8), (16, 2048, 1024, 64, 8), (256, 2048, 1024, 64, 8)]
    for T, H, I, E, k in shapes:
        args = random_moe(T, H, I, E, k)
        got, want = fused_moe(*args), moe_reference(*args)
        err = rel_error(got, want)
        print(f"  T={T:<4} H={H:<5} I={I:<5} E={E:<3} k={k}: relative error {err:.2e}")
        assert got.shape == want.shape and err < TOL


def test_tile_configs_agree():
    """Every tile configuration must give the same answer (padding/masking at all sizes)."""
    args = random_moe(9, 64, 40, 5, 2, seed=3)
    want = moe_reference(*args)
    for bm, bn, bk in ((16, 32, 32), (32, 64, 16), (64, 16, 64)):
        got = fused_moe(*args, config=dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=4, num_stages=2))
        assert rel_error(got, want) < TOL, (bm, bn, bk)


# ---------------------------------------------------------------- inside the model
def test_matches_transformers_experts_module():
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    cfg = OlmoeConfig(hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, num_experts=8, num_experts_per_tok=2, vocab_size=128)
    model = OlmoeForCausalLM(cfg).to(DEVICE, DTYPE).eval()
    ids = torch.randint(0, 128, (1, 12), device=DEVICE)
    with torch.no_grad():
        want = model(ids).logits
        assert apply_fused_moe(model) == 2, "both layers' experts should be swapped"
        assert isinstance(model.model.layers[0].mlp.experts, FusedExperts)
        got = model(ids).logits
    err = rel_error(got, want)
    print(f"  whole-model logits vs transformers' experts: relative error {err:.2e}")
    assert err < TOL


def test_engine_outputs_exact_with_fused_moe():
    """Custom engine with MOE_KERNEL=triton must still reproduce model.generate() (float32)."""
    if GPU:
        print("  (runs on CPU only: exact float32 comparison)")
        return
    from tests.test_batching import build_tiny_model, make_engine, reference_outputs, run_concurrently
    from transformers import PreTrainedTokenizerFast
    prompts = [("w5 w9 w14 w3", 6), ("w100 w200 w300 w400 w5 w6", 5), ("w42", 4)]
    with tempfile.TemporaryDirectory() as d:
        model = build_tiny_model(d)
        refs = reference_outputs(model, PreTrainedTokenizerFast.from_pretrained(d), prompts)
        saved = config.MOE_KERNEL
        config.MOE_KERNEL = "triton"
        try:
            engine = make_engine(d, "continuous", "paged")
        finally:
            config.MOE_KERNEL = saved
        assert engine.load_info.get("moe_kernel") == "triton"
        outs, _ = run_concurrently(engine, prompts)
        assert outs == refs, f"{outs} != {refs}"


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