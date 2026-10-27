"""
engine/core/fused_linear.py - fused projections and per-weight layout autotuning (FUSED_PROJ=1).

Round 3b. bench/micro/bench_gemm.py showed that in decode (1-64 rows per step) cuBLAS reads the
same weights at very different speeds depending on the matmul's shape and the weight's layout:
    - narrow matmuls waste the GPU: Qwen3-4B's k_proj/v_proj (2560 -> 1024) ran at ~17% of peak
      bandwidth at 16 rows, while one fused q+k+v matmul ran at ~70%
    - the kernel cuBLAS picks depends on the layout: the fused gate+up matmul at 64 rows ran at
      37% of peak with the weight stored [out, in] and 87% stored [in, out]
So, like vLLM's QKVParallelLinear / MergedColumnParallelLinear, this:
    1. fuses q/k/v into one weight and (dense MLPs) gate/up into one, at load time. The original
       modules' weights become views into the fused tensor, so memory doesn't grow.
    2. times each distinct weight shape in both layouts at a few decode batch sizes when the
       engine starts, and keeps the faster one per weight (WEIGHT_LAYOUT=auto). The choice depends
       on the GPU and cuBLAS version, which is why it's measured rather than hard-coded.
Quantized layers (QuantLinear) and the experts of MoE models are left alone (the fused MoE kernel
handles experts). A tied output layer (Qwen3-4B's lm_head IS its embedding matrix) keeps its
layout, since transposing it would need a second 0.8 GB copy.
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLinear:
    """y = x @ W^T + b for a weight that may be several projections side by side.
    layout "normal" stores W as [out, in] (PyTorch's own), "transposed" as [in, out]."""

    def __init__(self, weight, bias, sizes, layout="normal"):
        self.layout, self.sizes = layout, list(sizes)
        self.out_features, self.in_features = weight.shape
        if layout == "transposed":
            self.wt = weight.t().contiguous()            # [in, out]
            self.w = None
        else:
            self.w = weight.contiguous()                 # [out, in]
            self.wt = None
        self.bias = bias

    def weight_view(self):
        """The weight as [out, in], viewing the stored tensor (no copy)."""
        return self.w if self.w is not None else self.wt.t()

    def __call__(self, x):
        if self.w is not None:
            return F.linear(x, self.w, self.bias)
        if self.bias is not None:
            return torch.addmm(self.bias, x, self.wt)
        return x @ self.wt

    def split(self, y):
        return y.split(self.sizes, dim=-1)               # views, no copies


def _is_plain_linear(m):
    return type(m) is nn.Linear                           # not QuantLinear or other replacements


def _rebind(module, view):
    """Point an nn.Linear's weight at (a view of) the fused tensor, so memory isn't doubled and the
    original module still computes the same thing if anything calls it."""
    module.weight = nn.Parameter(view, requires_grad=False)


# ---------------------------------------------------------------- layout autotuning
L2_DEFEAT_BYTES = 256 * 1024 ** 2      # rotate weight copies so timings read GPU memory, not L2
TUNE_ROWS = (1, 16, 64)


def _time(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def tune_layout(out_features, in_features, dtype, device, rows=TUNE_ROWS):
    """Time both layouts for a weight shape; returns (best layout, {layout: total ms over rows})."""
    nbytes = out_features * in_features * torch.finfo(dtype).bits // 8
    copies = max(2, math.ceil(L2_DEFEAT_BYTES / nbytes))
    ws = [torch.randn(out_features, in_features, device=device, dtype=dtype) for _ in range(copies)]
    wts = [w.t().contiguous() for w in ws]
    totals = {"normal": 0.0, "transposed": 0.0}
    for m in rows:          # each layout timed 3 times (alternating), keeping its best: less noise
        best = {"normal": float("inf"), "transposed": float("inf")}
        x = torch.randn(m, in_features, device=device, dtype=dtype)
        i = {"n": 0}

        def normal():
            i["n"] = (i["n"] + 1) % copies
            return F.linear(x, ws[i["n"]])

        def transposed():
            i["n"] = (i["n"] + 1) % copies
            return x @ wts[i["n"]]
        for _ in range(3):
            best["normal"] = min(best["normal"], _time(normal))
            best["transposed"] = min(best["transposed"], _time(transposed))
        totals["normal"] += best["normal"]
        totals["transposed"] += best["transposed"]
    del ws, wts
    torch.cuda.empty_cache()
    # switch only when [in, out] is clearly faster (5%), so near-ties don't flip between runs
    layout = "transposed" if totals["transposed"] < 0.95 * totals["normal"] else "normal"
    return layout, totals


class LayoutChooser:
    """Picks each weight's layout. With WEIGHT_LAYOUT=auto, results are saved to a JSON file (keyed
    by GPU, PyTorch version, dtype and shape) and reused, so every run of the same setup uses the
    same layouts - timing noise on a busy GPU can't make two benchmark runs differ. Delete the file
    (config.TUNING_CACHE) to measure again."""

    def __init__(self, mode, dtype, device, cache_file=None):
        if mode not in ("auto", "normal", "transposed"):
            raise ValueError(f"Unknown WEIGHT_LAYOUT '{mode}' (use auto, normal or transposed)")
        self.mode, self.dtype, self.device = mode, dtype, torch.device(device)
        self.cache, self.report = {}, {}
        self.cache_file = cache_file
        self.saved = {}
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    self.saved = json.load(f)
            except (OSError, ValueError):
                self.saved = {}

    def _key(self, out_features, in_features):
        gpu = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        return f"{gpu} | torch {torch.__version__} | {self.dtype} | {out_features}x{in_features}"

    def _save(self):
        if not self.cache_file:
            return
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        tmp = self.cache_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.saved, f, indent=2, sort_keys=True)
        os.replace(tmp, self.cache_file)

    def __call__(self, name, out_features, in_features, can_transpose=True):
        if not can_transpose:
            layout, why = "normal", "kept (tied to the embeddings)"
        elif self.mode != "auto":
            layout, why = self.mode, "forced"
        elif self.device.type != "cuda":
            layout, why = "normal", "no GPU to time on"
        else:
            key = self._key(out_features, in_features)
            if key in self.saved:
                entry = self.saved[key]
                source = "from the tuning cache"
            else:
                if key not in self.cache:
                    layout, totals = tune_layout(out_features, in_features, self.dtype, self.device)
                    self.cache[key] = {"layout": layout, "ms": {k: round(v, 4) for k, v in totals.items()}}
                entry = self.cache[key]
                self.saved[key] = entry
                self._save()
                source = "measured now"
            layout, ms = entry["layout"], entry["ms"]
            why = (f"{source}, {'/'.join(map(str, TUNE_ROWS))} rows: [out,in] {ms['normal']:.3f} ms, "
                   f"[in,out] {ms['transposed']:.3f} ms")
        self.report.setdefault(name, {"shape": f"{out_features}x{in_features}", "layout": layout, "why": why})
        return layout


# ---------------------------------------------------------------- building the fused layers
@torch.no_grad()
def fuse_model(model, layout_mode="auto", cache_file=None):
    """Returns (per-layer dict of FusedLinear, output-layer FusedLinear or None, layout report)."""
    p = next(model.parameters())
    choose = LayoutChooser(layout_mode, p.dtype, p.device, cache_file)
    per_layer = []
    for layer in model.model.layers:
        entry = {}
        attn = layer.self_attn
        projs = [getattr(attn, n, None) for n in ("q_proj", "k_proj", "v_proj")]
        if all(_is_plain_linear(m) for m in projs):
            entry["qkv"] = _fuse(projs, choose, "qkv")
        if _is_plain_linear(getattr(attn, "o_proj", None)):
            entry["o"] = _fuse([attn.o_proj], choose, "o_proj")
        mlp = layer.mlp
        mlp_projs = [getattr(mlp, n, None) for n in ("gate_proj", "up_proj", "down_proj")]
        if all(_is_plain_linear(m) for m in mlp_projs) and hasattr(mlp, "act_fn"):   # dense MLP
            entry["gate_up"] = _fuse(mlp_projs[:2], choose, "gate_up")
            entry["down"] = _fuse([mlp_projs[2]], choose, "down_proj")
            entry["act"] = mlp.act_fn
        per_layer.append(entry)

    lm_head = None
    head = getattr(model, "lm_head", None)
    if _is_plain_linear(head):
        embed = model.get_input_embeddings()
        tied = embed is not None and embed.weight.data_ptr() == head.weight.data_ptr()
        lm_head = _fuse([head], choose, "lm_head", can_transpose=not tied)
    return per_layer, lm_head, choose.report


def _fuse(modules, choose, name, can_transpose=True):
    sizes = [m.out_features for m in modules]
    in_features = modules[0].in_features
    layout = choose(name, sum(sizes), in_features, can_transpose)
    if len(modules) == 1 and layout == "normal":          # nothing to fuse or move: use it as is
        m = modules[0]
        return FusedLinear(m.weight.data, None if m.bias is None else m.bias.data, sizes, "normal")
    weight = torch.cat([m.weight.data for m in modules], dim=0)
    bias = None
    if any(m.bias is not None for m in modules):
        bias = torch.cat([m.bias.data if m.bias is not None else
                          torch.zeros(m.out_features, dtype=weight.dtype, device=weight.device) for m in modules])
    fused = FusedLinear(weight, bias, sizes, layout)
    del weight
    view, start = fused.weight_view(), 0
    for m, n in zip(modules, sizes):                       # originals become views: no extra memory
        _rebind(m, view[start:start + n])
        if m.bias is not None:
            m.bias = nn.Parameter(fused.bias[start:start + n], requires_grad=False)
        start += n
    return fused