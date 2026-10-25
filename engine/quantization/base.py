"""
engine/quantization/base.py - the interface every quantization method implements.

A method knows how to:
  quantize(w)          turn a float weight [..., out, in] into a dict of tensors
                       (e.g. {"qweight": int8, "scale": bf16}). Leading dims (like the
                       expert dimension of OLMoE's fused experts) are kept as they are.
  linear(x, **q)       compute x @ W.T using those tensors (the hot path at inference)
  dequantize(**q)      rebuild an approximate float weight (for tests and debugging)

`linear` is the reference implementation in plain PyTorch. This is the function a
custom (e.g. Triton) kernel replaces later: same inputs, same outputs, faster.
"""
import torch


class QuantMethod:
    name = "base"

    def __init__(self, compute_dtype=torch.bfloat16):
        # dtype of activations and outputs (OLMoE runs in bfloat16)
        self.compute_dtype = compute_dtype

    def supports(self, weight: torch.Tensor) -> bool:
        """Whether this weight's shape can be quantized (otherwise it stays in bf16)."""
        return True

    def quantize(self, weight: torch.Tensor) -> dict:
        raise NotImplementedError

    def dequantize(self, **q) -> torch.Tensor:
        raise NotImplementedError

    def linear(self, x: torch.Tensor, **q) -> torch.Tensor:
        """Reference path: rebuild the float weight, then a normal matmul.
        Saves memory (weights are stored small) but does extra work every call."""
        w = self.dequantize(**q).to(x.dtype)
        return torch.nn.functional.linear(x, w)

    def describe(self) -> dict:
        return {"method": self.name}