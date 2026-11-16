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