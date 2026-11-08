"""
engine/quantization/fp8.py - FP8 weights and activations (W8A8, e4m3).

Weights: one scale per output channel, computed once at load time.
Activations: one scale per token, computed on the fly every forward pass ("dynamic").
    scale = max(|row|) / 448          (448 is the largest e4m3 value)
    w_fp8 = (w / scale) cast to float8_e4m3fn

On GPUs with FP8 tensor cores (compute capability 8.9+: Ada, Hopper, Blackwell) the
matmul runs natively with torch._scaled_mm. On other hardware it's emulated: values
are rounded to FP8 exactly as on real hardware, but the matmul runs in bf16. The
accuracy is the same either way; only the real path is faster.
"""
import torch
from engine.quantization.base import QuantMethod

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max   # 448.0


def native_fp8_available(device) -> bool:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device) >= (8, 9)
    return False


class FP8WeightAct(QuantMethod):
    name = "fp8"

    def __init__(self, backend: str = "auto", device="cpu", **kwargs):
        super().__init__(**kwargs)
        if backend == "auto":
            self.native = native_fp8_available(device)
        elif backend == "native":
            self.native = True
        else:
            self.native = False
        self._warned = False

    def supports(self, weight):
        # torch._scaled_mm needs both matrix dimensions to be multiples of 16
        return weight.shape[-1] % 16 == 0 and weight.shape[-2] % 16 == 0

    def quantize(self, weight):
        w = weight.float()
        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX   # [..., out, 1]
        q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        return {"qweight": q, "scale": scale.squeeze(-1)}   # scale kept in float32: [..., out]

    def dequantize(self, qweight, scale):
        return qweight.float() * scale.unsqueeze(-1)

    @staticmethod
    def quantize_activation(x2d):
        """Per-token dynamic quantization of a [tokens, features] activation."""
        scale = x2d.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX   # [M, 1]
        x_fp8 = (x2d.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        return x_fp8, scale

    def linear(self, x, qweight, scale):
        out_dtype = x.dtype
        x2d = x.reshape(-1, x.shape[-1])
        x_fp8, x_scale = self.quantize_activation(x2d)

        if self.native:
            try:
                y = torch._scaled_mm(
                    x_fp8,                         # [M, K] row-major
                    qweight.t(),                   # [K, N] column-major (weight stored [N, K])
                    scale_a=x_scale,               # [M, 1] float32
                    scale_b=scale.unsqueeze(0),    # [1, N] float32
                    out_dtype=torch.bfloat16,
                )
                return y.to(out_dtype).reshape(*x.shape[:-1], -1)
            except (RuntimeError, NotImplementedError) as e:
                if not self._warned:
                    print(f"[fp8] native FP8 matmul failed ({str(e)[:120]}); switching to emulation")
                    self._warned = True
                self.native = False

        # Emulation: same FP8 rounding, bf16 matmul
        x_deq = (x_fp8.to(out_dtype) * x_scale.to(out_dtype))
        y = torch.nn.functional.linear(x_deq, qweight.to(out_dtype))
        y = y * scale.to(out_dtype)
        return y.reshape(*x.shape[:-1], -1)

    def describe(self):
        return {"method": self.name, "fp8_backend": "native" if self.native else "emulated"}