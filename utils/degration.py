import math
from typing import Iterable, List, Sequence, Union

import torch
import torch.nn.functional as F

class DegradationOperator:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _ensure_batch(self, x: torch.Tensor):
        if x.dim() == 3:
            return x.unsqueeze(0), True
        return x, False

    def _restore_batch(self, x: torch.Tensor, squeezed: bool):
        if squeezed:
            return x.squeeze(0)
        return x


class KernelBlur(DegradationOperator):
    def __init__(self, kernel: torch.Tensor, padding_mode: str = "reflect"):
        if kernel.dim() != 2:
            raise ValueError("kernel must be a 2D tensor.")
        kernel_sum = kernel.sum().clamp_min(1e-12)
        self.kernel = kernel.float() / kernel_sum
        self.k = int(kernel.shape[-1])
        self.pad = self.k // 2
        self.mode = padding_mode

    def _expanded_kernel(self, x):
        _, channels, _, _ = x.shape
        return self.kernel.to(device=x.device, dtype=x.dtype).expand(channels, 1, *self.kernel.shape)

    def __call__(self, x):
        x, squeeze = self._ensure_batch(x)
        k = self._expanded_kernel(x)
        if self.mode in ("zeros", "constant"):
            x = F.pad(x, (self.pad,) * 4, mode="constant", value=0)
            out = F.conv2d(x, k, groups=x.shape[1])
        else:
            x_p = F.pad(x, (self.pad,) * 4, mode=self.mode)
            out = F.conv2d(x_p, k, groups=x.shape[1])
        return self._restore_batch(out, squeeze)

    def transpose(self, y):
        y, squeeze = self._ensure_batch(y)
        k = self._expanded_kernel(y)
        if self.mode in ("zeros", "constant"):
            k = torch.flip(k, dims=(-2, -1))
            y = F.pad(y, (self.pad,) * 4, mode="constant", value=0)
            out = F.conv2d(y, k, groups=y.shape[1])
        else:
            y = F.pad(y, (self.pad,) * 4, mode=self.mode)
            out = F.conv2d(y, k, groups=y.shape[1])
        return self._restore_batch(out, squeeze)


class GaussianBlur(KernelBlur):
    def __init__(self, kernel_size=7, sigma=1.0, padding_mode="reflect"):
        self.sigma = sigma
        ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss /= gauss.sum()
        super().__init__(gauss[:, None] * gauss[None, :], padding_mode=padding_mode)


class MotionBlur(KernelBlur):
    def __init__(self, kernel_size=7, angle=0.0, padding_mode="reflect"):
        base = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        base[kernel_size // 2] = 1.0
        theta = torch.tensor(
            [
                [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0.0],
                [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0.0],
            ],
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta.unsqueeze(0), size=(1, 1, kernel_size, kernel_size), align_corners=False)
        rot = F.grid_sample(base[None, None], grid, align_corners=False)
        super().__init__(rot.squeeze(0).squeeze(0), padding_mode=padding_mode)


class ThermalDiffusionBlur(DegradationOperator):
    def __init__(self, alpha=0.8, iterations=5):
        self.alpha = alpha
        self.iterations = iterations
        self.kernel = torch.tensor(
            [[0, alpha / 4, 0], [alpha / 4, 1 - alpha, alpha / 4], [0, alpha / 4, 0]],
            dtype=torch.float32,
        )

    def __call__(self, x):
        x, squeeze = self._ensure_batch(x)
        kernel = self.kernel.view(1, 1, 3, 3).to(device=x.device, dtype=x.dtype)
        for _ in range(self.iterations):
            x = F.conv2d(x, kernel.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
        return self._restore_batch(x, squeeze)

    def transpose(self, y):
        y, squeeze = self._ensure_batch(y)
        kernel = self.kernel.view(1, 1, 3, 3).to(device=y.device, dtype=y.dtype)
        for _ in range(self.iterations):
            y = F.conv_transpose2d(y, kernel.repeat(y.shape[1], 1, 1, 1), padding=1, groups=y.shape[1])
        return self._restore_batch(y, squeeze)


class AiryDiskBlur(KernelBlur):
    def __init__(self, radius: float = 5.0, size: int = 31, padding_mode: str = "reflect"):
        pad = size // 2
        coords = torch.arange(size, dtype=torch.float32) - pad
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rr = torch.sqrt(xx**2 + yy**2)
        kernel = (rr <= radius).float()
        super().__init__(kernel, padding_mode=padding_mode)


class ColumnStripeBlur(DegradationOperator):
    def __init__(self, kernel_len=7, padding_mode="reflect"):
        self.pad = kernel_len // 2
        self.mode = padding_mode
        self.kernel = torch.ones(1, 1, kernel_len, 1) / kernel_len

    def __call__(self, x):
        x, squeeze = self._ensure_batch(x)
        _, channels, _, _ = x.shape
        k = self.kernel.to(device=x.device, dtype=x.dtype).expand(channels, 1, *self.kernel.shape[-2:])
        if self.mode in ("zeros", "constant"):
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="constant", value=0)
            out = F.conv2d(x, k, groups=channels)
        else:
            x_p = F.pad(x, (0, 0, self.pad, self.pad), mode=self.mode)
            out = F.conv2d(x_p, k, groups=channels)
        return self._restore_batch(out, squeeze)

    def transpose(self, y):
        y, squeeze = self._ensure_batch(y)
        _, channels, _, _ = y.shape
        k = self.kernel.to(device=y.device, dtype=y.dtype).expand(channels, 1, *self.kernel.shape[-2:])
        if self.mode in ("zeros", "constant"):
            k = torch.flip(k, dims=(-2, -1))
            y = F.pad(y, (0, 0, self.pad, self.pad), mode="constant", value=0)
            out = F.conv2d(y, k, groups=channels)
        else:
            y = F.pad(y, (0, 0, self.pad, self.pad), mode=self.mode)
            out = F.conv2d(y, k, groups=channels)
        return self._restore_batch(out, squeeze)


class ContrastKernelReduction(KernelBlur):
    def __init__(self, kernel_size=7, alpha=0.5, padding_mode="reflect"):
        pad = kernel_size // 2
        delta = torch.zeros((kernel_size, kernel_size))
        delta[pad, pad] = 1.0
        uniform = torch.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        super().__init__((1 - alpha) * delta + alpha * uniform, padding_mode=padding_mode)


class LowLight(DegradationOperator):
    def __init__(self, gamma: float = 1.8, gain: float = 0.75):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, x):
        return (self.gain * x.clamp(0, 1).pow(self.gamma)).clamp(0, 1)

    def transpose(self, y):
        return (y.clamp(0, 1) / max(self.gain, 1e-6)).pow(1.0 / self.gamma).clamp(0, 1)


class OverExposure(DegradationOperator):
    def __init__(self, gamma: float = 0.65, gain: float = 1.25):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, x):
        return (self.gain * x.clamp(0, 1).pow(self.gamma)).clamp(0, 1)

    def transpose(self, y):
        return (y.clamp(0, 1) / max(self.gain, 1e-6)).pow(1.0 / self.gamma).clamp(0, 1)


class Null(DegradationOperator):
    def __call__(self, x):
        return x

    def transpose(self, y):
        return y


class DegradationPipeline(DegradationOperator):
    def __init__(self, ops: Sequence[DegradationOperator] = ()):
        self.ops = list(ops) if ops else [Null()]
        self.A = self

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x

    def transpose(self, y):
        for op in reversed(self.ops):
            y = op.transpose(y)
        return y

    def is_identity(self) -> bool:
        return all(isinstance(op, Null) for op in self.ops)

    def has_exposure_degradation(self) -> bool:
        return any(isinstance(op, (LowLight, OverExposure)) for op in self.ops)

    def exposure_modes(self):
        return {
            "lowlight": any(isinstance(op, LowLight) for op in self.ops),
            "overexposure": any(isinstance(op, OverExposure) for op in self.ops),
        }


def strip_exposure_operators(pipeline: DegradationPipeline):
    ops = [op for op in pipeline.ops if not isinstance(op, (LowLight, OverExposure))]
    return DegradationPipeline(ops or [Null()])


def _parse_value(text: str, default):
    if text is None or text == "":
        return default
    if isinstance(default, int):
        return int(text)
    return float(text)


def _build_named_operator(name: str) -> DegradationOperator:
    parts = [p.strip() for p in name.lower().replace(":", "@").split("@")]
    key = parts[0]
    args = parts[1:]
    if key in ("", "none", "null", "identity", "clean"):
        return Null()
    if key in ("gaussian", "gaussian_blur", "blur"):
        return GaussianBlur(kernel_size=_parse_value(args[0], 7) if args else 7,
                            sigma=_parse_value(args[1], 1.0) if len(args) > 1 else 1.0)
    if key in ("motion", "motion_blur"):
        return MotionBlur(kernel_size=_parse_value(args[0], 7) if args else 7,
                          angle=_parse_value(args[1], 0.0) if len(args) > 1 else 0.0)
    if key in ("low", "lowlight", "low_light", "under", "underexposure"):
        return LowLight(gamma=_parse_value(args[0], 1.8) if args else 1.8,
                        gain=_parse_value(args[1], 0.75) if len(args) > 1 else 0.75)
    if key in ("over", "overexposure", "over_exposure"):
        return OverExposure(gamma=_parse_value(args[0], 0.65) if args else 0.65,
                            gain=_parse_value(args[1], 1.25) if len(args) > 1 else 1.25)
    if key in ("thermal", "thermal_diffusion"):
        return ThermalDiffusionBlur(alpha=_parse_value(args[0], 0.8) if args else 0.8,
                                    iterations=_parse_value(args[1], 5) if len(args) > 1 else 5)
    if key in ("stripe", "column_stripe"):
        return ColumnStripeBlur(kernel_len=_parse_value(args[0], 5) if args else 5)
    if key in ("airy", "airy_disk"):
        return AiryDiskBlur(radius=_parse_value(args[0], 1.5) if args else 1.5,
                            size=_parse_value(args[1], 5) if len(args) > 1 else 5)
    if key in ("contrast", "contrast_kernel"):
        return ContrastKernelReduction(kernel_size=_parse_value(args[0], 7) if args else 7,
                                       alpha=_parse_value(args[1], 0.5) if len(args) > 1 else 0.5)
    raise ValueError(f"Unknown degradation operator: {name}")


def build_degradation_pipeline(spec: Union[str, DegradationOperator, Iterable[Union[str, DegradationOperator]], None]):
    if isinstance(spec, DegradationOperator):
        return DegradationPipeline([spec])
    if spec is None:
        return DegradationPipeline([Null()])
    if isinstance(spec, str):
        normalized = spec.replace("，", ",").replace(",", "+")
        tokens = [token.strip() for token in normalized.split("+") if token.strip()]
        return DegradationPipeline([_build_named_operator(token) for token in tokens] or [Null()])

    ops: List[DegradationOperator] = []
    for item in spec:
        if isinstance(item, DegradationOperator):
            ops.append(item)
        elif isinstance(item, str):
            ops.append(_build_named_operator(item))
        else:
            raise TypeError(f"Unsupported degradation spec item: {type(item)!r}")
    return DegradationPipeline(ops or [Null()])


def get_degradation_operator(spec=None) -> DegradationPipeline:
    return build_degradation_pipeline(spec)
