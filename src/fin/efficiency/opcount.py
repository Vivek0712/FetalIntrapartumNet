"""Exact per-call multiply-accumulate (MAC) counter.

Intercepts every executed call (``TorchFunctionMode``), so shared modules are counted once per call and transposed
convolutions are counted on their input grid. Counts the MACs of conv1d/2d, conv_transpose2d, linear,
matmul/bmm/baddbmm/addmm/@, scaled_dot_product_attention and ``nn.MultiheadAttention`` (counted as a whole:
in-projections, QK^T, AV, out-projection). Element-wise operations, norms, pooling and activations are excluded
(the same scope as thop/fvcore convolution + linear counts).
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


def _numel(s: tuple[int, ...] | torch.Size) -> int:
    return int(math.prod(s))


class MacCounter(TorchFunctionMode):
    """Torch function mode that accumulates MACs per operation family."""

    def __init__(self) -> None:
        super().__init__()
        self.macs: dict[str, int] = {}

    def _add(self, k: str, v: int) -> None:
        self.macs[k] = self.macs.get(k, 0) + int(v)

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        with contextlib.suppress(IndexError, KeyError, AttributeError, TypeError):
            self._count(func, args, kwargs, out)
        return out

    def _count(self, func: Any, args: tuple, kwargs: dict, out: Any) -> None:
        name = getattr(func, "__name__", "")
        if func in (F.conv2d, torch.conv2d, F.conv1d, torch.conv1d):
            w = args[1] if len(args) > 1 else kwargs["weight"]
            self._add("conv", out.numel() * _numel(w.shape[1:]))
        elif func in (F.conv_transpose2d, torch.conv_transpose2d):
            w = args[1] if len(args) > 1 else kwargs["weight"]
            self._add("conv_transpose", args[0].numel() * _numel(w.shape[1:]))
        elif func in (F.linear, torch._C._nn.linear):
            w = args[1] if len(args) > 1 else kwargs["weight"]
            self._add("linear", out.numel() * w.shape[1])
        elif func in (torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.bmm, torch.Tensor.bmm):
            self._add("matmul", out.numel() * args[0].shape[-1])
        elif func in (torch.baddbmm, torch.addmm, torch.addbmm):
            self._add("matmul", out.numel() * args[1].shape[-1])
        elif name in ("multi_head_attention_forward", "_native_multi_head_attention"):
            # the mode is disabled inside the handler, so the whole operation is counted here;
            # multi_head_attention_forward takes (L, B, E) inputs, _native_multi_head_attention (B, L, E)
            q, k = args[0], args[1]
            e = int(args[3])
            if name == "multi_head_attention_forward":
                n_l, b, s = q.shape[0], (q.shape[1] if q.dim() == 3 else 1), k.shape[0]
            else:
                b, n_l, s = (q.shape[0] if q.dim() == 3 else 1), q.shape[-2], k.shape[-2]
            self._add("mha", b * (n_l * e * e + 2 * s * e * e + 2 * n_l * s * e + n_l * e * e))
        elif func is F.scaled_dot_product_attention:
            q, k = args[0], args[1]
            n_l, s, e, ev = q.shape[-2], k.shape[-2], q.shape[-1], args[2].shape[-1]
            self._add("sdpa", _numel(q.shape[:-2]) * n_l * s * (e + ev))


def count_macs(model: nn.Module, x: torch.Tensor) -> tuple[int, dict[str, int]]:
    """Total MACs of one forward pass and the per-family breakdown."""
    model = model.eval()
    counter = MacCounter()
    with torch.no_grad(), counter:
        model(x)
    return sum(counter.macs.values()), dict(counter.macs)
