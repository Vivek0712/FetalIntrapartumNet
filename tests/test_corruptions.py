"""Robustness corruptions: deterministic per frame, range- and dtype-preserving, noise variance as configured."""

import numpy as np
import pytest

from fin.robustness.corruptions import conditions, corrupt

X = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)


@pytest.mark.parametrize("cond", conditions(), ids=[c[0] for c in conditions()])
def test_deterministic_range_and_dtype(cond):
    cid, kind, _, _ = cond
    a, b = corrupt(X, cid, "f1"), corrupt(X, cid, "f1")
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1
    if kind in ("speckle", "gaussian"):
        assert not np.array_equal(a, corrupt(X, cid, "f2"))  # differs between frames
    if kind == "none":
        assert np.array_equal(a, X)


def test_noise_levels():
    flat = np.full((512, 512), 0.5, np.float32)
    assert abs(float(np.var(corrupt(flat, "gauss-0.01", "f") - flat)) - 0.01) < 0.001
    assert abs(float(np.var(corrupt(flat, "speckle-0.05", "f") - flat)) - 0.05 * 0.25) < 0.0015


def test_uint8_input_and_blur_preserves_constant():
    u8 = np.full((32, 32), 128, np.uint8)
    out = corrupt(u8, "blur-9", "f")
    np.testing.assert_allclose(out, 128 / 255, rtol=1e-6)
