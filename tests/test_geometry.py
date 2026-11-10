"""AoP geometry on synthetic ellipses with known angles, failure handling and the challenge function."""

import math

import cv2
import numpy as np
import pytest

from fin import third_party
from fin.evaluation.geometry_check import challenge_drawline_aod
from fin.geometry import aop_from_mask, construct, landmark_aop, mask_construction, reference_aop

needs_challenge_code = pytest.mark.skipif(
    not third_party.is_available("aop_geometry"), reason="run `fin fetch-third-party` first"
)


def _known_configuration(r: float, d: float):
    """Horizontal PS axis (P2 = right end at (200, 128)) pointing at an FH circle centre d pixels to its left.

    Both tangents from P2 to the circle make the angle asin(r / d) with P2->P1, so the AoP is known exactly.
    """
    p2x, cy, long_axis = 200.0, 128.0, 60.0
    ps = ((p2x - long_axis / 2, cy), (12.0, long_axis), 90.0)  # cv2 box: angle 90 -> long axis along x
    fh = ((p2x - d, cy), (2 * r, 2 * r), 0.0)
    return fh, ps


@needs_challenge_code
@pytest.mark.parametrize(
    ("r", "d", "expected"),
    [(30.0, 60.0, 30.0), (40.0, 40.0 * math.sqrt(2), 45.0), (25.0, 100.0, math.degrees(math.asin(0.25)))],
)
def test_construction_known_angle(r, d, expected):
    fh, ps = _known_configuration(r, d)
    con = construct(fh, ps)
    assert con is not None
    assert con.angle == pytest.approx(expected, abs=1e-9)
    assert con.p2 == pytest.approx((200.0, 128.0))
    # T lies on the FH circle and P2-T is tangent to it
    tx, ty = con.t
    assert math.hypot(tx - fh[0][0], ty - fh[0][1]) == pytest.approx(r, rel=1e-9)
    radius = np.array([tx - fh[0][0], ty - fh[0][1]])
    tangent = np.array([tx - con.p2[0], ty - con.p2[1]])
    assert abs(radius @ tangent) < 1e-6 * np.linalg.norm(radius) * np.linalg.norm(tangent)
    # the reported angle is the angle at P2 between P1 and T
    assert con.angle == pytest.approx(landmark_aop(con.p2, con.p1, con.t), abs=1e-9)


def _rasterised(r: float = 40.0, d: float = 80.0) -> tuple[np.ndarray, float]:
    """Label map with an FH disc and a PS ellipse (long axis 60 px along x, P2 at x = 200); known AoP."""
    mask = np.zeros((256, 256), np.uint8)
    cv2.circle(mask, (int(200 - d), 128), int(r), 1, -1)
    cv2.ellipse(mask, (170, 128), (30, 5), 0, 0, 360, 2, -1)
    return mask, math.degrees(math.asin(r / d))


@needs_challenge_code
def test_aop_from_rasterised_masks_close_to_truth():
    mask, expected = _rasterised()
    assert aop_from_mask(mask) == pytest.approx(expected, abs=1.0)


def test_secondary_reference_close_to_truth():
    mask, expected = _rasterised()
    assert reference_aop(mask == 2, mask == 1) == pytest.approx(expected, abs=2.0)


def test_failures_return_nan():
    mask = np.zeros((64, 64), np.uint8)
    assert math.isnan(aop_from_mask(mask))
    cv2.circle(mask, (32, 32), 10, 1, -1)
    assert math.isnan(aop_from_mask(mask))  # PS missing
    assert mask_construction(mask) is None


def test_landmark_aop_right_angle():
    assert landmark_aop((0, 0), (10, 0), (0, 10)) == pytest.approx(90.0)
    assert landmark_aop((0, 0), (10, 0), (10, 10)) == pytest.approx(45.0)


@needs_challenge_code
@pytest.mark.parametrize(("r", "d"), [(30.0, 60.0), (40.0, 90.0)])
def test_matches_unchanged_challenge_function(r, d):
    fh, ps = _known_configuration(r, d)
    assert construct(fh, ps).angle == pytest.approx(challenge_drawline_aod()(fh, ps), abs=1e-9)


@needs_challenge_code
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_undefined_angle_is_none():
    fh, ps = _known_configuration(30.0, 60.0)
    degenerate_fh = (fh[0], (0.0, 0.0), 0.0)
    assert construct(degenerate_fh, ps) is None
