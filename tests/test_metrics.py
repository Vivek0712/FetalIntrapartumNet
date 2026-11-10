"""Segmentation, AoP and classification metrics, including empty-mask edge cases."""

import math

import numpy as np
import pytest

from fin.evaluation.cls_metrics import cluster_bootstrap, fast_metrics, point_metrics, youden_threshold
from fin.evaluation.seg_metrics import aop_stats, bf1, hd95, per_image, pixel_accuracy, summarise_seg


def _square(n=32, x0=8, y0=8, s=10, value=1):
    m = np.zeros((n, n), np.uint8)
    m[y0 : y0 + s, x0 : x0 + s] = value
    return m


def test_dice_iou_identical_empty_and_one_sided():
    gt = np.stack([_square(), _square(), np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.uint8)])
    pred = np.stack([_square(), np.zeros((32, 32), np.uint8), _square(), np.zeros((32, 32), np.uint8)])
    r = per_image(pred, gt)
    assert r["dice_FH"][0] == 1.0 and r["iou_FH"][0] == 1.0
    assert r["dice_FH"][1] == 0.0 and r["iou_FH"][1] == 0.0  # prediction empty
    assert r["dice_FH"][2] == 0.0  # ground truth empty
    assert math.isnan(r["dice_FH"][3]) and math.isnan(r["iou_FH"][3])  # both empty: excluded
    assert np.isnan(r["hd95_FH"][1]) and np.isnan(r["hd95_FH"][2])
    s = summarise_seg(r)
    assert s["dice_FH_n"] == 3
    assert s["dice_FH_mean"] == pytest.approx(1 / 3)
    assert s["n_pred_empty_FH"] == 2 and s["n_hd_undefined_FH"] == 3
    assert s["pooled_dice_FH"] == pytest.approx(2 * 100 / (200 + 200))


def test_partial_overlap_dice_iou():
    gt, pred = _square(x0=8), _square(x0=13)  # overlap 5 x 10 of two 10 x 10 squares
    r = per_image(pred[None], gt[None])
    assert r["dice_FH"][0] == pytest.approx(2 * 50 / 200)
    assert r["iou_FH"][0] == pytest.approx(50 / 150)


def test_hd95_shift_and_sampling():
    a, b = _square(x0=8).astype(bool), _square(x0=11).astype(bool)
    assert hd95(a, a) == 0.0
    assert hd95(a, b) == pytest.approx(3.0)
    assert hd95(a, b, sampling=(1.0, 2.0)) == pytest.approx(6.0)  # columns scaled by 2
    assert math.isnan(hd95(a, np.zeros_like(a)))


def test_bf1_edge_cases():
    a = _square().astype(bool)
    empty = np.zeros_like(a)
    assert bf1(a, a)[2] == 1.0
    assert bf1(a, empty)[2] == 0.0
    assert math.isnan(bf1(empty, empty)[2])


def test_pixel_accuracy():
    gt = np.stack([_square(), _square()])
    pred = np.stack([_square(), np.zeros((32, 32), np.uint8)])
    np.testing.assert_allclose(pixel_accuracy(pred, gt), [1.0, 1 - 100 / 1024])


def test_aop_stats_failures_and_errors():
    ref = np.array([90.0, 100.0, 110.0, np.nan, 95.0])
    pred = np.array([92.0, 94.0, np.nan, 80.0, 95.0])
    s = aop_stats(pred, ref, n_boot=50)
    assert s["n"] == 4 and s["n_fail"] == 1 and s["n_used"] == 3
    assert s["mae"] == pytest.approx((2 + 6 + 0) / 3)
    assert s["bias"] == pytest.approx((2 - 6 + 0) / 3)
    assert s["within5"] == pytest.approx(2 / 3)
    assert s["max_ae"] == 6.0
    assert s["mae_ci95"][0] <= s["mae"] <= s["mae_ci95"][1]


def test_point_metrics_and_fast_metrics_agree():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    m, fm = point_metrics(y, p), fast_metrics(y, p)
    assert fm["acc"] == pytest.approx(m["accuracy"])
    assert fm["sens"] == pytest.approx(m["standard"]["recall"])
    assert fm["spec"] == pytest.approx(m["standard"]["specificity"])
    assert fm["mf1"] == pytest.approx(m["macro_f1"])
    assert fm["auc"] == pytest.approx(m["auc"])
    assert np.array(m["confusion_matrix"]).sum() == 200


def test_perfect_classifier_and_youden():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = point_metrics(y, p)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["auc"] == 1.0
    assert 0.2 < youden_threshold(y, p) <= 0.8


def test_cluster_bootstrap_resamples_clusters():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.4, 0.9, 0.1, 0.6])
    res = cluster_bootstrap(y, p, np.array(["a", "a", "b", "b", "c", "c"]), n_boot=100)
    assert res["n_clusters"] == 3 and res["n_boot"] == 100
    lo, hi = res["ci95"]["accuracy"]
    assert 0 <= lo <= hi <= 1
