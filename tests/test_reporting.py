"""Reporting: number formatting, result keys, README table freshness and efficiency values."""

import pytest

from fin.config import DOCS_ASSETS, RESULT_RUNS, ROOT, SPLITS
from fin.io import read_json, write_json
from fin.reporting import readme_table, repo_figures, summary, values
from fin.reporting.fmt import Store, f, fint, pm
from fin.reporting.keys import format_hint, normalise


def test_number_formatting():
    assert f(-1.234, 2) == "\\ensuremath{-}1.23"
    assert f(0.5, 1, 100) == "50.0"
    assert f(2.0, 2, sign=True) == "+2.00"
    assert f(float("nan")) is None
    assert fint(12345) == "12{,}345" and fint(9999) == "9999"
    assert pm([1.0, 3.0], 1) == "2.0 \\ensuremath{\\pm} 1.4"
    assert pm([1.0], 1) == "1.0"


def test_store_mean_sd():
    v = Store()
    v.ms("x", [0.5, 0.7], 1, 100)
    assert v["x.mean"] == "60.0" and v["x.sd"] == "14.1"
    assert v.raw["x"][0] == [0.5, 0.7]


def test_key_normalisation_and_hints():
    assert normalise("cls.deepfetalnet.test.macrof1") == "cls.deepfetalnet.mf1"
    assert normalise("cls.gate.purity") == "cls.deepfetalnet.gate.purity"
    assert normalise("seg.unet.jnu.dice_fh") == "seg.unet.jnu.dice.fh"
    assert format_hint("accuracy, mean over seeds (%, 1 dp)") == (1, True, "mean")


def _results():
    return summary.load(RESULT_RUNS, SPLITS / "frames.csv")


def test_readme_table_is_up_to_date():
    readme = (ROOT / "README.md").read_text()
    assert readme_table.replace_block(readme, readme_table.build(_results())) == readme


def test_summary_covers_every_model():
    r = _results()
    assert set(summary.CLS_ORDER) <= set(r.cv_mf1) and set(summary.CLS_ORDER) <= set(r.test_mf1)
    for m in summary.SEG_ORDER:
        for k in summary.AOP_SETS:
            assert (m, k) in r.aop
    assert set(summary.AOP_SETS) <= set(r.const) and r.const_value > 0


@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("pipeline.svg", lambda r: repo_figures.pipeline_svg()),
        ("cv_macro_f1.svg", repo_figures.cv_chart),
        ("seg_dice.svg", repo_figures.dice_chart),
        ("aop_mae.svg", repo_figures.aop_chart),
    ],
)
def test_repo_figures_are_up_to_date(name, make):
    assert (DOCS_ASSETS / name).read_text() == make(_results()) + "\n", f"run `fin repo-figures`: {name} is stale"


def _efficiency_as_on(gpu: str | None, tmp_path):
    """The committed efficiency results with the MPS entries renamed to ``gpu`` (or dropped) and no MPS memory."""
    d = read_json(RESULT_RUNS / "efficiency" / "results.json")
    for e in d["models"]:
        for k in ("latency", "throughput_b16"):
            mps = e[k].pop("mps")
            if gpu:
                e[k] = {gpu: mps, **e[k]}
        e["mps_memory_b1"] = e["mps_memory_b16"] = None
    return write_json(tmp_path / "results.json", d)


def test_efficiency_values_on_mps_results():
    v = Store()
    values.WARNINGS.clear()
    values.fill_eff(v, RESULT_RUNS / "efficiency" / "results.json")
    assert not values.WARNINGS
    assert "eff.deepfetalnet.lat.mps" in v and "eff.deepfetalnet.mem.weights" in v


@pytest.mark.parametrize("gpu", ["cuda", None])
def test_efficiency_values_without_mps(gpu, tmp_path):
    v = Store()
    values.WARNINGS.clear()
    values.fill_eff(v, _efficiency_as_on(gpu, tmp_path))
    assert "eff.deepfetalnet.lat.cpu" in v and "eff.deepfetalnet.lat.mps" not in v
    assert ("eff.deepfetalnet.lat.cuda" in v) == (gpu == "cuda")
    assert "eff.deepfetalnet.mem.weights" not in v
    assert any("no mps timings" in w for w in values.WARNINGS)
