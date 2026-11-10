"""Result I/O: portable paths, JSON round trip and repository-root discovery."""

import pytest

from fin import RepositoryNotFoundError
from fin.config import ROOT, repo_root
from fin.io import portable_path, portable_text, read_json, write_json


@pytest.fixture
def work(tmp_path, monkeypatch):
    w = tmp_path / "work"
    (w / "runs").mkdir(parents=True)
    monkeypatch.setenv("FIN_WORK", str(w))
    monkeypatch.delenv("FIN_RUNS", raising=False)
    monkeypatch.delenv("FIN_DATA", raising=False)
    return w


def test_portable_path(work, tmp_path):
    assert portable_path(work / "runs" / "cls" / "m" / "seed0") == "$FIN_WORK/runs/cls/m/seed0"
    assert portable_path(work) == "$FIN_WORK"
    assert portable_path(ROOT / "results" / "runs") == "results/runs"
    assert portable_path(tmp_path / "elsewhere") == str(tmp_path / "elsewhere")


def test_portable_path_through_symlink(work, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(work, target_is_directory=True)
    assert portable_path(link / "runs" / "x") == "$FIN_WORK/runs/x"


def test_portable_text(work):
    text = f'{{"run": "{work}/runs/seg/unet/seed0", "root": "{work}"}}\nwrote {ROOT}/docs/x.svg\n'
    assert portable_text(text) == '{"run": "$FIN_WORK/runs/seg/unet/seed0", "root": "$FIN_WORK"}\nwrote docs/x.svg\n'
    assert portable_text(f"{work}2/runs") == f"{work}2/runs"  # a different folder sharing the prefix is kept


def test_json_round_trip(tmp_path):
    p = write_json(tmp_path / "a" / "b.json", {"x": [1, 2.5, None]})
    assert read_json(p) == {"x": [1, 2.5, None]}
    assert not list(p.parent.glob("*.tmp"))


def test_repo_root(monkeypatch, tmp_path):
    assert repo_root() == ROOT
    monkeypatch.setenv("FIN_REPO", str(ROOT))
    assert repo_root() == ROOT
    monkeypatch.setenv("FIN_REPO", str(tmp_path))
    with pytest.raises(RepositoryNotFoundError, match="not a FetalIntrapartumNet checkout"):
        repo_root()
