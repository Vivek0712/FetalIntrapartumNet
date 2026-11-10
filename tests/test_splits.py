"""Split integrity from the committed split files: patient-disjoint splits and CV folds."""

import json

import pandas as pd
import pytest

from fin.config import SPLITS, load_config
from fin.io import read_frames_csv

FRAMES = read_frames_csv(SPLITS / "frames.csv")


def test_every_frame_once_and_counts():
    assert FRAMES.frame_id.is_unique
    assert len(FRAMES) == 6224
    assert FRAMES.split.value_counts().to_dict() == {"train": 4958, "val": 636, "test": 630}
    assert FRAMES.groupby("split").patient_id.nunique().to_dict() == {"train": 38, "val": 6, "test": 7}
    assert FRAMES.groupby("split").video_id.nunique().to_dict() == {"train": 61, "val": 9, "test": 8}
    assert FRAMES.category_name.isin(["None", "OnlySP", "OnlyHead", "SPHead"]).all()


@pytest.mark.parametrize("key", ["patient_id", "video_id"])
def test_no_woman_or_video_in_two_splits(key):
    per = FRAMES.groupby(key).split.nunique()
    assert (per == 1).all()


def test_split_videos_json_matches_frames():
    sv = json.loads((SPLITS / "split_videos.json").read_text())
    for split in ("train", "val", "test"):
        assert sorted(FRAMES[FRAMES.split == split].video_id.unique()) == sv["videos"][split]
        assert sorted(int(p) for p in FRAMES[FRAMES.split == split].patient_id.unique()) == sv["patients"][split]
    assert sv["seed"] == load_config("split")["seed"]


def test_cv_folds_partition_women_and_frames():
    k = load_config("cv")["k"]
    folds = [pd.read_csv(SPLITS / "cv" / f"fold{i}.csv") for i in range(k)]
    tested = pd.concat([f[f.split == "test"] for f in folds])
    assert len(tested) == len(FRAMES) and tested.frame_id.is_unique
    assert set(tested.frame_id) == set(FRAMES.frame_id)
    test_women = [set(f[f.split == "test"].patient_id) for f in folds]
    assert set().union(*test_women) == set(FRAMES.patient_id)
    assert sum(len(w) for w in test_women) == FRAMES.patient_id.nunique()
    for f in folds:
        assert set(f.frame_id) == set(FRAMES.frame_id) and f.frame_id.is_unique
        roles = f.groupby("patient_id").split.nunique()
        assert (roles == 1).all()  # no woman in two roles within a fold
    meta = json.loads((SPLITS / "cv" / "folds.json").read_text())
    assert meta["k"] == k and len(meta["folds"]) == k


def test_video_level_split_assigns_every_frame_once():
    v = pd.read_csv(SPLITS / "video_level_split.csv")
    assert v.frame_id.is_unique and set(v.frame_id) == set(FRAMES.frame_id)
    assert v.split.value_counts().to_dict() == {"train": 4979, "val": 623, "test": 622}
    assert (v.groupby("video_id").split.nunique() == 1).all()


def test_ext469_sources_cover_the_release():
    s = pd.read_csv(SPLITS / "ext469_sources.csv")
    assert len(s) == 469 and s.Filename.is_unique
