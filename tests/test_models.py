"""Forward shapes and parameter counts of the seven classifiers and four segmenters."""

import pytest
import torch

from fin import third_party
from fin.models import (
    CLASSIFIERS,
    SEGMENTERS,
    THIRD_PARTY_SEGMENTERS,
    build_classifier,
    build_segmenter,
    count_params,
    split_output,
)

CLS_PARAMS = {
    "deepfetalnet": 659_433,
    "psfhspnet": 387_906,
    "resnet152": 58_147_906,
    "squeezenet11": 723_522,
    "shufflenetv2": 1_255_654,
    "mobilenetv2": 2_226_434,
    "mobilenetv3": 4_204_594,
}
SEG_PARAMS = {"unet": 31_036_611, "fatnet": 28_759_235, "malunet": 177_817, "aopgaunet": 7_795_211}


def _available(name: str) -> bool:
    return name not in THIRD_PARTY_SEGMENTERS or third_party.is_available(THIRD_PARTY_SEGMENTERS[name])


@pytest.mark.parametrize("name", list(CLASSIFIERS))
@pytest.mark.parametrize("init", ["scratch", "pretrained"])
def test_classifier_forward_and_params(name, init):
    if name == "deepfetalnet" and init == "pretrained":
        with pytest.raises(ValueError):
            build_classifier(name, init, load_weights=False)
        return
    model, resolved, _ = build_classifier(name, init, load_weights=False)
    assert resolved == init
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)
    assert count_params(model) == CLS_PARAMS[name]


@pytest.mark.parametrize("name", ["shufflenetv2", "mobilenetv2", "resnet152"])
def test_pretrained_tree_is_the_torchvision_tree(name):
    model, _, _ = build_classifier(name, "pretrained", load_weights=False)
    assert all(k.startswith("backbone.") for k in model.state_dict())


@pytest.mark.parametrize("name", list(SEGMENTERS))
def test_segmenter_forward_and_params(name):
    if not _available(name):
        pytest.skip("third-party source not fetched (fin fetch-third-party)")
    model = build_segmenter(name, pretrained=False).eval()
    with torch.no_grad():
        seg, aop = split_output(model(torch.rand(2, 1, 256, 256)))
    assert seg.shape == (2, 3, 256, 256)
    assert count_params(model) == SEG_PARAMS[name]
    if name == "aopgaunet":
        assert aop.shape == (2,) and bool(((aop > 0) & (aop < 180)).all())
    else:
        assert aop is None


def test_default_inits():
    assert CLASSIFIERS["deepfetalnet"].default_init == "scratch"
    assert all(CLASSIFIERS[m].default_init == "pretrained" for m in CLASSIFIERS if m != "deepfetalnet")


def test_third_party_pins():
    for name, src in third_party.SOURCES.items():
        assert len(src.commit) == 40 and len(src.sha256) == 64, name
        assert src.url.startswith("https://raw.githubusercontent.com/")
    unchanged, adapted = third_party.SOURCES["challenge_ellipse"], third_party.SOURCES["aop_geometry"]
    assert (unchanged.repo, unchanged.commit, unchanged.path) == (adapted.repo, adapted.commit, adapted.path)
    assert not unchanged.patches and len(adapted.patches) == 5
