"""Third-party source files without a licence: fetched at a pinned commit, verified, adapted in memory.

The repositories of FAT-Net, MALUNet and the FH-PS-AoP challenge AoP code carry no licence (the GitHub API reports
none at the pinned commits), so their source is not redistributed here. ``fetch`` downloads each file from the
original repository at a pinned commit into ``$FIN_THIRD_PARTY`` and checks its SHA-256. ``adapted_source`` applies
our documented adaptations as exact text substitutions (each must match exactly once, so a changed upstream file is
detected) and ``load_module`` executes the result as an in-memory module. Nothing is written back to the cache.

Adaptations
    FAT-Net (Wu et al., Medical Image Analysis 76 (2022) 102327), ``FAT_Net.py``
        * DeiT-tiny-distilled ImageNet weights from timm (``deit_tiny_distilled_patch16_224.fb_in1k``) instead of
          ``torch.hub.load("facebookresearch/deit:main", ...)`` (same weights, no remote code execution), with
          ``patch_embed.strict_img_size`` off; ResNet-34 ImageNet weights from torchvision (``IMAGENET1K_V1``);
          a ``pretrained`` argument switches both downloads off (used before loading a trained checkpoint).
        * The token grid is reshaped to (H/16, W/16) instead of the hard-coded 14 x 14, so a 256 x 256 input
          (16 x 16 tokens) works. As published, no positional embedding or class/distillation tokens are used.
        * ``n_classes = 3``; ``fin.models.third_party.FATNetGray`` replicates the grayscale input to three channels
          and applies the ImageNet normalisation. The published code reuses one FAMBlock instance inside each
          ModuleList; kept.
    MALUNet (Ruan et al., IEEE BIBM 2022), ``models/malunet.py``
        * ``input_channels = 1``, ``num_classes = 3`` (constructor arguments); default c_list, ``split_att="fc"``,
          ``bridge=True``.
        * The final ``torch.sigmoid`` is removed so the model returns logits for softmax/cross-entropy.
        * ``trunc_normal_`` is imported from ``timm.layers`` (its current location); a construction-time print
          is removed.
    FH-PS-AoP challenge AoP function ``drawline_AOD`` (MICCAI 2023), ``ellipse.py``
        The file as distributed in the ``maskoffs/PS-FH-MICCAI23`` collection of released challenge-team code
        (identical in all eight team folders). Two modules are built from it:
        * ``challenge_ellipse``: unchanged; ``fin check-geometry`` compares the adapted function against it.
        * ``aop_geometry``: the function behind every reported mask-derived AoP (driven by ``fin.geometry``):
          - the three distances use ``math.hypot`` and the angle ``math.degrees``;
          - the cosine of the law-of-cosines step is clipped to [-1, 1] (no domain error from rounding);
          - the function returns every point of the construction (P1, P2, T and the tangent-frame values) as a
            dictionary instead of only the angle, so figures draw exactly the construction that was evaluated.
          The first two changes alter the angle by less than 1e-13 degrees on all 4043 ground-truth mask pairs
          (``fin check-geometry``); they reproduce the arithmetic of the implementation that produced the
          reported values bit for bit.

    fin fetch-third-party [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import types
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fin.config import third_party_dir
from fin.io import setup_logging

log = logging.getLogger(__name__)


class ThirdPartyMissingError(RuntimeError):
    """A third-party source file has not been fetched yet."""


@dataclass(frozen=True)
class Source:
    """One upstream file pinned to a commit, with the exact text substitutions applied before use."""

    repo: str
    commit: str
    path: str
    sha256: str
    patches: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def url(self) -> str:
        """Raw download URL of the pinned file."""
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{self.path}"

    @property
    def cache_path(self) -> Path:
        """Location of the fetched file in the third-party cache."""
        return third_party_dir() / self.repo.replace("/", "__") / self.commit / self.path


_CHALLENGE = {
    "repo": "maskoffs/PS-FH-MICCAI23",
    "commit": "da34d616165d66eb89da6c43faf7313ed03cf969",
    "path": "1/nnUNetv2/ellipse.py",
    "sha256": "4396eddd5a4e8a3a3e0556e62f693d775bea381f79bfb024e135ba5222343ed2",
}

SOURCES: dict[str, Source] = {
    "fatnet": Source(
        repo="SZUcsh/FAT-Net",
        commit="732db6ead0b8056c5a82f2702d3cbd8216dd8b59",
        path="FAT_Net.py",
        sha256="23355e4736f51b142dc2943bdb13138e531500b10e588f729a94deb58a3cf4e9",
        patches=(
            (
                "def __init__(self, n_channels=3, n_classes=1):",
                "def __init__(self, n_channels=3, n_classes=1, pretrained=True):",
            ),
            (
                "transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', "
                "pretrained=True)",
                "transformer = _fin_deit_tiny_distilled(pretrained)",
            ),
            ("resnet = resnet_model.resnet34(pretrained=True)", "resnet = _fin_resnet34(pretrained)"),
            (
                "feature_tf = feature_tf.view(b, 192, 14, 14)",
                "feature_tf = feature_tf.reshape(b, 192, h // 16, w // 16)",
            ),
        ),
    ),
    "malunet": Source(
        repo="JCruan519/MALUNet",
        commit="e184b47de99b3fda6b7880fcf3ff99b806308d2b",
        path="models/malunet.py",
        sha256="c7c1bacd4f7b434670c97416e490316edeac9b1ab19ec626a070ac6d73308dc0",
        patches=(
            ("from timm.models.layers import trunc_normal_", "from timm.layers import trunc_normal_"),
            ("print('SC_Att_Bridge was used')", "pass"),
            ("return torch.sigmoid(out0)", "return out0"),
        ),
    ),
    "challenge_ellipse": Source(**_CHALLENGE),
    "aop_geometry": Source(
        **_CHALLENGE,
        patches=(
            (
                "ld1d3 = math.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2)",
                "ld1d3 = math.hypot(d11 - d21, d12 - d22)",
            ),
            (
                "ld3x4 = math.sqrt((d21 - qie1) ** 2 + (d22 - qie2) ** 2)",
                "ld3x4 = math.hypot(d21 - qie1, d22 - qie2)",
            ),
            (
                "ld1x4 = math.sqrt((d11 - qie1) ** 2 + (d12 - qie2) ** 2)",
                "ld1x4 = math.hypot(d11 - qie1, d12 - qie2)",
            ),
            (
                "aod = math.acos((ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / (2 * ld1d3 * ld3x4)) / math.pi * 180",
                "aod = math.degrees(math.acos(max(-1.0, min(1.0, "
                "(ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / (2 * ld1d3 * ld3x4)))))",
            ),
            (
                "return aod",
                "return {'angle': aod, 'p1': (d11, d12), 'p2': (d21, d22), 't': (qie1, qie2), 'a': a, 'b': b, "
                "'x': dpz21, 'y': dpz22, 'q1': qiepz1, 'q2': qiepz2, "
                "'ps_long': element1[1][0], 'ps_short': element1[1][1]}",
            ),
        ),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, force: bool = False) -> Path:
    """Download one pinned source file into the cache (if needed) and verify its checksum."""
    src = SOURCES[name]
    path = src.cache_path
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(src.url, timeout=60) as resp:
            data = resp.read()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    digest = _sha256(path)
    if digest != src.sha256:
        raise RuntimeError(f"{path}: SHA-256 {digest} does not match the pinned {src.sha256}")
    return path


def is_available(name: str) -> bool:
    """True when the pinned file is in the cache with the expected checksum."""
    path = SOURCES[name].cache_path
    return path.exists() and _sha256(path) == SOURCES[name].sha256


def adapted_source(name: str) -> str:
    """Pinned upstream source with our substitutions applied."""
    src = SOURCES[name]
    if not is_available(name):
        raise ThirdPartyMissingError(
            f"{src.repo}@{src.commit[:7]}/{src.path} is not fetched; run `fin fetch-third-party`"
        )
    text = src.cache_path.read_text()
    for old, new in src.patches:
        if text.count(old) != 1:
            raise RuntimeError(f"{name}: adaptation target found {text.count(old)} times: {old!r}")
        text = text.replace(old, new)
    return text


def load_module(name: str, namespace: Mapping[str, Any] | None = None) -> types.ModuleType:
    """Execute the adapted source of ``name`` as a new in-memory module; ``namespace`` is defined first."""
    src = SOURCES[name]
    module = types.ModuleType(f"fin_third_party_{name}")
    module.__file__ = str(src.cache_path)
    module.__dict__.update(namespace or {})
    exec(compile(adapted_source(name), str(src.cache_path), "exec"), module.__dict__)
    return module


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: fetch every pinned file."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if cached")
    a = ap.parse_args(argv)
    setup_logging()
    for name, src in SOURCES.items():
        path = fetch(name, a.force)
        log.info("%s: %s@%s %s -> %s (sha256 ok)", name, src.repo, src.commit[:7], src.path, path)


if __name__ == "__main__":
    main()
