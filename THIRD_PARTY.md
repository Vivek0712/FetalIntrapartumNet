# Third-party code, weights and data

The code in this repository is released under the MIT licence (`LICENSE`). It uses the sources below. Code
whose repository carries no licence is **not redistributed**: `fin fetch-third-party` downloads the exact file
from the original repository at a pinned commit, checks its SHA-256, and `fin.third_party` applies our
documented adaptations in memory (each adaptation is an exact text substitution that must match once, so a
changed upstream file is detected). Nothing derived from that code is stored in this repository.

## Code fetched at a pinned commit (not redistributed)

| Component | Source | Pinned commit | File (SHA-256) | Licence status | Use and adaptations |
|---|---|---|---|---|---|
| FAT-Net (Wu et al., *Med. Image Anal.* 76 (2022) 102327) | https://github.com/SZUcsh/FAT-Net | `732db6ead0b8056c5a82f2702d3cbd8216dd8b59` | `FAT_Net.py` (`23355e47…a3cf4e9`) | No licence (GitHub API reports none) | Segmentation baseline. DeiT-tiny-distilled weights from timm instead of `torch.hub` (same weights, no remote code), ResNet-34 weights from torchvision, a `pretrained` switch, token grid reshaped to H/16 x W/16 for 256 x 256 input, 3 classes; grayscale replication and ImageNet normalisation in our wrapper `FATNetGray`. |
| MALUNet (Ruan et al., IEEE BIBM 2022) | https://github.com/JCruan519/MALUNet | `e184b47de99b3fda6b7880fcf3ff99b806308d2b` | `models/malunet.py` (`c7c1bacd…6d73308dc0`) | No licence (GitHub API reports none) | Segmentation baseline. 1 input channel and 3 classes (constructor arguments), final sigmoid removed (logits for cross-entropy), `trunc_normal_` imported from `timm.layers`, a construction-time print removed. |
| FH-PS-AoP challenge AoP function `drawline_AOD` in `ellipse.py` (MICCAI 2023 challenge) | https://github.com/maskoffs/PS-FH-MICCAI23, a collection of the released code of the challenge teams (the file is identical in all eight team folders). The repository is not confirmed to belong to the challenge organisers; the paper credits the function to the FH-PS-AoP challenge and cites the challenge data release (Bai and Ou, Zenodo 7851339) | `da34d616165d66eb89da6c43faf7313ed03cf969` | `1/nnUNetv2/ellipse.py` (`4396eddd…22343ed2`) | No licence (GitHub API reports none) | The AoP of every reported mask-derived measurement. Module `aop_geometry`: distances with `math.hypot`, angle with `math.degrees`, cosine clipped to [-1, 1], all construction points returned (not only the angle). Module `challenge_ellipse`: unchanged, used by `fin check-geometry`. |

`fin.geometry` is our driver around the fetched function: it fits an ellipse to the largest external contour of
each structure's own class, calls the adapted `drawline_AOD` and returns NaN when no angle exists (missing class,
contour under five points, degenerate ellipse) instead of raising or returning 0. On the ground-truth masks of
all 3743 JNU-IFM SPHead frames and 300 IUGC 2024 test frames the adapted and the unchanged function agree to
within 9 x 10^-14 degrees (`results/checks/geometry_vs_challenge.json`).

## Code included in this repository

| Component | Origin | Licence |
|---|---|---|
| DeepFetalNet, AoP-GAU-Net, PSFHSP-Net re-implementation, torchvision wrappers of the baselines | The authors' original model code (commit `631c725`), module names unchanged so that trained checkpoints load | MIT (this repository) |
| MobileNetV2 scratch module tree (`fin.models.classifiers`: `_make_divisible`, `InvertedResidual`, `MobileNetV2Classifier` scratch path) | Follows https://github.com/d-li14/mobilenetv2.pytorch (commit `1733532bd43743442077326e1efc556d7cfd025d`) | Apache-2.0; licence text in `third_party_licenses/mobilenetv2.pytorch-Apache-2.0.txt` |
| ShuffleNetV2 scratch module tree (`fin.models.classifiers`: `channel_shuffle`, `ShuffleBlock`) | Adapted from torchvision's `torchvision/models/shufflenetv2.py` (`channel_shuffle`, `InvertedResidual`), copyright Soumith Chintala 2016. The authors' original header credited https://github.com/ZhuYun97/ShuffleNet-v2-Pytorch (commit `78bf811fd628d07eed68bca51136eae3927cc573`, no licence), whose implementation differs in structure; none of it is included | BSD-3-Clause (torchvision), adapted; licence text in `third_party_licenses/torchvision-BSD-3-Clause.txt` |
| U-Net, losses, metrics, Grad-CAM, efficiency counter, data pipeline, reporting | Written for this study (U-Net from Ronneberger et al. 2015; Grad-CAM from Selvaraju et al. 2017) | MIT (this repository) |

The scratch trees of ShuffleNetV2 and MobileNetV2 are used only by the efficiency benchmark and the
`--init scratch` option; every reported baseline was trained from the torchvision ImageNet trees.

## Libraries and pretrained weights

| Item | Source | Licence |
|---|---|---|
| ResNet-18/34/152, SqueezeNet 1.1, ShuffleNetV2 x1.0, MobileNetV2, MobileNetV3-Large architectures and `IMAGENET1K_V1` weights | torchvision | BSD-3-Clause |
| DeiT-tiny-distilled (`deit_tiny_distilled_patch16_224.fb_in1k`) | timm / facebookresearch/deit | Apache-2.0 |
| PyTorch, NumPy, pandas, SciPy, scikit-learn, OpenCV, Matplotlib, SimpleITK, PyYAML, thop, fvcore | PyPI (versions in `requirements.txt`) | Their respective open-source licences |

## Data (not redistributed; see docs/DATA.md)

| Dataset | Source | Licence | Use |
|---|---|---|---|
| JNU-IFM (Lu et al., *Data in Brief* 41 (2022) 107904) | figshare, doi:10.6084/m9.figshare.14371652.v2 | CC BY 4.0 | Training, validation and internal test (all tasks) |
| FH-PS-AoP challenge training set (MICCAI 2023) | Zenodo, doi:10.5281/zenodo.7851339 | CC BY 4.0 | Identification of the SPHead frames, mask cross-check, reference AoP |
| IUGC 2024 test set | Zenodo, doi:10.5281/zenodo.17655183 | CC BY 4.0 | External test (segmentation and AoP); masks of labelled training/validation frames only for the mask floor of the second external set |
| IUGC 2025 release (469 images) | Zenodo, doi:10.5281/zenodo.17356295 | CC BY 4.0 | Second external AoP test |

Derived tables committed under `splits/` and `results/` (frame lists, split assignments, reference AoPs,
per-frame predictions) are derived from these CC BY 4.0 datasets and are shared under the same terms with
attribution to the dataset creators.
