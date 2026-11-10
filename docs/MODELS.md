# Models

Eleven networks: seven Standard-plane classifiers and four FH/PS segmenters. Parameter counts and per-call GMACs
are those of `results/runs/efficiency/results.json` (`fin efficiency`) and are checked by `tests/test_models.py`.

## Classifiers (input 3 x 224 x 224, ImageNet mean/std; two logits, index 1 = Standard)

| Model | Code | Parameters | GMACs | Initialisation in every reported run |
|---|---|---:|---:|---|
| DeepFetalNet (proposed) | `fin.models.deepfetalnet.DeepFetalNet` | 659,433 | 0.113 | random (no ImageNet weights exist) |
| PSFHSP-Net | `fin.models.classifiers.PSFHSPNet` | 387,906 | 0.760 | ImageNet ResNet-18 copied into conv1, bn1, layer1, layer2 (48 of 50 tensors; fc new) |
| ResNet-152 | `ResNet152Classifier` (torchvision) | 58,147,906 | 11.512 | torchvision `IMAGENET1K_V1`, new 2-way FC |
| SqueezeNet 1.1 | `SqueezeNet11Classifier` | 723,522 | 0.263 | `IMAGENET1K_V1`, new 2-way final 1x1 conv |
| ShuffleNetV2 1.0x | `ShuffleNetV2Classifier` | 1,255,654 | 0.144 | `IMAGENET1K_V1`, new 2-way FC |
| MobileNetV2 | `MobileNetV2Classifier` | 2,226,434 | 0.299 | `IMAGENET1K_V1`, new 2-way FC |
| MobileNetV3-Large | `MobileNetV3LargeClassifier` | 4,204,594 | 0.215 | `IMAGENET1K_V1`, new 2-way FC |

**DeepFetalNet**: stem (3 -> 32, stride 2); four depthwise-separable stages with squeeze-and-excitation (64, 128,
256, 256; strides 2, 2, 2, 1); local branch AMFAM (three dilated depthwise-separable branches, dilations 1/2/3,
mixed by softmax weights from pooled statistics); global branch (1x1 projection to 64, 4-head self-attention over
the 14 x 14 tokens, residual + LayerNorm, projection back to 256); adaptive fusion (softmax weights of the two
branches); attention refinement (channel + spatial attention added to the input); head GAP -> FC 256->128 -> ReLU
-> dropout 0.5 -> FC 128->2. The supplement (Section S1) gives the layer-by-layer definition.

`fin.models.build_classifier(name, init, load_weights)`: `init` is `default` (the column above), `pretrained` or
`scratch`. ShuffleNetV2 and MobileNetV2 have different module trees for `pretrained` (torchvision) and `scratch`
(the authors' trees, used only by the efficiency benchmark), so a checkpoint is loaded into the tree it was trained
with (`config.json` records the init); `load_weights=False` builds a tree without downloading weights.

## Segmenters (input 1 x 256 x 256 in [0, 1]; logits for 0 bg, 1 FH, 2 PS)

| Model | Code | Parameters | GMACs | Notes |
|---|---|---:|---:|---|
| U-Net | `fin.models.unet.UNet` | 31,036,611 | 48.10 | 64-128-256-512-1024, two 3x3 conv + BN + ReLU per level, transposed-conv up-sampling |
| FAT-Net | `fin.models.third_party.FATNetGray` | 28,759,235 | 30.05 | published code fetched at a pinned commit, adapted (below); ResNet-34 + DeiT-tiny dual encoder |
| MALUNet | `fin.models.third_party.build_malunet` | 177,817 | 0.074 | published code fetched at a pinned commit, adapted (below); with the SC_Att_Bridge |
| AoP-GAU-Net (proposed) | `fin.models.aopgaunet.AoPGAUNet` | 7,795,211 | 7.93 | also returns AoP = 180 * sigmoid(head) in degrees |

**AoP-GAU-Net**: four encoder blocks (base width 32, depthwise-separable convolutions, max-pooling); bottleneck
convolutions to 512 channels; graph bottleneck at 16 x 16 (noise-aware attention, dynamic cosine-similarity graph,
sparse k-nearest-neighbour graph attention with k = 6, 4 heads of 32, 2-D positional encoding, convolutional
feed-forward); four decoder blocks with attention-gated skips; 1x1 segmentation head; attention-pooling regression
head (hidden 64). Kaiming initialisation for convolutions, Xavier for linear layers.

### Adaptations of the third-party segmenters

Both repositories carry no licence, so their files are fetched at pinned commits (`fin fetch-third-party`,
`THIRD_PARTY.md`) and adapted in memory by exact text substitutions (`fin.third_party.SOURCES`); the wrappers are
in `fin.models.third_party`:

- **FAT-Net** (`SZUcsh/FAT-Net@732db6e`, `FAT_Net.py`): DeiT-tiny-distilled weights from timm instead of
  `torch.hub` (same weights, no remote code), `patch_embed.strict_img_size` off; ResNet-34 weights from torchvision;
  `pretrained` switch; token grid reshaped to (H/16, W/16) instead of the hard-coded 14 x 14; 3 classes. Wrapper
  `FATNetGray` replicates the grayscale input to 3 channels and applies the ImageNet normalisation. The published
  code reuses one FAMBlock instance inside each ModuleList (weight sharing); kept, so `count_params` counts it once.
- **MALUNet** (`JCruan519/MALUNet@e184b47`, `models/malunet.py`): 1 input channel, 3 classes; the final
  `torch.sigmoid` removed (logits for softmax/cross-entropy); `trunc_normal_` from `timm.layers`; a construction-time
  print removed.

## Training protocols (`configs/`)

| | Classification (`classification.yaml`) | Segmentation (`segmentation.yaml`) |
|---|---|---|
| Data | locked split, all frames (4958 / 636 / 630) | SPHead frames (2994 / 370 / 379) |
| Input | 224 x 224 x 3, ImageNet mean/std | 256 x 256 x 1 in [0, 1] |
| Augmentation (train only; `augmentation` block) | rotation +-10 deg, scale 0.9-1.1, shift +-4 %, brightness +-0.1, contrast x0.9-1.1; no flips | same, one affine shared by image and mask |
| Loss | cross-entropy (unweighted) | CE + Dice (baselines); AoP-GAU-Net: Dice + CE + 0.5 boundary + 0.2 AoP |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-2 | AdamW, lr 3e-4, weight decay 1e-4 |
| Schedule | cosine, T_max 30, eta_min 1e-6, per epoch | cosine, T_max 60, eta_min 0, per epoch |
| Batch | 32 (validation 64) | 8, drop_last (validation 16) |
| Epochs / early stopping | max 30, patience 7 on validation macro-F1 | max 60, patience 10 on validation mean FG Dice |
| Selection | best validation macro-F1 at threshold 0.5 | best validation mean of FH and PS Dice |
| Seeds | 0, 1, 2 (CV: 0 per fold) | 0, 1, 2 for each of the four segmenters |

AoP-GAU-Net loss corrections relative to the released code (`fin.training.losses`): the boundary term is computed
per foreground class on one-hot targets; the AoP term is the MSE on the normalised scale (angle / 180); frames
without a reference AoP contribute no AoP loss.

Classification evaluation: threshold 0.5 on p(Standard) (primary) and the Youden-J threshold chosen on the run's
validation predictions (secondary); 95% CIs by video-level cluster bootstrap (2000 resamples). Segmentation
evaluation: per-image Dice/IoU/HD95/BF1 on the 256 x 256 grid; AoP from the predicted masks with `fin.geometry`
(the FH-PS-AoP challenge function `drawline_AOD`, fetched at a pinned commit and adapted in memory; `THIRD_PARTY.md`).
