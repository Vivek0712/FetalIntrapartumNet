# Data

All datasets are public under CC BY 4.0 and are not redistributed in this repository. Prepared data live under
`$FIN_DATA` (default `$FIN_WORK/data`, with `FIN_WORK` defaulting to `./work`). Every count below is produced by the
preparation commands; their logs are committed in `results/data/`.

## 1. Downloads

| Dataset | Use | Source | File(s) and checksum | Licence | Cite |
|---|---|---|---|---|---|
| JNU-IFM: 78 videos, 51 women, 6224 frames (NanFang Hospital, Youkey D8 probe) | training, validation and internal test of all tasks | figshare, doi:10.6084/m9.figshare.14371652.v2 | `us_data.zip`, 1,791,469,933 bytes, MD5 `0562df615326604c3b883ab224ec37ea` | CC BY 4.0 | Lu Y. et al., *Data in Brief* 41 (2022) 107904 (correction: 42 (2022) 108128) |
| FH-PS-AoP challenge training set (MICCAI 2023) | identification of the JNU-IFM SPHead frames, mask cross-check, reference AoP | Zenodo, doi:10.5281/zenodo.7851339 (open access; the later record doi:10.5281/zenodo.10829116 of the same dataset is restricted) | `Pubic Symphysis-Fetal Head Segmentation and Angle of Progression.zip` (362,910,106 bytes, MD5 `46d44052dfaa6ea095dfc0c1d0a9870c`) with `image_mha/` and `label_mha/` | CC BY 4.0 | Bai J. and Ou Z., FH-PS-AoP challenge data (2024) |
| IUGC 2024 ("Maternal-fetal ultrasound video dataset for end-to-end intrapartum biometry and multi-task learning", DatasetV3) | external test of segmentation and AoP (test archive); mask floor of the second external set (masks of labelled training/validation frames) | Zenodo, doi:10.5281/zenodo.17655183 | `DatasetV3.zip` (1,084,176,447 bytes, MD5 `548cd5d52e3459521522a526d0d25b1b`) holding `train-*.zip`, `val-*.zip`, `test-*.zip` | CC BY 4.0 | IUGC 2024 challenge and dataset papers |
| IUGC 2025 release ("A dataset with 169 images for IUGC 2025"): 469 images | second external AoP test | Zenodo, doi:10.5281/zenodo.17356295 | `A Dataset with 169 images for Intrapartum Ultrasound Grand Challenge (IUGC 2025).zip` (31,289,648 bytes, MD5 `8ee67c566b3f4afddfe06bd94c602f6c`): `label.csv` + 469 JPEG images | CC BY 4.0 | IUGC 2025 challenge |

The IUGC training set contains JNU-era videos (names such as `20190909T155747I1`); it is never used for training,
selection or thresholds. None of the 300 IUGC 2024 test frames comes from a JNU-IFM video (checked by video
identifier and nearest-neighbour image search).

## 2. Folder layout expected by the commands

```
$FIN_DATA/jnu/raw/us_data/<video>/{frame_label.csv, image/<video>_<frame>.png, mask/<video>_<frame>_mask.png}
<FHPSAOP_DIR>/{image_mha, label_mha}/NNNNN.mha                    (unzipped challenge release)
<IUGC24_ROOT>/{train,val,test}/...                                (the three archives of DatasetV3 unzipped here;
                                                                   test/videos, test/seg, test/test_info.csv, ...)
<FIN_EXT469>/{label.csv, *.jpg}                                   (unzipped IUGC 2025 release)
```

## 3. Preparation

```bash
fin jnu build  $FIN_DATA/jnu/raw/us_data $FIN_DATA/jnu/processed          # frame table, masks, locked split
fin match-aop  $FIN_DATA/jnu/processed <FHPSAOP_DIR>                       # challenge ids, reference AoPs -> frames.csv
fin build-iugc <IUGC24_ROOT> $FIN_DATA/iugc/processed                      # 300 external test frames
fin cache cls && fin cache seg                                             # 224 / 256 pixel caches
```

`scripts/run_pipeline.sh data` runs the same steps (`JNU_RAW`, `FHPSAOP_DIR`, `IUGC24_ROOT`). Image and mask paths in
the generated tables are stored relative to `$FIN_DATA`. The committed `splits/frames.csv` is the resulting frame
table without the two path columns; `fin jnu build` reproduces its split column exactly.

## 4. JNU-IFM: checks against the data descriptor

| Category (`frame_label`) | Descriptor | Found |
|---|---|---|
| 3 None | 1022 | 1022 |
| 4 OnlySP | 323 | 323 |
| 5 OnlyHead | 1136 | 1136 |
| 6 SPHead | 3743 | 3743 |
| Total | 6224 | 6224 |

- 78 video folders; every `frame_label.csv` row has an image and a mask; no unlabelled images.
- Images are 8-bit grayscale PNG, 1295 x 1026 (W x H). Masks are 2-D uint8 with values {0, 7, 8} (7 = PS, 8 = FH).
- Per-woman counts match Table 1 of the descriptor for all 51 women (patient-to-video map:
  `splits/descriptor_table1_patients.json`, parsed with `fin jnu parse-descriptor`).
- Three frames' labels disagree with their masks (kept with the official label; masks unchanged):
  `20190909T161601_69` and `20190909T161601_79` (None, mask contains PS) and `20191115T112747_169` (OnlySP, mask
  contains PS and FH). Columns `has_fh` / `has_ps` record what each mask contains.

## 5. Label conventions

| Item | Value |
|---|---|
| Processed mask (all datasets, all tasks) | **0 background, 1 FH, 2 PS** |
| JNU-IFM raw mask | 0 bg, 7 PS, 8 FH |
| FH-PS-AoP challenge label | 0 bg, 1 PS, 2 FH |
| IUGC 2024 test/validation mask | 0 bg, 127 PS, 255 FH (all 600 PS landmarks fall on 127) |
| `binary_label` | 1 = Standard = SPHead; 0 = Non-Standard = None / OnlySP / OnlyHead |
| `paper_class` | 0 = Standard, 1 = Non-Standard (= 1 - `binary_label`) |

Standard = SPHead is the only definition consistent with the task (the AoP needs both structures).

## 6. Locked split (all tasks)

`splits/frames.csv` (column `split`) and `splits/split_videos.json`. Patient-disjoint (hence video-disjoint), target
80/10/10 of frames, all four categories in every split. 20,000 random permutations of the women from
`np.random.default_rng(42)`, each filled greedily test -> validation -> training; the draw with the lowest cost
(frame-count deviation + L1 distance of the category mix) is kept (draw 14577, cost 0.081); `configs/split.yaml`.

| Split | Women | Videos | Frames | None | OnlySP | OnlyHead | SPHead (Standard) | Non-Standard |
|---|---|---|---|---|---|---|---|---|
| train | 38 | 61 | 4958 | 818 | 238 | 908 | 2994 | 1964 |
| val | 6 | 9 | 636 | 106 | 47 | 113 | 370 | 266 |
| test | 7 | 8 | 630 | 98 | 38 | 115 | 379 | 251 |
| total | 51 | 78 | 6224 | 1022 | 323 | 1136 | 3743 | 2481 |

Validation women: 5, 10, 27, 33, 48, 51; test women: 8, 19, 20, 23, 25, 32, 35. Task sizes: classification
4958/636/630; segmentation and AoP (SPHead frames) 2994/370/379; IUGC 2024 300.

Cross-validation folds (`splits/cv/fold{0..4}.csv`, `folds.json`): `fin splits cv`, parameters in
`configs/cv.yaml`; every woman is in exactly one test fold. `splits/video_level_split.csv` is a video-level split
(4979/623/622 frames, 61/9/8 videos; the same video can share a woman across subsets), used only for the
sensitivity analysis of the supplement.

## 7. Reference AoP for the JNU-IFM SPHead frames

- The FH-PS-AoP challenge images 00001-03743 are exactly the 3743 JNU-IFM SPHead frames: challenge id k is the k-th
  SPHead frame when the names `<video>_<frame>.png` are sorted as strings (so `_109` < `_19`). Verified on every
  frame (`results/data/fhpsaop_match_log.json`): image MAE at 256 x 256 mean 2.76 / max 5.05 grey levels; JNU mask
  vs remapped challenge label Dice 1.000 for FH and PS on every frame.
- `aop_ref_challenge_code` (primary JNU reference): `fin.geometry` (the fetched challenge function) on the challenge label at
  256 x 256.
- `aop_ref` (secondary): the independent ellipse–tangent implementation `fin.geometry.reference_aop` on the same
  label. |difference| between the two: mean 0.46, median 0.38, p95 1.06, max 1.89 degrees.
- JNU-IFM has no sonographer-measured AoP; both references are mask-derived on 256 x 256 (anisotropically
  resized) masks, as in the challenge.

## 8. IUGC 2024 external test set

`$FIN_DATA/iugc/processed/{images,masks,aop.csv}`: 300 frames, one per test video, decoded with OpenCV at
`labeled_frame_index` (identical to the pre-extracted frames of the release), 512 x 512 grayscale. Every labelled
frame lies in its video's standard-plane list and shows both FH and PS. `aop_official` is the landmark AoP of the
release (mean 97.73, SD 14.76 degrees); `aop_mask_challenge_code` is the challenge geometry on the ground-truth mask
(|difference| to the official AoP: mean 3.45, median 2.98, p95 8.51 degrees), the error floor of any mask-ellipse
AoP method on this set.

## 9. IUGC 2025 release (second external AoP set)

`label.csv` holds landmarks for all 469 images and an AoP for 300. The reference for every image is the landmark AoP
(`fin.geometry.landmark_aop`), which reproduces the given AoP column to below 10^-7 degrees. `splits/ext469_sources.csv`
records the provenance of each image (IUGC 2024 test/validation/training frame, JNU-IFM recording, matching
woman and split), established by landmark matching against the IUGC 2024 annotations and nearest-neighbour image
search against JNU-IFM; it defines the subset of frames not from a woman of our training or validation split.
