"""Label conventions shared by every task.

Processed masks (all datasets, all tasks): 0 background, 1 fetal head (FH), 2 pubic symphysis (PS).

Source conventions
    JNU-IFM raw mask        0 background, 7 PS, 8 FH
    FH-PS-AoP challenge     0 background, 1 PS, 2 FH
    IUGC 2024 test/val mask 0 background, 127 PS, 255 FH

Frame labels (JNU-IFM ``frame_label``): 3 None, 4 OnlySP, 5 OnlyHead, 6 SPHead.
    binary_label = 1 for Standard (SPHead: both PS and FH visible), 0 for Non-Standard.
    paper_class  = 1 - binary_label (the manuscript's class indices: 0 Standard, 1 Non-Standard).
"""

from __future__ import annotations

import numpy as np

BACKGROUND, FH, PS = 0, 1, 2
STRUCTURES = {FH: "FH", PS: "PS"}
CATEGORY_NAMES = {3: "None", 4: "OnlySP", 5: "OnlyHead", 6: "SPHead"}
STANDARD_CATEGORY = 6
# (has FH, has PS) implied by each frame category
CATEGORY_CONTENT = {3: (False, False), 4: (False, True), 5: (True, False), 6: (True, True)}

JNU_RAW = {"fh": 8, "ps": 7}
CHALLENGE = {"fh": 2, "ps": 1}
IUGC = {"fh": 255, "ps": 127}


def remap(mask: np.ndarray, fh_value: int, ps_value: int) -> np.ndarray:
    """Map a source label map with the given FH and PS values to 0 bg / 1 FH / 2 PS (uint8)."""
    m = np.asarray(mask)
    out = np.zeros(m.shape, np.uint8)
    out[m == fh_value] = FH
    out[m == ps_value] = PS
    return out


def remap_jnu(mask: np.ndarray) -> np.ndarray:
    """JNU-IFM raw mask (7 PS, 8 FH) to the processed map."""
    return remap(mask, JNU_RAW["fh"], JNU_RAW["ps"])


def remap_challenge(label: np.ndarray) -> np.ndarray:
    """FH-PS-AoP challenge label (1 PS, 2 FH) to the processed map."""
    return remap(label, CHALLENGE["fh"], CHALLENGE["ps"])


def remap_iugc(mask: np.ndarray) -> np.ndarray:
    """IUGC 2024 mask (127 PS, 255 FH) to the processed map."""
    return remap(mask, IUGC["fh"], IUGC["ps"])


def binary_label(category: int) -> int:
    """1 for Standard (SPHead), 0 for Non-Standard."""
    return int(category == STANDARD_CATEGORY)


def video_of(frame_id: str, dataset: str) -> str:
    """Bootstrap cluster of a frame: its video for JNU-IFM (``<video>_<frame>``), the frame itself for IUGC."""
    return str(frame_id).rsplit("_", 1)[0] if dataset == "jnu" else str(frame_id)
