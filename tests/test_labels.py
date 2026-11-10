"""Label conventions: JNU-IFM mask remap (7 = PS -> 2, 8 = FH -> 1), challenge/IUGC remap, categories."""

import numpy as np

from fin.data.labels import (
    CATEGORY_CONTENT,
    CATEGORY_NAMES,
    binary_label,
    remap_challenge,
    remap_iugc,
    remap_jnu,
    video_of,
)


def test_jnu_mask_remap():
    raw = np.array([[0, 7, 8], [8, 7, 0]], np.uint8)
    out = remap_jnu(raw)
    np.testing.assert_array_equal(out, [[0, 2, 1], [1, 2, 0]])  # 7 = PS -> 2, 8 = FH -> 1
    assert out.dtype == np.uint8


def test_challenge_and_iugc_remap():
    np.testing.assert_array_equal(remap_challenge(np.array([0, 1, 2])), [0, 2, 1])  # 1 PS, 2 FH
    np.testing.assert_array_equal(remap_iugc(np.array([0, 127, 255])), [0, 2, 1])  # 127 PS, 255 FH


def test_categories_and_binary_label():
    assert CATEGORY_NAMES == {3: "None", 4: "OnlySP", 5: "OnlyHead", 6: "SPHead"}
    assert [binary_label(c) for c in (3, 4, 5, 6)] == [0, 0, 0, 1]
    assert CATEGORY_CONTENT[6] == (True, True) and CATEGORY_CONTENT[3] == (False, False)


def test_video_of():
    assert video_of("20190830T115515_169", "jnu") == "20190830T115515"
    assert video_of("abc_6.png", "iugc") == "abc_6.png"
