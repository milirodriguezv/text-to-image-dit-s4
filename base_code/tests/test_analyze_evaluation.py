import copy
import random

import numpy as np
import pytest

from analyze_evaluation import (GROUPS, clip_mean, fid_from_features, indices_for_ids,
                                load_array, save_array, select_groups)


def sample_data():
    records = [{"image_id": i} for i in range(1, 9)]
    annotations = {"images": [{"id": i} for i in range(1, 10)],
                   "categories": [{"id": 1, "name": "person"}],
                   "annotations": [{"image_id": i, "category_id": 1, "num_keypoints": 0}
                                   for i in (1, 2, 3, 4, 5, 9)]}
    return records, annotations


def test_groups_are_balanced_reproducible_and_use_only_reference_ids():
    records, annotations = sample_data()
    state = random.getstate()
    result = select_groups(records, annotations, 0)
    assert random.getstate() == state
    assert result["population_counts"] == {GROUPS[0]: 5, GROUPS[1]: 3}
    assert result["n_per_group"] == 3
    assert set(result["selected_ids"][GROUPS[1]]) == {6, 7, 8}
    assert set(result["selected_ids"][GROUPS[0]]) <= {1, 2, 3, 4, 5}
    assert not set(result["selected_ids"][GROUPS[0]]) & set(result["selected_ids"][GROUPS[1]])
    assert select_groups(records[::-1], annotations, 0) == result
    # Zero visible keypoints still counts as an annotation, per the requested rule.
    assert 1 in result["population_ids"][GROUPS[0]]


def test_selection_rejects_missing_duplicate_and_nonperson_inputs():
    records, annotations = sample_data()
    with pytest.raises(ValueError, match="Duplicate"):
        select_groups(records + records[:1], annotations, 0)
    with pytest.raises(ValueError, match="absent"):
        select_groups(records + [{"image_id": 100}], annotations, 0)
    other = copy.deepcopy(annotations)
    other["categories"][0]["name"] = "car"
    with pytest.raises(ValueError, match="person"):
        select_groups(records, other, 0)


def test_clip_clamps_after_averaging_and_preserves_selected_pair_indices():
    assert clip_mean(np.array([-10., 30.])) == 10.  # not 15 (per-row clamp)
    assert clip_mean(np.array([-30., 10.])) == 0.
    records = [{"image_id": i} for i in (50, 10, 30, 20)]
    indices = indices_for_ids(records, [10, 20])
    assert indices.tolist() == [1, 3]
    assert clip_mean(np.array([100., 10., 100., 30.]), indices) == 20.
    with pytest.raises(ValueError, match="Invalid"):
        indices_for_ids(records, [10, 10])
    with pytest.raises(ValueError, match="Invalid"):
        indices_for_ids(records, [99])
    with pytest.raises(ValueError, match="finite"):
        clip_mean([np.nan, 1.])


def test_fid_uses_matched_subsets_and_detects_a_shift():
    real = np.array([[0., 1.], [1., 0.], [2., 3.], [3., 2.], [4., 5.]])
    generated = real.copy()
    generated[3:] += 10
    assert abs(fid_from_features(real, generated, np.array([0, 1, 2]))) < 1e-6
    assert fid_from_features(real, generated) > 1
    assert fid_from_features(real, real + 2) == pytest.approx(8.)
    with pytest.raises(ValueError, match="aligned"):
        fid_from_features(real, generated[:-1])


def test_cache_checks_saved_shape_and_hash(tmp_path):
    path = tmp_path / "features.npy"
    save_array(path, np.ones((3, 2), dtype=np.float32))
    assert load_array(path, (3, 2)).shape == (3, 2)
    with pytest.raises(ValueError, match="Unexpected"):
        load_array(path, (2, 3))
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Corrupted"):
        load_array(path, (3, 2))
