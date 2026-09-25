import copy

import numpy as np

from persistence_operators import (
    raw_scores,
    robust_z,
    scale_space_matches,
    score_candidates,
    topological_persistence,
)
from run_controlled_study import candidate_signature, select_threshold


def test_topological_single_peak_and_global_min_death():
    assert topological_persistence(np.array([0, 1, 3, 1, 0.]), np.array([2])).tolist() == [3.0]


def test_topological_two_peaks_different_height():
    scores = topological_persistence(np.array([0, 4, 1, 3, 0.]), np.array([1, 3]))
    assert scores.tolist() == [4.0, 2.0]


def test_topological_two_peaks_different_saddle():
    low = topological_persistence(np.array([0, 5, 1, 4, 0.]), np.array([1, 3]))[1]
    high = topological_persistence(np.array([0, 5, 3, 4, 0.]), np.array([1, 3]))[1]
    assert low == 3.0 and high == 1.0


def test_plateau_maps_to_deterministic_left_middle_birth():
    curve = np.array([0, 3, 3, 3, 1, 0.])
    assert topological_persistence(curve, np.array([1, 2, 3])).tolist() == [3.0, 3.0, 3.0]


def test_boundary_peak_and_constant_curve():
    assert topological_persistence(np.array([3, 2, 0.]), np.array([0])).tolist() == [3.0]
    assert topological_persistence(np.ones(5), np.array([0, 2, 4])).tolist() == [0.0] * 3


def test_no_peak_and_small_candidate_normalization_fallback():
    assert score_candidates(np.arange(5.), np.array([], dtype=int), 2, "P2_topological_0d").size == 0
    assert robust_z(np.array([1., 2.])).tolist() == [0.0, 0.0]
    assert robust_z(np.ones(4)).tolist() == [0.0] * 4


def test_scale_space_matching_is_deterministic_and_preserves_base_peak():
    curve = np.array([0, 1, 3, 1, 0, 1, 3, 1, 0.], dtype=float)
    peaks = np.array([2, 6])
    first = scale_space_matches(curve, peaks, 2)
    second = scale_space_matches(curve, peaks, 2)
    assert first == second
    assert peaks.tolist() == [2, 6]
    assert raw_scores(curve, peaks, 2, "P3_scale_space_survival").shape == (2,)


def test_same_candidate_signature_is_operator_independent():
    events = [{"peak": 2, "onset": 0, "offset": 4}, {"peak": 7, "onset": 5, "offset": 9}]
    signatures = {name: candidate_signature(events) for name in (
        "P0_no_persistence", "P1_peak_prominence", "P2_topological_0d",
        "P3_scale_space_survival")}
    assert len(set(signatures.values())) == 1


def test_held_out_labels_do_not_enter_threshold_selection():
    bank = [
        {"subject": "train", "samples": [[0, 1, 2]], "events": [
            {"peak": 1, "onset": 0, "offset": 2, "norm": 1.0}]},
        {"subject": "held", "samples": [[0, 1, 2]], "events": [
            {"peak": 10, "onset": 9, "offset": 11, "norm": 0.0}]},
    ]
    before = select_threshold(bank, ["train"], [0.0, 0.5, 1.0])
    changed = copy.deepcopy(bank)
    changed[1]["samples"] = [[9, 10, 11]] * 20
    after = select_threshold(changed, ["train"], [0.0, 0.5, 1.0])
    assert before == after


def test_rerun_deterministic():
    curve = np.array([0, 4, 1, 3, 0, 2, 0.], dtype=float)
    peaks = np.array([1, 3, 5])
    for operator in ("P1_peak_prominence", "P2_topological_0d", "P3_scale_space_survival"):
        assert np.array_equal(
            score_candidates(curve, peaks, 2, operator),
            score_candidates(curve, peaks, 2, operator),
        )


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_") and callable(globals()[name]))
    for name in tests:
        globals()[name]()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
