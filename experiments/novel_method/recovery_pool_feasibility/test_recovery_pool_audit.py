import copy

import numpy as np

from run_recovery_pool_audit import (
    cluster_window_candidates,
    coverage,
    proposal,
    union_pool,
)


def test_low_gate_changes_only_peak_membership():
    record = {"score": np.array([0, 1, 4, 1, 0, 1, 2, 1, 0.]), "samples": []}
    _g0, native, _ = proposal(record, 1, 0.55)
    _g1, low, _ = proposal(record, 1, 0.25)
    assert {event["peak"] for event in native} <= {event["peak"] for event in low}
    assert all(event["offset"] - event["peak"] == 1 for event in low)


def test_window_clustering_support_first_and_deterministic():
    raw = [{"peak": 10, "window_id": 0, "height": 1.0},
           {"peak": 10, "window_id": 1, "height": 1.0},
           {"peak": 12, "window_id": 2, "height": 5.0}]
    indices = np.tile(np.arange(30), (3, 1))
    first = cluster_window_candidates(copy.deepcopy(raw), indices, 3)
    second = cluster_window_candidates(copy.deepcopy(raw), indices, 3)
    assert first == second and first[0]["peak"] == 10
    assert first[0]["support_window_count"] == 3
    assert 0.0 <= first[0]["support_ratio"] <= 1.0


def test_union_preserves_native_and_prefers_window_support_for_extras():
    native = [{"peak": 5, "onset": 3, "offset": 7, "sources": {"native"}}]
    low = [{"peak": 5, "onset": 3, "offset": 7, "sources": {"low_gate"}},
           {"peak": 15, "onset": 13, "offset": 17, "sources": {"low_gate"}}]
    window = [{"peak": 16, "onset": 14, "offset": 18, "sources": {"window"},
               "support_window_count": 2, "eligible_window_count": 2,
               "support_ratio": 1.0, "window_peak_height": 1.0}]
    merged = union_pool(native, low, window, 2)
    assert merged[0]["peak"] == 5
    assert merged[1]["peak"] == 16
    assert merged[1]["sources"] == {"low_gate", "window"}


def test_oracle_labels_do_not_enter_proposal():
    record = {"score": np.array([0, 1, 4, 1, 0.]), "samples": [[0, 2, 4]]}
    before = proposal(record, 1, 0.25)[1]
    changed = copy.deepcopy(record)
    changed["samples"] = [[100, 101, 102]] * 20
    after = proposal(changed, 1, 0.25)[1]
    assert before == after


def test_coverage_is_per_gt_oracle_not_greedy():
    records = [{"subject": "s", "video": "v", "samples": [[0, 1, 2], [0, 1, 2]]}]
    bank = {("s", "v"): [{"peak": 1, "onset": 0, "offset": 2}]}
    assert len(coverage(bank, records)) == 2


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_") and callable(globals()[name]))
    for name in tests:
        globals()[name]()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
