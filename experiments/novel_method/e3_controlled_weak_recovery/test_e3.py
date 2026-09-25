import copy

import numpy as np

from run_e3 import (
    decode,
    g_margin,
    local_evidence,
    make_bank,
    select_threshold,
    signature,
)


def test_g_margin_endpoints_and_clip():
    curve = np.array([0.0, 1.0, 2.0])
    assert g_margin(curve, 0, 1.0, 2.0) == 0.0
    assert g_margin(curve, 1, 1.0, 2.0) == 0.0
    assert g_margin(curve, 2, 1.0, 2.0) == 1.0


def test_local_evidence_is_deterministic_and_normalized():
    score = np.array([0, 1, 4, 1, 0, 1, 3, 1, 0.], dtype=float)
    first = local_evidence(score, 2, 2)
    second = local_evidence(score, 2, 2)
    assert first == second
    assert 0 <= first[1] <= 1


def test_weak_membership_uses_temporal_dedup_without_gt():
    record = {"subject": "s", "video": "v",
              "score": np.array([0, 1, 5, 1, 0, 1, 2, 1, 0.], dtype=float),
              "samples": [[0, 2, 4]]}
    before = make_bank([record], 1)
    changed = copy.deepcopy(record)
    changed["samples"] = [[100, 101, 102]] * 10
    after = make_bank([changed], 1)
    assert signature(before[0]["strict"]) == signature(after[0]["strict"])
    assert signature(before[0]["weak"]) == signature(after[0]["weak"])


def test_decode_never_changes_native_geometry():
    record = {"strict": [{"peak": 5, "onset": 3, "offset": 7}],
              "weak": [{"peak": 10, "onset": 8, "offset": 12,
                        "G_norm": 1.0, "L_norm": 1.0,
                        "mean_score": 1.0, "consensus_score": 1.0}]}
    expected = signature(record["strict"])
    for method in ("V1_global", "V2_local", "V3_mean", "V4_consensus"):
        events, _ = decode(record, method, 0.5)
        assert signature([event for event in events if event["source"] == "native"]) == expected


def test_held_out_labels_do_not_select_threshold():
    base = {"subject": "train", "video": "a", "samples": [[0, 1, 2]],
            "strict": [], "weak": [{"peak": 1, "onset": 0, "offset": 2,
            "G_norm": 0.5, "L_norm": 0.5, "mean_score": 0.5, "consensus_score": 0.5}]}
    held = {"subject": "held", "video": "b", "samples": [], "strict": [], "weak": []}
    before = select_threshold([base, held], "V4_consensus", ["train"])
    held["samples"] = [[0, 1, 2]] * 100
    after = select_threshold([base, held], "V4_consensus", ["train"])
    assert before == after


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_") and callable(globals()[name]))
    for name in tests:
        globals()[name]()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
