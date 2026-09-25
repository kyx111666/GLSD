"""Independently recount the saved fair-validation predictions and controls."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from equiscale_fair_validation import EXPECTED, LABELS, WORKSPACE, digest, metrics


def rematch(predictions, gt):
    matched = set()
    output = []
    for prediction in predictions:
        start, end = prediction["onset"], prediction["offset"]
        overlaps = []
        for event in gt:
            intersection = max(0, min(end, event[2]) - max(start, event[0]) + 1)
            union = end - start + 1 + event[2] - event[0] + 1 - intersection
            overlaps.append(intersection / union)
        index = int(np.argmax(overlaps)) if overlaps else -1
        if index < 0 or overlaps[index] < 0.5 or index in matched:
            index = -1
        else:
            matched.add(index)
        assert index == prediction["matched_gt"], "Saved GT match differs from independent matcher"
        output.append(index)
    return np.array([len(matched), len(output)-len(matched), len(gt)-len(matched)])


def verify(backbone, dataset):
    target = WORKSPACE / backbone / "results" / "equiscale_fair_validation" / dataset
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((target / "PROTOCOL.json").read_text(encoding="utf-8"))
    assert manifest["input_sha256"] == digest(manifest["input_path"]), "Input changed during experiment"
    assert manifest["engine_sha256"] == digest(Path(__file__).with_name("equiscale_fair_validation.py"))
    assert manifest["helper_sha256"] == digest(Path(__file__).with_name("tune_equiscale.py"))
    assert manifest["signature"] == report["manifest_signature"]
    assert tuple(report["legacy_baseline_audit"]["metrics"][key] for key in ("TP", "FP", "FN")) == EXPECTED[backbone, dataset]
    predictions = json.loads((target / "selected_predictions.json").read_text(encoding="utf-8"))
    priors = json.loads((target / "fold_duration_priors.json").read_text(encoding="utf-8"))
    subjects = priors["subjects"]
    records = [{"subject": r["subject"], "gt": r["gt"]} for r in predictions]
    sid = {s: i for i, s in enumerate(subjects)}
    counts = {name: np.zeros((len(subjects), 3), dtype=np.int64) for name in LABELS}
    for record in predictions:
        for name, detections in record["predictions"].items():
            counts[name][sid[record["subject"]]] += rematch(detections, record["gt"])
    for name in LABELS:
        actual = metrics(counts[name].sum(axis=0))
        assert actual == report["metrics"][name], (name, actual, report["metrics"][name])
        assert sum(report["selection_frequency"][name].values()) == len(subjects)

    # Recompute all fold priors directly from exported GT and exclusion sets.
    def expected_k(excluded):
        d = sorted(e[2]-e[0] for r in records if r["subject"] not in excluded for e in r["gt"])
        value = sum(d)/len(d) if backbone == "metst" else d[len(d)//2]
        return max(1, int((value+1)/2))
    for i, subject in enumerate(subjects):
        assert priors["outer"][i] == expected_k({subject})
        for j, validation in enumerate(subjects):
            if i != j:
                assert priors["inner"][i][j] == expected_k({subject, validation})

    import csv
    with (target / "outer_loso_selections.csv").open(encoding="utf-8-sig", newline="") as handle:
        selections = list(csv.DictReader(handle))
    config_list = json.loads((target / "configurations.json").read_text(encoding="utf-8"))
    saved_stats = {}
    for cache_path in target.glob("stats_k*.npz"):
        with np.load(cache_path, allow_pickle=False) as package:
            assert str(package["signature"]) == manifest["signature"]
            saved_stats[int(cache_path.stem[len("stats_k"):])] = package["stats"]
    pools = {
        "single_tuned": [i for i, c in enumerate(config_list) if c["kind"] == "single"],
        "multiscale_union": [i for i, c in enumerate(config_list) if c["kind"] == "union"],
        "multiscale_height": [i for i, c in enumerate(config_list) if c["kind"] == "score" and c["weight"] == 1],
        "multiscale_height_support": [i for i, c in enumerate(config_list) if c["kind"] == "score"],
        "multiscale_fusion_only": [i for i, c in enumerate(config_list) if c["kind"] == "score" and c["weight"] < 1],
    }

    # Rebuild selection objectives without calling the experiment's selector.
    for held, subject in enumerate(subjects):
        inner_counts = np.zeros((len(config_list), 3), dtype=np.int64)
        for validation in range(len(subjects)):
            if validation != held:
                inner_counts += saved_stats[priors["inner"][held][validation]][:, validation]
        subject_rows = {row["family"]: row for row in selections if row["subject"] == subject}
        height_id = int(subject_rows["multiscale_height"]["config_id"])
        height_settings = {k: v for k, v in config_list[height_id].items() if k not in ("config_id", "weight")}
        matched_pool = [i for i, c in enumerate(config_list)
                        if {k: v for k, v in c.items() if k not in ("config_id", "weight")} == height_settings]
        for family, candidates in {**pools, "support_matched_height": matched_pool}.items():
            def rank(index):
                m = metrics(inner_counts[index])
                return (-m["F1"], -m["precision"], m["FP"], 1-config_list[index]["weight"], index)
            expected_id = min(candidates, key=rank)
            row = subject_rows[family]
            assert int(row["config_id"]) == expected_id, (subject, family, "Selection mismatch")
            assert abs(float(row["inner_F1"])-metrics(inner_counts[expected_id])["F1"]) < 1e-12
            assert int(row["k_train_only"]) == priors["outer"][held]
    matched_rows = {row["subject"]: row for row in selections if row["family"] == "support_matched_height"}
    for row in selections:
        if row["family"] == "multiscale_height":
            first = dict(config_list[int(row["config_id"])])
            second = dict(config_list[int(matched_rows[row["subject"]]["config_id"])])
            for config in (first, second):
                config.pop("config_id")
                config.pop("weight")
            assert first == second, "Matched ablation changed another parameter"
        m = metrics(counts[row["family"]][sid[row["subject"]]])
        for key in ("TP", "FP", "FN"):
            assert int(row[key]) == m[key]
    result = {"backbone": backbone, "dataset": dataset, "status": "PASS",
              "videos_recounted": len(predictions), "subjects": len(subjects),
              "checks": ["source_and_code_hashes", "legacy_native", "independent_GT_rematching",
                         "all_family_counts", "fold_duration_exclusion", "matched_support_changes_weight_only",
                         "selection_counts_and_frequencies", "independent_outer_excluded_config_selection"]}
    (target / "VERIFICATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS {backbone}/{dataset}: {len(predictions)} videos, {len(LABELS)} methods", flush=True)
    return result


if __name__ == "__main__":
    for backbone in ("metst", "boostingvrme"):
        for dataset in ("sammlv", "casme3"):
            verify(backbone, dataset)
