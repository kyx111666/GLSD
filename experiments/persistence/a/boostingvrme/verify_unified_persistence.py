"""Independently recount saved predictions and audit held-subject selection."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import equiscale_fair_validation as fair
import scheme11_persistence_selector as reference
import unified_persistence as unified


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_experiment(backbone, dataset):
    root = fair.WORKSPACE / backbone / "results" / unified.RESULT_NAME / dataset
    records, subjects, _, input_path = fair.load_data(backbone, dataset)
    manifest = read_json(root / "PROTOCOL.json")
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    signature = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    assert signature == manifest["signature"]
    assert fair.digest(input_path) == manifest["input_sha256"]
    for filename, digest in manifest["source_sha256"].items():
        assert fair.digest(Path(__file__).with_name(filename)) == digest
    configs = [unified.Config(**row) for row in manifest["protocol"]["configurations"]]
    assert configs == unified.configuration_grid()
    outer, inner = fair.fold_priors(records, subjects, backbone)
    priors = read_json(root / "fold_duration_priors.json")
    assert priors == {"subjects": subjects, "outer": outer.tolist(), "inner": inner.tolist()}
    modes = ("native", "fixed") if backbone == "boostingvrme" else ("fixed",)
    stats = {mode: {} for mode in modes}
    for k in sorted(set(outer.tolist()) | set(inner[inner > 0].tolist())):
        with np.load(root / f"stats_k{k}.npz", allow_pickle=False) as cache:
            assert str(cache["signature"]) == signature
            np.testing.assert_array_equal(cache["needed"], fair.required_subjects(k, outer, inner))
            for mode in modes:
                stats[mode][k] = cache[mode]
    summaries = []
    pools = unified.selection_pools(configs)
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    for mode in modes:
        target = root / mode
        report = read_json(target / "report.json")
        videos = read_json(target / "selected_predictions.json")
        with (target / "outer_loso_selections.csv").open(encoding="utf-8-sig", newline="") as handle:
            selections = {(row["family"], row["subject"]): row for row in csv.DictReader(handle)}
        assert len(selections) == len(subjects) * len(pools)
        totals = {family: np.zeros((len(subjects), 3), dtype=int) for family in pools}
        for held, subject in enumerate(subjects):
            train = np.zeros((len(configs), 3), dtype=int)
            for validation in range(len(subjects)):
                if validation != held:
                    train += stats[mode][int(inner[held, validation])][:, validation]
            for family, indexes in pools.items():
                def key(index):
                    counts = reference.Counts(*map(int, train[index]))
                    return counts.f1(), counts.precision(), -counts.fp, -index
                expected = max(indexes, key=key)
                selection = selections[family, subject]
                assert int(selection["config_id"]) == expected
                assert int(selection["k_train_only"]) == int(outer[held])
                assert selection["config"] == configs[expected].identifier
        assert len(videos) == len(records)
        for record, video in zip(records, videos):
            assert (record["subject"], record["video"], record["gt"]) == (video["subject"], video["video"], video["gt"])
            sid = subject_ids[record["subject"]]
            for family, output in video["predictions"].items():
                detections = [(p["onset"], p["offset"], p["peak"]) for p in output]
                counts = reference.count_video(detections, record)
                totals[family][sid] += (counts.tp, counts.fp, counts.fn)
                ids = [p["matched_gt"] for p in output if p["matched_gt"] >= 0]
                assert len(ids) == len(set(ids)) == counts.tp
                if family == "unified":
                    config_id = int(selections[family, record["subject"]]["config_id"])
                    predicted = unified.detect_curve(record["curve"], int(outer[sid]), configs[config_id], backbone, mode)
                    assert predicted == [{k: p[k] for k in ("onset", "offset", "peak")} for p in output]
        for family in pools:
            assert fair.metrics(totals[family].sum(axis=0)) == report["metrics"][family]
            for sid, subject in enumerate(subjects):
                selected = int(selections[family, subject]["config_id"])
                np.testing.assert_array_equal(totals[family][sid], stats[mode][int(outer[sid])][selected, sid])
        check = {"input_and_code_signatures": True, "fold_priors_recomputed": True,
                 "selection_independently_recomputed": True, "all_saved_predictions_recounted": True,
                 "unified_label_free_inference_replayed": True}
        fair.json_write(target / "independent_verification.json", check)
        summaries.append({"backbone": backbone, "dataset": dataset, "intervals": mode, "checks": check})
        fair.log(f"VERIFIED {backbone}/{dataset}/{mode}")
    return summaries


def main():
    checks = []
    for backbone in ("metst", "boostingvrme"):
        for dataset in ("sammlv", "casme3"):
            checks.extend(audit_experiment(backbone, dataset))
    target = fair.WORKSPACE / "boostingvrme" / "results" / unified.RESULT_NAME
    fair.json_write(target / "independent_verification.json", checks)
    write_summary()


def write_summary():
    reports = []
    lines = ["# Unified Persistence: Four-Setting Results", "",
             "The primary comparison uses each backbone's native interval decoder. "
             "All four settings use the same formula and search grid; configurations are "
             "selected separately from each outer fold's training subjects.", "",
             "| Backbone | Dataset | Original native F1 | Train-k native F1 | Unified F1 | TP | FP | FN |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for backbone in ("metst", "boostingvrme"):
        for dataset in ("sammlv", "casme3"):
            mode = "fixed" if backbone == "metst" else "native"
            path = fair.WORKSPACE / backbone / "results" / unified.RESULT_NAME / dataset / mode / "report.json"
            report = read_json(path)
            reports.append(report)
            original, baseline, result = report["legacy_baseline"], report["metrics"]["native"], report["metrics"]["unified"]
            lines.append(f"| {backbone} | {dataset} | {original['F1']:.6f} | {baseline['F1']:.6f} | "
                         f"{result['F1']:.6f} | {result['TP']} | {result['FP']} | {result['FN']} |")
    lines += ["", "ME-TST/CASME3 has a different training-only duration prior in three outer folds. "
              "The original native result is also shown so the baseline change is explicit.", "",
              "## Paired Uncertainty", "",
              "| Backbone | Dataset | Delta F1 vs train-k native | Subject-paired 95% CI | Net TP | Net FP |",
              "| --- | --- | ---: | --- | ---: | ---: |"]
    for report in reports:
        c = report["comparisons"]["unified_vs_native"]
        lines.append(f"| {report['backbone']} | {report['dataset']} | {c['delta_F1']:+.6f} | "
                     f"[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {c['net_TP']:+d} | {c['net_FP']:+d} |")
    lines += ["", "All four point estimates improve, but every interval crosses zero. "
              "These are exploratory improvements, not evidence of established significance or independent generalization.", "",
              "## Unified Rule", "",
              "`S = w*H + (1-w)*(G+L)/2`; accept when `S >= tau`.", "",
              "- H is continuous reference-curve height `(peak-mean)/(max-mean)`, clipped below at zero.",
              "- G is global prominence divided by the reference smoothed curve's range.",
              "- L is the median aligned local prominence across smoothing scales 1, 1.5, and 2 times k. "
              "Each scale uses its own range; missing matches contribute zero.",
              "- Local radii are selected from k, 2k, and 3k. Reference smoothing is selected from k, 1.5k, and 2k.",
              "- Height weights are 0, 0.25, 0.5, and 0.75. Thresholds are 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, and 0.75.",
              "- There is no P-based candidate gate, high/low routing, strict/recovery priority, or global-peak snapping.",
              "- Peak spacing is k. Rounded duplicate smoothing scales count once. Constant input produces no peaks.",
              "- The unified pool has 360 configurations. Including native and ablations, 511 configurations are recorded.", "",
              "## Complete Ablations", "",
              "The primary family was unified for every setting; no per-dataset best-family substitution is used.", "",
              "| Backbone | Dataset | Native | Unified | Persistence only | Height only | Global only | Local only |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        values = [f"{report['metrics'][name]['F1']:.6f}" for name in unified.selection_pools(unified.configuration_grid())]
        lines.append(f"| {report['backbone']} | {report['dataset']} | " + " | ".join(values) + " |")
    lines += ["", "## Verification and Scope", "",
              "Duration priors exclude the outer subject, and additionally exclude the inner validation subject "
              "during inner selection. Configurations maximize aggregate inner raw F1; ties use precision, fewer FP, then grid order.", "",
              "The verification script independently recomputes selected configurations, recounts every saved prediction "
              "using the old scalar IoU matcher, checks input/code hashes, and replays the label-free unified detector.", "",
              "Frozen OOF backbone curves are reused. Inner-specific backbone predictions are unavailable, so this is not "
              "fully nested network retraining. Both datasets informed method design. Bootstrap uses 10,000 subject-paired "
              "draws and conditions on the selected configurations and frozen curves. Only raw spotting is evaluated.", "",
              "Cached labels are preserved, including pre-existing malformed/out-of-bounds CASME3 intervals. "
              "BoostingVRME uses the existing reference-kernel curve cache; this work does not validate binary-kernel equivalence.", "",
              "## Run", "", "```powershell",
              "& 'D:\\Anaconda3\\envs\\ME-TST\\python.exe' -B 'D:\\workspace\\a\\boostingvrme\\unified_persistence.py' --backbone both --dataset both",
              "& 'D:\\Anaconda3\\envs\\ME-TST\\python.exe' -B -m unittest discover -s 'D:\\workspace\\a\\boostingvrme' -p 'test_unified_persistence.py' -v",
              "& 'D:\\Anaconda3\\envs\\ME-TST\\python.exe' -B 'D:\\workspace\\a\\boostingvrme\\verify_unified_persistence.py'",
              "```", "", "Each project writes `results/unified_persistence_final/<dataset>/<intervals>/`. "
              "`report.json`, per-subject selections/counts, selected predictions, and independent verification are included. "
              "BoostingVRME additionally has a fixed-interval sensitivity analysis.", "",
              "The original scripts and results are preserved. `unified_persistence_v1` records the first, one-radius attempt "
              "that failed on ME-TST/SAMMLV; `unified_persistence_v2` records the wider-radius experiment. "
              "The final run adds a constant-input guard and reproduces v2's results.", ""]
    (fair.WORKSPACE / "UNIFIED_PERSISTENCE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    fair.json_write(fair.WORKSPACE / "boostingvrme" / "results" / unified.RESULT_NAME / "summary.json", reports)


if __name__ == "__main__":
    main()
