"""Regression gate and paper-ready benchmark for FINAL-GL-SKILL."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from .adapters import load_boostingvrme, load_metst
from .evaluation import evaluate, metrics, overlap_matrix
from .evidence import Config, configuration_grid, pure_config_indexes
from .selection import select_all
from .skill import GLSaliencySkill


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
HISTORICAL_ROOT = WORKSPACE / "historical_gl_exact_fresh_reproduction"
SIGNED_ROOT = HISTORICAL_ROOT / "source_archive" / "a"
REFERENCE_ROOT = HISTORICAL_ROOT / "fresh_run"
RESULT_ROOT = PROJECT / "results" / "final_gl_skill"
SOURCE_ROOT = SIGNED_ROOT / "boostingvrme"
SOURCE_SHA256 = {
    "pure_persistence_validation.py": "4ef11bab525c351eb93efc814f8e43ed05ef3fe77a0b8a22212b96e47b7b6b98",
    "unified_persistence.py": "56c531b3d63649c5173ffb7aa647fac6843a6f075db23b5f175f66fda1513723",
    "equiscale_fair_validation.py": "c04c98aa3ba887459e03482886c183a76002341d8ad2656d49d08d9f2b8c206c",
    "tune_equiscale.py": "bc3aa386570fa3de8d9d56c0fd2a57e9c12d7b594765c7095fbbb40ee205263d",
}
GROUPS = (
    ("metst", "sammlv"),
    ("metst", "casme3"),
    ("boostingvrme", "sammlv"),
    ("boostingvrme", "casme3"),
)
OFFSETS = (-0.1, -0.05, 0.0, 0.05, 0.1)
METHODS = ("Native", "G", "L", "GL")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_bundle(backbone: str, dataset: str):
    if backbone == "metst":
        path = PROJECT / "caches" / "me_tst" / f"{dataset}_strategy1_outputs.pkl"
        return load_metst(path, dataset)
    stem = "casme_3" if dataset == "casme3" else dataset
    path = SIGNED_ROOT / "boostingvrme" / "curve_cache" / f"{stem}_curves.pkl"
    return load_boostingvrme(path, dataset)


def duration_k(records, excluded: set[str], backbone: str) -> int:
    durations = sorted(
        event[2] - event[0]
        for record in records
        if record.subject not in excluded
        for event in record.ground_truth
    )
    if not durations:
        raise ValueError("No training durations")
    value = float(np.mean(durations)) if backbone == "metst" else durations[len(durations) // 2]
    return max(1, int((value + 1) / 2))


def fold_priors(bundle) -> tuple[np.ndarray, np.ndarray]:
    subjects = bundle.subjects
    outer = np.array(
        [duration_k(bundle.records, {subject}, bundle.backbone) for subject in subjects],
        dtype=int,
    )
    inner = np.zeros((len(subjects), len(subjects)), dtype=int)
    for first_index, first in enumerate(subjects):
        for second_index in range(first_index + 1, len(subjects)):
            second = subjects[second_index]
            inner[first_index, second_index] = inner[second_index, first_index] = duration_k(
                bundle.records, {first, second}, bundle.backbone
            )
    return outer, inner


def signed_context(bundle, configs: list[Config]):
    root = SIGNED_ROOT / bundle.backbone / "results" / "unified_persistence_final" / bundle.dataset
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    if protocol["protocol"]["configurations"] != [asdict(config) for config in configs]:
        raise AssertionError("SKILL_REFACTOR_REGRESSION_MISMATCH: configuration grid")
    if protocol["input_sha256"] != digest(Path(bundle.cache_path)):
        raise AssertionError("SKILL_REFACTOR_REGRESSION_MISMATCH: input cache SHA")
    for name, expected in SOURCE_SHA256.items():
        if digest(SOURCE_ROOT / name) != expected:
            raise AssertionError(f"SKILL_REFACTOR_REGRESSION_MISMATCH: source SHA {name}")
    outer, inner = fold_priors(bundle)
    saved = json.loads((root / "fold_duration_priors.json").read_text(encoding="utf-8"))
    if saved["subjects"] != bundle.subjects:
        raise AssertionError("SKILL_REFACTOR_REGRESSION_MISMATCH: subject order")
    np.testing.assert_array_equal(outer, saved["outer"])
    np.testing.assert_array_equal(inner, saved["inner"])
    mode = "native" if bundle.backbone == "boostingvrme" else "fixed"
    stats = {}
    for path in sorted(root.glob("stats_k*.npz")):
        with np.load(path, allow_pickle=False) as values:
            if str(values["signature"]) != protocol["signature"]:
                raise AssertionError("SKILL_REFACTOR_REGRESSION_MISMATCH: stats signature")
            stats[int(path.stem.removeprefix("stats_k"))] = values[mode]
    return root, protocol, outer, inner, stats, mode


def strip_event(event: dict) -> dict:
    return {key: event[key] for key in ("onset", "offset", "peak", "matched_gt")}


def regression_gate(bundle, configs, selected, selection_rows, outer, mode) -> dict:
    reference = (
        REFERENCE_ROOT
        / bundle.backbone
        / "results/pure_persistence_matched_v1"
        / bundle.dataset
        / mode
    )
    old_selection = [
        row for row in read_csv(reference / "outer_loso_selections.csv") if row["family"] == "pure"
    ]
    new_selection = [row for row in selection_rows if row["family"] == "GL"]
    selection_exact = len(old_selection) == len(new_selection)
    first_divergence = None
    for old, new in zip(old_selection, new_selection):
        exact = (
            old["subject"] == new["subject"]
            and int(old["config_id"]) == int(new["config_id"])
            and old["config"] == new["config"]
            and int(old["k"]) == int(new["k"])
            and abs(float(old["inner_F1"]) - float(new["inner_F1"])) <= 1e-15
        )
        if not exact:
            selection_exact = False
            first_divergence = {
                "stage": "selected_config_per_subject",
                "subject": new["subject"],
                "historical": old,
                "skill": new,
            }
            break
    historical_predictions = json.loads(
        (reference / "selected_predictions.json").read_text(encoding="utf-8")
    )
    historical_lookup = {
        (str(row["subject"]), str(row["video"])): row["predictions"]["pure"]
        for row in historical_predictions
    }
    subject_index = {subject: index for index, subject in enumerate(bundle.subjects)}
    totals = Counter(TP=0, FP=0, FN=0)
    prediction_exact = True
    for record in bundle.records:
        subject_id = subject_index[record.subject]
        config = configs[selected[record.subject]["GL"]]
        events = GLSaliencySkill(config).decode(
            record.score,
            int(outer[subject_id]),
            {"interval_adapter": bundle.interval_adapter},
        )
        counts, details = evaluate(events, record.ground_truth)
        totals.update(counts)
        current = [strip_event(event) for event in details]
        historical = historical_lookup[(record.subject, record.video)]
        if current != historical:
            prediction_exact = False
            first_divergence = first_divergence or {
                "stage": "per_video_prediction_replay",
                "subject": record.subject,
                "video": record.video,
                "historical": historical,
                "skill": current,
            }
            break
    report = json.loads((reference / "report.json").read_text(encoding="utf-8"))
    expected = report["metrics"]["pure"]
    observed = metrics(totals)
    counts_exact = all(observed[key] == expected[key] for key in ("TP", "FP", "FN"))
    f1_difference = abs(observed["F1"] - expected["F1"])
    passed = selection_exact and prediction_exact and counts_exact and f1_difference <= 1e-12
    return {
        "backbone": bundle.backbone,
        "dataset": bundle.dataset,
        "interval_protocol": mode,
        "selected_configs_exact": selection_exact,
        "selected_subjects": len(new_selection),
        "predictions_exact": prediction_exact,
        "TP_FP_FN_exact": counts_exact,
        "historical": expected,
        "skill": observed,
        "F1_absolute_difference": f1_difference,
        "first_divergence": first_divergence,
        "status": "PASS" if passed else "SKILL_REFACTOR_REGRESSION_MISMATCH",
    }


def aggregate_counts(rows: list[dict]) -> dict:
    return {
        key: sum(int(row[key]) for row in rows)
        for key in ("TP", "FP", "FN")
    }


def bootstrap(first: np.ndarray, second: np.ndarray, seed: int = 100, repeats: int = 10_000) -> dict:
    rng = np.random.default_rng(seed)
    count = len(first)
    draws = rng.integers(0, count, size=(repeats, count))

    def sampled_f1(values):
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0], denominator, out=np.zeros(repeats), where=denominator > 0
        )

    deltas = sampled_f1(first) - sampled_f1(second)
    point = metrics(dict(zip(("TP", "FP", "FN"), first.sum(axis=0))))["F1"] - metrics(
        dict(zip(("TP", "FP", "FN"), second.sum(axis=0)))
    )["F1"]
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "point_delta_F1": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "crosses_zero": bool(low <= 0 <= high),
        "resamples": repeats,
        "seed": seed,
        "label": "FINAL_SKILL_RECOMPUTED_BOOTSTRAP",
    }


def decode_method(bundle, record, subject_id, configs, selected, outer, method):
    if method == "Native":
        config = configs[0]
        temporal_scale = bundle.legacy_temporal_scale
    else:
        config = configs[selected[record.subject][method]]
        temporal_scale = int(outer[subject_id])
    skill = GLSaliencySkill(config)
    return skill.decode_with_audit(
        record.score,
        temporal_scale,
        {"interval_adapter": bundle.interval_adapter},
    )


def analyze_group(bundle, configs, selected, outer, candidate_writer) -> dict:
    subject_index = {subject: index for index, subject in enumerate(bundle.subjects)}
    per_subject = {
        method: np.zeros((len(bundle.subjects), 3), dtype=np.int64) for method in METHODS
    }
    mechanism = Counter(
        retained_native_GT=0, lost_native_GT=0, rescued_GT=0, removed_FP=0, new_FP=0
    )
    robustness = {
        offset: np.zeros((len(bundle.subjects), 3), dtype=np.int64) for offset in OFFSETS
    }
    for record in bundle.records:
        subject_id = subject_index[record.subject]
        method_details = {}
        gl_audits = None
        for method in METHODS:
            events, audits = decode_method(bundle, record, subject_id, configs, selected, outer, method)
            counts, details = evaluate(events, record.ground_truth)
            per_subject[method][subject_id] += np.array(
                [counts["TP"], counts["FP"], counts["FN"]], dtype=np.int64
            )
            method_details[method] = details
            if method == "GL":
                gl_audits = audits
        native_gt = {event["matched_gt"] for event in method_details["Native"] if event["matched_gt"] >= 0}
        gl_gt = {event["matched_gt"] for event in method_details["GL"] if event["matched_gt"] >= 0}
        mechanism["retained_native_GT"] += len(native_gt & gl_gt)
        mechanism["lost_native_GT"] += len(native_gt - gl_gt)
        mechanism["rescued_GT"] += len(gl_gt - native_gt)
        native_fp = {
            (event["onset"], event["peak"], event["offset"])
            for event in method_details["Native"]
            if event["matched_gt"] < 0
        }
        gl_fp = {
            (event["onset"], event["peak"], event["offset"])
            for event in method_details["GL"]
            if event["matched_gt"] < 0
        }
        mechanism["removed_FP"] += len(native_fp - gl_fp)
        mechanism["new_FP"] += len(gl_fp - native_fp)

        gl_match = {
            (event["onset"], event["peak"], event["offset"]): event["matched_gt"]
            for event in method_details["GL"]
        }
        for candidate in gl_audits or []:
            values = overlap_matrix([(candidate.onset, candidate.offset)], record.ground_truth)
            best_iou = float(values.max()) if values.size else 0.0
            geometry = (candidate.onset, candidate.peak, candidate.offset)
            candidate_writer.writerow(
                {
                    "backbone": bundle.backbone,
                    "dataset": bundle.dataset,
                    "subject": record.subject,
                    "video": record.video,
                    "onset": candidate.onset,
                    "peak": candidate.peak,
                    "offset": candidate.offset,
                    "G": candidate.global_score,
                    "L": candidate.local_score,
                    "fused_score": candidate.fused_score,
                    "prediction_status": (
                        "TP" if gl_match.get(geometry, -1) >= 0 else "FP"
                    )
                    if candidate.selected
                    else "rejected",
                    "best_GT_IoU": best_iou,
                }
            )

        base_config = configs[selected[record.subject]["GL"]]
        for offset in OFFSETS:
            changed = replace(
                base_config,
                threshold=max(0.0, round(base_config.threshold + offset, 10)),
            )
            events = GLSaliencySkill(changed).decode(
                record.score,
                int(outer[subject_id]),
                {"interval_adapter": bundle.interval_adapter},
            )
            counts, _ = evaluate(events, record.ground_truth)
            robustness[offset][subject_id] += np.array(
                [counts["TP"], counts["FP"], counts["FN"]], dtype=np.int64
            )

    method_metrics = {
        method: metrics(dict(zip(("TP", "FP", "FN"), counts.sum(axis=0))))
        for method, counts in per_subject.items()
    }
    return {
        "metrics": method_metrics,
        "per_subject": per_subject,
        "mechanism": dict(mechanism),
        "robustness": {
            offset: metrics(dict(zip(("TP", "FP", "FN"), values.sum(axis=0))))
            for offset, values in robustness.items()
        },
    }


def distribution(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
    }


def complexity(bundle, configs, selected, outer, repeats: int = 10) -> list[dict]:
    subject_index = {subject: index for index, subject in enumerate(bundle.subjects)}

    def execute(method: str) -> None:
        for record in bundle.records:
            decode_method(
                bundle,
                record,
                subject_index[record.subject],
                configs,
                selected,
                outer,
                method,
            )

    runtimes = {"Native": [], "GL": []}
    peaks = {"Native": [], "GL": []}
    for _ in range(repeats):
        for method in ("Native", "GL"):
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            execute(method)
            runtimes[method].append(time.perf_counter() - started)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[method].append(peak / (1024 * 1024))
    extra_runtime = [gl - native for gl, native in zip(runtimes["GL"], runtimes["Native"])]
    extra_ram = [gl - native for gl, native in zip(peaks["GL"], peaks["Native"])]
    rows = []
    for method, runtime_values, ram_values in (
        ("Native", runtimes["Native"], peaks["Native"]),
        ("GL Skill", runtimes["GL"], peaks["GL"]),
        ("Extra GL-Native", extra_runtime, extra_ram),
    ):
        runtime_stats = distribution(runtime_values)
        ram_stats = distribution(ram_values)
        rows.append(
            {
                "backbone": bundle.backbone,
                "dataset": bundle.dataset,
                "method": method,
                "repeats": repeats,
                "runtime_mean_seconds": runtime_stats["mean"],
                "runtime_median_seconds": runtime_stats["median"],
                "runtime_std_seconds": runtime_stats["std"],
                "peak_ram_mean_MiB": ram_stats["mean"],
                "peak_ram_median_MiB": ram_stats["median"],
                "peak_ram_std_MiB": ram_stats["std"],
            }
        )
    return rows


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = configuration_grid()
    pure_config_indexes(configs)
    contexts = []
    regression = []
    for backbone, dataset in GROUPS:
        bundle = load_bundle(backbone, dataset)
        signed_root, protocol, outer, inner, stats, mode = signed_context(bundle, configs)
        selected, selection_rows = select_all(configs, stats, bundle.subjects, outer, inner)
        gate = regression_gate(bundle, configs, selected, selection_rows, outer, mode)
        regression.append(gate)
        write_json(
            RESULT_ROOT / "prediction_regression.json",
            {"groups": regression, "status": gate["status"] if gate["status"] != "PASS" else "RUNNING"},
        )
        if gate["status"] != "PASS":
            raise SystemExit("SKILL_REFACTOR_REGRESSION_MISMATCH")
        contexts.append((bundle, protocol, outer, selected))
    write_json(
        RESULT_ROOT / "prediction_regression.json",
        {
            "groups": regression,
            "F1_tolerance": 1e-12,
            "status": "ALL_FOUR_EXACT_REPLAY",
        },
    )

    candidate_fields = [
        "backbone", "dataset", "subject", "video", "onset", "peak", "offset",
        "G", "L", "fused_score", "prediction_status", "best_GT_IoU",
    ]
    analyses = []
    candidate_path = RESULT_ROOT / "candidate_analysis.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        candidate_writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        candidate_writer.writeheader()
        for bundle, protocol, outer, selected in contexts:
            analyses.append((bundle, protocol, outer, selected, analyze_group(
                bundle, configs, selected, outer, candidate_writer
            )))

    main_rows = []
    ablation_rows = []
    bootstrap_rows = []
    robustness_rows = []
    mechanism_rows = []
    complexity_rows = []
    for bundle, protocol, outer, selected, analysis in analyses:
        native = analysis["metrics"]["Native"]
        gl = analysis["metrics"]["GL"]
        main_rows.append(
            {
                "Backbone": bundle.backbone,
                "Dataset": bundle.dataset,
                **{f"Native_{key}": native[key] for key in ("TP", "FP", "FN", "F1")},
                **{f"GL_Skill_{key}": gl[key] for key in ("TP", "FP", "FN", "F1")},
                "Delta_F1": gl["F1"] - native["F1"],
            }
        )
        for method in METHODS:
            ablation_rows.append(
                {"Backbone": bundle.backbone, "Dataset": bundle.dataset, "Method": method, **analysis["metrics"][method]}
            )
        for reference in ("Native", "G", "L"):
            bootstrap_rows.append(
                {
                    "Backbone": bundle.backbone,
                    "Dataset": bundle.dataset,
                    "Comparison": f"GL-{reference}",
                    **bootstrap(analysis["per_subject"]["GL"], analysis["per_subject"][reference]),
                }
            )
        for offset, result in analysis["robustness"].items():
            robustness_rows.append(
                {
                    "Backbone": bundle.backbone,
                    "Dataset": bundle.dataset,
                    "threshold_offset": offset,
                    **result,
                    "Native_F1": native["F1"],
                    "above_Native": result["F1"] > native["F1"],
                }
            )
        mechanism_rows.append(
            {"Backbone": bundle.backbone, "Dataset": bundle.dataset, **analysis["mechanism"]}
        )
        complexity_rows.extend(complexity(bundle, configs, selected, outer))

    write_csv(RESULT_ROOT / "main_results.csv", main_rows)
    write_csv(RESULT_ROOT / "ablation.csv", ablation_rows)
    write_csv(RESULT_ROOT / "bootstrap.csv", bootstrap_rows)
    write_csv(RESULT_ROOT / "threshold_robustness.csv", robustness_rows)
    write_csv(RESULT_ROOT / "mechanism_summary.csv", mechanism_rows)
    write_csv(RESULT_ROOT / "complexity.csv", complexity_rows)
    portability = {
        "same_skill_class": True,
        "same_evidence_implementation": True,
        "same_fusion_formula": True,
        "fusion_formula": "S=(G+L)/2",
        "same_selection_logic": True,
        "backbone_specific_adapter_only": True,
        "backbone_specific_G_or_L_branch": False,
        "trainable_parameters": 0,
        "test_leakage": False,
        "evidence_signature": str(inspect.signature(GLSaliencySkill.decode)),
        "historical_source_sha256": SOURCE_SHA256,
        "regression_gate": "ALL_FOUR_EXACT_REPLAY",
    }
    write_json(RESULT_ROOT / "skill_portability.json", portability)
    output_files = [
        "combined_report.json", "main_results.csv", "ablation.csv", "bootstrap.csv",
        "threshold_robustness.csv", "mechanism_summary.csv", "candidate_analysis.csv",
        "complexity.csv", "skill_portability.json", "prediction_regression.json", "run_summary.json",
    ]
    combined = {
        "status": "FINAL_GL_SKILL_COMPLETE",
        "historical_algorithm_semantics": "unchanged",
        "regression": regression,
        "main_results": main_rows,
        "ablation": ablation_rows,
        "bootstrap": bootstrap_rows,
        "threshold_robustness": robustness_rows,
        "mechanism": mechanism_rows,
        "complexity": complexity_rows,
        "portability": portability,
        "result_files": [str((RESULT_ROOT / name).resolve()) for name in output_files],
        "stop": "No method exploration performed after requested outputs",
    }
    write_json(RESULT_ROOT / "combined_report.json", combined)
    write_json(
        RESULT_ROOT / "run_summary.json",
        {
            "implementation_target": "python",
            "status": "success",
            "regression_gate": "4/4 exact",
            "bootstrap_seed": 100,
            "complexity_repeats": 10,
            "input_files": [context[0].cache_path for context in contexts],
            "output_files": [str((RESULT_ROOT / name).resolve()) for name in output_files],
            "dependencies": {"numpy": np.__version__},
        },
    )
    print(json.dumps({"status": "FINAL_GL_SKILL_COMPLETE", "results": str(RESULT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
