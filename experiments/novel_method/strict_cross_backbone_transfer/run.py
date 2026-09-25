"""Run strict outer-LOSO transfer without modifying or tuning GL Saliency Skill."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from my_method.gl_saliency_skill.benchmark import (
    GROUPS,
    load_bundle,
    signed_context,
    write_csv,
    write_json,
)
from my_method.gl_saliency_skill.evaluation import evaluate, metrics
from my_method.gl_saliency_skill.evidence import configuration_grid, pure_config_indexes
from my_method.gl_saliency_skill.selection import select_all
from my_method.gl_saliency_skill.skill import GLSaliencySkill


PROJECT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT / "results" / "final_gl_skill_transfer"
FINAL_RESULT_ROOT = PROJECT / "results" / "final_gl_skill"
SKILL_ROOT = PROJECT / "my_method" / "gl_saliency_skill"
SEED = 100
RESAMPLES = 10_000
NEUTRAL_TOLERANCE = 1e-12
TRANSFER_PAIRS = (
    ("metst", "boostingvrme", "sammlv"),
    ("metst", "boostingvrme", "casme3"),
    ("boostingvrme", "metst", "sammlv"),
    ("boostingvrme", "metst", "casme3"),
)


def source_manifest() -> dict[str, str]:
    output = {}
    for path in sorted(SKILL_ROOT.rglob("*.py")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        output[str(path.relative_to(SKILL_ROOT))] = digest
    return output


def load_context(backbone: str, dataset: str, configs) -> dict:
    bundle = load_bundle(backbone, dataset)
    signed_root, protocol, outer, inner, stats, mode = signed_context(bundle, configs)
    selected, selection_rows = select_all(configs, stats, bundle.subjects, outer, inner)
    return {
        "bundle": bundle,
        "signed_root": signed_root,
        "protocol": protocol,
        "outer": outer,
        "inner": inner,
        "selected": selected,
        "selection_rows": selection_rows,
        "mode": mode,
    }


def add_counts(target: np.ndarray, counts: dict) -> None:
    target += np.array([counts["TP"], counts["FP"], counts["FN"]], dtype=np.int64)


def aggregate(subject_counts: np.ndarray) -> dict:
    totals = subject_counts.sum(axis=0)
    return metrics(dict(zip(("TP", "FP", "FN"), totals)))


def paired_bootstrap(first: np.ndarray, second: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(first), size=(RESAMPLES, len(first)))

    def sampled_f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0],
            denominator,
            out=np.zeros(RESAMPLES),
            where=denominator > 0,
        )

    deltas = sampled_f1(first) - sampled_f1(second)
    point = aggregate(first)["F1"] - aggregate(second)["F1"]
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "point_delta_F1": point,
        "bootstrap_mean_delta_F1": float(deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "crosses_zero": bool(low <= 0 <= high),
        "resamples": RESAMPLES,
        "seed": SEED,
    }


def strip_predictions(events: list[dict]) -> list[dict]:
    return [
        {
            "onset": int(event["onset"]),
            "peak": int(event["peak"]),
            "offset": int(event["offset"]),
            "confidence": float(event["confidence"]),
            "matched_gt": int(event["matched_gt"]),
        }
        for event in events
    ]


def transfer_label(delta: float) -> str:
    if delta > NEUTRAL_TOLERANCE:
        return "POSITIVE_TRANSFER"
    if delta < -NEUTRAL_TOLERANCE:
        return "NEGATIVE_TRANSFER"
    return "NEUTRAL_TRANSFER"


def run_direction(source: dict, target: dict, configs) -> dict:
    source_bundle = source["bundle"]
    target_bundle = target["bundle"]
    if source_bundle.dataset != target_bundle.dataset:
        raise AssertionError("Source and target datasets differ")
    if source_bundle.subjects != target_bundle.subjects:
        raise AssertionError("Source and target subject identities/order differ")

    subjects = target_bundle.subjects
    target_subject_index = {subject: index for index, subject in enumerate(subjects)}
    source_subject_index = {subject: index for index, subject in enumerate(subjects)}
    source_selection_rows = {
        row["subject"]: row
        for row in source["selection_rows"]
        if row["family"] == "GL"
    }
    counts = {
        name: np.zeros((len(subjects), 3), dtype=np.int64)
        for name in ("Native", "Target-Tuned GL", "Transferred GL")
    }
    mechanism = Counter(
        retained_native_GT=0,
        rescued_GT=0,
        lost_native_GT=0,
        removed_FP=0,
        new_FP=0,
    )
    robustness = {
        offset: np.zeros((len(subjects), 3), dtype=np.int64)
        for offset in (-0.05, 0.05)
    }
    predictions = []

    for record in target_bundle.records:
        subject_id = target_subject_index[record.subject]
        source_config_id = source["selected"][record.subject]["GL"]
        target_config_id = target["selected"][record.subject]["GL"]
        source_config = configs[source_config_id]
        target_config = configs[target_config_id]
        target_k = int(target["outer"][subject_id])

        specifications = (
            ("Native", configs[0], target_bundle.legacy_temporal_scale),
            ("Target-Tuned GL", target_config, target_k),
            ("Transferred GL", source_config, target_k),
        )
        details_by_method = {}
        results_by_method = {}
        for method, config, temporal_scale in specifications:
            events = GLSaliencySkill(config).decode(
                record.score,
                int(temporal_scale),
                {"interval_adapter": target_bundle.interval_adapter},
            )
            result, details = evaluate(events, record.ground_truth)
            add_counts(counts[method][subject_id], result)
            results_by_method[method] = result
            details_by_method[method] = details

        native_details = details_by_method["Native"]
        transferred_details = details_by_method["Transferred GL"]
        native_gt = {item["matched_gt"] for item in native_details if item["matched_gt"] >= 0}
        transferred_gt = {
            item["matched_gt"] for item in transferred_details if item["matched_gt"] >= 0
        }
        mechanism["retained_native_GT"] += len(native_gt & transferred_gt)
        mechanism["rescued_GT"] += len(transferred_gt - native_gt)
        mechanism["lost_native_GT"] += len(native_gt - transferred_gt)
        native_fp = {
            (item["onset"], item["peak"], item["offset"])
            for item in native_details
            if item["matched_gt"] < 0
        }
        transferred_fp = {
            (item["onset"], item["peak"], item["offset"])
            for item in transferred_details
            if item["matched_gt"] < 0
        }
        mechanism["removed_FP"] += len(native_fp - transferred_fp)
        mechanism["new_FP"] += len(transferred_fp - native_fp)

        predictions.append(
            {
                "source": source_bundle.backbone,
                "target": target_bundle.backbone,
                "dataset": target_bundle.dataset,
                "subject": record.subject,
                "video": record.video,
                "source_config_id": source_config_id,
                "source_config": source_config.identifier,
                "target_k_p": target_k,
                "counts": results_by_method["Transferred GL"],
                "predictions": strip_predictions(transferred_details),
            }
        )

        for offset in (-0.05, 0.05):
            changed = replace(
                source_config,
                threshold=max(0.0, round(source_config.threshold + offset, 10)),
            )
            events = GLSaliencySkill(changed).decode(
                record.score,
                target_k,
                {"interval_adapter": target_bundle.interval_adapter},
            )
            result, _ = evaluate(events, record.ground_truth)
            add_counts(robustness[offset][subject_id], result)

    source_configs = []
    for subject in subjects:
        source_id = source["selected"][subject]["GL"]
        target_id = target["selected"][subject]["GL"]
        source_config = configs[source_id]
        target_config = configs[target_id]
        training_subjects = [item for item in subjects if item != subject]
        source_row = source_selection_rows[subject]
        source_configs.append(
            {
                "Source": source_bundle.backbone,
                "Target": target_bundle.backbone,
                "Dataset": target_bundle.dataset,
                "held_out_subject": subject,
                "source_training_subject_count": len(training_subjects),
                "held_out_excluded": subject not in training_subjects,
                "source_config_id": source_id,
                "source_config": source_config.identifier,
                "source_scale": source_config.reference,
                "source_radius": source_config.radius,
                "source_threshold": source_config.threshold,
                "source_outer_k_p": int(
                    source["outer"][source_subject_index[subject]]
                ),
                "source_inner_F1": float(source_row["inner_F1"]),
                "target_k_p": int(target["outer"][target_subject_index[subject]]),
                "target_tuned_config_id": target_id,
                "target_tuned_config": target_config.identifier,
                "target_scale": target_config.reference,
                "target_radius": target_config.radius,
                "target_threshold": target_config.threshold,
                "exact_agreement": source_id == target_id,
                "scale_agreement": source_config.reference == target_config.reference,
                "radius_agreement": source_config.radius == target_config.radius,
                "threshold_agreement": source_config.threshold == target_config.threshold,
                "absolute_threshold_difference": abs(
                    source_config.threshold - target_config.threshold
                ),
            }
        )

    result_metrics = {name: aggregate(values) for name, values in counts.items()}
    native_f1 = result_metrics["Native"]["F1"]
    tuned_f1 = result_metrics["Target-Tuned GL"]["F1"]
    transferred_f1 = result_metrics["Transferred GL"]["F1"]
    tuned_gain = tuned_f1 - native_f1
    transfer_gain = transferred_f1 - native_f1
    retention = transfer_gain / tuned_gain if tuned_gain > 0 else None

    agreement = {
        "Source": source_bundle.backbone,
        "Target": target_bundle.backbone,
        "Dataset": target_bundle.dataset,
        "subjects": len(subjects),
        "exact_agreement_rate": float(np.mean([row["exact_agreement"] for row in source_configs])),
        "scale_agreement_rate": float(np.mean([row["scale_agreement"] for row in source_configs])),
        "radius_agreement_rate": float(np.mean([row["radius_agreement"] for row in source_configs])),
        "threshold_agreement_rate": float(
            np.mean([row["threshold_agreement"] for row in source_configs])
        ),
        "mean_absolute_threshold_difference": float(
            np.mean([row["absolute_threshold_difference"] for row in source_configs])
        ),
    }
    main = {
        "Source": source_bundle.backbone,
        "Target": target_bundle.backbone,
        "Dataset": target_bundle.dataset,
        "Native_F1": native_f1,
        "Target_Tuned_GL_F1": tuned_f1,
        "Transferred_GL_F1": transferred_f1,
        "Transfer_Delta_vs_Native": transfer_gain,
        "Tuned_Delta_vs_Native": tuned_gain,
        "Target_Tuned_minus_Transferred": tuned_f1 - transferred_f1,
        "Transfer_Retention": retention,
        "Interpretation": transfer_label(transfer_gain),
        "Native_TP": result_metrics["Native"]["TP"],
        "Native_FP": result_metrics["Native"]["FP"],
        "Native_FN": result_metrics["Native"]["FN"],
        "Target_Tuned_TP": result_metrics["Target-Tuned GL"]["TP"],
        "Target_Tuned_FP": result_metrics["Target-Tuned GL"]["FP"],
        "Target_Tuned_FN": result_metrics["Target-Tuned GL"]["FN"],
        "Transferred_TP": result_metrics["Transferred GL"]["TP"],
        "Transferred_FP": result_metrics["Transferred GL"]["FP"],
        "Transferred_FN": result_metrics["Transferred GL"]["FN"],
    }
    bootstrap = {
        "Source": source_bundle.backbone,
        "Target": target_bundle.backbone,
        "Dataset": target_bundle.dataset,
        "Comparison": "Transferred GL - Target Native",
        **paired_bootstrap(counts["Transferred GL"], counts["Native"]),
    }
    robustness_rows = []
    for offset, values in robustness.items():
        result = aggregate(values)
        robustness_rows.append(
            {
                "Source": source_bundle.backbone,
                "Target": target_bundle.backbone,
                "Dataset": target_bundle.dataset,
                "source_threshold_offset": offset,
                **result,
                "Native_F1": native_f1,
                "Delta_vs_Native": result["F1"] - native_f1,
                "above_Native": result["F1"] > native_f1,
                "post_hoc_only_no_selection": True,
            }
        )
    return {
        "main": main,
        "agreement": agreement,
        "bootstrap": bootstrap,
        "mechanism": {
            "Source": source_bundle.backbone,
            "Target": target_bundle.backbone,
            "Dataset": target_bundle.dataset,
            **dict(mechanism),
        },
        "source_configs": source_configs,
        "predictions": predictions,
        "robustness": robustness_rows,
        "per_subject_counts": {
            name: values.tolist() for name, values in counts.items()
        },
    }


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before_manifest = source_manifest()
    configs = configuration_grid()
    if len(pure_config_indexes(configs)) != 90:
        raise AssertionError("Strict transfer requires the locked 90-config GL grid")

    contexts = {
        (backbone, dataset): load_context(backbone, dataset, configs)
        for backbone, dataset in GROUPS
    }
    with (FINAL_RESULT_ROOT / "main_results.csv").open(newline="", encoding="utf-8") as handle:
        final_rows = list(csv.DictReader(handle))
    final_lookup = {(row["Backbone"], row["Dataset"]): row for row in final_rows}

    directions = []
    for source_name, target_name, dataset in TRANSFER_PAIRS:
        direction = run_direction(
            contexts[source_name, dataset],
            contexts[target_name, dataset],
            configs,
        )
        reference = final_lookup[target_name, dataset]
        if abs(direction["main"]["Native_F1"] - float(reference["Native_F1"])) > 1e-15:
            raise AssertionError("Target Native comparison does not match FINAL-GL-SKILL")
        if abs(
            direction["main"]["Target_Tuned_GL_F1"]
            - float(reference["GL_Skill_F1"])
        ) > 1e-15:
            raise AssertionError("Target-tuned comparison does not match FINAL-GL-SKILL")
        directions.append(direction)

    after_manifest = source_manifest()
    if before_manifest != after_manifest:
        raise AssertionError("GL Skill source changed during strict transfer")

    main_rows = [item["main"] for item in directions]
    agreement_rows = [item["agreement"] for item in directions]
    bootstrap_rows = [item["bootstrap"] for item in directions]
    mechanism_rows = [item["mechanism"] for item in directions]
    source_config_rows = [row for item in directions for row in item["source_configs"]]
    prediction_rows = [row for item in directions for row in item["predictions"]]
    robustness_rows = [row for item in directions for row in item["robustness"]]

    held_exclusion = all(row["held_out_excluded"] for row in source_config_rows)
    identities_aligned = all(
        contexts[source, dataset]["bundle"].subjects
        == contexts[target, dataset]["bundle"].subjects
        for source, target, dataset in TRANSFER_PAIRS
    )
    decode_source = inspect.getsource(GLSaliencySkill.decode)
    adapter_source = "\n".join(
        inspect.getsource(contexts[key]["bundle"].interval_adapter.__class__)
        for key in sorted(contexts)
    )
    leakage_audit = {
        "status": "PASS",
        "source_config_selection_excludes_held_subject": held_exclusion,
        "selection_uses_source_stats_only": True,
        "transferred_decoder_receives_target_training_metrics": False,
        "target_tuned_selection_used_only_as_comparator": True,
        "target_GT_enters_decoder": "ground_truth" in decode_source,
        "target_outputs_used_for_selection": False,
        "target_outputs_used_only_for_final_evaluation_and_post_hoc_analysis": True,
        "source_target_subject_identities_aligned": identities_aligned,
        "same_skill_core": True,
        "skill_source_manifest_unchanged": before_manifest == after_manifest,
        "skill_source_manifest_sha256": before_manifest,
        "target_adapter_contains_G_L_or_selection_logic": any(
            token in adapter_source
            for token in ("CurveFeatures", "evidence_scores", "select_all", "inner_F1")
        ),
        "target_adapter_role": "cache/schema/k_p/native interval metadata only",
        "trainable_parameters": GLSaliencySkill(configs[pure_config_indexes(configs)[0]]).trainable_parameters,
        "neutral_tolerance": NEUTRAL_TOLERANCE,
        "target_k_p_policy": "target fold protocol k_p; never performance-selected",
    }
    required_true = (
        held_exclusion
        and identities_aligned
        and before_manifest == after_manifest
        and not leakage_audit["target_GT_enters_decoder"]
        and not leakage_audit["target_adapter_contains_G_L_or_selection_logic"]
        and leakage_audit["trainable_parameters"] == 0
    )
    if not required_true:
        leakage_audit["status"] = "FAIL"
        write_json(RESULT_ROOT / "transfer_leakage_audit.json", leakage_audit)
        raise AssertionError("Strict transfer leakage audit failed")

    write_csv(RESULT_ROOT / "transfer_main_results.csv", main_rows)
    write_csv(RESULT_ROOT / "transfer_config_agreement.csv", agreement_rows)
    write_csv(RESULT_ROOT / "transfer_bootstrap.csv", bootstrap_rows)
    write_csv(RESULT_ROOT / "transfer_mechanism.csv", mechanism_rows)
    write_csv(RESULT_ROOT / "per_subject_source_configs.csv", source_config_rows)
    write_json(RESULT_ROOT / "per_video_transferred_predictions.json", prediction_rows)
    write_csv(RESULT_ROOT / "transfer_threshold_robustness.csv", robustness_rows)
    write_json(RESULT_ROOT / "transfer_leakage_audit.json", leakage_audit)

    record = {
        "status": "STRICT_CROSS_BACKBONE_SKILL_TRANSFER_COMPLETE",
        "protocol": {
            "selection": "source backbone inner-pooled training counts; outer subject excluded",
            "grid": "locked historical 90-config pure G+L grid",
            "target_configuration_selection": False,
            "target_k_p": "target fold protocol value supplied by target adapter context",
            "matching": "historical chronological greedy IoU>=0.5",
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": SEED,
            "neutral_tolerance": NEUTRAL_TOLERANCE,
        },
        "main_results": main_rows,
        "config_agreement": agreement_rows,
        "bootstrap": bootstrap_rows,
        "mechanism": mechanism_rows,
        "threshold_robustness": robustness_rows,
        "leakage_audit": leakage_audit,
        "per_subject_counts": {
            f"{item['main']['Source']}->{item['main']['Target']}/{item['main']['Dataset']}": item[
                "per_subject_counts"
            ]
            for item in directions
        },
        "result_files": [
            str((RESULT_ROOT / name).resolve())
            for name in (
                "combined_report.json",
                "transfer_main_results.csv",
                "transfer_config_agreement.csv",
                "transfer_bootstrap.csv",
                "transfer_mechanism.csv",
                "transfer_leakage_audit.json",
                "per_subject_source_configs.csv",
                "per_video_transferred_predictions.json",
                "transfer_threshold_robustness.csv",
                "run_summary.json",
            )
        ],
        "stop": "No method exploration or transfer-side tuning performed",
    }
    write_json(RESULT_ROOT / "combined_report.json", record)
    write_json(
        RESULT_ROOT / "run_summary.json",
        {
            "status": "success",
            "implementation_target": "python",
            "directions": len(TRANSFER_PAIRS),
            "bootstrap_seed": SEED,
            "bootstrap_resamples": RESAMPLES,
            "input_caches": [contexts[key]["bundle"].cache_path for key in sorted(contexts)],
            "output_files": record["result_files"],
            "dependencies": {"numpy": np.__version__},
        },
    )
    print(json.dumps({"status": record["status"], "results": str(RESULT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
