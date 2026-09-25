"""EXP-2A: expand only GLSD's searched multiscale set from 3 to 7 values."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RETHINK = ROOT / "RethinkFuse_reproduction"
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
BASELINE = ROOT / "controlled_exploration/baseline_snapshot"

sys.path.insert(0, str(SIGNED))
import equiscale_fair_validation as fair  # noqa: E402
import unified_persistence as engine  # noqa: E402


GROUPS = (
    ("metst", "sammlv", "ME-TST/SAMMLV"),
    ("metst", "casme3", "ME-TST/CAS(ME)3"),
    ("boostingvrme", "sammlv", "BoostingVRME/SAMMLV"),
    ("boostingvrme", "casme3", "BoostingVRME/CAS(ME)3"),
)
BASELINE_SCALES = (1.0, 1.5, 2.0)
EXPANDED_SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
BASELINE_CONFIGURATIONS = 90
SEED = 100


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def config_reference(config_identifier: str) -> float:
    """Extract the reference scale from the archived stable config identifier."""
    return float(config_identifier.split("|scale=", 1)[1].split("|", 1)[0])


def expanded_configs() -> list[engine.Config]:
    configs = [
        engine.Config("unified", reference, 0.0, threshold, radius)
        for reference in EXPANDED_SCALES
        for radius in engine.LOCAL_RADII
        for threshold in engine.THRESHOLDS
    ]
    expected = len(EXPANDED_SCALES) * len(engine.LOCAL_RADII) * len(engine.THRESHOLDS)
    if len(configs) != expected or expected != 210:
        raise AssertionError("EXP-2A expanded grid must contain exactly 210 configurations")
    return configs


def baseline_subject_counts(backbone: str, dataset: str) -> np.ndarray:
    mode = "native" if backbone == "boostingvrme" else "fixed"
    path = (
        ROOT
        / "historical_gl_exact_fresh_reproduction/fresh_run"
        / backbone
        / "results/pure_persistence_matched_v1"
        / dataset
        / mode
        / "subject_counts.csv"
    )
    rows = [row for row in load_csv(path) if row["family"] == "pure"]
    return np.array([[int(row[key]) for key in ("TP", "FP", "FN")] for row in rows])


def baseline_selection_rows(backbone: str, dataset: str) -> list[dict]:
    mode = "native" if backbone == "boostingvrme" else "fixed"
    path = (
        ROOT
        / "historical_gl_exact_fresh_reproduction/fresh_run"
        / backbone
        / "results/pure_persistence_matched_v1"
        / dataset
        / mode
        / "outer_loso_selections.csv"
    )
    return [row for row in load_csv(path) if row["family"] == "pure"]


def paired_bootstrap(expanded: np.ndarray, baseline: np.ndarray) -> dict:
    if expanded.shape != baseline.shape:
        raise AssertionError("Expanded and baseline subject-count shapes differ")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(expanded), size=(10_000, len(expanded)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0],
            denominator,
            out=np.zeros(10_000),
            where=denominator > 0,
        )

    deltas = f1(expanded) - f1(baseline)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "ci95_low": float(low),
        "ci95_high": float(high),
        "positive_resample_fraction": float(np.mean(deltas > 0)),
        "resamples": 10_000,
        "seed": SEED,
    }


def run_group(backbone: str, dataset: str, setting: str, configs: list[engine.Config]):
    records, subjects, _, input_path = fair.load_data(backbone, dataset)
    outer, inner = fair.fold_priors(records, subjects, backbone)
    mode = "native" if backbone == "boostingvrme" else "fixed"
    stats: dict[int, np.ndarray] = {}
    needed_k = sorted(set(outer.tolist()) | set(inner[inner > 0].tolist()))
    for k in needed_k:
        needed = fair.required_subjects(k, outer, inner)
        values = engine.evaluate_k(records, subjects, backbone, k, needed, configs, (mode,))
        stats[k] = values[mode]

    selected: list[int] = []
    counts = np.zeros((len(subjects), 3), dtype=np.int64)
    selection_rows = []
    indexes = list(range(len(configs)))
    for held, subject in enumerate(subjects):
        training = engine.inner_counts(stats, held, inner)
        choice = engine.choose(training, indexes)
        config = configs[choice]
        selected.append(choice)
        counts[held] = stats[int(outer[held])][choice, held]
        selection_rows.append(
            {
                "setting": setting,
                "backbone": backbone,
                "dataset": dataset,
                "outer_subject": subject,
                "selected_scale": config.reference,
                "selected_radius": config.radius,
                "selected_threshold": config.threshold,
                "selected_configuration": config.identifier,
                "inner_F1": fair.metrics(training[choice])["F1"],
                "outer_TP": int(counts[held, 0]),
                "outer_FP": int(counts[held, 1]),
                "outer_FN": int(counts[held, 2]),
            }
        )

    # Independently replay every selected outer-fold configuration.
    replay = np.zeros_like(counts)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for record in records:
        held = subject_index[record["subject"]]
        config = configs[selected[held]]
        features = engine.CurveFeatures(record["curve"], int(outer[held]))
        prepared = engine.prepare(
            record, features, config.reference, backbone, mode, config.radius
        )
        replay[held] += prepared.evaluate([config])[0]
    np.testing.assert_array_equal(replay, counts)

    baseline_rows = baseline_selection_rows(backbone, dataset)
    if [row["subject"] for row in baseline_rows] != subjects:
        raise AssertionError(f"Baseline subject order mismatch for {setting}")
    for row, baseline in zip(selection_rows, baseline_rows):
        row["baseline_selected_configuration"] = baseline["config"]
        row["baseline_inner_F1"] = baseline["inner_F1"]

    return fair.metrics(counts.sum(axis=0)), counts, selection_rows, {
        "input": str(input_path),
        "input_sha256": digest(Path(input_path)),
        "subjects": len(subjects),
        "videos": len(records),
        "outer_k_frequency": dict(Counter(str(value) for value in outer)),
        "inner_k_frequency": dict(Counter(str(value) for value in inner[inner > 0])),
        "selected_configuration_frequency": dict(
            Counter(configs[index].identifier for index in selected)
        ),
        "outer_count_replay": "PASS",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = RETHINK / "caches/me_tst"

    # Isolated process-local expansion; no source model file is modified.
    engine.SCALES = EXPANDED_SCALES
    engine.REFERENCE_SCALES = EXPANDED_SCALES
    configs = expanded_configs()

    baseline_main = {
        (row["Backbone"], row["Dataset"]): row
        for row in load_csv(BASELINE / "results/main_results.csv")
    }
    result_rows = []
    distribution_rows = []
    selection_rows = []
    bootstrap_rows = []
    audits = {}

    for backbone, dataset, setting in GROUPS:
        expanded_metrics, expanded_counts, selections, audit = run_group(
            backbone, dataset, setting, configs
        )
        baseline = baseline_main[backbone, dataset]
        baseline_f1 = float(baseline["GL_Skill_F1"])
        delta_f1 = expanded_metrics["F1"] - baseline_f1
        result_rows.append(
            {
                "setting": setting,
                "backbone": backbone,
                "dataset": dataset,
                "baseline_configurations": BASELINE_CONFIGURATIONS,
                "expanded_configurations": len(configs),
                "baseline_TP": int(baseline["GL_Skill_TP"]),
                "baseline_FP": int(baseline["GL_Skill_FP"]),
                "baseline_FN": int(baseline["GL_Skill_FN"]),
                "baseline_F1": baseline_f1,
                "expanded_TP": expanded_metrics["TP"],
                "expanded_FP": expanded_metrics["FP"],
                "expanded_FN": expanded_metrics["FN"],
                "expanded_F1": expanded_metrics["F1"],
                "delta_F1": delta_f1,
            }
        )
        scale_counts = Counter(float(row["selected_scale"]) for row in selections)
        baseline_scale_counts = Counter(
            config_reference(row["baseline_selected_configuration"])
            for row in selections
        )
        for scale in EXPANDED_SCALES:
            baseline_count = baseline_scale_counts[scale]
            expanded_count = scale_counts[scale]
            distribution_rows.append(
                {
                    "setting": setting,
                    "backbone": backbone,
                    "dataset": dataset,
                    "scale": scale,
                    "baseline_selected_outer_folds": baseline_count,
                    "baseline_selection_fraction": baseline_count / len(selections),
                    "expanded_selected_outer_folds": expanded_count,
                    "expanded_selection_fraction": expanded_count / len(selections),
                    "total_outer_folds": len(selections),
                    "in_baseline_scale_set": scale in BASELINE_SCALES,
                }
            )
        selection_rows.extend(selections)
        bootstrap_rows.append(
            {
                "setting": setting,
                "delta_F1": delta_f1,
                **paired_bootstrap(
                    expanded_counts, baseline_subject_counts(backbone, dataset)
                ),
            }
        )
        audits[setting] = {"expanded_metrics": expanded_metrics, **audit}

    write_csv(
        OUT / "results.csv",
        result_rows,
        [
            "setting", "backbone", "dataset", "baseline_configurations",
            "expanded_configurations", "baseline_TP", "baseline_FP", "baseline_FN",
            "baseline_F1", "expanded_TP", "expanded_FP", "expanded_FN",
            "expanded_F1", "delta_F1",
        ],
    )
    write_csv(
        OUT / "selected_scale_distribution.csv",
        distribution_rows,
        [
            "setting", "backbone", "dataset", "scale",
            "baseline_selected_outer_folds", "baseline_selection_fraction",
            "expanded_selected_outer_folds", "expanded_selection_fraction",
            "total_outer_folds", "in_baseline_scale_set",
        ],
    )
    write_csv(
        OUT / "outer_fold_selections.csv",
        selection_rows,
        [
            "setting", "backbone", "dataset", "outer_subject", "selected_scale",
            "selected_radius", "selected_threshold", "selected_configuration",
            "inner_F1", "outer_TP", "outer_FP", "outer_FN",
            "baseline_selected_configuration", "baseline_inner_F1",
        ],
    )
    write_csv(
        OUT / "bootstrap.csv",
        bootstrap_rows,
        [
            "setting", "delta_F1", "ci95_low", "ci95_high",
            "positive_resample_fraction", "resamples", "seed",
        ],
    )

    protocol = {
        "experiment": "GLSD controlled exploration EXP-2A",
        "interpretation": "configuration-space expansion; not a new model",
        "only_change": "scale set and searched reference-scale choices",
        "baseline_scales": list(BASELINE_SCALES),
        "expanded_scales": list(EXPANDED_SCALES),
        "baseline_configuration_budget": BASELINE_CONFIGURATIONS,
        "expanded_configuration_budget": len(configs),
        "local_aggregation": "median (unchanged)",
        "global_evidence": "unchanged",
        "fusion": "S(c)=(G(c)+L(c))/2 (unchanged)",
        "local_radii": list(engine.LOCAL_RADII),
        "thresholds": list(engine.THRESHOLDS),
        "outer": "LOSO",
        "inner": "LOSO excluding outer test and inner validation subjects",
        "selection_tiebreak": [
            "higher F1", "higher precision", "fewer FP", "lower config order"
        ],
        "primary_intervals": {"metst": "fixed", "boostingvrme": "native"},
        "main_glsd_modified": False,
        "bootstrap_resamples": 10_000,
        "seed": SEED,
        "source_sha256": {
            "run_exp2a.py": digest(Path(__file__)),
            "unified_persistence.py": digest(SIGNED / "unified_persistence.py"),
            "equiscale_fair_validation.py": digest(
                SIGNED / "equiscale_fair_validation.py"
            ),
        },
        "audits": audits,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "EXP2A_COMPLETE", "results": result_rows}, indent=2))


if __name__ == "__main__":
    main()
