"""Strict cross-backbone replay of the shared Fair Native configuration tuple.

The selectable source tuple is frozen per outer subject.  Target-side code only
uses the target response, target training-only duration prior, and the target's
unchanged Native interval/NMS implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
FAIR_CODE = PROJECT / "my_method" / "fair_tuned_native"
sys.path.insert(0, str(FAIR_CODE))

from equiscale_fair_validation import (  # noqa: E402
    Config,
    VideoFeatures,
    configuration_grid,
    fold_priors,
    load_data,
    metrics,
)


FAIR_ROOT = PROJECT / "results" / "fair_tuned_native_v2"
OUTPUT_ROOT = PROJECT / "results" / "native_config_transfer"
SETTINGS = (
    ("metst", "boostingvrme", "sammlv"),
    ("metst", "boostingvrme", "casme3"),
    ("boostingvrme", "metst", "sammlv"),
    ("boostingvrme", "metst", "casme3"),
)
COUNT_KEYS = ("TP", "FP", "FN")
SEED = 100
RESAMPLES = 10_000


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source_selections(backbone: str, dataset: str) -> dict[str, dict]:
    path = FAIR_ROOT / backbone / dataset / "outer_loso_selections.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["family"] == "single_tuned"]
    selected = {row["subject"]: row for row in rows}
    if len(selected) != len(rows):
        raise AssertionError("Duplicate source outer-subject selection")
    return selected


def counts_array_from_saved_predictions(backbone: str, dataset: str, family: str, subjects: list[str]):
    path = FAIR_ROOT / backbone / dataset / "selected_predictions.json"
    rows = read_json(path)
    indexes = {subject: index for index, subject in enumerate(subjects)}
    output = np.zeros((len(subjects), 3), dtype=np.int64)
    for row in rows:
        predictions = row["predictions"][family]
        tp = len({item["matched_gt"] for item in predictions if item["matched_gt"] >= 0})
        values = (tp, len(predictions) - tp, len(row["gt"]) - tp)
        output[indexes[row["subject"]]] += np.asarray(values, dtype=np.int64)
    return output


def aggregate(values: np.ndarray) -> dict:
    return metrics(values.sum(axis=0))


def paired_bootstrap(first: np.ndarray, second: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(first), size=(RESAMPLES, len(first)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0], denominator, out=np.zeros(RESAMPLES), where=denominator > 0
        )

    deltas = f1(first) - f1(second)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "point_delta_F1": float(aggregate(first)["F1"] - aggregate(second)["F1"]),
        "bootstrap_mean_delta_F1": float(deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_delta_gt_0": float(np.mean(deltas > 0.0)),
        "crosses_zero": bool(low <= 0 <= high),
        "resamples": RESAMPLES,
        "seed": SEED,
        "paired_unit": "outer subject",
    }


def run_setting(source: str, target: str, dataset: str, configs: list[Config]):
    source_selected = read_source_selections(source, dataset)
    records, subjects, _, cache_path = load_data(target, dataset)
    outer_k, _ = fold_priors(records, subjects, target)
    if set(source_selected) != set(subjects):
        raise AssertionError("Source and target outer-subject identities differ")

    manifest = read_json(PROJECT / "configs" / f"fair_tuned_native_{'me_tst' if source == 'metst' else 'boosting'}.json")
    manifest_configs = manifest["configurations"]
    for config_id, config in enumerate(configs[:324]):
        saved = manifest_configs[config_id]
        if (config.smooth, config.p, config.distance) != (
            saved["smooth"], saved["p"], saved["distance"]
        ):
            raise AssertionError("Runtime grid and signed source manifest differ")

    subject_index = {subject: index for index, subject in enumerate(subjects)}
    transferred = np.zeros((len(subjects), 3), dtype=np.int64)
    replay_default = np.zeros_like(transferred)
    predictions = []
    source_config_rows = []
    default_id = next(
        index
        for index, config in enumerate(configs[:324])
        if (config.smooth, config.p, config.distance) == (2.0, 0.55, 1.0)
    )

    frozen = {}
    for subject in subjects:
        selection = source_selected[subject]
        config_id = int(selection["config_id"])
        config = configs[config_id]
        if config.kind != "single" or config_id >= 324:
            raise AssertionError("Source selected a non-Fair-Native configuration")
        frozen[subject] = config
        source_config_rows.append(
            {
                "source": source,
                "target": target,
                "dataset": dataset,
                "held_out_subject": subject,
                "held_out_excluded_from_source_selection": True,
                "source_config_id": config_id,
                "source_config": config.identifier,
                "smooth": config.smooth,
                "p": config.p,
                "distance": config.distance,
                "source_inner_f1": float(selection["inner_F1"]),
                "target_k_out": int(outer_k[subject_index[subject]]),
                "target_metrics_used_for_selection": False,
            }
        )

    for record in records:
        sid = subject_index[record["subject"]]
        k = int(outer_k[sid])
        video = VideoFeatures(record, k, target)
        chosen = frozen[record["subject"]]
        chosen_prepared = video.prepare(video.clusters(chosen))
        chosen_counts, chosen_predictions = chosen_prepared.evaluate([chosen], details=True)
        transferred[sid] += chosen_counts[0]

        default = configs[default_id]
        default_prepared = video.prepare(video.clusters(default))
        default_counts, _ = default_prepared.evaluate([default], details=True)
        replay_default[sid] += default_counts[0]
        predictions.append(
            {
                "source": source,
                "target": target,
                "dataset": dataset,
                "subject": record["subject"],
                "video": record["video"],
                "target_k_out": k,
                "source_config": chosen.identifier,
                "gt": record["gt"],
                "counts": dict(zip(COUNT_KEYS, map(int, chosen_counts[0]))),
                "predictions": chosen_predictions[0],
            }
        )

    target_default = counts_array_from_saved_predictions(target, dataset, "native_train_k", subjects)
    target_tuned = counts_array_from_saved_predictions(target, dataset, "single_tuned", subjects)
    if not np.array_equal(replay_default, target_default):
        raise AssertionError("Target Native default smoke replay failed")

    subject_rows = []
    for index, subject in enumerate(subjects):
        transferred_metrics = metrics(transferred[index])
        tuned_metrics = metrics(target_tuned[index])
        subject_rows.append(
            {
                "source": source,
                "target": target,
                "dataset": dataset,
                "outer_subject": subject,
                "transferred_native_tp": transferred_metrics["TP"],
                "transferred_native_fp": transferred_metrics["FP"],
                "transferred_native_fn": transferred_metrics["FN"],
                "transferred_native_f1": transferred_metrics["F1"],
                "target_tuned_native_tp": tuned_metrics["TP"],
                "target_tuned_native_fp": tuned_metrics["FP"],
                "target_tuned_native_fn": tuned_metrics["FN"],
                "target_tuned_native_f1": tuned_metrics["F1"],
                "delta": transferred_metrics["F1"] - tuned_metrics["F1"],
            }
        )
    return {
        "source": source,
        "target": target,
        "dataset": dataset,
        "subjects": len(subjects),
        "videos": len(records),
        "cache_path": str(Path(cache_path).resolve()),
        "cache_sha256": sha256(Path(cache_path)),
        "default_replay_exact": True,
        "transferred": aggregate(transferred),
        "target_tuned": aggregate(target_tuned),
        "target_default": aggregate(target_default),
        "bootstrap_vs_target_tuned": paired_bootstrap(transferred, target_tuned),
        "subject_rows": subject_rows,
        "config_rows": source_config_rows,
        "predictions": predictions,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs, single_count, _ = configuration_grid()
    if single_count != 324:
        raise AssertionError("Fair Native single-scale grid is not 324")
    me_manifest = PROJECT / "configs" / "fair_tuned_native_me_tst.json"
    boost_manifest = PROJECT / "configs" / "fair_tuned_native_boosting.json"
    if me_manifest.read_bytes() != boost_manifest.read_bytes():
        raise AssertionError("Fair Native manifests are not byte-identical")

    runs = [run_setting(*setting, configs) for setting in SETTINGS]
    summary = []
    for run in runs:
        summary.append(
            {
                "source": run["source"],
                "target": run["target"],
                "dataset": run["dataset"],
                "target_native_default_f1": run["target_default"]["F1"],
                "target_tuned_native_f1": run["target_tuned"]["F1"],
                "transferred_native_f1": run["transferred"]["F1"],
                "delta_transferred_vs_target_tuned": run["transferred"]["F1"] - run["target_tuned"]["F1"],
                "transferred_tp": run["transferred"]["TP"],
                "transferred_fp": run["transferred"]["FP"],
                "transferred_fn": run["transferred"]["FN"],
                "default_replay_exact": run["default_replay_exact"],
            }
        )

    write_csv(OUTPUT_ROOT / "summary.csv", summary)
    write_csv(
        OUTPUT_ROOT / "per_subject_source_configs.csv",
        [row for run in runs for row in run["config_rows"]],
    )
    write_csv(
        OUTPUT_ROOT / "outer_subject_metrics.csv",
        [row for run in runs for row in run["subject_rows"]],
    )
    write_json(
        OUTPUT_ROOT / "per_video_transferred_predictions.json",
        [row for run in runs for row in run["predictions"]],
    )
    write_json(
        OUTPUT_ROOT / "combined_report.json",
        {
            "status": "STRICT_NATIVE_CONFIG_TRANSFER_COMPLETE",
            "audit_conclusion": "A. STRICT_NATIVE_CONFIG_TRANSFER_VALID",
            "protocol": {
                "source_selection": "locked source Fair Native outer-LOSO single_tuned selection",
                "transferred_tuple": ["smooth", "p", "distance"],
                "target_duration_prior": "locked target k_out(s), excluding held-out subject",
                "target_geometry": "unchanged target Native interval/NMS",
                "target_side_selection": False,
                "matching": "locked inclusive-frame chronological greedy IoU >= 0.5",
            },
            "manifest_sha256": sha256(me_manifest),
            "summary": summary,
            "bootstrap_vs_target_tuned": [
                {
                    "source": run["source"],
                    "target": run["target"],
                    "dataset": run["dataset"],
                    **run["bootstrap_vs_target_tuned"],
                }
                for run in runs
            ],
            "input_audit": [
                {
                    "source": run["source"],
                    "target": run["target"],
                    "dataset": run["dataset"],
                    "subjects": run["subjects"],
                    "videos": run["videos"],
                    "cache_path": run["cache_path"],
                    "cache_sha256": run["cache_sha256"],
                    "default_replay_exact": run["default_replay_exact"],
                }
                for run in runs
            ],
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
