#!/usr/bin/env python3
"""Nested frozen-cache evaluation of the H+L candidate score.

This is a frozen-response experiment.  Candidate geometry, interval construction,
and IoU evaluation come from ``gl_saliency_skill``; only the scalar score used to
threshold each candidate is changed.  H+L is ``(H + L) / 2`` with the exact H
normalization already used by the package.  G, L, H, and GL share one grid:
three reference scales, fixed radius 3, and thresholds 0.01--0.99.

The outer subject is never used to select its configuration.  The output is a
formal nested result under the frozen-cache evaluator, not a replacement for the
separately sealed official-response run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Allow direct execution from any working directory, matching the other
# frozen-cache experiment entry points.
PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from my_method.gl_saliency_skill.adapters.metst import load_metst
from my_method.gl_saliency_skill.evaluation import evaluate, metrics
from my_method.gl_saliency_skill.evidence import CurveFeatures


ROOT = PROJECT
DATASET_SPECS = {
    "sammlv": {"cache": ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl", "subjects": 29, "videos": 79, "gt": 159, "k": 5, "official_gt": 159},
    # This local CAS(ME)3 cache is the historical 858-GT cache.  It is useful
    # only as a transfer probe; the current official target uses 853 GT events.
    "casme3": {"cache": ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl", "subjects": 94, "videos": 462, "gt": 858, "k": 17, "official_gt": 853},
}
SCALES = (1.0, 1.5, 2.0)
RADIUS = 3.0
TAUS = tuple(round(i / 100.0, 2) for i in range(1, 100))
MODES = ("G", "L", "H", "HL", "GL")
SEED = 100
BOOTSTRAP_REPEATS = 10_000


@dataclass(frozen=True)
class Config:
    mode: str
    reference: float
    radius: float
    threshold: float
    weight: float | None = None

    @property
    def identifier(self) -> str:
        suffix = "" if self.weight is None else f"|weight={self.weight:g}"
        return (
            f"{self.mode}|reference={self.reference:g}|"
            f"radius={self.radius:g}|threshold={self.threshold:g}{suffix}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def f1(counts) -> float:
    tp, fp, fn = (int(x) for x in counts)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def mode_score(evidence: np.ndarray, mode: str, weight: float | None = None) -> np.ndarray:
    height, global_score, local_score = evidence.T
    if mode == "G":
        return global_score
    if mode == "L":
        return local_score
    if mode == "H":
        return height
    if mode == "HL":
        return 0.5 * (height + local_score)
    if mode.startswith("HL_w"):
        weight = float(mode.removeprefix("HL_w"))
        return (1.0 - weight) * height + weight * local_score
    if mode == "HL_joint":
        if weight is None:
            raise ValueError("HL_joint requires an explicit weight")
        return (1.0 - float(weight)) * height + float(weight) * local_score
    if mode == "GL":
        return 0.5 * (global_score + local_score)
    raise ValueError(mode)


def decode(record, feature: CurveFeatures, config: Config, k: int):
    peaks, evidence = feature.evidence(config.reference, config.radius)
    geometry = record["adapter"].geometry(record["score"], k, peaks)
    ordered_peaks = peaks[geometry.order]
    ordered_evidence = evidence[geometry.order]
    scores = mode_score(ordered_evidence, config.mode, config.weight)
    keep = scores >= config.threshold
    events = []
    for index, peak in enumerate(ordered_peaks):
        if not keep[index]:
            continue
        onset, offset = geometry.intervals[index]
        events.append(
            {
                "onset": int(onset),
                "peak": int(peak),
                "offset": int(offset),
                "confidence": float(scores[index]),
            }
        )
    return events


def choose(pooled: np.ndarray, configs: list[Config]) -> int:
    values = np.asarray(pooled, dtype=float)
    tp, fp, fn = values.T
    scores = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(values)), where=(2 * tp + fp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(values)), where=(tp + fp) > 0)
    indexes = np.arange(len(configs), dtype=int)
    return int(np.lexsort((indexes, fp, -precision, -scores))[0])


def evaluate_record(record, feature: CurveFeatures, config: Config, k: int) -> np.ndarray:
    counts, _ = evaluate(decode(record, feature, config, k), record["ground_truth"])
    return np.asarray([counts["TP"], counts["FP"], counts["FN"]], dtype=np.int64)


def paired_bootstrap(subject_counts: dict[str, np.ndarray], reference: str, output: Path, modes: tuple[str, ...]) -> None:
    subjects = sorted(subject_counts[reference])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPEATS, len(subjects)))

    def sample(mode: str) -> np.ndarray:
        values = np.asarray([subject_counts[mode][subject] for subject in subjects], dtype=np.int64)
        totals = values[draws].sum(axis=1)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(2 * totals[:, 0], denominator, out=np.zeros(len(totals)), where=denominator > 0)

    rows = []
    for mode in modes:
        if mode == reference:
            continue
        delta = sample(mode) - sample(reference)
        point = f1(np.asarray([subject_counts[mode][s] for s in subjects]).sum(axis=0)) - f1(
            np.asarray([subject_counts[reference][s] for s in subjects]).sum(axis=0)
        )
        rows.append({
            "comparison": f"{mode}-minus-{reference}",
            "point_delta_F1": point,
            "ci95_low": float(np.quantile(delta, 0.025)),
            "ci95_high": float(np.quantile(delta, 0.975)),
            "crosses_zero": bool(np.quantile(delta, 0.025) <= 0 <= np.quantile(delta, 0.975)),
            "resamples": BOOTSTRAP_REPEATS,
            "seed": SEED,
            "unit": "subject",
        })
    write_csv(output / "paired_bootstrap.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), default="sammlv")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--radii", nargs="+", type=float, default=[RADIUS])
    parser.add_argument("--hl-weights", nargs="+", type=float, default=[0.5])
    parser.add_argument("--joint-hl", action="store_true", help="select H/L weight jointly inside each outer training fold")
    args = parser.parse_args()
    spec = DATASET_SPECS[args.dataset]
    cache = (args.cache or spec["cache"]).resolve()
    output = (args.output or ROOT / f"results/height_local_nested_{args.dataset}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")

    bundle = load_metst(cache, args.dataset)
    package = pickle.loads(cache.read_bytes())
    k = int(package["k_p"])
    if k != spec["k"] or len(bundle.subjects) != spec["subjects"] or len(bundle.records) != spec["videos"] or sum(len(x.ground_truth) for x in bundle.records) != spec["gt"]:
        raise RuntimeError(f"frozen {args.dataset} cache cardinality or k_p mismatch")

    subjects = list(bundle.subjects)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    records = []
    features = {}
    for source, record in zip(package["records"], bundle.records):
        key = (record.subject, record.video)
        feature = CurveFeatures(record.score, k)
        features[key] = feature
        records.append({
            "subject": record.subject,
            "video": record.video,
            "score": record.score,
            "ground_truth": record.ground_truth,
            "adapter": bundle.interval_adapter,
            "feature": feature,
        })

    radii = tuple(float(x) for x in args.radii)
    weights = tuple(float(x) for x in args.hl_weights)
    if any(x <= 0 for x in radii) or any(not 0 < x < 1 for x in weights):
        raise RuntimeError("radii must be positive and H+L weights must lie in (0, 1)")
    if args.joint_hl:
        run_modes = ("G", "L", "H", "GL", "HL_joint")
    elif len(weights) == 1 and abs(weights[0] - 0.5) < 1e-12:
        run_modes = MODES
    else:
        run_modes = ("G", "L", "H", "GL") + tuple(f"HL_w{weight:g}" for weight in weights)
    configs = {}
    for mode in run_modes:
        if mode == "HL_joint":
            configs[mode] = [Config(mode, scale, radius, tau, weight) for weight in weights for scale in SCALES for radius in radii for tau in TAUS]
        else:
            configs[mode] = [Config(mode, scale, radius, tau) for scale in SCALES for radius in radii for tau in TAUS]
    output.mkdir(parents=True)

    # Counts are computed once per mode/config/subject, then reused for every outer fold.
    tables = {}
    for mode in run_modes:
        grid = configs[mode]
        table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
        for record in records:
            si = subject_index[record["subject"]]
            feature = record["feature"]
            for config_id, config in enumerate(grid):
                table[config_id, si] += evaluate_record(record, feature, config, k)
        tables[mode] = table

    selected_rows = []
    outer_rows = []
    per_subject_rows = []
    totals = {}
    subject_counts = {}
    for mode in run_modes:
        grid = configs[mode]
        selected = {}
        counts_by_subject = {}
        for held, subject in enumerate(subjects):
            train = np.delete(tables[mode], held, axis=1).sum(axis=1)
            winner = choose(train, grid)
            selected[subject] = winner
            inner_counts = train[winner]
            selected_rows.append({
                "mode": mode,
                "outer_subject": subject,
                "config_id": winner,
                "config": grid[winner].identifier,
                "inner_TP": int(inner_counts[0]),
                "inner_FP": int(inner_counts[1]),
                "inner_FN": int(inner_counts[2]),
                "inner_F1": f1(inner_counts),
            })
            test_counts = tables[mode][winner, held]
            counts_by_subject[subject] = test_counts.copy()
            outer_rows.append({
                "mode": mode,
                "outer_subject": subject,
                "config_id": winner,
                "reference": grid[winner].reference,
                "radius": grid[winner].radius,
                "threshold": grid[winner].threshold,
                "TP": int(test_counts[0]),
                "FP": int(test_counts[1]),
                "FN": int(test_counts[2]),
                "F1": f1(test_counts),
            })
            per_subject_rows.append({"mode": mode, "subject": subject, "TP": int(test_counts[0]), "FP": int(test_counts[1]), "FN": int(test_counts[2]), "F1": f1(test_counts)})
        subject_counts[mode] = counts_by_subject
        total = np.asarray(list(counts_by_subject.values()), dtype=np.int64).sum(axis=0)
        totals[mode] = total

    summary = []
    for mode in run_modes:
        summary.append({"mode": mode, "TP": int(totals[mode][0]), "FP": int(totals[mode][1]), "FN": int(totals[mode][2]), **metrics(dict(zip(("TP", "FP", "FN"), totals[mode])))})
    write_csv(output / "nested_summary.csv", summary)
    write_csv(output / "outer_selected_configs.csv", selected_rows)
    write_csv(output / "outer_folds.csv", outer_rows)
    write_csv(output / "per_subject_counts.csv", per_subject_rows)
    reference_mode = "HL" if "HL" in run_modes else "HL_w0.5" if "HL_w0.5" in run_modes else "HL_joint" if "HL_joint" in run_modes else run_modes[-1]
    paired_bootstrap(subject_counts, reference_mode, output, run_modes)

    selected_frequency = []
    for mode in run_modes:
        for config_id, config in enumerate(configs[mode]):
            count = sum(row["config_id"] == config_id and row["mode"] == mode for row in selected_rows)
            if count:
                selected_frequency.append({"mode": mode, "config_id": config_id, "config": config.identifier, "count": count})
    write_csv(output / "selected_config_frequency.csv", selected_frequency)

    write_json(output / "run_manifest.json", {
        "status": "PASS",
        "protocol": "nested_LOSO_subject_selection_on_frozen_SAMM-LV_cache",
        "dataset": args.dataset,
        "subjects": len(subjects),
        "videos": len(records),
        "GT_events": int(sum(len(x["ground_truth"]) for x in records)),
        "k_p": k,
        "cache": str(cache),
        "cache_sha256": sha256(cache),
        "grid": {"modes": list(run_modes), "reference_scales": list(SCALES), "radii": list(radii), "hl_weights": list(weights), "thresholds": list(TAUS), "configs_per_mode": {mode: len(configs[mode]) for mode in run_modes}},
        "score_formulas": {"H": "max(0, (smooth[peak]-mean)/(smooth.max()-mean))", "HL": "(H+L)/2", "HL_joint": "(1-w)H+wL with w selected only on outer-training subjects", "GL": "(G+L)/2"},
        "selection": "outer subject held out; pooled training TP/FP/FN; F1, precision, fewer FP, fixed config order",
        "evaluator": "gl_saliency_skill.evaluation.evaluate; inclusive IoU >= 0.5; chronological one-to-one matching",
        "official_response_status": "This local frozen-cache run does not replace the separately sealed official-response evaluation.",
        "official_target_GT": spec["official_gt"],
        "transfer_probe_only": args.dataset == "casme3",
        "no_fabrication": True,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OUTPUT =", output)


if __name__ == "__main__":
    main()
