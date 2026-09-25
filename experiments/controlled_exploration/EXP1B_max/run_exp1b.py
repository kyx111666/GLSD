"""EXP-1B: replace only GLSD's cross-scale local median with a maximum."""

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
SEED = 100


class MaxCurveFeatures(engine.CurveFeatures):
    """Historical evidence with only median(aligned) changed to max(aligned)."""

    def evidence(self, reference, radius=1.0):
        if (reference, radius) in self.evidence_cache:
            return self.evidence_cache[reference, radius]
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))
        for width, (other_peaks, _, _, smooth, spread) in self.effective_scales.items():
            if (width, window) not in self.local_cache:
                self.local_cache[width, window] = np.array(
                    [
                        max(
                            0.0,
                            float(smooth[peak])
                            - max(
                                float(np.min(smooth[max(0, peak - window) : peak + 1])),
                                float(
                                    np.min(
                                        smooth[
                                            peak : min(len(smooth), peak + window + 1)
                                        ]
                                    )
                                ),
                            ),
                        )
                        / spread
                        for peak in other_peaks
                    ]
                )
        local = []
        for point in peaks:
            aligned = []
            for width, (other_peaks, _, _, _, _) in self.effective_scales.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    aligned.append(0.0)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (
                        abs(int(other_peaks[index]) - point),
                        -values[index],
                    ),
                )
                aligned.append(float(values[chosen]))
            local.append(float(np.max(aligned)))
        output = peaks, np.column_stack((height, global_values, local))
        self.evidence_cache[reference, radius] = output
        return output


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


def pure_configs() -> list[engine.Config]:
    configs = [
        engine.Config("unified", reference, 0.0, threshold, radius)
        for reference in engine.REFERENCE_SCALES
        for radius in engine.LOCAL_RADII
        for threshold in engine.THRESHOLDS
    ]
    if len(configs) != 90:
        raise AssertionError("EXP-1B must preserve the locked 90-config budget")
    return configs


def baseline_rows(backbone: str, dataset: str) -> list[dict]:
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


def bootstrap(first: np.ndarray, second: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(first), size=(10_000, len(first)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0],
            denominator,
            out=np.zeros(10_000),
            where=denominator > 0,
        )

    deltas = f1(first) - f1(second)
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
        values = engine.evaluate_k(
            records,
            subjects,
            backbone,
            k,
            needed,
            configs,
            (mode,),
        )
        stats[k] = values[mode]

    selected = []
    counts = np.zeros((len(subjects), 3), dtype=np.int64)
    max_rows = []
    indexes = list(range(len(configs)))
    for held, subject in enumerate(subjects):
        training = engine.inner_counts(stats, held, inner)
        choice = engine.choose(training, indexes)
        selected.append(choice)
        counts[held] = stats[int(outer[held])][choice, held]
        max_rows.append(
            {
                "subject": subject,
                "config": configs[choice].identifier,
                "inner_F1": fair.metrics(training[choice])["F1"],
            }
        )

    # Exact outer-count replay for the chosen configuration on every subject.
    replay = np.zeros_like(counts)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for record in records:
        held = subject_index[record["subject"]]
        config = configs[selected[held]]
        features = MaxCurveFeatures(record["curve"], int(outer[held]))
        prepared = engine.prepare(
            record, features, config.reference, backbone, mode, config.radius
        )
        replay[held] += prepared.evaluate([config])[0]
    np.testing.assert_array_equal(replay, counts)

    baseline = baseline_rows(backbone, dataset)
    if [row["subject"] for row in baseline] != subjects:
        raise AssertionError(f"Baseline subject order mismatch for {setting}")
    differences = []
    for base, maximum in zip(baseline, max_rows):
        differences.append(
            {
                "setting": setting,
                "outer_subject": maximum["subject"],
                "baseline_selected_configuration": base["config"],
                "max_selected_configuration": maximum["config"],
                "baseline_inner_F1": base["inner_F1"],
                "max_inner_F1": maximum["inner_F1"],
                "changed": base["config"] != maximum["config"],
            }
        )
    return fair.metrics(counts.sum(axis=0)), counts, differences, {
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
    engine.CurveFeatures = MaxCurveFeatures
    configs = pure_configs()
    baseline_main = {
        (row["Backbone"], row["Dataset"]): row
        for row in load_csv(BASELINE / "results/main_results.csv")
    }

    result_rows = []
    difference_rows = []
    bootstrap_rows = []
    mechanism_rows = []
    audits = {}
    for backbone, dataset, setting in GROUPS:
        max_metrics, max_counts, differences, audit = run_group(
            backbone, dataset, setting, configs
        )
        baseline = baseline_main[backbone, dataset]
        baseline_f1 = float(baseline["GL_Skill_F1"])
        delta = max_metrics["F1"] - baseline_f1
        result_rows.append(
            {
                "setting": setting,
                "baseline_F1": baseline_f1,
                "max_L_F1": max_metrics["F1"],
                "delta_F1": delta,
            }
        )
        difference_rows.extend(differences)
        changed_subjects = sum(row["changed"] for row in differences)
        baseline_tp = int(baseline["GL_Skill_TP"])
        baseline_fp = int(baseline["GL_Skill_FP"])
        baseline_fn = int(baseline["GL_Skill_FN"])
        mechanism_rows.append(
            {
                "setting": setting,
                "baseline_TP": baseline_tp,
                "max_L_TP": max_metrics["TP"],
                "delta_TP": max_metrics["TP"] - baseline_tp,
                "baseline_FP": baseline_fp,
                "max_L_FP": max_metrics["FP"],
                "delta_FP": max_metrics["FP"] - baseline_fp,
                "baseline_FN": baseline_fn,
                "max_L_FN": max_metrics["FN"],
                "delta_FN": max_metrics["FN"] - baseline_fn,
                "changed_selected_configs": changed_subjects,
                "subjects": len(differences),
                "changed_config_fraction": changed_subjects / len(differences),
            }
        )
        bootstrap_rows.append(
            {
                "setting": setting,
                "delta_F1": delta,
                **bootstrap(max_counts, baseline_subject_counts(backbone, dataset)),
            }
        )
        audits[setting] = {"max_metrics": max_metrics, **audit}

    write_csv(
        OUT / "results.csv",
        result_rows,
        ["setting", "baseline_F1", "max_L_F1", "delta_F1"],
    )
    write_csv(
        OUT / "config_difference.csv",
        difference_rows,
        [
            "setting",
            "outer_subject",
            "baseline_selected_configuration",
            "max_selected_configuration",
            "baseline_inner_F1",
            "max_inner_F1",
            "changed",
        ],
    )
    write_csv(
        OUT / "mechanism_analysis.csv",
        mechanism_rows,
        [
            "setting",
            "baseline_TP",
            "max_L_TP",
            "delta_TP",
            "baseline_FP",
            "max_L_FP",
            "delta_FP",
            "baseline_FN",
            "max_L_FN",
            "delta_FN",
            "changed_selected_configs",
            "subjects",
            "changed_config_fraction",
        ],
    )
    write_csv(
        OUT / "bootstrap.csv",
        bootstrap_rows,
        [
            "setting",
            "delta_F1",
            "ci95_low",
            "ci95_high",
            "positive_resample_fraction",
            "resamples",
            "seed",
        ],
    )
    protocol = {
        "experiment": "GLSD controlled exploration EXP-1B",
        "baseline_local_aggregation": "median",
        "experimental_local_aggregation": "max",
        "only_algorithmic_change": "np.median(aligned) -> np.max(aligned)",
        "score": "S(c)=(G(c)+L(c))/2",
        "scales": list(engine.SCALES),
        "reference_scales": list(engine.REFERENCE_SCALES),
        "local_radii": list(engine.LOCAL_RADII),
        "thresholds": list(engine.THRESHOLDS),
        "configuration_budget": len(configs),
        "outer": "LOSO",
        "inner": "LOSO excluding outer test and inner validation subjects",
        "selection_tiebreak": ["higher F1", "higher precision", "fewer FP", "lower config order"],
        "bootstrap_resamples": 10_000,
        "seed": SEED,
        "primary_intervals": {"metst": "fixed", "boostingvrme": "native"},
        "main_glsd_modified": False,
        "source_sha256": {
            "run_exp1b.py": digest(Path(__file__)),
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
    print(json.dumps({"status": "EXP1B_COMPLETE", "results": result_rows}, indent=2))


if __name__ == "__main__":
    main()
