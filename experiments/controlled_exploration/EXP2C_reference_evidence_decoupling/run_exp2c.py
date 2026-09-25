"""EXP-2C: decouple reference-scale search from local evidence scale breadth."""

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
REFERENCE_SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
BASELINE_CONFIGURATIONS = 90
EXP2A = ROOT / "controlled_exploration/EXP2A_scale"
SEED = 100


def three_nearest_scales(reference: float) -> tuple[float, float, float]:
    """Return the fixed three-nearest evidence neighborhood for a reference."""
    selected = sorted(REFERENCE_SCALES, key=lambda scale: (abs(scale - reference), scale))[:3]
    if reference not in selected:
        raise AssertionError(f"Reference scale {reference} missing from its evidence triplet")
    return tuple(sorted(selected))


EVIDENCE_SCALE_MAP = {
    reference: three_nearest_scales(reference) for reference in REFERENCE_SCALES
}


BaseCurveFeatures = engine.CurveFeatures


class DecoupledCurveFeatures(BaseCurveFeatures):
    """Canonical features with L restricted to the reference's fixed triplet."""

    def evidence(self, reference, radius=1.0):
        if (reference, radius) in self.evidence_cache:
            return self.evidence_cache[reference, radius]
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))

        # Preserve the canonical one-vote-per-distinct-integer-width rule while
        # restricting the available evidence to the deterministic triplet.
        evidence_by_width = {}
        for scale in EVIDENCE_SCALE_MAP[reference]:
            width = min(len(self.curve), max(1, round(scale * self.k)))
            evidence_by_width.setdefault(width, self.scales[scale])
        if len(evidence_by_width) != 3:
            raise AssertionError(
                f"Evidence triplet collapsed to {len(evidence_by_width)} physical widths "
                f"for reference={reference}, k={self.k}"
            )

        for width, (other_peaks, _, _, smooth, spread) in evidence_by_width.items():
            if (width, window) not in self.local_cache:
                self.local_cache[width, window] = np.array([
                    max(
                        0.0,
                        float(smooth[p])
                        - max(
                            float(np.min(smooth[max(0, p-window):p+1])),
                            float(np.min(smooth[p:min(len(smooth), p+window+1)])),
                        ),
                    ) / spread
                    for p in other_peaks
                ])

        local = []
        for point in peaks:
            aligned = []
            for width, (other_peaks, _, _, _, _) in evidence_by_width.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    aligned.append(0.0)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (abs(int(other_peaks[index]) - point), -values[index]),
                )
                aligned.append(float(values[chosen]))
            local.append(float(np.median(aligned)))
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


def config_reference(config_identifier: str) -> float:
    """Extract the reference scale from the archived stable config identifier."""
    return float(config_identifier.split("|scale=", 1)[1].split("|", 1)[0])


def exp2c_configs() -> list[engine.Config]:
    configs = [
        engine.Config("unified", reference, 0.0, threshold, radius)
        for reference in REFERENCE_SCALES
        for radius in engine.LOCAL_RADII
        for threshold in engine.THRESHOLDS
    ]
    expected = len(REFERENCE_SCALES) * len(engine.LOCAL_RADII) * len(engine.THRESHOLDS)
    if len(configs) != expected or expected != 210:
        raise AssertionError("EXP-2C grid must contain exactly 210 configurations")
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


def exp2a_subject_counts(setting: str) -> np.ndarray:
    rows = [
        row for row in load_csv(EXP2A / "outer_fold_selections.csv")
        if row["setting"] == setting
    ]
    return np.array([
        [int(row[key]) for key in ("outer_TP", "outer_FP", "outer_FN")]
        for row in rows
    ])


def paired_bootstrap(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.shape != reference.shape:
        raise AssertionError("Candidate and reference subject-count shapes differ")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(candidate), size=(10_000, len(candidate)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0],
            denominator,
            out=np.zeros(10_000),
            where=denominator > 0,
        )

    deltas = f1(candidate) - f1(reference)
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

    # Isolated process-local override; no canonical model source is modified.
    engine.SCALES = REFERENCE_SCALES
    engine.REFERENCE_SCALES = REFERENCE_SCALES
    engine.CurveFeatures = DecoupledCurveFeatures
    configs = exp2c_configs()

    baseline_main = {
        (row["Backbone"], row["Dataset"]): row
        for row in load_csv(BASELINE / "results/main_results.csv")
    }
    exp2a_main = {
        row["setting"]: row for row in load_csv(EXP2A / "results.csv")
    }
    result_rows = []
    distribution_rows = []
    selection_rows = []
    bootstrap_rows = []
    audits = {}

    for backbone, dataset, setting in GROUPS:
        exp2c_metrics, exp2c_counts, selections, audit = run_group(
            backbone, dataset, setting, configs
        )
        baseline = baseline_main[backbone, dataset]
        baseline_f1 = float(baseline["GL_Skill_F1"])
        exp2a = exp2a_main[setting]
        exp2a_f1 = float(exp2a["expanded_F1"])
        baseline_counts = baseline_subject_counts(backbone, dataset)
        exp2a_counts = exp2a_subject_counts(setting)
        vs_baseline = paired_bootstrap(exp2c_counts, baseline_counts)
        vs_exp2a = paired_bootstrap(exp2c_counts, exp2a_counts)
        result_rows.append(
            {
                "setting": setting,
                "baseline_f1": baseline_f1,
                "exp2a_f1": exp2a_f1,
                "exp2c_f1": exp2c_metrics["F1"],
                "delta_exp2c_vs_baseline": exp2c_metrics["F1"] - baseline_f1,
                "delta_exp2c_vs_exp2a": exp2c_metrics["F1"] - exp2a_f1,
                "paired_ci_low_vs_baseline": vs_baseline["ci95_low"],
                "paired_ci_high_vs_baseline": vs_baseline["ci95_high"],
                "paired_ci_low_vs_exp2a": vs_exp2a["ci95_low"],
                "paired_ci_high_vs_exp2a": vs_exp2a["ci95_high"],
            }
        )
        for row in selections:
            row["subject"] = row.pop("outer_subject")
            row["selected_a0"] = row.pop("selected_scale")
            row["selected_tau"] = row.pop("selected_threshold")
            row["evidence_scales"] = "|".join(
                f"{scale:g}" for scale in EVIDENCE_SCALE_MAP[float(row["selected_a0"])]
            )
            row["inner_f1"] = row.pop("inner_F1")
            row["outer_tp"] = row.pop("outer_TP")
            row["outer_fp"] = row.pop("outer_FP")
            row["outer_fn"] = row.pop("outer_FN")
            distribution_rows.append({
                key: row[key] for key in (
                    "setting", "subject", "selected_a0", "selected_radius",
                    "selected_tau", "evidence_scales", "inner_f1",
                    "outer_tp", "outer_fp", "outer_fn",
                )
            })
        selection_rows.extend(selections)
        bootstrap_rows.append(
            {
                "setting": setting,
                "comparison": "EXP2C-minus-baseline",
                "delta_F1": exp2c_metrics["F1"] - baseline_f1,
                **vs_baseline,
            }
        )
        bootstrap_rows.append(
            {
                "setting": setting,
                "comparison": "EXP2C-minus-EXP2A",
                "delta_F1": exp2c_metrics["F1"] - exp2a_f1,
                **vs_exp2a,
            }
        )
        audits[setting] = {
            "baseline_counts": {
                "TP": int(baseline["GL_Skill_TP"]),
                "FP": int(baseline["GL_Skill_FP"]),
                "FN": int(baseline["GL_Skill_FN"]),
            },
            "exp2a_counts": {
                "TP": int(exp2a["expanded_TP"]),
                "FP": int(exp2a["expanded_FP"]),
                "FN": int(exp2a["expanded_FN"]),
            },
            "exp2c_metrics": exp2c_metrics,
            **audit,
        }

    write_csv(
        OUT / "results.csv",
        result_rows,
        [
            "setting", "baseline_f1", "exp2a_f1", "exp2c_f1",
            "delta_exp2c_vs_baseline", "delta_exp2c_vs_exp2a",
            "paired_ci_low_vs_baseline", "paired_ci_high_vs_baseline",
            "paired_ci_low_vs_exp2a", "paired_ci_high_vs_exp2a",
        ],
    )
    fold_fields = [
        "setting", "subject", "selected_a0", "selected_radius", "selected_tau",
        "evidence_scales", "inner_f1", "outer_tp", "outer_fp", "outer_fn",
    ]
    write_csv(
        OUT / "selected_reference_scale_distribution.csv",
        distribution_rows,
        fold_fields,
    )
    write_csv(
        OUT / "selected_evidence_triplet_distribution.csv",
        sorted(distribution_rows, key=lambda row: (row["evidence_scales"], row["setting"], row["subject"])),
        fold_fields,
    )
    write_csv(
        OUT / "bootstrap.csv",
        bootstrap_rows,
        [
            "setting", "comparison", "delta_F1", "ci95_low", "ci95_high",
            "positive_resample_fraction", "resamples", "seed",
        ],
    )

    protocol = {
        "experiment": "GLSD controlled exploration EXP-2C",
        "name": "Reference-Scale / Evidence-Scale Decoupling",
        "interpretation": "controlled decoupling experiment; not a new model",
        "only_changes": [
            "reference-scale candidate set",
            "fixed three-nearest evidence-scale neighborhood for each reference",
        ],
        "baseline_scales": list(BASELINE_SCALES),
        "reference_scale_candidates": list(REFERENCE_SCALES),
        "evidence_scale_rule": (
            "select a0 plus the three scales with smallest (absolute distance, scale); "
            "reported in ascending scale order"
        ),
        "evidence_scale_map": {
            f"{reference:g}": list(scales)
            for reference, scales in EVIDENCE_SCALE_MAP.items()
        },
        "baseline_configuration_budget": BASELINE_CONFIGURATIONS,
        "exp2a_configuration_budget": 210,
        "exp2c_configuration_budget": len(configs),
        "configuration_budget_formula": "7 reference scales x 3 radii x 10 thresholds = 210",
        "local_aggregation": "median of the three aligned local evidences; missing scale is zero",
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
            "run_exp2c.py": digest(Path(__file__)),
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
    print(json.dumps({"status": "EXP2C_COMPLETE", "results": result_rows}, indent=2))


if __name__ == "__main__":
    main()
