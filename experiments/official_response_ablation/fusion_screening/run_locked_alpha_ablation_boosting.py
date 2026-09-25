"""Locked Table 3 alpha ablation for the official BoostingVRME responses.

The runner reuses the sealed BoostingVRME Table 1 GLSD-90 outer-fold
configuration for every subject and changes only
``S_alpha = alpha * G + (1 - alpha) * L``.  The alpha=.5 full counts must
match the sealed Table 1 per-subject counts before any result is written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# The sealed BoostingVRME evaluator still calls DataFrame.append.  Colab's
# current pandas removed that method, so install the compatibility alias before
# importing or invoking either dataset-specific official runner.
try:
    import pandas as pd
    if not hasattr(pd.DataFrame, "append"):
        def _dataframe_append(self, other, ignore_index=False,
                              verify_integrity=False, sort=False):
            if isinstance(other, dict):
                other = pd.DataFrame([other])
            elif isinstance(other, pd.Series):
                other = other.to_frame().T
            return pd.concat([self, other], ignore_index=ignore_index,
                             verify_integrity=verify_integrity, sort=sort)
        pd.DataFrame.append = _dataframe_append
except ImportError:
    pass

import official_response_component_ablation as ora


def _load_boosting_adapter(spec: dict):
    """Load the dataset-specific official runner from the package or Colab."""
    return ora.load_module(spec["source"], f"sealed_boosting_runner_{spec['label']}")


class CASWeightedFeature:
    """CAS(ME)3 adapter: preserve the sealed candidate universe and score G/L."""
    def __init__(self, base):
        self.base = base

    def selected_peaks(self, reference_scale, local_radius, threshold, alpha):
        evidence = self.base.scores(float(reference_scale), float(local_radius))
        peaks = np.asarray(evidence["peaks"], dtype=int)
        if not len(peaks):
            return []
        g = np.asarray(evidence["G"], dtype=float)
        l = np.asarray(evidence["L"], dtype=float)
        score = float(alpha) * g + (1.0 - float(alpha)) * l
        return [int(x) for x in peaks[score >= float(threshold)]]


def parse_alphas(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(x) or not 0.0 <= x <= 1.0 for x in values):
        raise argparse.ArgumentTypeError("alphas must be non-empty values in [0, 1]")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("alpha grid contains duplicates")
    return values


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")


@dataclass(frozen=True)
class WeightedConfig:
    method: str
    config_id: int
    reference_scale: float
    local_radius: float
    threshold: float
    alpha: float


def make_weighted_features(base_features):
    class WeightedFeatures(base_features):
        def selected_peaks(self, config):
            peaks, evidence = self.evidence(config.reference_scale, config.local_radius)
            peaks = np.asarray(peaks, dtype=int)
            values = np.asarray(evidence, dtype=float)
            if values.shape != (len(peaks), 2):
                raise RuntimeError("sealed Boosting evidence must expose G and L")
            g, l = values.T
            score = float(config.alpha) * g + (1.0 - float(config.alpha)) * l
            return peaks[score >= float(config.threshold)]

    return WeightedFeatures


class WeightedCore:
    def __init__(self, base_runner):
        base_features = getattr(base_runner, "GLSDFeatures", None)
        if base_features is None or not hasattr(base_features, "evidence"):
            raise RuntimeError("Boosting runner lacks the sealed GLSDFeatures.evidence interface")
        self.GLSDFeatures = make_weighted_features(base_features)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(counts) -> dict:
    tp, fp, fn = (int(value) for value in counts)
    denom = 2 * tp + fp + fn
    return dict(TP=tp, FP=fp, FN=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                F1=2 * tp / denom if denom else 0.0)


def subject_gt(context, index: int) -> int:
    grouped = context[2]
    if "gt_intervals" in grouped[index][0]:
        return sum(len(video["gt_intervals"]) for video in grouped[index])
    return sum(len(video) for video in context[0].final_samples(context[4])[index])


def locked_config_index(config, grid) -> int:
    for index, candidate in enumerate(grid):
        if (float(candidate.reference_scale) == float(config.reference_scale)
                and float(candidate.local_radius) == float(config.local_radius)
                and float(candidate.threshold) == float(config.threshold)):
            return index
    raise RuntimeError("sealed outer configuration is absent from the 90-point grid")


def weighted_config(base_config, alpha: float, config_id: int) -> WeightedConfig:
    return WeightedConfig(
        method="WeightedMean", config_id=int(config_id),
        reference_scale=float(base_config.reference_scale),
        local_radius=float(base_config.local_radius),
        threshold=float(base_config.threshold), alpha=float(alpha),
    )


def validate_counts(counts, gt_count: int, label: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in counts)
    if len(values) != 3 or min(values) < 0 or values[0] + values[2] != gt_count:
        raise RuntimeError(f"GT conservation failed for {label}: {values}, GT={gt_count}")
    return values


def paired_alpha_comparisons(setting: str, counts_by_alpha: dict[float, np.ndarray], subjects: list[str], baseline: float = 0.5) -> list[dict]:
    """Bootstrap full-count F1 differences on the same held subjects."""
    rng = np.random.default_rng(100)
    draws = rng.integers(0, len(subjects), size=(10_000, len(subjects)))

    def distribution(counts):
        totals = counts[draws].sum(axis=1).astype(float)
        denom = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(2 * totals[:, 0], denom, out=np.zeros(len(totals)), where=denom > 0)

    target = distribution(counts_by_alpha[baseline])
    point_target = metrics(counts_by_alpha[baseline].sum(axis=0))["F1"]
    rows = []
    for alpha, counts in counts_by_alpha.items():
        if alpha == baseline:
            continue
        reference = distribution(counts)
        point_reference = metrics(counts.sum(axis=0))["F1"]
        delta = target - reference
        rows.append({
            "setting": setting,
            "target_alpha": baseline, "reference_alpha": alpha,
            "metric_stage": "full", "delta_F1": point_target - point_reference,
            "CI_low": float(np.quantile(delta, 0.025)),
            "CI_high": float(np.quantile(delta, 0.975)),
            "seed": 100, "resamples": 10_000, "unit": "subject",
        })
    return rows


def run_setting(name: str, output: Path, alphas: tuple[float, ...], probe_only: bool) -> None:
    output.mkdir(parents=False, exist_ok=False)
    spec = ora.SPECS[name]
    # Both datasets use the same official response cache contract, but CAS(ME)3
    # has a dedicated runner implementation.  The adapter in the package
    # selects the correct one through SPECS['source'].
    runner_module = _load_boosting_adapter(spec)
    casme3 = name == "boosting_casme3"
    if casme3:
        cache, videos = runner_module.load_cache(spec["cache"])
        grouped = runner_module.group_videos(videos)
        subjects = [str(subject[0]["subject"]) for subject in grouped]
        context = (runner_module, cache, grouped, subjects)
        metric, official = runner_module.tu.MeanAveragePrecision2d, runner_module.tu
    else:
        context = ora.boosting_context(spec)
        runner_module, cache, grouped, subjects, metric, official, _native = context
        subjects = [str(x) for x in subjects]
    actual_counts = (sum(len(group) for group in grouped),
                     sum(len(video["gt_intervals"]) for group in grouped for video in group))
    if actual_counts != (spec["videos"], spec["gt"]):
        raise RuntimeError(f"{spec['label']}: official video/GT counts do not match sealed protocol")
    base_grid = runner_module.build_grid() if casme3 else runner_module.configuration_grid()
    if len(base_grid) != 90:
        raise RuntimeError(f"{spec['label']}: expected a 90-point Table 1 grid")
    with (spec["locked_results"] / "outer_selected_configs.csv").open(newline="", encoding="utf-8-sig") as handle:
        config_rows = list(csv.DictReader(handle))
    if len(config_rows) != len(subjects):
        raise RuntimeError("locked outer_selected_configs.csv has the wrong fold count")
    selected, selected_ids = {}, {}
    for row in config_rows:
        subject = str(row.get("outer_subject", row.get("subject")))
        config_id = int(row["config_id"])
        if subject not in subjects or not 0 <= config_id < len(base_grid):
            raise RuntimeError("locked selected configuration does not match Boosting subjects/grid")
        selected[subject] = base_grid[config_id]
        selected_ids[subject] = config_id
    sealed_full = ora.existing_glsd_subject_counts(spec["run_evidence"] / "per_subject_counts.csv", len(subjects))
    core = None if casme3 else WeightedCore(runner_module)
    protocol = {
        "experiment": "locked_table3_alpha_ablation_boosting_v1",
        "setting": name,
        "formula": "S_alpha = alpha*G + (1-alpha)*L",
        "alphas": list(alphas), "baseline_alpha": 0.5,
        "fixed_source": "Table 1 GLSD-90 outer_selected_configs.csv",
        "configs_per_alpha": len(subjects),
        "selection": "no selection; reuse Table 1 outer-fold configuration for each subject",
        "evaluation": "held-subject full official BoostingVRME recognition/result-synergy counts",
        "matching_protocol": "event_one_to_one_v1_prediction_order",
        "gt_events": spec["gt"],
        "alpha_zero": "L-only / w/o G",
        "alpha_one": "G-only / w/o L",
        "alpha_half_alignment": "required to match sealed Table 1 GLSD-90 full counts",
        "source": str(spec["source"]),
        "source_sha256": ora.sha256(spec["source"]),
        "table1_glsd_full_counts": list(spec["glsd_full"]),
        "cache": str(spec["cache"]),
    }
    write_json(output / "protocol.json", protocol)

    if 0.5 not in alphas:
        raise RuntimeError("alpha grid must include 0.5")

    def decode(index, subject, alpha):
        base_config = selected[subject]
        config = (weighted_config(base_config, alpha, selected_ids[subject])
                  if not casme3 else base_config)
        if casme3:
            # CAS runner's native config is a dict and its decode_subject API
            # accepts the feature bank explicitly.
            feature_bank = [[CASWeightedFeature(runner_module.GLSDFeatures(video["spot_response"], int(cache["config"]["k_p"])))
                             for video in group] for group in grouped]
            config_item = {"reference_scale": float(base_config["reference_scale"]),
                           "local_radius": float(base_config["local_radius"]),
                           "threshold": float(base_config["threshold"]),
                           "alpha": float(alpha)}
            class AlphaFeature:
                def __init__(self, base, a): self.base, self.alpha = base, a
                def selected_peaks(self, ref, rho, tau): return self.base.selected_peaks(ref, rho, tau, self.alpha)
            feature_bank = [[AlphaFeature(feature, alpha) for feature in group] for group in feature_bank]
            final_samples = [[video["gt_intervals"] for video in group] for group in grouped]
            final_emotions = [[video["emotion_labels"] for video in group] for group in grouped]
            decoded = runner_module.decode_subject(index, config_item, grouped, feature_bank,
                                                   final_samples, final_emotions, cache["config"],
                                                   need_recognition=True)
            raw, full = tuple(decoded["raw_counts"]), tuple(decoded["full_counts"])
        else:
            raw, full = ora.decode_component("boosting", context, core, index, config, True)
        gt_count = subject_gt(context, index)
        return (validate_counts(raw, gt_count, f"alpha={alpha}/raw/{subject}"),
                validate_counts(full, gt_count, f"alpha={alpha}/full/{subject}"), config)

    alignment = {}
    for index, subject in enumerate(subjects):
        raw, full, _config = decode(index, subject, 0.5)
        expected = tuple(sealed_full[subject])
        alignment[subject] = {
            "expected_TP": expected[0], "expected_FP": expected[1], "expected_FN": expected[2],
            "actual_TP": full[0], "actual_FP": full[1], "actual_FN": full[2],
            "aligned": tuple(full) == expected,
        }
    mismatches = [subject for subject, row in alignment.items() if not row["aligned"]]
    if mismatches:
        raise RuntimeError(f"alpha=.5 does not reproduce Table 1 for subjects: {mismatches[:10]}")
    write_json(output / "alpha_0p5_alignment.json", {
        "completed": True, "subjects": len(subjects), "mismatches": [],
        "expected_source": str(spec["run_evidence"] / "per_subject_counts.csv"),
    })
    if probe_only:
        write_json(output / "completion.json", {
            "completed": True, "probe_only": True, "setting": name,
            "alignment_alpha": 0.5, "subject_count": len(subjects),
            "matching_protocol": "event_one_to_one_v1_prediction_order",
        })
        return

    start = time.monotonic()
    pooled_rows, subject_rows, selected_rows = [], [], []
    counts_by_alpha = {}
    with gzip.open(output / "selected_predictions.jsonl.gz", "wt", encoding="utf-8") as pred_file:
        for alpha in alphas:
            variant = f"alpha_{alpha_tag(alpha)}"
            raw_rows, full_rows = [], []
            for index, subject in enumerate(subjects):
                raw, full, config = decode(index, subject, alpha)
                raw_rows.append(raw); full_rows.append(full)
                if casme3:
                    reference_scale = config["reference_scale"]
                    local_radius = config["local_radius"]
                    threshold = config["threshold"]
                else:
                    reference_scale = config.reference_scale
                    local_radius = config.local_radius
                    threshold = config.threshold
                selected_rows.append({
                    "setting": name, "variant": variant, "alpha": alpha, "subject": subject,
                    "source_config_id": selected_ids[subject],
                    "reference_scale": reference_scale, "local_radius": local_radius,
                    "threshold": threshold, "selection": "locked_table1_outer_config",
                })
                for stage, counts in (("raw", raw), ("full", full)):
                    subject_rows.append({"setting": name, "variant": variant, "alpha": alpha,
                                         "subject": subject, "metric_stage": stage, **metrics(counts)})
                config_record = ({"method": "WeightedMean", "config_id": selected_ids[subject],
                                  "reference_scale": reference_scale, "local_radius": local_radius,
                                  "threshold": threshold, "alpha": alpha}
                                 if casme3 else asdict(config))
                pred_file.write(json.dumps({
                    "setting": name, "variant": variant, "alpha": alpha,
                    "subject": subject, "config": config_record,
                    "raw_counts": raw, "full_counts": full,
                }, allow_nan=False) + "\n")
            counts_by_alpha[alpha] = np.asarray(full_rows, dtype=np.int64)
            for stage, rows in (("raw", raw_rows), ("full", full_rows)):
                pooled_rows.append({"setting": name, "variant": variant, "alpha": alpha,
                                    "metric_stage": stage, "configs": "locked_per_subject",
                                    **metrics(np.sum(rows, axis=0))})

    if not np.array_equal(counts_by_alpha[0.5], np.asarray([sealed_full[s] for s in subjects], dtype=np.int64)):
        raise RuntimeError("alpha=.5 full counts do not match Table 1 sealed per-subject counts")
    write_csv(output / "alpha_summary.csv", pooled_rows)
    write_csv(output / "alpha_per_subject_counts.csv", subject_rows)
    write_csv(output / "alpha_locked_configs.csv", selected_rows)
    write_csv(output / "alpha_paired_comparisons.csv", paired_alpha_comparisons(name, counts_by_alpha, subjects))
    write_json(output / "completion.json", {
        "completed": True, "probe_only": False, "setting": name,
        "alphas": list(alphas), "baseline_alpha": 0.5, "subject_count": len(subjects),
        "elapsed_seconds": time.monotonic() - start,
        "matching_protocol": "event_one_to_one_v1_prediction_order", "alpha_0p5_aligned": True,
    })


def main() -> None:
    print("LOCKED_ALPHA_BOOSTING_RUNNER = v3_pandas_append_compat", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("boosting_sammlv", "boosting_casme3", "both"), default="boosting_sammlv")
    parser.add_argument("--alphas", type=parse_alphas, default=(0.0, 0.3, 0.5, 0.7, 1.0))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if 0.5 not in args.alphas:
        parser.error("the alpha grid must include 0.5")
    settings = list(ora.SPECS) if args.setting == "both" else [args.setting]
    settings = [name for name in settings if name.startswith("boosting_")]
    for name in settings:
        ora.verify_sealed_inputs(ora.SPECS[name])
    args.output.resolve().mkdir(parents=False, exist_ok=False)
    for name in settings:
        run_setting(name, args.output.resolve() / name, args.alphas, args.probe_only)
    write_json(args.output.resolve() / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "alphas": list(args.alphas), "baseline_alpha": 0.5,
        "mode": "locked_table1_outer_configs", "probe_only": args.probe_only,
    })


if __name__ == "__main__":
    main()
