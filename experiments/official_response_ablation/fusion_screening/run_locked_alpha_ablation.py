"""Locked alpha sweep for the Table 3 sensitivity/component analysis.

This runner reuses the exact per-subject outer configurations stored with the
Table 1 GLSD-90 evidence.  It changes only the score weight in
S_alpha = alpha*G + (1-alpha)*L.  In particular, alpha=0.5 is replayed and
must match the locked Table 1 full counts before the other alpha values are
accepted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# The archived ME-TST evaluator still calls DataFrame.append.  Colab's
# current pandas removed it, so install the same compatibility alias used by
# the original runner before constructing any ME-TST context.
try:
    import pandas as pd
    if not hasattr(pd.DataFrame, "append"):
        def _dataframe_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
            return pd.concat([self, other], ignore_index=ignore_index,
                             verify_integrity=verify_integrity, sort=sort)
        pd.DataFrame.append = _dataframe_append
except ImportError:
    pass

import one_to_one_evaluator as evaluator
import run_alpha_hparam_ablation as nested
import run_full_fusion_tuning as full

shared = full.shared


def parse_alphas(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(x) or not 0.0 <= x <= 1.0 for x in values):
        raise argparse.ArgumentTypeError("alphas must be non-empty values in [0, 1]")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("alpha grid contains duplicates")
    return values


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def locked_weighted_config(locked, alpha: float, config_id: int):
    """Convert a sealed Table 1 structure config to a weighted config."""
    return full.Config(
        method="WeightedMean",
        config_id=int(config_id),
        reference_scale=float(locked.reference_scale),
        local_radius=float(locked.local_radius),
        threshold=float(locked.threshold),
        alpha=float(alpha),
        beta=None,
    )


def locked_config_index(locked, grid) -> int:
    """Find the sealed grid index without relying on a source-specific ID field."""
    for index, candidate in enumerate(grid):
        if (
            float(candidate.reference_scale) == float(locked.reference_scale)
            and float(candidate.local_radius) == float(locked.local_radius)
            and float(candidate.threshold) == float(locked.threshold)
        ):
            return index
    raise RuntimeError(
        "locked outer configuration is absent from the 90-point grid: "
        f"a0={locked.reference_scale}, rho={locked.local_radius}, tau={locked.threshold}"
    )


def validate_counts(counts, gt_count: int, label: str):
    values = tuple(int(x) for x in counts)
    if len(values) != 3 or min(values) < 0 or values[0] + values[2] != gt_count:
        raise RuntimeError(f"GT conservation failed for {label}: {values}, GT={gt_count}")
    return values


def subject_gt(context, index: int) -> int:
    runner, _, _, _, records, *_ = context
    return sum(len(video) for video in runner.final_samples(records)[index])


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    shared.write_csv(path, rows)


def run_setting(ora, name: str, output: Path, alphas: tuple[float, ...], *, prepared_context=None,
                probe_only=False):
    output.mkdir(parents=False, exist_ok=False)
    spec = ora.SPECS[name]
    context = ora.metst_context(spec) if prepared_context is None else prepared_context
    _, base, _, _, _, _, subjects, _ = context
    if ora.context_video_and_gt_counts("metst", context) != (spec["videos"], spec["gt"]):
        raise RuntimeError("unexpected official video/GT counts")

    full.SCALES = full.LEGACY_SCALES
    full.RADII = full.LEGACY_RADII
    full.THRESHOLDS = full.LEGACY_THRESHOLDS
    full.ALPHAS = tuple(alphas)
    core = full.FusionCore(base)
    locked_grid = ora.locked_grid(base)
    if len(locked_grid) != 90:
        raise RuntimeError(f"Table 1 locked grid has {len(locked_grid)} rows, expected 90")
    selected = ora.selected_locked_configs(
        spec["locked_results"] / "outer_selected_configs.csv", locked_grid, subjects
    )
    selected_ids = {subject: locked_config_index(selected[subject], locked_grid) for subject in subjects}
    sealed_full = ora.existing_glsd_subject_counts(
        spec["run_evidence"] / "per_subject_counts.csv", len(subjects)
    )
    protocol = dict(
        experiment="locked_table3_alpha_ablation_v1",
        setting=name,
        formula="S_alpha = alpha*G + (1-alpha)*L",
        alphas=list(alphas),
        baseline_alpha=0.5,
        fixed_source="Table 1 GLSD-90 outer_selected_configs.csv",
        configs_per_alpha=len(subjects),
        selection="no selection in this run; reuse the Table 1 outer-fold configuration for each subject",
        evaluation="held-subject full official counts using unchanged official recognition/result synergy",
        matching_protocol=evaluator.PROTOCOL,
        gt_events=spec["gt"],
        alpha_zero="L-only / w/o G at the locked Table 1 configuration",
        alpha_one="G-only / w/o L at the locked Table 1 configuration",
        alpha_half_alignment="required to match sealed Table 1 GLSD-90 full counts",
        source=str(spec["source"]), source_sha256=ora.sha256(spec["source"]),
        core_sha256=ora.sha256(spec["core"]),
    )
    shared.write_json(output / "protocol.json", protocol)

    if probe_only:
        # Probe all held subjects at alpha=.5 so alignment is checked globally,
        # rather than on a single arbitrary subject.
        if 0.5 not in alphas:
            raise RuntimeError("alpha grid must include 0.5")
        actual = {}
        for index, subject in enumerate(subjects):
            gt_count = subject_gt(context, index)
            config = locked_weighted_config(selected[subject], 0.5, selected_ids[subject])
            raw, full_counts, _, _ = full.decode_with_context(
                context, core, index, subject, config, True, name, "alpha_0p5_probe"
            )
            validate_counts(full_counts, gt_count, f"alpha=.5/{subject}")
            expected = tuple(sealed_full[subject])
            actual[subject] = dict(
                expected_TP=expected[0], expected_FP=expected[1], expected_FN=expected[2],
                actual_TP=full_counts[0], actual_FP=full_counts[1], actual_FN=full_counts[2],
                aligned=tuple(full_counts) == expected,
            )
        mismatches = [s for s, row in actual.items() if not row["aligned"]]
        if mismatches:
            raise RuntimeError(f"alpha=.5 does not reproduce Table 1 for subjects: {mismatches[:10]}")
        shared.write_json(output / "alpha_0p5_alignment.json", {
            "completed": True, "subjects": len(subjects), "mismatches": [],
            "expected_source": str(spec["run_evidence"] / "per_subject_counts.csv"),
        })
        shared.write_json(output / "completion.json", {
            "completed": True, "probe_only": True, "setting": name,
            "alignment_alpha": 0.5, "subject_count": len(subjects),
            "matching_protocol": evaluator.PROTOCOL,
        })
        return [], []

    pooled_rows, subject_rows, selected_rows = [], [], []
    counts_by_alpha = {}
    start = time.monotonic()
    with gzip.open(output / "selected_predictions.jsonl.gz", "wt", encoding="utf-8") as pred_file:
        for alpha in alphas:
            variant = f"alpha_{alpha_tag(alpha)}"
            raw_rows, full_rows = [], []
            for index, subject in enumerate(subjects):
                gt_count = subject_gt(context, index)
                base_config = selected[subject]
                config = locked_weighted_config(base_config, alpha, selected_ids[subject])
                raw, full_counts, predictions, traces = full.decode_with_context(
                    context, core, index, subject, config, True, name, variant
                )
                raw = validate_counts(raw, gt_count, f"{variant}/raw/{subject}")
                full_counts = validate_counts(full_counts, gt_count, f"{variant}/full/{subject}")
                raw_rows.append(raw)
                full_rows.append(full_counts)
                selected_rows.append(dict(
                    setting=name, variant=variant, alpha=alpha, subject=subject,
                    source_config_id=selected_ids[subject],
                    reference_scale=base_config.reference_scale,
                    local_radius=base_config.local_radius,
                    threshold=base_config.threshold,
                    selection="locked_table1_outer_config",
                ))
                for stage, counts in (("raw", raw), ("full", full_counts)):
                    subject_rows.append(dict(
                        setting=name, variant=variant, alpha=alpha, subject=subject,
                        metric_stage=stage, **shared.metrics(counts),
                    ))
                pred_file.write(json.dumps(shared.serializable(dict(
                    setting=name, variant=variant, alpha=alpha, subject=subject,
                    config=asdict(config), raw_counts=raw, full_counts=full_counts,
                    predictions=predictions,
                )), allow_nan=False) + "\n")
            counts_by_alpha[alpha] = np.asarray(full_rows, dtype=np.int64)
            for stage, rows in (("raw", raw_rows), ("full", full_rows)):
                pooled_rows.append(dict(
                    setting=name, variant=variant, alpha=alpha,
                    metric_stage=stage, configs="locked_per_subject", **shared.metrics(np.sum(rows, axis=0)),
                ))

    # This is the decisive alignment gate for the full run.
    baseline = counts_by_alpha[0.5]
    expected = np.asarray([sealed_full[s] for s in subjects], dtype=np.int64)
    if not np.array_equal(baseline, expected):
        raise RuntimeError("alpha=.5 full counts do not match Table 1 sealed per-subject counts")
    shared.write_json(output / "alpha_0p5_alignment.json", {
        "completed": True, "subjects": len(subjects), "mismatches": [],
        "expected_source": str(spec["run_evidence"] / "per_subject_counts.csv"),
    })
    comparisons = nested.paired_alpha_comparisons(counts_by_alpha, subjects, 0.5)
    write_csv(output / "alpha_summary.csv", pooled_rows)
    write_csv(output / "alpha_per_subject_counts.csv", subject_rows)
    write_csv(output / "alpha_locked_configs.csv", selected_rows)
    write_csv(output / "alpha_paired_comparisons.csv", comparisons)
    shared.write_json(output / "completion.json", {
        "completed": True, "probe_only": False, "setting": name,
        "alphas": list(alphas), "baseline_alpha": 0.5,
        "subject_count": len(subjects), "elapsed_seconds": time.monotonic() - start,
        "matching_protocol": evaluator.PROTOCOL, "alpha_0p5_aligned": True,
    })
    return [row for row in pooled_rows if row["metric_stage"] == "full"], comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("sammlv", "casme3", "both"), default="sammlv")
    parser.add_argument("--alphas", type=parse_alphas, default=(0.0, 0.3, 0.5, 0.7, 1.0))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if 0.5 not in args.alphas:
        parser.error("the alpha grid must include 0.5")
    ora, helper_path = shared.load_helper()
    selected = list(shared.SETTINGS.values()) if args.setting == "both" else [shared.SETTINGS[args.setting]]
    for name in selected:
        ora.verify_sealed_inputs(ora.SPECS[name])
    output = (args.output or Path("/content/drive/MyDrive/GLSD_LOCKED_TABLE3_ALPHA") /
              datetime.now(timezone.utc).strftime("alpha_%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=False, exist_ok=False)
    inventory = {name: [dict(path=str(p), exists=p.exists())
                        for p in ora.required_paths(ora.SPECS[name])] for name in selected}
    shared.write_json(output / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "settings": selected,
        "alphas": list(args.alphas), "baseline_alpha": 0.5,
        "runner_sha256": ora.sha256(Path(__file__)),
        "evaluator_sha256": ora.sha256(Path(evaluator.__file__)),
        "helper_sha256": ora.sha256(helper_path), "input_inventory": inventory,
        "mode": "locked_table1_outer_configs",
    })
    all_rows, all_comparisons = [], []
    for name in selected:
        context = ora.metst_context(ora.SPECS[name])
        audit = output / f"{name}_evaluation_audit"
        audit.mkdir()
        with evaluator.install(context[2]) as info:
            shared.write_json(audit / "matching_protocol.json", info)
            rows, comparisons = run_setting(
                ora, name, output / name, tuple(args.alphas),
                prepared_context=context, probe_only=args.probe_only,
            )
        all_rows.extend(rows)
        all_comparisons.extend(comparisons)
    if not args.probe_only:
        write_csv(output / "alpha_summary_full.csv", all_rows)
        write_csv(output / "alpha_paired_comparisons.csv", all_comparisons)
    shared.write_json(output / "completion.json", {
        "completed": True, "probe_only": args.probe_only, "settings": selected,
        "alphas": list(args.alphas), "baseline_alpha": 0.5,
        "matching_protocol": evaluator.PROTOCOL,
    })
    print("LOCKED_TABLE3_ALPHA_PROBE = PASS" if args.probe_only else "LOCKED_TABLE3_ALPHA = PASS")
    print("OUTPUT =", output)


if __name__ == "__main__":
    main()
