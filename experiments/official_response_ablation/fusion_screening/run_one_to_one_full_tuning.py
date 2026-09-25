"""One-to-one evaluation fork of ME-TST+ full fusion tuning.

Legacy replay runs first without patches. The known failing case is checked
before search. Native and all five methods are then evaluated consistently
under the separately versioned one-to-one protocol, including inner search.
"""
import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import io
from pathlib import Path

import numpy as np
import run_full_fusion_tuning as full
import one_to_one_evaluator as evaluator

shared = full.shared


def configure_threshold_grid(mode):
    if mode == "legacy":
        full.THRESHOLDS = full.LEGACY_THRESHOLDS
    elif mode == "refined":
        full.THRESHOLDS = tuple(i / 100 for i in range(5, 100, 5))
    else:
        raise ValueError(mode)


def configure_structure_grid(mode):
    if mode == "legacy":
        full.SCALES, full.RADII = full.LEGACY_SCALES, full.LEGACY_RADII
    elif mode == "expanded":
        from structure_expansion import EXPANDED_SCALES, EXPANDED_RADII
        full.SCALES, full.RADII = EXPANDED_SCALES, EXPANDED_RADII
    else:
        raise ValueError(mode)


def configure_beta_grid(mode):
    if mode == "all":
        full.BETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    elif mode == "fixed_max":
        full.BETAS = (1.0,)
    else:
        raise ValueError(mode)


def validate(raw, final, gt, label):
    for stage, values in (("raw", raw), ("full", final)):
        if values is None:
            continue
        values = tuple(values)
        if len(values) != 3 or any(not np.isfinite(v) or v < 0 or int(v) != v for v in values):
            raise RuntimeError(f"Invalid {label} {stage} counts: {values}")
        if values[0] + values[2] != gt or values[0] > gt:
            raise RuntimeError(f"GT conservation failed for {label} {stage}: {values}, GT={gt}")


def subject_gt(context, index):
    runner, _, _, _, records, *_ = context
    return sum(len(v) for v in runner.final_samples(records)[index])


def failing_case(context, core, with_recognition):
    runner, _, metric, official, records, _, subjects, _ = context
    si = next(i for i, s in enumerate(subjects) if str(s).lstrip("0") == "37")
    config = next(c for c in full.grids()["L"]
                  if (c.reference_scale, c.local_radius, c.threshold) == (1.0, 2.0, 0.05))
    assert (config.reference_scale, config.local_radius, config.threshold) == (1.0, 2.0, 0.05)
    # Call directly so the legacy negative FN can be recorded, never used for selection.
    with contextlib.redirect_stdout(io.StringIO()):
        decoded = runner.decode_glsd_subject(records, si, config, core, metric, official, with_recognition)
    raw, predictions, pred_list, gt_list = decoded[:4]
    final = runner.full_counts_from_official_synergy(raw, pred_list, gt_list) if with_recognition else None
    return dict(subject=str(subjects[si]), subject_index=si, config=asdict(config),
                GT=subject_gt(context, si), raw=list(raw), full=None if final is None else list(final),
                predictions=shared.serializable(predictions))


def native_counts(ora, context, setting, output):
    subjects = context[6]
    rows, raw_all, full_all = [], [], []
    for i, subject in enumerate(subjects):
        with contextlib.redirect_stdout(io.StringIO()):
            raw, final = ora.native_subject_counts("metst", context, i)
        validate(raw, final, subject_gt(context, i), f"Native {subject}")
        raw_all.append(raw)
        full_all.append(final)
        for stage, counts in (("raw", raw), ("full", final)):
            rows.append(dict(setting=setting, method="Native", subject=subject,
                             metric_stage=stage, **shared.metrics(counts)))
    shared.write_csv(output / "native_per_subject_counts.csv", rows)
    summaries = [dict(setting=setting, method="Native", metric_stage=stage, configs=1,
                      **shared.metrics(np.sum(counts, axis=0)))
                 for stage, counts in (("raw", raw_all), ("full", full_all))]
    shared.write_csv(output / "native_summary_raw_full.csv", summaries)
    return summaries[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("sammlv", "casme3", "both"), default="sammlv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold-grid", choices=("legacy", "refined"), default="legacy")
    parser.add_argument("--structure-grid", choices=("legacy", "expanded"), default="legacy")
    parser.add_argument("--beta-grid", choices=("all", "fixed_max"), default="all")
    parser.add_argument("--mean-selection", choices=("legacy", "near_optimal_mean_synergy_v1"), default="legacy")
    parser.add_argument("--mean-epsilon", type=float, default=None,
                        help="Near-optimal Mean tolerance; defaults to 0.002 for the frozen rule")
    parser.add_argument("--probe-only", action="store_true", help="Legacy replay, failure regression and corrected Native only")
    args = parser.parse_args()
    configure_threshold_grid(args.threshold_grid)
    configure_structure_grid(args.structure_grid)
    configure_beta_grid(args.beta_grid)
    if args.structure_grid == "expanded" and args.threshold_grid != "refined":
        parser.error("Expanded structure requires --threshold-grid refined for the previous-run replay")
    if args.mean_epsilon is not None and (not np.isfinite(args.mean_epsilon) or args.mean_epsilon < 0):
        parser.error("--mean-epsilon must be a finite non-negative number")
    ora, helper_path = shared.load_helper()
    settings = list(shared.SETTINGS.values()) if args.setting == "both" else [shared.SETTINGS[args.setting]]
    for name in settings:
        ora.verify_sealed_inputs(ora.SPECS[name])
    output = (args.output or Path("/content/drive/MyDrive/GLSD_ONE_TO_ONE_FULL_TUNING") /
              datetime.now(timezone.utc).strftime("oto_%Y%m%dT%H%M%S_%fZ")).resolve()
    for name in settings:
        for key in ("dump", "evidence", "run_evidence", "results", "locked_results"):
            protected = ora.SPECS[name][key].resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise RuntimeError(f"Output overlaps input: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(full.__file__), Path(shared.__file__), Path(evaluator.__file__), helper_path]
    if args.structure_grid == "expanded":
        import structure_expansion
        sources.extend([Path(structure_expansion.__file__), structure_expansion.REFERENCE_FILE])
    shared.write_json(output / "run_manifest.json", dict(
        matching_protocol=evaluator.PROTOCOL, probe_only=args.probe_only, settings=settings,
        threshold_grid=args.threshold_grid, thresholds=full.THRESHOLDS,
        structure_grid=args.structure_grid, reference_scales=full.SCALES, local_radii=full.RADII,
        beta_grid=args.beta_grid, betas=full.BETAS,
        support_scales=full.LEGACY_SCALES,
        mean_selection_rule=args.mean_selection,
        mean_selection_epsilon=(
            (full.MEAN_SELECTION_EPSILON if args.mean_epsilon is None else args.mean_epsilon)
            if args.mean_selection == "near_optimal_mean_synergy_v1" else None),
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={p.name: ora.sha256(p) for p in sources},
        note="Legacy replay is provenance only; all new inner and outer counts use one-to-one matching"))
    print("OUTPUT =", output, flush=True)
    print("THRESHOLD_GRID =", args.threshold_grid, full.THRESHOLDS, flush=True)
    print("STRUCTURE_GRID =", args.structure_grid, "a0=", full.SCALES, "rho=", full.RADII, flush=True)
    preflight = output / "legacy_preflight"
    preflight.mkdir()
    replay_rows = [ora.run_preflight_setting(name, preflight) for name in settings]
    shared.write_csv(preflight / "locked_replay_summary.csv", replay_rows)
    print("LEGACY_NATIVE_AND_GLSD_REPLAY = PASS", flush=True)
    all_rows, all_paired = [], []
    for name in settings:
        context = ora.metst_context(ora.SPECS[name])  # Still original evaluator/native gate.
        metric, subjects = context[2], context[6]
        audit = output / (name + "_evaluation_audit")
        audit.mkdir()
        before = None
        if name == "metst_sammlv":
            before = failing_case(context, full.FusionCore(context[1]), False)
            shared.write_json(audit / "subject037_legacy.json", before)
            print("SUBJECT037_LEGACY_RAW =", before["raw"], flush=True)
        with evaluator.install(metric) as info:
            shared.write_json(audit / "matching_protocol.json", info)
            if before is not None:
                after = failing_case(context, full.FusionCore(context[1]), True)
                shared.write_json(audit / "subject037_one_to_one.json", after)
                validate(after["raw"], after["full"], after["GT"], "subject037 regression")
                if before["predictions"] != after["predictions"]:
                    raise RuntimeError("Evaluation patch changed predictions in regression case")
                print("SUBJECT037_ONE_TO_ONE_REGRESSION = PASS; raw=", after["raw"], "full=", after["full"], flush=True)
            native = native_counts(ora, context, name, audit)
            all_rows.append(native)
            print(f"{name} ONE_TO_ONE_NATIVE_FULL_F1 = {native['F1']:.6f}", flush=True)
            if args.probe_only:
                continue
            # Reuse all search, scoring, geometry, subject exclusion and full conversion.
            # Every metric instance, including the official recognition path, uses
            # the same patched class binding throughout this context.
            result, paired = full.run_setting(ora, name, output / name,
                prepared_context=context, evaluation_protocol=evaluator.PROTOCOL,
                expanded_structure=args.structure_grid == "expanded",
                selection_rule=args.mean_selection,
                selection_epsilon=args.mean_epsilon)
            all_rows.extend(result)
            all_paired.extend(paired)
            shared.write_csv(output / "one_to_one_summary_full.csv", all_rows)
            shared.write_csv(output / "paired_comparisons.csv", all_paired)
    if not args.probe_only:
        shared.write_json(output / "decision.json", full.decision(all_rows))
    shared.write_json(output / "completion.json", dict(completed=True, probe_only=args.probe_only,
                      settings=settings, matching_protocol=evaluator.PROTOCOL))
    print("ONE_TO_ONE_PROBE = PASS" if args.probe_only else "ONE_TO_ONE_FULL_TUNING = PASS", flush=True)
    print("PASS means execution, not a positive method result. OUTPUT =", output, flush=True)


if __name__ == "__main__":
    main()
