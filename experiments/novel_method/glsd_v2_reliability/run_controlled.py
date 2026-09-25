"""Stage-1 controlled same-candidate gate for the fixed GLSD-v2 score."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from my_method.gl_saliency_skill.benchmark import GROUPS, load_bundle, signed_context
from my_method.gl_saliency_skill.evaluation import evaluate, metrics
from my_method.gl_saliency_skill.evidence import Config, THRESHOLDS, configuration_grid
from my_method.glsd_v2_reliability.reliability import evidence_and_reliability, v2_score


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT / "results" / "glsd_v2_reliability"
PROTOCOL = OUTPUT_ROOT / "GLSD_V2_CONTROLLED_PROTOCOL.md"
REFERENCE = 2.0
RADIUS = 1.0
METHODS = ("H", "G", "L", "v1", "v2")
COMPARATORS = ("H", "G", "L", "v1")
SEED = 100
RESAMPLES = 10_000
LARGE_DEGRADATION = 0.01
COUNT_KEYS = ("TP", "FP", "FN")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def fixed_v1_pools(configs: list[Config]) -> dict[str, list[int]]:
    families = {"H": "height", "G": "global", "L": "local", "v1": "unified"}
    output = {}
    for method, family in families.items():
        indexes = [
            index
            for index, config in enumerate(configs)
            if config.family == family
            and config.reference == REFERENCE
            and (family not in ("local", "unified") or config.radius == RADIUS)
            and (family != "unified" or config.height_weight == 0.0)
        ]
        if len(indexes) != len(THRESHOLDS):
            raise AssertionError(f"Unexpected {method} fixed-grid size")
        if tuple(configs[index].threshold for index in indexes) != THRESHOLDS:
            raise AssertionError(f"Unexpected {method} threshold order")
        output[method] = indexes
    return output


def shared_candidate_scores(curve, k: int, adapter):
    peaks, evidence, reliability = evidence_and_reliability(curve, k, REFERENCE, RADIUS)
    geometry = adapter.geometry(curve, k, peaks)
    peaks = peaks[geometry.order]
    evidence = evidence[geometry.order]
    reliability = reliability[geometry.order]
    scores = {
        "H": evidence[:, 0],
        "G": evidence[:, 1],
        "L": evidence[:, 2],
        "v1": 0.5 * (evidence[:, 1] + evidence[:, 2]),
        "v2": v2_score(evidence, reliability),
    }
    for name, values in {"G": scores["G"], "L": scores["L"], "A": reliability, "v2": scores["v2"]}.items():
        if np.any(~np.isfinite(values)) or np.any(values < -1e-12) or np.any(values > 1 + 1e-12):
            raise AssertionError(f"{name} range failure in controlled engine")
    return peaks, geometry, scores


def threshold_events(peaks, geometry, scores: np.ndarray, threshold: float) -> list[dict]:
    keep = np.asarray(scores >= threshold, dtype=bool)
    for index in range(len(peaks)):
        if geometry.conflicts[index, index + 1 :].any():
            keep[index + 1 :] &= ~(
                keep[index] & geometry.conflicts[index, index + 1 :]
            )
    return [
        {
            "onset": int(geometry.intervals[index, 0]),
            "peak": int(peaks[index]),
            "offset": int(geometry.intervals[index, 1]),
            "confidence": float(scores[index]),
        }
        for index in range(len(peaks))
        if keep[index]
    ]


def compute_stats(bundle, required_k: list[int]):
    subject_index = {subject: index for index, subject in enumerate(bundle.subjects)}
    stats = {
        k: {
            method: np.zeros((len(THRESHOLDS), len(bundle.subjects), 3), dtype=np.int64)
            for method in METHODS
        }
        for k in required_k
    }
    for k in required_k:
        for record in bundle.records:
            sid = subject_index[record.subject]
            peaks, geometry, scores = shared_candidate_scores(
                record.score, k, bundle.interval_adapter
            )
            for method in METHODS:
                for threshold_id, threshold in enumerate(THRESHOLDS):
                    events = threshold_events(peaks, geometry, scores[method], threshold)
                    counts, _ = evaluate(events, record.ground_truth)
                    stats[k][method][threshold_id, sid] += np.asarray(
                        [counts[key] for key in COUNT_KEYS], dtype=np.int64
                    )
    return stats


def choose_threshold(counts: np.ndarray) -> int:
    tp, fp, fn = counts.astype(float).T
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    return int(np.lexsort((np.arange(len(THRESHOLDS)), fp, -precision, -f1))[0])


def aggregate(values: np.ndarray) -> dict:
    totals = values.sum(axis=0)
    return metrics(dict(zip(COUNT_KEYS, totals)))


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
        "crosses_zero": bool(low <= 0.0 <= high),
        "resamples": RESAMPLES,
        "seed": SEED,
        "paired_unit": "outer subject",
    }


def run_group(backbone: str, dataset: str, configs, pools):
    bundle = load_bundle(backbone, dataset)
    _, protocol, outer, inner, signed_stats, mode = signed_context(bundle, configs)
    required_k = sorted(
        {int(value) for value in outer}
        | {int(value) for value in inner.ravel() if int(value) > 0}
    )
    stats = compute_stats(bundle, required_k)

    for k in required_k:
        required_subjects = (outer == k) | np.any(inner == k, axis=0)
        for method in ("H", "G", "L", "v1"):
            for threshold_id, config_id in enumerate(pools[method]):
                np.testing.assert_array_equal(
                    stats[k][method][threshold_id, required_subjects],
                    signed_stats[k][config_id, required_subjects],
                    err_msg=f"shared candidate replay differs: {backbone}/{dataset}/{k}/{method}",
                )

    selected = {method: [] for method in METHODS}
    subject_counts = {
        method: np.zeros((len(bundle.subjects), 3), dtype=np.int64) for method in METHODS
    }
    selection_rows = []
    for held, subject in enumerate(bundle.subjects):
        for method in METHODS:
            training = np.zeros((len(THRESHOLDS), 3), dtype=np.int64)
            for validation in range(len(bundle.subjects)):
                if validation != held:
                    training += stats[int(inner[held, validation])][method][:, validation]
            threshold_id = choose_threshold(training)
            selected[method].append(threshold_id)
            test = stats[int(outer[held])][method][threshold_id, held]
            subject_counts[method][held] = test
            inner_metric = metrics(dict(zip(COUNT_KEYS, training[threshold_id])))
            outer_metric = metrics(dict(zip(COUNT_KEYS, test)))
            selection_rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "outer_subject": subject,
                    "method": method,
                    "tau_id": threshold_id,
                    "tau": THRESHOLDS[threshold_id],
                    "a0": REFERENCE,
                    "rho": RADIUS,
                    "k_out": int(outer[held]),
                    "inner_tp": inner_metric["TP"],
                    "inner_fp": inner_metric["FP"],
                    "inner_fn": inner_metric["FN"],
                    "inner_f1": inner_metric["F1"],
                    "outer_tp": outer_metric["TP"],
                    "outer_fp": outer_metric["FP"],
                    "outer_fn": outer_metric["FN"],
                    "outer_f1": outer_metric["F1"],
                }
            )

    group_metrics = {method: aggregate(subject_counts[method]) for method in METHODS}
    bootstrap = [
        {
            "backbone": backbone,
            "dataset": dataset,
            "comparison": f"v2-{comparator}",
            **paired_bootstrap(subject_counts["v2"], subject_counts[comparator]),
        }
        for comparator in COMPARATORS
    ]
    return {
        "backbone": backbone,
        "dataset": dataset,
        "subjects": len(bundle.subjects),
        "videos": len(bundle.records),
        "interval_mode": mode,
        "protocol_signature": protocol["signature"],
        "metrics": group_metrics,
        "bootstrap": bootstrap,
        "selection_rows": selection_rows,
    }


def gate_decision(runs: list[dict]) -> dict:
    criterion_a_count = sum(
        run["metrics"]["v2"]["F1"]
        >= max(run["metrics"][method]["F1"] for method in ("H", "G", "L"))
        for run in runs
    )
    v2_v1 = [
        run["metrics"]["v2"]["F1"] - run["metrics"]["v1"]["F1"] for run in runs
    ]
    criterion_b_positive = sum(delta > 0.0 for delta in v2_v1)
    criterion_b_no_large_drop = all(delta > -LARGE_DEGRADATION for delta in v2_v1)
    v1_bootstrap = [
        row
        for run in runs
        for row in run["bootstrap"]
        if row["comparison"] == "v2-v1"
    ]
    criterion_c_positive = sum(row["ci95_low"] > 0.0 for row in v1_bootstrap)
    criterion_c_no_significant_negative = all(row["ci95_high"] >= 0.0 for row in v1_bootstrap)
    passed_a = criterion_a_count >= 3
    passed_b = criterion_b_positive >= 3 and criterion_b_no_large_drop
    passed_c = criterion_c_positive >= 2 and criterion_c_no_significant_negative
    passed = passed_a or passed_b or passed_c
    return {
        "A": {"passed": passed_a, "settings_at_or_above_best_single_score": criterion_a_count},
        "B": {
            "passed": passed_b,
            "positive_v2_minus_v1_settings": criterion_b_positive,
            "no_drop_ge_0.01": criterion_b_no_large_drop,
            "deltas": v2_v1,
        },
        "C": {
            "passed": passed_c,
            "strictly_positive_v2_minus_v1_CIs": criterion_c_positive,
            "no_other_significantly_negative_CI": criterion_c_no_significant_negative,
        },
        "passed": passed,
        "status": "GLSD_V2_CONTROLLED_GATE_PASSED" if passed else "GLSD_V2_CONTROLLED_GATE_FAILED",
    }


def render_report(runs: list[dict], gate: dict) -> str:
    labels = {
        ("metst", "sammlv"): "ME-TST+ / SAMMLV",
        ("metst", "casme3"): "ME-TST+ / CAS(ME)3",
        ("boostingvrme", "sammlv"): "BoostingVRME / SAMMLV",
        ("boostingvrme", "casme3"): "BoostingVRME / CAS(ME)3",
    }
    lines = [
        "# GLSD-v2 Controlled Same-Candidate Gate",
        "",
        f"Final status: `{gate['status']}`",
        "",
        "| Setting | H | G | L | GL-v1 | GLSD-v2 | v2-H | v2-G | v2-L | v2-v1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        m = run["metrics"]
        lines.append(
            f"| {labels[(run['backbone'],run['dataset'])]} | {m['H']['F1']:.6f} | "
            f"{m['G']['F1']:.6f} | {m['L']['F1']:.6f} | {m['v1']['F1']:.6f} | "
            f"{m['v2']['F1']:.6f} | {m['v2']['F1']-m['H']['F1']:+.6f} | "
            f"{m['v2']['F1']-m['G']['F1']:+.6f} | {m['v2']['F1']-m['L']['F1']:+.6f} | "
            f"{m['v2']['F1']-m['v1']['F1']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate evaluation",
            "",
            f"- A: {gate['A']['settings_at_or_above_best_single_score']}/4 at or above max(H,G,L): "
            f"`{'PASS' if gate['A']['passed'] else 'FAIL'}`.",
            f"- B: {gate['B']['positive_v2_minus_v1_settings']}/4 positive vs v1; "
            f"no ≥0.01 degradation={gate['B']['no_drop_ge_0.01']}: "
            f"`{'PASS' if gate['B']['passed'] else 'FAIL'}`.",
            f"- C: {gate['C']['strictly_positive_v2_minus_v1_CIs']} strictly positive v2-v1 CIs; "
            f"no significant-negative remainder={gate['C']['no_other_significantly_negative_CI']}: "
            f"`{'PASS' if gate['C']['passed'] else 'FAIL'}`.",
            "",
            "All H/G/L/v1 counts from the shared candidate engine exactly matched the signed historical stats. "
            "A failed gate terminates the study before full Nested LOSO and transfer.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not PROTOCOL.exists():
        raise AssertionError("Controlled protocol must be frozen before execution")
    configs = configuration_grid()
    pools = fixed_v1_pools(configs)
    runs = [run_group(backbone, dataset, configs, pools) for backbone, dataset in GROUPS]
    gate = gate_decision(runs)
    summary_rows = []
    for run in runs:
        row = {"backbone": run["backbone"], "dataset": run["dataset"]}
        for method in METHODS:
            metric = run["metrics"][method]
            for key in COUNT_KEYS:
                row[f"{method}_{key.lower()}"] = metric[key]
            row[f"{method}_f1"] = metric["F1"]
        for comparator in COMPARATORS:
            row[f"v2_minus_{comparator}"] = row["v2_f1"] - row[f"{comparator}_f1"]
        summary_rows.append(row)
    write_csv(OUTPUT_ROOT / "controlled_summary.csv", summary_rows)
    write_csv(
        OUTPUT_ROOT / "controlled_outer_selections.csv",
        [row for run in runs for row in run["selection_rows"]],
    )
    write_json(
        OUTPUT_ROOT / "controlled_bootstrap.json",
        {
            "protocol": {"resamples": RESAMPLES, "seed": SEED, "paired_unit": "outer subject"},
            "results": [row for run in runs for row in run["bootstrap"]],
        },
    )
    write_json(OUTPUT_ROOT / "controlled_gate.json", gate)
    (OUTPUT_ROOT / "CONTROLLED_GATE_REPORT_CN.md").write_text(
        render_report(runs, gate), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
