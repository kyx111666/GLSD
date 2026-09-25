#!/usr/bin/env python3
"""One locked peak-relative morphology refinement on the SAMMLV frozen cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIRST3 = ROOT / "my_method/first3_candidate_recovery"
LEGACY = ROOT / "my_method/multi_scale_candidate_rescue"
RGR = ROOT / "my_method/rgr1_sammlv"
PSED = ROOT / "my_method/psed_feasibility"
MORPH = ROOT / "my_method/temporal_morphology_feasibility"
sys.path[:0] = [str(FIRST3), str(LEGACY), str(RGR), str(PSED), str(MORPH)]

from run_first3_nested import (  # noqa: E402
    add_counts, author_native, empty_counts, evaluate_subset, f1, key,
    label_training, metrics, model_predict, model_fit, union,
)
from run_mscr_nested_loso import (  # noqa: E402
    K_P, EPS, evaluate_decoding, match_events, sha256, threshold, tuned_native_decode,
)
from run_rgr1_nested_loso import decode_low  # noqa: E402
from run_morphology_audit import extract_patch, resample_patch  # noqa: E402

SEED = 20260902
DELTAS = (0.15, 0.20, 0.25, 0.30)
L_RATIOS = (0.25, 0.50, 0.75, 1.00)
CS = (0.1, 1.0, 10.0)
GAMMAS = (0.60, 0.70, 0.80, 0.90)
CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
NATIVE_REPORT = ROOT / "results/rgr1_sammlv_nested/report.json"
V1_REPORT = ROOT / "results/first3_candidate_recovery_nested/outputs/first3_summary.json"
V1_OUT = ROOT / "results/first3_candidate_recovery_nested/outputs"
CACHE_HASH = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
FIELDS = ("TP", "FP", "FN", "event_count", "Precision", "Recall", "F1",
          "weak_candidate_count", "rescued_candidate_count", "rescue_TP",
          "rescue_FP", "native_missed_GT_recovered", "rescue_precision")
CTX = None


def grid():
    return [
        {"Delta_p": d, "requested_L_ratio": ratio,
         "actual_integer_radius": max(1, int(round(ratio * K_P))),
         "C": c, "gamma": gamma}
        for d, ratio, c, gamma in itertools.product(DELTAS, L_RATIOS, CS, GAMMAS)
    ]


def rank(counts, config):
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    score = Fraction(2 * counts["TP"], denominator) if denominator else Fraction(0)
    return (-score, counts["FP"], counts["rescued_candidate_count"], config["Delta_p"],
            -config["gamma"], config["requested_L_ratio"], config["actual_integer_radius"],
            config["C"])


def patch_features(curve, peak, height, radius):
    patch = resample_patch(extract_patch(curve, int(peak), int(radius)))
    patch = patch - patch[15]
    patch = patch / (float(np.linalg.norm(patch)) + EPS)
    result = np.concatenate(([float(height)], patch)).astype(float)
    assert result.shape == (32,) and np.isfinite(result).all()
    return result


def native_config(config):
    return {"c_s": config["c_s"], "p": config["p_s"],
            "c_d": config["c_d"], "c_b": config["c_b"]}


def build_pool(record, config):
    """Build all weak pools from score only; GT and logits are absent here."""
    score = np.asarray(record["score"], dtype=float)
    signal = {"score": score, "R2": np.zeros(len(score), dtype=float)}
    decoded = {d: decode_low(signal, config, d) for d in DELTAS}
    high = decoded[DELTAS[0]]["native"]["events"]
    curve = decoded[DELTAS[0]]["native"]["curve"]
    pools = {}
    for delta, result in decoded.items():
        assert result["native"]["events"] == high
        weak = result["rescued"]
        tau_low = threshold(curve, result["p_w"])
        tau_high = float(decoded[DELTAS[0]]["native"]["threshold"])
        heights = np.asarray([
            np.clip((curve[event["peak"]] - tau_low) /
                    (tau_high - tau_low + EPS), 0.0, 1.0)
            for event in weak
        ], dtype=float)
        patches = {
            radius: np.asarray([
                patch_features(curve, event["peak"], height, radius)
                for event, height in zip(weak, heights)
            ], dtype=float).reshape(-1, 32)
            for radius in sorted({x["actual_integer_radius"] for x in grid()})
        }
        pools[delta] = {"events": weak, "H": heights, "patches": patches,
                        "low_events": result["events"]}
    return {"high": high, "curve": curve, "pools": pools}


def build_banks(records, folds):
    banks = {}
    for fold in folds.values():
        config = fold["selected_strong_config"]
        config_key = key(config)
        if config_key in banks:
            continue
        banks[config_key] = {}
        for record in records:
            rid = (str(record["subject"]), str(record["video"]))
            banks[config_key][rid] = build_pool(record, config)
    return banks


def init_worker(context):
    global CTX
    CTX = context
    threadpool_limits(limits=1)
    warnings.simplefilter("error", ConvergenceWarning)


def fit(x, y, c):
    assert x.ndim == 2 and x.shape[1] == 32 and len(np.unique(y)) == 2
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                               max_iter=2000, random_state=SEED)
    model.fit(scaler.transform(x), y)
    assert int(max(model.n_iter_)) < 2000
    return scaler, model


def fit_predict_inner(x, y, owner, subjects, held, c):
    oof = np.full(len(y), np.nan, dtype=float)
    audit = []
    for validation in subjects:
        train_mask = owner != validation
        validation_mask = ~train_mask
        assert held not in set(owner[train_mask])
        assert validation not in set(owner[train_mask])
        no_validation_candidates = not validation_mask.any()
        single_class = len(np.unique(y[train_mask])) < 2
        if no_validation_candidates:
            # A subject with no weak candidates contributes no OOF rows.  Do
            # not call sklearn on an empty validation matrix; this is a
            # deterministic structural case and does not involve labels.
            iterations = -1
        elif single_class:
            probability = float(y[train_mask][0]) if train_mask.any() else 0.0
            oof[validation_mask] = probability
            iterations = 0
        else:
            scaler, model = fit(x[train_mask], y[train_mask], c)
            oof[validation_mask] = model.predict_proba(
                scaler.transform(x[validation_mask])
            )[:, 1]
            iterations = int(max(model.n_iter_))
        audit.append({
            "outer_subject": held, "inner_validation_subject": validation,
            "inner_training_subjects": ";".join(s for s in subjects if s != validation),
            "train_candidate_count": int(train_mask.sum()),
            "train_positive_count": int(y[train_mask].sum()),
            "validation_candidate_count": int(validation_mask.sum()),
            "max_iterations_used": iterations, "single_class_constant": single_class,
            "no_validation_candidates": no_validation_candidates,
            "train_x_sha256": hashlib.sha256(x[train_mask].tobytes()).hexdigest(),
            "train_y_sha256": hashlib.sha256(y[train_mask].tobytes()).hexdigest(),
        })
    assert np.isfinite(oof).all()
    return oof, audit


def worker(held):
    context = CTX
    fold = context["folds"][held]
    bank = context["banks"][key(fold["selected_strong_config"])]
    train = [r for r in context["records"] if str(r["subject"]) != held]
    test = [r for r in context["records"] if str(r["subject"]) == held]
    subjects = sorted({str(r["subject"]) for r in train})
    assert set(subjects) == set(context["subjects"]) - {held}
    assert len(subjects) == 28
    configs = grid()
    scored = []
    inner_rows = []
    fit_audit = []
    labels_by_delta = {}
    training_ids = [(str(r["subject"]), str(r["video"])) for r in train]
    probability_bank = {}
    for delta in DELTAS:
        labels = label_training(train, bank, delta)
        labels_by_delta[delta] = labels
        y = np.concatenate([labels[rid] for rid in training_ids])
        owner = np.concatenate([np.repeat(rid[0], len(labels[rid])) for rid in training_ids])
        offsets = np.cumsum([0] + [len(labels[rid]) for rid in training_ids])
        for radius in sorted({x["actual_integer_radius"] for x in configs}):
            for c in CS:
                x = np.concatenate([bank[rid]["pools"][delta]["patches"][radius]
                                    for rid in training_ids])
                oof, audits = fit_predict_inner(x, y, owner, subjects, held, c)
                for audit_row in audits:
                    audit_row.update({"Delta_p": delta, "actual_integer_radius": radius,
                                      "C": c})
                fit_audit.extend(audits)
                probabilities = {
                    rid: oof[offsets[i]:offsets[i + 1]]
                    for i, rid in enumerate(training_ids)
                }
                probability_bank[(delta, radius, c)] = probabilities
                for gamma in GAMMAS:
                    config = {"Delta_p": delta, "requested_L_ratio": next(
                        ratio for ratio in L_RATIOS
                        if max(1, int(round(ratio * K_P))) == radius
                    ), "actual_integer_radius": radius, "C": c, "gamma": gamma}
                    counts, _ = evaluate_subset(
                        train, bank, "Morphology Rescue", config, probabilities
                    )
                    scored.append((rank(counts, config), config, counts))
    _, selected, inner = min(scored, key=lambda row: row[0])
    delta, radius, c = (selected["Delta_p"], selected["actual_integer_radius"], selected["C"])
    labels = labels_by_delta[delta]
    x_train = np.concatenate([bank[rid]["pools"][delta]["patches"][radius]
                              for rid in training_ids])
    y_train = np.concatenate([labels[rid] for rid in training_ids])
    scaler, model = fit(x_train, y_train, c)
    test_probabilities = {}
    for record in test:
        rid = (held, str(record["video"]))
        test_x = bank[rid]["pools"][delta]["patches"][radius]
        test_probabilities[rid] = (
            model.predict_proba(scaler.transform(test_x))[:, 1]
            if len(test_x) else np.empty(0, dtype=float)
        )
    outer, predictions = evaluate_subset(
        test, bank, "Morphology Rescue", selected, test_probabilities, trace=True
    )
    weak_trace = []
    for record in test:
        rid = (held, str(record["video"]))
        pool = bank[rid]["pools"][delta]
        values = test_probabilities[rid]
        prediction_matches = {
            (row["video"], row["peak"]): row["matched_gt"]
            for row in predictions
        }
        for event, value in zip(pool["events"], values):
            weak_trace.append({
                "outer_subject": held, "subject": held, "video": rid[1],
                "candidate_id": f"{held}/{rid[1]}/weak_peak_{event['peak']}",
                "source": "rescue", "onset": event["onset"], "peak": event["peak"],
                "offset": event["offset"], "Delta_p": delta,
                "actual_integer_radius": radius, "C": c, "gamma": selected["gamma"],
                "evidence": float(value), "selected": bool(value >= selected["gamma"]),
                "matched_gt_posthoc": prediction_matches.get((rid[1], event["peak"]), -1),
            })
    selected_oof = []
    oof = probability_bank[(delta, radius, c)]
    for rid in training_ids:
        for event, label, probability in zip(
            bank[rid]["pools"][delta]["events"], labels[rid], oof[rid]
        ):
            selected_oof.append({
                "outer_subject": held, "validation_subject": rid[0], "video": rid[1],
                "weak_peak": event["peak"], "training_label_for_audit": int(label),
                "oof_probability": float(probability), "Delta_p": delta,
                "actual_integer_radius": radius, "C": c,
            })
    model_state = {
        "outer_subject": held, "train_subjects": subjects, "selected_config": selected,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(), "train_candidates": len(y_train),
        "train_positive": int(y_train.sum()), "max_iter_used": int(max(model.n_iter_)),
    }
    for row in inner_rows:
        row["outer_subject"] = held
    inner_rows = [{"outer_subject": held, "config_json": key(config),
                   **{k: metrics(counts)[k] for k in FIELDS},
                   "selected": config == selected}
                  for _, config, counts in scored]
    return {
        "outer_subject": held, "selected_config": selected, "inner_metrics": metrics(inner),
        "outer_metrics": metrics(outer), "train_subjects": subjects,
        "inner_rows": inner_rows, "fit_audit": fit_audit,
        "predictions": predictions, "weak_trace": weak_trace,
        "selected_oof": selected_oof, "model_state": model_state,
    }


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def baseline(records, folds):
    totals = {"Author Native": empty_counts(), "Tuned Native": empty_counts()}
    by_subject = {}
    for subject, fold in folds.items():
        by_subject[subject] = {}
        config = native_config(fold["selected_strong_config"])
        for method in totals:
            counts = empty_counts()
            for record in records:
                if str(record["subject"]) != subject:
                    continue
                decoded = (author_native({"score": record["score"]}) if method == "Author Native"
                           else tuned_native_decode({"score": record["score"]}, config))
                add_counts(counts, evaluate_decoding(record, decoded))
            by_subject[subject][method] = metrics(counts)
            add_counts(totals[method], counts)
    expected = {"Author Native": (53, 184, 106), "Tuned Native": (49, 143, 110)}
    for method, anchor in expected.items():
        if tuple(totals[method][k] for k in ("TP", "FP", "FN")) != anchor:
            raise RuntimeError("BLOCKED-BASELINE-MISMATCH")
    return {m: metrics(c) for m, c in totals.items()}, by_subject


def f1_from_array(array):
    array = np.asarray(array, dtype=float)
    den = 2 * array[..., 0] + array[..., 1] + array[..., 2]
    return np.divide(2 * array[..., 0], den, out=np.zeros_like(den), where=den != 0)


def finish(results, anchors, baseline_subjects, v1_subjects, subjects, output, metadata):
    out = output / "outputs"
    method_rows = {"Author Native": [], "Tuned Native": [], "Morphology v1": [],
                   "Morphology Refined": []}
    for subject in subjects:
        method_rows["Author Native"].append(baseline_subjects[subject]["Author Native"])
        method_rows["Tuned Native"].append(baseline_subjects[subject]["Tuned Native"])
        method_rows["Morphology v1"].append(v1_subjects[subject])
        method_rows["Morphology Refined"].append(results[subject]["outer_metrics"])
    summary = []
    aggregate = {}
    for method, rows in method_rows.items():
        counts = empty_counts()
        for row in rows:
            add_counts(counts, row)
        aggregate[method] = metrics(counts)
        summary.append({"Method": method, **{k: metrics(counts)[k] for k in FIELDS}})
    tuned_array = np.asarray([[baseline_subjects[s]["Tuned Native"][k] for k in ("TP", "FP", "FN")] for s in subjects])
    v1_array = np.asarray([[v1_subjects[s][k] for k in ("TP", "FP", "FN")] for s in subjects])
    refined_array = np.asarray([[results[s]["outer_metrics"][k] for k in ("TP", "FP", "FN")] for s in subjects])
    delta_rows = []
    stability = {}
    for reference, array in (("TunedNative", tuned_array), ("Morphology-v1", v1_array)):
        deltas = f1_from_array(refined_array) - f1_from_array(array)
        leaveout = f1_from_array(refined_array.sum(axis=0) - refined_array) - f1_from_array(array.sum(axis=0) - array)
        stability[reference] = {
            "improved": int(np.sum(deltas > 0)), "equal": int(np.sum(deltas == 0)),
            "worse": int(np.sum(deltas < 0)),
            "leave_one_subject_out_aggregate_delta": dict(zip(subjects, map(float, leaveout))),
            "minimum_leave_one_subject_out_delta": float(np.min(leaveout)),
            "gain_survives_every_subject_removal": bool(np.all(leaveout > 0)),
        }
        for subject, delta in zip(subjects, deltas):
            delta_rows.append({"outer_subject": subject, "comparison": reference,
                               "delta_F1": float(delta)})
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(subjects), size=(1000, len(subjects)))
    bootstrap = {}
    tuned_boot = f1_from_array(tuned_array[indices].sum(axis=1))
    v1_boot = f1_from_array(v1_array[indices].sum(axis=1))
    refined_boot = f1_from_array(refined_array[indices].sum(axis=1))
    for label, values in (("Refined_minus_TunedNative", refined_boot - tuned_boot),
                          ("Refined_minus_Morphology_v1", refined_boot - v1_boot)):
        bootstrap[label] = {"repetitions": 1000, "seed": SEED,
                            "mean_delta": float(np.mean(values)),
                            "ci95": list(map(float, np.percentile(values, [2.5, 97.5])))}
    dump(out / "morph_refined_bootstrap.json", {"subjects": subjects,
                                                  "indices": indices.tolist(), **bootstrap})
    selected = [results[s]["selected_config"] for s in subjects]
    parameter_stability = {"requested_L_ratio_to_radius": {
        str(ratio): max(1, int(round(ratio * K_P))) for ratio in L_RATIOS}}
    for field in ("Delta_p", "actual_integer_radius", "C", "gamma"):
        frequency = Counter(str(config[field]) for config in selected)
        possible = sorted({str(config[field]) for config in grid()})
        endpoint_count = sum(frequency.get(value, 0) for value in (possible[0], possible[-1]))
        endpoint_max = max(frequency.get(value, 0) for value in (possible[0], possible[-1]))
        parameter_stability[field] = {
            "frequency": dict(sorted(frequency.items())),
            "endpoint_frequency_total": int(endpoint_count),
            "maximum_single_endpoint_frequency": int(endpoint_max),
            "status": "BOUNDARY-SELECTION-REMAINS" if endpoint_max > len(subjects) / 2 else None,
        }
    refined = aggregate["Morphology Refined"]
    v1 = aggregate["Morphology v1"]
    decision = "MORPHOLOGY-REFINED-WINNER" if refined["F1"] > v1["F1"] else "MORPHOLOGY-V1-RETAINED"
    report = {**metadata, "aggregate_metrics": aggregate, "summary": summary,
              "refined_vs_TunedNative": {
                  "delta": {k: refined[k] - aggregate["Tuned Native"][k] for k in FIELDS},
                  **stability["TunedNative"]},
              "refined_vs_v1": {
                  "delta": {k: refined[k] - v1[k] for k in FIELDS},
                  **stability["Morphology-v1"]},
              "bootstrap": bootstrap, "parameter_stability": parameter_stability,
              "decision": decision, "method_development_status": "SAMMLV-METHOD-DEVELOPMENT-FROZEN",
              "completed_outer_folds": len(subjects), "incomplete": False}
    dump(out / "morph_refined_parameter_stability.json", parameter_stability)
    dump(out / "morph_refined_summary.json", report)
    write_csv(out / "morph_refined_outer_metrics.csv", [
        {"outer_subject": s, **results[s]["outer_metrics"],
         **{f"selected_{k}": v for k, v in results[s]["selected_config"].items()},
         "inner_F1": results[s]["inner_metrics"]["F1"]}
        for s in subjects
    ])
    write_csv(out / "morph_refined_selected_configs.csv", [
        {"outer_subject": s, **results[s]["selected_config"],
         "inner_F1": results[s]["inner_metrics"]["F1"],
         "inner_TP": results[s]["inner_metrics"]["TP"],
         "inner_FP": results[s]["inner_metrics"]["FP"],
         "inner_FN": results[s]["inner_metrics"]["FN"],
         "train_subjects": ";".join(results[s]["train_subjects"])}
        for s in subjects
    ])
    write_csv(out / "morph_refined_inner_scores", []) if False else None
    write_csv(out / "morph_refined_inner_scores.csv", [
        row for s in subjects for row in results[s]["inner_rows"]
    ])
    prediction_rows = []
    for s in subjects:
        prediction_rows.extend(results[s]["predictions"])
        prediction_rows.extend({**row, "trace_role": "weak_pool"} for row in results[s]["weak_trace"])
    for row in prediction_rows:
        row.setdefault("trace_role", "final_prediction")
    write_csv(out / "morph_refined_prediction_trace.csv", prediction_rows)
    write_csv(out / "morph_refined_subject_deltas.csv", delta_rows)
    write_csv(out / "morph_refined_inner_fit_audit.csv", [
        row for s in subjects for row in results[s]["fit_audit"]
    ])
    write_csv(out / "morph_refined_selected_inner_oof.csv", [
        row for s in subjects for row in results[s]["selected_oof"]
    ])
    dump(out / "morph_refined_fitted_models.json", [results[s]["model_state"] for s in subjects])
    lines = [
        "# Morphology Rescue Local Refinement — SAMMLV", "",
        "本实验是在 Morphology Rescue v1 之后按预注册有限 grid 进行的唯一 SAMMLV local refinement。完成后状态为 `SAMMLV-METHOD-DEVELOPMENT-FROZEN`。", "",
        "## Integrity", "",
        f"- V1 reproduction gate: PASS；54/152/105，F1={v1['F1']:.12f}。Tuned Native anchor: {aggregate['Tuned Native']['TP']}/{aggregate['Tuned Native']['FP']}/{aggregate['Tuned Native']['FN']}，F1={aggregate['Tuned Native']['F1']:.12f}。",
        f"- Cache SHA-256: `{metadata['cache_sha256']}`；29/29 outer folds complete。",
        "- 完整复用 Native evaluator、weak labels、inner subject LOSO、LR/scaler 与单类别处理；peak-relative、31点、192 configs。",
        "- 所有 high Tuned-Native events 均保留；prediction count 与每折 Native 相同；未读取 recognition 作为 feature，没有新增 NMS。",
        "- 详细输入及代码哈希见 `morph_refined_summary.json`；v1 artifact 未写回。", "",
        "## Main comparison", "",
        "| Method | TP | FP | FN | Precision | Recall | F1 | weak | selected rescue | rescue TP | rescue FP | rescue precision | missed GT recovered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| " + " | ".join([
            row["Method"], str(row["TP"]), str(row["FP"]), str(row["FN"]),
            f"{row['Precision']:.6f}", f"{row['Recall']:.6f}", f"{row['F1']:.6f}",
            str(row["weak_candidate_count"]), str(row["rescued_candidate_count"]),
            str(row["rescue_TP"]), str(row["rescue_FP"]),
            f"{row['rescue_precision']:.6f}", str(row["native_missed_GT_recovered"])
        ]) + " |")
    lines += ["", "## Subject stability", "",
              "| Comparison | improved | equal | worse | min leave-one-out ΔF1 | all leave-one-out gains positive |",
              "|---|---:|---:|---:|---:|---|"]
    for label in ("TunedNative", "Morphology-v1"):
        s = stability[label]
        lines.append(f"| Refined − {label} | {s['improved']} | {s['equal']} | {s['worse']} | {s['minimum_leave_one_subject_out_delta']:+.6f} | {s['gain_survives_every_subject_removal']} |")
    lines += ["", "## Shared bootstrap", "",
              "1000 次 subject bootstrap，seed=20260902；两个比较使用完全相同的抽样 indices。", "",
              "| Contrast | mean ΔF1 | 95% CI |", "|---|---:|---|"]
    for label, row in bootstrap.items():
        lines.append(f"| {label} | {row['mean_delta']:+.6f} | [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] |")
    lines += ["", "## Parameter stability", "",
              "| Parameter | Frequency | Status |", "|---|---|---|"]
    for field, row in parameter_stability.items():
        if field == "requested_L_ratio_to_radius":
            lines.append(f"| requested_L_ratio → actual radius | `{row}` | fixed mapping |")
        else:
            lines.append(f"| {field} | `{row['frequency']}` | {row['status'] or 'stable'} |")
    lines += ["", "## Decision", "",
              f"- Refined vs v1: ΔF1={refined['F1'] - v1['F1']:+.6f}。",
              f"- **{decision}**", "- **SAMMLV-METHOD-DEVELOPMENT-FROZEN**", "",
              "本报告只给本轮预注册 refinement 的实验事实与决定；未自动运行 CASME3，也未扩大 grid。", "",
              "## Files", ""]
    lines += [f"- `{p.resolve()}`" for p in sorted(out.iterdir())]
    lines += [f"- `{(output / 'MORPHOLOGY_REFINEMENT_SAMMLV_REPORT_CN.md').resolve()}`",
              f"- `{Path(__file__).resolve()}`", f"- `{(HERE / 'LOCKED_PROTOCOL_CN.md').resolve()}`"]
    (output / "MORPHOLOGY_REFINEMENT_SAMMLV_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.time()
    output = ROOT / "results" / ("morphology_refinement_sammlv" if args.mode == "full" else "morphology_refinement_smoke_006")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    protocol = HERE / "LOCKED_PROTOCOL_CN.md"
    paths = [Path(__file__), protocol, CACHE, NATIVE_REPORT, V1_REPORT,
             V1_OUT / "morphology_outer_metrics.csv", V1_OUT / "morphology_prediction_trace.csv",
             LEGACY / "run_mscr_nested_loso.py", RGR / "run_rgr1_nested_loso.py",
             FIRST3 / "run_first3_nested.py", MORPH / "run_morphology_audit.py"]
    provenance = {str(path.resolve()): sha256(path) for path in paths}
    assert provenance[str(CACHE.resolve())] == CACHE_HASH
    if args.mode == "full":
        smoke = json.loads((ROOT / "results/morphology_refinement_smoke_006/outputs/morph_refined_summary.json").read_text())
        assert smoke["smoke_pass"] and smoke["provenance"] == provenance
    source = json.loads(NATIVE_REPORT.read_text())
    v1_report = json.loads(V1_REPORT.read_text())
    assert not source["incomplete"] and source["cache_sha256"] == CACHE_HASH
    v1_metric = v1_report["aggregate_metrics"]["Morphology Rescue"]
    if not (v1_metric["TP"] == 54 and v1_metric["FP"] == 152 and v1_metric["FN"] == 105 and
            abs(v1_metric["F1"] - 0.2958904109589041) < 1e-15):
        raise RuntimeError("BLOCKED-V1-REPRODUCTION")
    folds = {fold["subject"]: fold for fold in source["outer_folds"]}
    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    records = list(payload["records"])
    subjects = sorted(folds)
    assert payload["dataset"] == "SAMMLV" and int(payload["k_p"]) == K_P
    assert len(subjects) == 29 and len(records) == 79 and sum(len(r["samples"]) for r in records) == 159
    assert all(set(folds[s]["train_subjects"]) == set(subjects) - {s} for s in subjects)
    anchors, baseline_subjects = baseline(records, folds)
    print("BASELINE AND V1 GATES PASS", json.dumps({"anchors": anchors, "v1": v1_metric}), flush=True)
    v1_rows = read_csv(V1_OUT / "morphology_outer_metrics.csv")
    assert len(v1_rows) == 29
    v1_subjects = {row["outer_subject"]: {k: (int(row[k]) if k in ("TP", "FP", "FN", "event_count",
        "weak_candidate_count", "rescued_candidate_count", "rescue_TP", "rescue_FP", "native_missed_GT_recovered")
        else float(row[k])) for k in FIELDS} for row in v1_rows}
    assert set(v1_subjects) == set(subjects)
    assert sum(row["TP"] for row in v1_subjects.values()) == 54
    assert sum(row["FP"] for row in v1_subjects.values()) == 152
    banks = build_banks(records, folds)
    held_subjects = ["006"] if args.mode == "smoke" else subjects
    context = {"records": records, "subjects": subjects, "folds": folds, "banks": banks}
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(context,)) as executor:
        futures = {executor.submit(worker, subject): subject for subject in held_subjects}
        for future in as_completed(futures):
            result = future.result()
            results[result["outer_subject"]] = result
            print(f"FOLD {result['outer_subject']} complete ({len(results)}/{len(held_subjects)})", flush=True)
    assert set(results) == set(held_subjects)
    output.mkdir(parents=True)
    out = output / "outputs"
    out.mkdir()
    common_metadata = {
        "experiment": "Morphology Rescue Local Refinement - SAMMLV", "mode": args.mode,
        "cache_path": str(CACHE.resolve()), "cache_sha256": CACHE_HASH,
        "provenance": provenance, "grid_count": len(grid()),
        "grid": {"Delta_p": DELTAS, "requested_L_ratio": L_RATIOS,
                 "actual_integer_radius": sorted({x["actual_integer_radius"] for x in grid()}),
                 "normalization": "peak-relative", "patch_points": 31,
                 "C": CS, "gamma": GAMMAS},
        "inner_protocol": "subject-disjoint inner LOSO, pooled validation F1, only inner-training fit",
        "single_class_handling": "constant probability of sole inner-training class; recorded",
        "baseline_anchors": anchors, "v1_anchor": v1_metric,
        "no_backbone_forward": True, "no_training_backbone": True,
        "no_recognition_feature": True, "no_new_nms": True,
        "completed_outer_folds": len(results), "incomplete": False,
    }
    if args.mode == "smoke":
        dump(out / "morph_refined_summary.json", {**common_metadata, "smoke_pass": True,
                                                    "outer_metrics": results["006"]["outer_metrics"]})
        write_csv(out / "morph_refined_outer_metrics.csv", [{"outer_subject": "006",
            **results["006"]["outer_metrics"], **{f"selected_{k}": v for k, v in results["006"]["selected_config"].items()}}])
        write_csv(out / "morph_refined_selected_configs.csv", [{"outer_subject": "006",
            **results["006"]["selected_config"], "inner_F1": results["006"]["inner_metrics"]["F1"]}])
        write_csv(out / "morph_refined_inner_scores.csv", results["006"]["inner_rows"])
        write_csv(out / "morph_refined_prediction_trace.csv", results["006"]["predictions"])
        write_csv(out / "morph_refined_inner_fit_audit.csv", results["006"]["fit_audit"])
        print("SMOKE PASS", flush=True)
    else:
        report = finish(results, anchors, baseline_subjects, v1_subjects, subjects, output,
                        common_metadata)
        print(json.dumps({"aggregate": report["aggregate_metrics"], "decision": report["decision"],
                          "status": report["method_development_status"]}, indent=2), flush=True)
    print(f"FINISHED {output} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
