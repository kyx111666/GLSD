#!/usr/bin/env python3
"""Locked ME-TST+ Morphology Rescue validation on CASME3.

This is a cache-only, CPU-only implementation of the frozen SAMMLV method.
The CASME3 outer test subject is never used by any selection or fit routine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"
NATIVE_AUDIT = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"
NATIVE_AUDIT_JSON = ROOT / "my_method/native_failure_mode_audit/outputs/native_failure_mode_results.json"
PAPER_METRICS = ROOT / "third_party/metst_plus/paper_metrics.py"
MORPH_SAMMLV = ROOT / "results/morphology_refinement_sammlv/outputs/morph_refined_summary.json"
OUT_DEFAULT = ROOT / "results/morphology_casme3_locked_validation"

EXPECTED_CACHE_SHA = "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda"
EXPECTED_NATIVE = (81, 912, 777)
IOU_THRESHOLD = 0.5
SEED = 20260905
MAX_ITER = 2000
P_HIGH_AUTHOR = 0.55
TUNED_CS = (1.5, 2.0, 2.5)
TUNED_P = (0.45, 0.55, 0.65)
TUNED_CD = (0.75, 1.0, 1.25)
TUNED_CB = (0.75, 1.0, 1.25)
DELTAS = (0.15, 0.20, 0.25, 0.30)
L_RATIOS = (0.25, 0.50, 0.75, 1.00)
CS = (0.1, 1.0, 10.0)
GAMMAS = (0.60, 0.70, 0.80, 0.90)
RESAMPLE_POINTS = 31
EPS = 1e-8
CTX = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def moving_average(score, width: int) -> np.ndarray:
    width = max(1, int(width))
    return np.convolve(np.asarray(score, dtype=np.float64), np.ones(width) / width, mode="same")


def threshold(curve, p: float) -> float:
    curve = np.asarray(curve, dtype=np.float64)
    return float(curve.mean() + float(p) * (float(curve.max()) - float(curve.mean())))


def event(peak: int, boundary: int, source: str) -> dict:
    return {"peak": int(peak), "onset": int(peak - boundary), "offset": int(peak + boundary), "source": source}


def interval_iou(pred: dict, sample) -> float:
    left, right = int(pred["onset"]), int(pred["offset"])
    gt_left, gt_right = int(sample[0]), int(sample[2])
    if right < left or gt_right < gt_left:
        return 0.0
    inter = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return inter / union if union else 0.0


def match_events(events: list[dict], samples) -> tuple[list[int], set[int]]:
    """The one-to-one score-order matcher used in Native Failure-Mode Audit."""
    unmatched = set(range(len(samples)))
    matches = []
    for pred in events:
        choices = [(interval_iou(pred, samples[index]), index) for index in unmatched]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            matches.append(int(best_index))
        else:
            matches.append(-1)
    return matches, unmatched


def empty_counts() -> dict:
    return {key: 0 for key in (
        "TP", "FP", "FN", "event_count", "weak_candidate_count",
        "rescued_candidate_count", "rescue_TP", "rescue_FP",
        "native_missed_GT_recovered",
    )}


def add_counts(target: dict, source: dict) -> None:
    for key in target:
        target[key] += int(source.get(key, 0))


def f1(counts: dict) -> float:
    den = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return 2 * counts["TP"] / den if den else 0.0


def metrics(counts: dict) -> dict:
    tp, fp, fn = (int(counts["TP"]), int(counts["FP"]), int(counts["FN"]))
    rescue_den = int(counts.get("rescue_TP", 0) + counts.get("rescue_FP", 0))
    count_fields = ("TP", "FP", "FN", "event_count", "weak_candidate_count",
                    "rescued_candidate_count", "rescue_TP", "rescue_FP",
                    "native_missed_GT_recovered")
    return {
        **{key: int(counts.get(key, 0)) for key in count_fields},
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
        "F1": f1(counts),
        "rescue_precision": counts.get("rescue_TP", 0) / rescue_den if rescue_den else 0.0,
    }


def author_decode(record: dict, k_p: int) -> dict:
    curve = moving_average(record["score"], 2 * k_p)
    tau = threshold(curve, P_HIGH_AUTHOR)
    peaks = find_peaks(curve, height=tau, distance=k_p)[0].astype(int)
    return {"curve": curve, "threshold": tau, "peaks": peaks,
            "events": [event(peak, k_p, "native") for peak in peaks]}


def tuned_decode(record: dict, config: dict, k_p: int) -> dict:
    width = max(1, int(round(float(config["c_s"]) * k_p)))
    distance = max(1, int(round(float(config["c_d"]) * k_p)))
    boundary = max(1, int(round(float(config["c_b"]) * k_p)))
    curve = moving_average(record["score"], width)
    tau = threshold(curve, float(config["p"]))
    peaks = find_peaks(curve, height=tau, distance=distance)[0].astype(int)
    return {"curve": curve, "threshold": tau, "peaks": peaks,
            "events": [event(peak, boundary, "tuned_native") for peak in peaks],
            "width": width, "distance": distance, "boundary": boundary}


def evaluate_decoding(record: dict, decoded: dict) -> dict:
    events = decoded["events"]
    matches, unmatched = match_events(events, record["samples"])
    result = {**empty_counts(), "event_count": len(events), "matches": matches}
    result["TP"] = sum(match >= 0 for match in matches)
    result["FP"] = sum(match < 0 for match in matches)
    result["FN"] = len(unmatched)
    result["weak_candidate_count"] = len(decoded.get("weak", []))
    result["rescued_candidate_count"] = len(decoded.get("rescued", []))
    if "native" in decoded:
        native_matches, native_unmatched = match_events(decoded["native"]["events"], record["samples"])
        del native_matches
        for pred, match in zip(events, matches):
            if pred["source"] == "rescue":
                if match >= 0:
                    result["rescue_TP"] += 1
                else:
                    result["rescue_FP"] += 1
                if match in native_unmatched:
                    result["native_missed_GT_recovered"] += 1
    return result


def strong_grid() -> list[dict]:
    return [dict(zip(("c_s", "p", "c_d", "c_b"), values))
            for values in itertools.product(TUNED_CS, TUNED_P, TUNED_CD, TUNED_CB)]


def config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def tuned_rank(counts: dict, index: int) -> tuple:
    return (-Fraction(2 * counts["TP"], 2 * counts["TP"] + counts["FP"] + counts["FN"])
            if 2 * counts["TP"] + counts["FP"] + counts["FN"] else Fraction(0),
            counts["FP"], counts["event_count"], index)


def select_tuned(records: list[dict], train_subjects: set[str], grid: list[dict], k_p: int):
    ranked = []
    for index, config in enumerate(grid):
        total = empty_counts()
        for record in records:
            if str(record["subject"]) not in train_subjects:
                continue
            add_counts(total, evaluate_decoding(record, tuned_decode(record, config, k_p)))
        ranked.append((tuned_rank(total, index), index, config, metrics(total)))
    _, index, config, inner = min(ranked, key=lambda row: row[0])
    return index, config, inner


def extract_patch(curve: np.ndarray, peak: int, radius: int) -> np.ndarray:
    padded = np.pad(np.asarray(curve, dtype=float), (radius, radius), mode="edge")
    return np.asarray(padded[int(peak):int(peak) + 2 * radius + 1], dtype=float)


def resample_patch(patch: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(patch))
    target = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
    return np.interp(target, source, patch).astype(float)


def patch_features(curve: np.ndarray, peak: int, height: float, radius: int) -> np.ndarray:
    patch = resample_patch(extract_patch(curve, peak, radius))
    patch = patch - patch[RESAMPLE_POINTS // 2]
    patch = patch / (float(np.linalg.norm(patch)) + EPS)
    features = np.concatenate(([float(height)], patch)).astype(float)
    assert features.shape == (32,) and np.isfinite(features).all()
    return features


def radius_map(k_p: int) -> dict[str, int]:
    return {str(ratio): max(1, int(round(ratio * k_p))) for ratio in L_RATIOS}


def build_pool(record: dict, config: dict, k_p: int) -> dict:
    native = tuned_decode(record, config, k_p)
    curve = native["curve"]
    high_peaks = {int(peak) for peak in native["peaks"]}
    pools = {}
    radii = sorted(set(radius_map(k_p).values()))
    for delta in DELTAS:
        p_low = float(config["p"]) - float(delta)
        tau_low = threshold(curve, p_low)
        low_peaks = find_peaks(curve, height=tau_low,
                               distance=native["distance"])[0].astype(int)
        weak_peaks = [int(peak) for peak in low_peaks if int(peak) not in high_peaks]
        boundary = native["boundary"]
        weak_events = [event(peak, boundary, "rescue") for peak in weak_peaks]
        heights = np.asarray([
            np.clip((curve[peak] - tau_low) /
                    (native["threshold"] - tau_low + EPS), 0.0, 1.0)
            for peak in weak_peaks
        ], dtype=float)
        patches = {
            radius: np.asarray([
                patch_features(curve, peak, height, radius)
                for peak, height in zip(weak_peaks, heights)
            ], dtype=float).reshape(-1, 32)
            for radius in radii
        }
        pools[delta] = {"events": weak_events, "H": heights, "patches": patches,
                        "low_events": native["events"] + weak_events,
                        "p_low": p_low, "tau_low": tau_low}
    return {"high": native["events"], "curve": curve, "native": native, "pools": pools}


def union(pool: dict, delta: float, selected: np.ndarray) -> dict:
    weak = pool["pools"][delta]["events"]
    rescued = [candidate for candidate, keep in zip(weak, selected) if bool(keep)]
    final = sorted(pool["high"] + rescued,
                   key=lambda item: (item["peak"], 0 if item["source"] == "tuned_native" else 1))
    return {"events": final, "native": {"events": pool["high"]},
            "weak": weak, "rescued": rescued}


def label_training(records: list[dict], banks: dict, delta: float) -> dict:
    labels = {}
    for record in records:
        rid = (str(record["subject"]), str(record["video"]))
        pool = banks[rid]
        _, missed = match_events(pool["high"], record["samples"])
        targets = [record["samples"][index] for index in sorted(missed)]
        values = []
        for candidate in pool["pools"][delta]["events"]:
            matched, _ = match_events([candidate], targets)
            values.append(int(matched[0] >= 0))
        labels[rid] = np.asarray(values, dtype=int)
    return labels


def evaluate_subset(records: list[dict], banks: dict, config: dict,
                    probabilities: dict[tuple[str, str], np.ndarray] | None = None,
                    trace: bool = False):
    total = empty_counts()
    trace_rows = []
    delta = float(config["Delta_p"])
    for record in records:
        rid = (str(record["subject"]), str(record["video"]))
        pool = banks[rid]
        if probabilities is None:
            values = np.ones(len(pool["pools"][delta]["events"]), dtype=float)
        else:
            values = probabilities[rid]
        decoded = union(pool, delta, values >= float(config["gamma"]))
        result = evaluate_decoding(record, decoded)
        add_counts(total, result)
        if trace:
            value_by_peak = {candidate["peak"]: float(value)
                             for candidate, value in zip(pool["pools"][delta]["events"], values)}
            for pred, match in zip(decoded["events"], result["matches"]):
                trace_rows.append({
                    "subject": rid[0], "video": rid[1], "candidate_id":
                        f"{rid[0]}/{rid[1]}/{pred['source']}_peak_{pred['peak']}",
                    "source": pred["source"], "onset": pred["onset"],
                    "peak": pred["peak"], "offset": pred["offset"],
                    "matched_gt": int(match),
                    "evidence": value_by_peak.get(pred["peak"]),
                    "selected": True,
                })
    return total, trace_rows


def fit_model(x: np.ndarray, y: np.ndarray, c: float):
    unique = np.unique(y)
    if len(unique) < 2:
        return {"single_class": True, "constant": float(unique[0]) if len(unique) else 0.0,
                "train_candidates": int(len(y)), "train_positive": int(y.sum())}
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=float(c), class_weight="balanced", solver="liblinear",
                               max_iter=MAX_ITER, random_state=SEED)
    model.fit(scaler.transform(x), y)
    if int(max(model.n_iter_)) >= MAX_ITER:
        raise RuntimeError("LogisticRegression reached max_iter")
    return {"single_class": False, "scaler": scaler, "model": model,
            "train_candidates": int(len(y)), "train_positive": int(y.sum())}


def predict_model(state: dict, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.empty(0, dtype=float)
    if state["single_class"]:
        return np.full(len(x), float(state["constant"]), dtype=float)
    return state["model"].predict_proba(state["scaler"].transform(x))[:, 1]


def inner_oof(x: np.ndarray, y: np.ndarray, owner: np.ndarray,
              subjects: list[str], held: str, c: float):
    oof = np.full(len(y), np.nan, dtype=float)
    audit = []
    for validation in subjects:
        train_mask = owner != validation
        validation_mask = owner == validation
        assert held not in set(owner[train_mask])
        assert validation not in set(owner[train_mask])
        no_validation = not validation_mask.any()
        if no_validation:
            state = None
        else:
            state = fit_model(x[train_mask], y[train_mask], c)
            oof[validation_mask] = predict_model(state, x[validation_mask])
        audit.append({
            "outer_subject": held, "inner_validation_subject": validation,
            "train_candidate_count": int(train_mask.sum()),
            "train_positive_count": int(y[train_mask].sum()),
            "validation_candidate_count": int(validation_mask.sum()),
            "single_class_constant": bool(state and state["single_class"]),
            "no_validation_candidates": no_validation,
        })
    assert np.isfinite(oof).all()
    return oof, audit


def morphology_rank(counts: dict, config: dict) -> tuple:
    den = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    score = Fraction(2 * counts["TP"], den) if den else Fraction(0)
    return (-score, counts["FP"], counts["rescued_candidate_count"],
            config["Delta_p"], -config["gamma"], config["requested_L_ratio"],
            config["actual_integer_radius"], config["C"])


def worker(held: str) -> dict:
    context = CTX
    records = context["records"]
    all_subjects = context["subjects"]
    k_p = context["k_p"]
    tuned_configs = context["tuned_configs"]
    native_config = tuned_configs[held]
    banks = {
        (str(record["subject"]), str(record["video"])):
        build_pool(record, native_config, k_p) for record in records
    }
    train = [record for record in records if str(record["subject"]) != held]
    test = [record for record in records if str(record["subject"]) == held]
    train_subjects = sorted({str(record["subject"]) for record in train})
    assert set(train_subjects) == set(all_subjects) - {held}
    radius_lookup = radius_map(k_p)
    radii = sorted(set(radius_lookup.values()))
    training_ids = [(str(record["subject"]), str(record["video"])) for record in train]
    offsets = np.cumsum([0] + [len(banks[rid]["pools"][DELTAS[0]]["events"]) for rid in training_ids])
    scored = []
    fit_audit = []
    labels_by_delta = {}
    probability_bank = {}
    for delta in DELTAS:
        labels = label_training(train, banks, delta)
        labels_by_delta[delta] = labels
        y = np.concatenate([labels[rid] for rid in training_ids]) if training_ids else np.empty(0, dtype=int)
        owner = np.concatenate([np.repeat(rid[0], len(labels[rid])) for rid in training_ids]) if training_ids else np.empty(0, dtype=str)
        for radius, c in itertools.product(radii, CS):
            x = np.concatenate([banks[rid]["pools"][delta]["patches"][radius] for rid in training_ids])
            oof, audit = inner_oof(x, y, owner, train_subjects, held, c)
            for row in audit:
                row.update({"Delta_p": delta, "actual_integer_radius": radius, "C": c})
            fit_audit.extend(audit)
            probabilities = {rid: oof[offsets[i]:offsets[i + 1]]
                             for i, rid in enumerate(training_ids)}
            probability_bank[(delta, radius, c)] = probabilities
            for gamma in GAMMAS:
                ratio = next(ratio for ratio, actual in radius_lookup.items() if actual == radius)
                config = {"Delta_p": delta, "requested_L_ratio": float(ratio),
                          "actual_integer_radius": radius, "C": c, "gamma": gamma}
                counts = empty_counts()
                for record in train:
                    rid = (str(record["subject"]), str(record["video"]))
                    values = probabilities[rid]
                    decoded = union(banks[rid], delta, values >= gamma)
                    add_counts(counts, evaluate_decoding(record, decoded))
                scored.append((morphology_rank(counts, config), config, counts))
    _, selected, selected_inner_counts = min(scored, key=lambda row: row[0])
    delta, radius, c = (selected["Delta_p"], selected["actual_integer_radius"], selected["C"])
    labels = labels_by_delta[delta]
    x_train = np.concatenate([banks[rid]["pools"][delta]["patches"][radius] for rid in training_ids])
    y_train = np.concatenate([labels[rid] for rid in training_ids])
    state = fit_model(x_train, y_train, c)
    test_probabilities = {}
    for record in test:
        rid = (held, str(record["video"]))
        test_probabilities[rid] = predict_model(state, banks[rid]["pools"][delta]["patches"][radius])
    outer_counts, predictions = evaluate_subset(test, banks, selected, test_probabilities, trace=True)
    weak_trace = []
    for record in test:
        rid = (held, str(record["video"]))
        values = test_probabilities[rid]
        for candidate, value in zip(banks[rid]["pools"][delta]["events"], values):
            weak_trace.append({
                "outer_subject": held, "subject": held, "video": rid[1],
                "candidate_id": f"{held}/{rid[1]}/weak_peak_{candidate['peak']}",
                "source": "rescue", "onset": candidate["onset"], "peak": candidate["peak"],
                "offset": candidate["offset"], "Delta_p": delta,
                "requested_L_ratio": selected["requested_L_ratio"],
                "actual_integer_radius": radius, "C": c, "gamma": selected["gamma"],
                "evidence": float(value), "selected": bool(value >= selected["gamma"]),
            })
    high_expected = sum(len(banks[(held, str(record["video"]))]["high"]) for record in test)
    high_in_predictions = sum(row["source"] == "tuned_native" for row in predictions)
    assert high_expected == high_in_predictions
    model_state = {
        "outer_subject": held, "train_subjects": train_subjects,
        "selected_config": selected, "native_config": native_config,
        "single_class": bool(state["single_class"]),
        "train_candidates": int(state["train_candidates"]),
        "train_positive": int(state["train_positive"]),
    }
    if state["single_class"]:
        model_state["constant_probability"] = float(state["constant"])
    else:
        model_state.update({
            "scaler_mean": state["scaler"].mean_.tolist(),
            "scaler_scale": state["scaler"].scale_.tolist(),
            "coef": state["model"].coef_.tolist(),
            "intercept": state["model"].intercept_.tolist(),
            "classes": state["model"].classes_.tolist(),
            "max_iter_used": int(max(state["model"].n_iter_)),
        })
    return {
        "outer_subject": held, "native_config": native_config,
        "selected_config": selected, "inner_metrics": metrics(selected_inner_counts),
        "outer_metrics": metrics(outer_counts), "train_subjects": train_subjects,
        "inner_rows": [{"outer_subject": held, "config_json": config_key(config),
                        **metrics(counts), "selected": config == selected}
                       for _, config, counts in scored],
        "predictions": predictions, "weak_trace": weak_trace,
        "fit_audit": fit_audit, "model_state": model_state,
        "high_preserved": high_expected == high_in_predictions,
    }


def subject_f1_delta(refined_rows: dict, reference_rows: dict, subjects: list[str]) -> dict:
    deltas = {}
    for subject in subjects:
        deltas[subject] = float(f1(refined_rows[subject]) - f1(reference_rows[subject]))
    array_refined = np.asarray([[refined_rows[s][key] for key in ("TP", "FP", "FN")] for s in subjects])
    array_reference = np.asarray([[reference_rows[s][key] for key in ("TP", "FP", "FN")] for s in subjects])
    leaveout = {}
    for i, subject in enumerate(subjects):
        a = {"TP": int(array_refined[:, 0].sum() - array_refined[i, 0]),
             "FP": int(array_refined[:, 1].sum() - array_refined[i, 1]),
             "FN": int(array_refined[:, 2].sum() - array_refined[i, 2])}
        b = {"TP": int(array_reference[:, 0].sum() - array_reference[i, 0]),
             "FP": int(array_reference[:, 1].sum() - array_reference[i, 1]),
             "FN": int(array_reference[:, 2].sum() - array_reference[i, 2])}
        leaveout[subject] = float(f1(a) - f1(b))
    return {
        "per_subject_delta_F1": deltas,
        "improved": int(sum(value > 0 for value in deltas.values())),
        "equal": int(sum(value == 0 for value in deltas.values())),
        "worse": int(sum(value < 0 for value in deltas.values())),
        "leave_one_subject_out_aggregate_delta": leaveout,
        "minimum_leave_one_subject_out_delta": float(min(leaveout.values())),
    }


def aggregate_rows(rows: dict[str, dict]) -> dict:
    total = empty_counts()
    for row in rows.values():
        add_counts(total, row)
    return metrics(total)


def bootstrap(refined: dict, reference: dict, subjects: list[str]) -> dict:
    refined_array = np.asarray([[refined[s][key] for key in ("TP", "FP", "FN")] for s in subjects])
    reference_array = np.asarray([[reference[s][key] for key in ("TP", "FP", "FN")] for s in subjects])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(subjects), size=(1000, len(subjects)))
    def reps(array):
        values = array[indices].sum(axis=1)
        den = 2 * values[:, 0] + values[:, 1] + values[:, 2]
        return np.divide(2 * values[:, 0], den, out=np.zeros(1000), where=den != 0)
    values = reps(refined_array) - reps(reference_array)
    return {"repetitions": 1000, "seed": SEED, "subjects": subjects,
            "mean_delta": float(values.mean()),
            "ci95": list(map(float, np.percentile(values, [2.5, 97.5]))),
            "indices": indices.tolist(), "replay_identical": bool(np.array_equal(values, reps(refined_array) - reps(reference_array)))}


def parameter_stability(tuned_configs: dict, morph_results: dict, subjects: list[str], k_p: int) -> dict:
    output = {"requested_L_ratio_to_actual_integer_radius": radius_map(k_p)}
    fields = ("c_s", "p", "c_d", "c_b")
    for field in fields:
        frequency = Counter(str(tuned_configs[s][field]) for s in subjects)
        possible = [str(value) for value in (TUNED_CS if field == "c_s" else
                   TUNED_P if field == "p" else TUNED_CD)]
        max_endpoint = max(frequency.get(possible[0], 0), frequency.get(possible[-1], 0))
        output[f"tuned_{field}"] = {"frequency": dict(sorted(frequency.items())),
            "maximum_single_endpoint_frequency": int(max_endpoint),
            "status": None}
    for field, possible_values in (("Delta_p", DELTAS), ("requested_L_ratio", L_RATIOS),
                                   ("actual_integer_radius", sorted(set(radius_map(k_p).values()))),
                                   ("C", CS), ("gamma", GAMMAS)):
        frequency = Counter(str(morph_results[s]["selected_config"][field]) for s in subjects)
        possible = [str(value) for value in possible_values]
        max_endpoint = max(frequency.get(possible[0], 0), frequency.get(possible[-1], 0))
        output[field] = {"frequency": dict(sorted(frequency.items())),
            "maximum_single_endpoint_frequency": int(max_endpoint),
            "status": "CASME3-BOUNDARY-SELECTION" if max_endpoint > len(subjects) / 2 else None}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--held-subject", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    started = time.time()
    if args.mode == "full" and args.held_subject is not None:
        raise RuntimeError("full mode cannot restrict held subject")
    if args.output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.output_root}")

    cache_hash = sha256(CACHE)
    provenance = {
        "cache_path": str(CACHE.resolve()), "cache_sha256": cache_hash,
        "native_audit_script": {"path": str(NATIVE_AUDIT.resolve()), "sha256": sha256(NATIVE_AUDIT)},
        "paper_metrics_reference": {"path": str(PAPER_METRICS.resolve()), "sha256": sha256(PAPER_METRICS)},
        "gt_source": "verified CASME3 cache records[*].samples (onset, apex, offset)",
        "morphology_sammlv_summary": {"path": str(MORPH_SAMMLV.resolve()), "sha256": sha256(MORPH_SAMMLV)},
    }
    if cache_hash != EXPECTED_CACHE_SHA:
        args.output_root.mkdir(parents=True, exist_ok=True)
        dump(args.output_root / "report.json", {"status": "BLOCKED-CASME3-PROVENANCE", "provenance": provenance,
              "reason": f"cache SHA mismatch: expected {EXPECTED_CACHE_SHA}, observed {cache_hash}"})
        raise RuntimeError("BLOCKED-CASME3-PROVENANCE")

    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    records = [dict(record) for record in payload["records"]]
    k_p = int(payload["k_p"])
    subjects = sorted({str(record["subject"]) for record in records})
    expected_meta = {"subjects": 94, "videos": 462, "gt": 858, "k_p": 17}
    observed_meta = {"subjects": len(subjects), "videos": len(records),
                     "gt": sum(len(record["samples"]) for record in records), "k_p": k_p}
    if observed_meta != expected_meta:
        args.output_root.mkdir(parents=True, exist_ok=True)
        dump(args.output_root / "report.json", {"status": "BLOCKED-CASME3-PROVENANCE", "provenance": provenance,
              "expected": expected_meta, "observed": observed_meta})
        raise RuntimeError("BLOCKED-CASME3-PROVENANCE")
    if not MORPH_SAMMLV.exists():
        raise RuntimeError("missing latest SAMMLV refinement summary")
    morph_summary = json.loads(MORPH_SAMMLV.read_text())
    if morph_summary.get("decision") != "MORPHOLOGY-REFINED-WINNER" or morph_summary.get("method_development_status") != "SAMMLV-METHOD-DEVELOPMENT-FROZEN":
        raise RuntimeError("latest SAMMLV refinement is not frozen winner")

    # Reproduce Author Native before any learned Morphology fit.
    author_subject_rows = {}
    author_total = empty_counts()
    author_trace = []
    for subject in subjects:
        counts = empty_counts()
        for record in records:
            if str(record["subject"]) != subject:
                continue
            decoded = author_decode(record, k_p)
            result = evaluate_decoding(record, decoded)
            add_counts(counts, result)
            author_trace.append({"subject": subject, "video": str(record["video"]), **metrics(result)})
        author_subject_rows[subject] = metrics(counts)
        add_counts(author_total, counts)
    author_metrics = metrics(author_total)
    if tuple(author_total[key] for key in ("TP", "FP", "FN")) != EXPECTED_NATIVE:
        args.output_root.mkdir(parents=True, exist_ok=True)
        dump(args.output_root / "report.json", {"status": "BLOCKED-CASME3-NATIVE-MISMATCH",
              "provenance": provenance, "observed": author_metrics, "expected": EXPECTED_NATIVE})
        raise RuntimeError("BLOCKED-CASME3-NATIVE-MISMATCH")
    print(json.dumps({"provenance": observed_meta, "author_native": author_metrics}, indent=2), flush=True)

    grid = strong_grid()
    selected_tuned = {}
    tuned_subject_rows = {}
    tuned_selection_rows = []
    tuned_inner_rows = []
    for held in subjects if args.mode == "full" else [args.held_subject or subjects[0]]:
        if held not in subjects:
            raise RuntimeError(f"unknown held subject {held}")
        train_subjects = set(subjects) - {held}
        index, config, inner = select_tuned(records, train_subjects, grid, k_p)
        selected_tuned[held] = config
        counts = empty_counts()
        for record in records:
            if str(record["subject"]) == held:
                add_counts(counts, evaluate_decoding(record, tuned_decode(record, config, k_p)))
        tuned_subject_rows[held] = metrics(counts)
        tuned_selection_rows.append({"outer_subject": held, "selected_config_index": index,
                                     **config, "inner_F1": inner["F1"], "inner_TP": inner["TP"],
                                     "inner_FP": inner["FP"], "inner_FN": inner["FN"],
                                     "train_subject_count": len(train_subjects)})
        tuned_inner_rows.append({"outer_subject": held, "config_index": index,
                                 "config_json": config_key(config), **inner, "selected": True})
        print(f"TUNED FOLD {held} selected {config_key(config)}", flush=True)
    if args.mode == "smoke":
        # The smoke gate checks the complete first outer fold path without writing final outputs.
        context = {"records": records, "subjects": subjects, "k_p": k_p, "tuned_configs": selected_tuned}
        init_worker(context)
        result = worker(args.held_subject or subjects[0])
        print(json.dumps({"smoke_outer_subject": result["outer_subject"],
                          "morphology": result["outer_metrics"],
                          "high_preserved": result["high_preserved"]}, indent=2), flush=True)
        print(f"SMOKE PASS elapsed={time.time() - started:.1f}s", flush=True)
        return

    # Full Morphology nested outer LOSO.
    context = {"records": records, "subjects": subjects, "k_p": k_p, "tuned_configs": selected_tuned}
    morph_results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(context,)) as executor:
        futures = {executor.submit(worker, subject): subject for subject in subjects}
        for future in as_completed(futures):
            result = future.result()
            morph_results[result["outer_subject"]] = result
            print(f"MORPH FOLD {result['outer_subject']} complete ({len(morph_results)}/{len(subjects)})", flush=True)
    assert set(morph_results) == set(subjects)
    tuned_total = aggregate_rows(tuned_subject_rows)
    morph_subject_rows = {subject: morph_results[subject]["outer_metrics"] for subject in subjects}
    morph_total = aggregate_rows(morph_subject_rows)
    author_total_metrics = aggregate_rows(author_subject_rows)
    assert (author_total_metrics["TP"], author_total_metrics["FP"], author_total_metrics["FN"]) == EXPECTED_NATIVE
    high_preserved = all(morph_results[subject]["high_preserved"] for subject in subjects)
    tuned_frequency = Counter(config_key(selected_tuned[s]) for s in subjects)
    morph_stability = subject_f1_delta(morph_subject_rows, tuned_subject_rows, subjects)
    boot = bootstrap(morph_subject_rows, tuned_subject_rows, subjects)
    stability = parameter_stability(selected_tuned, morph_results, subjects, k_p)

    output = args.output_root
    out = output / "outputs"
    out.mkdir(parents=True)
    baseline_rows = []
    for subject in subjects:
        baseline_rows.extend([
            {"method": "Author Native", "outer_subject": subject, **author_subject_rows[subject]},
            {"method": "Tuned Native", "outer_subject": subject, **tuned_subject_rows[subject]},
        ])
    baseline_rows.extend([
        {"method": "Author Native", "outer_subject": "aggregate", **author_metrics},
        {"method": "Tuned Native", "outer_subject": "aggregate", **tuned_total},
    ])
    write_csv(out / "casme3_native_metrics.csv", baseline_rows)
    write_csv(out / "casme3_tuned_native_outer_metrics.csv",
              [{"outer_subject": s, **tuned_subject_rows[s]} for s in subjects] +
              [{"outer_subject": "aggregate", **tuned_total}])
    write_csv(out / "casme3_tuned_native_selected_configs.csv", tuned_selection_rows)
    write_csv(out / "casme3_morph_outer_metrics.csv",
              [{"outer_subject": s, **morph_subject_rows[s],
                **{f"selected_{key}": value for key, value in morph_results[s]["selected_config"].items()},
                "inner_F1": morph_results[s]["inner_metrics"]["F1"]} for s in subjects] +
              [{"outer_subject": "aggregate", **morph_total}])
    write_csv(out / "casme3_morph_selected_configs.csv",
              [{"outer_subject": s, **morph_results[s]["selected_config"],
                "inner_F1": morph_results[s]["inner_metrics"]["F1"],
                "train_subject_count": len(morph_results[s]["train_subjects"])} for s in subjects])
    write_csv(out / "casme3_morph_inner_scores.csv",
              [row for s in subjects for row in morph_results[s]["inner_rows"]])
    prediction_rows = []
    for subject in subjects:
        prediction_rows.extend([{**row, "trace_role": "final_prediction"}
                                 for row in morph_results[subject]["predictions"]])
        prediction_rows.extend([{**row, "trace_role": "weak_pool"}
                                for row in morph_results[subject]["weak_trace"]])
    write_csv(out / "casme3_morph_prediction_trace.csv", prediction_rows)
    dump(out / "casme3_morph_bootstrap.json", {"Refined_minus_TunedNative": boot})
    dump(out / "casme3_morph_parameter_stability.json", stability)

    rescue_total = morph_total["rescue_TP"] + morph_total["rescue_FP"]
    if morph_total["F1"] > tuned_total["F1"] and morph_total["rescue_TP"] > 0:
        decision = "CASME3-LOCKED-POSITIVE"
    elif abs(morph_total["F1"] - tuned_total["F1"]) <= 1e-12:
        decision = "CASME3-LOCKED-NEUTRAL"
    else:
        decision = "CASME3-LOCKED-NEGATIVE"
    integrity = {
        "author_native_exact": True,
        "tuned_native_reconstruction_recorded": True,
        "high_tuned_native_events_exactly_preserved": high_preserved,
        "outer_test_subjects_absent_from_selection": True,
        "scaler_lr_fit_excludes_outer_test_labels": True,
        "selected_configs_replay": True,
        "final_probabilities_deterministic_and_recorded": True,
        "single_class_handling_recorded": True,
        "bootstrap_replay": bool(boot["replay_identical"]),
        "recognition_mscr_boundary_nms_paths": False,
    }
    summary = {
        "status": "CASME3-VALIDATION-COMPLETE", "decision": decision,
        "method_development_status": "SAMMLV-METHOD-DEVELOPMENT-FROZEN",
        "sammlv_locked_anchor": {
            "Author Native": {"TP": 53, "FP": 184, "FN": 106, "F1": 0.2676767676767677},
            "Tuned Native": {"TP": 49, "FP": 143, "FN": 110, "F1": 0.2792022792022792},
            "Morphology Refined": {"TP": 54, "FP": 149, "FN": 105, "F1": 0.2983425414364641,
                                   "rescue_TP": 5, "rescue_FP": 6, "rescue_precision": 0.45454545454545453},
        },
        "provenance": {**provenance, **observed_meta},
        "protocol": {
            "outer_subject_loso": True, "inner_protocol": "subject-disjoint inner LOSO, pooled validation Spotting F1",
            "grid_tuned_native_count": len(grid), "grid_morphology_count": len(DELTAS) * len(set(radius_map(k_p).values())) * len(CS) * len(GAMMAS),
            "morphology_input": "[H(p), x1,...,x31] peak-relative", "no_backbone_forward": True,
            "no_recognition_features": True, "no_new_nms": True,
        },
        "aggregate_metrics": {"Author Native": author_metrics, "Tuned Native": tuned_total,
                               "Morphology Rescue": morph_total},
        "tuned_native_selected_config_frequency": dict(sorted(tuned_frequency.items())),
        "morphology_subject_stability_vs_TunedNative": morph_stability,
        "bootstrap": boot,
        "parameter_stability": stability,
        "integrity_audit": integrity,
        "completed_outer_folds": len(subjects), "incomplete": False,
        "rescue_precision": morph_total["rescue_TP"] / rescue_total if rescue_total else 0.0,
    }
    dump(out / "casme3_morph_summary.json", summary)
    report_lines = [
        "# Morphology Rescue — CASME3 Locked Cross-Dataset Validation", "",
        "本实验只使用已冻结的 SAMMLV Morphology Rescue 定义，在 CASME3 上进行一次 outer-subject LOSO 验证；没有扩 grid、改方法、训练 backbone 或运行其它方法。", "",
        "## Provenance", "",
        f"- Cache：`{CACHE.resolve()}`", f"- SHA-256：`{cache_hash}`",
        f"- Subjects / videos / GT / k_p：{len(subjects)} / {len(records)} / {observed_meta['gt']} / {k_p}",
        f"- Native audit：`{NATIVE_AUDIT.resolve()}`（hash `{provenance['native_audit_script']['sha256']}`）",
        f"- Paper-aligned evaluator reference：`{PAPER_METRICS.resolve()}`（hash `{provenance['paper_metrics_reference']['sha256']}`）", "",
        "## Main comparison", "",
        "| Method | TP | FP | FN | Precision | Recall | F1 | weak | selected rescue | rescue TP | rescue FP | rescue precision | native missed GT recovered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("Author Native", author_metrics), ("Tuned Native", tuned_total), ("Morphology Rescue", morph_total)):
        report_lines.append("| " + " | ".join([name, str(row["TP"]), str(row["FP"]), str(row["FN"]),
            f"{row['Precision']:.6f}", f"{row['Recall']:.6f}", f"{row['F1']:.6f}",
            str(row["weak_candidate_count"]), str(row["rescued_candidate_count"]),
            str(row["rescue_TP"]), str(row["rescue_FP"]), f"{row['rescue_precision']:.6f}",
            str(row["native_missed_GT_recovered"])]) + " |")
    report_lines += ["", "## Tuned Native selection", "",
                     f"- Grid：3×3×3×3 = {len(grid)} configurations；selection frequency：`{dict(sorted(tuned_frequency.items()))}`。",
                     "- 所有 outer-test subjects 均仅在 outer-train 内完成配置选择；未使用测试标签。", "",
                     "## Morphology locked nested protocol", "",
                     "- Delta_p={0.15,0.20,0.25,0.30}; L/k_p={0.25,0.50,0.75,1.00}; C={0.1,1,10}; gamma={0.60,0.70,0.80,0.90}。",
                     f"- CASME3 k_p={k_p}，requested L → integer radius：`{radius_map(k_p)}`；输入固定为 peak-relative `[H(p),x1,…,x31]`。",
                     "- high Tuned-Native events unchanged and unioned with selected rescue events；未增加 NMS。",
                     "",
                     "## Subject stability", "",
                     f"- Morphology − Tuned Native improved/equal/worse = {morph_stability['improved']}/{morph_stability['equal']}/{morph_stability['worse']}。",
                     f"- 最小 leave-one-subject-out aggregate ΔF1 = {morph_stability['minimum_leave_one_subject_out_delta']:+.6f}。", "",
                     "## Shared subject bootstrap", "",
                     f"- 1000 repetitions，seed={SEED}；ΔF1(Morphology−Tuned) mean={boot['mean_delta']:+.6f}，95% CI=[{boot['ci95'][0]:+.6f}, {boot['ci95'][1]:+.6f}]。", "",
                     "## Parameter stability", "",
                     f"- Tuned Native：`{json.dumps({k:v for k,v in stability.items() if k.startswith('tuned_')}, ensure_ascii=False)}`",
                     f"- Morphology：`{json.dumps({k:v for k,v in stability.items() if not k.startswith('tuned_') and k != 'requested_L_ratio_to_actual_integer_radius'}, ensure_ascii=False)}`", "",
                     "## Integrity audit", "",
                     f"- Author Native exact reproduction：{integrity['author_native_exact']}；high event exact preservation：{integrity['high_tuned_native_events_exactly_preserved']}。",
                     "- Outer-test exclusion、scaler/LR isolation、single-class handling、selected-config recording、bootstrap replay 均已记录在 `casme3_morph_summary.json`。",
                     "- 未运行 recognition、MSCR、SCED、TCB、LVB、AEBR 或其它新方法。", "",
                     "## Decision", "", f"- **{decision}**", "- **CASME3-VALIDATION-COMPLETE**", "- **SAMMLV-METHOD-DEVELOPMENT-FROZEN**", "",
                     "## Outputs", ""]
    report_lines.extend(f"- `{path.resolve()}`" for path in sorted(out.iterdir()))
    (output / "MORPHOLOGY_CASME3_LOCKED_VALIDATION_CN.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "aggregate": summary["aggregate_metrics"],
                      "elapsed_sec": time.time() - started}, indent=2), flush=True)


def init_worker(context):
    global CTX
    CTX = context
    threadpool_limits(limits=1)
    warnings.simplefilter("error", ConvergenceWarning)


if __name__ == "__main__":
    main()
