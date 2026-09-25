"""EXP-6C — frozen-reference candidate diagnostic.

This is a read-only diagnostic over the saved EXP-2A/EXP-2C LOSO choices.
It deliberately reuses the EXP-6B replay semantics, fixes evidence to E_7,
and never selects a decoder rule from outer-test labels.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP6B = ROOT / "controlled_exploration/EXP6B_reference_evidence_cross_replay"
EXP2A = ROOT / "controlled_exploration/EXP2A_scale"
EXP2C = ROOT / "controlled_exploration/EXP2C_reference_evidence_decoupling"
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
BOOTSTRAPS, SEED = 10_000, 100


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PARENT = load_module("exp6b_parent", EXP6B / "run_exp6b.py")
ENGINE = PARENT.engine


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    overlap = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = (first[1] - first[0] + 1) + (second[1] - second[0] + 1) - overlap
    return overlap / union if union else 0.0


def as_number(value):
    return "" if value is None else value


def build_state(record: dict, backbone: str, k: int, ref: float, rho: float, tau: float) -> dict:
    """Use the parent E_7 decoder, then retain its feature objects for audit only."""
    state = PARENT.replay(record, backbone, k, ref, rho, tau, "E_7")
    features = ENGINE.CurveFeatures(record["curve"], k)
    mode = "native" if backbone == "boostingvrme" else "fixed"
    prepared = ENGINE.prepare(record, features, ref, backbone, mode, rho)
    # The repeated preparation is an explicit guard against descriptor/replay drift.
    peaks = [int(c[0]) for c in prepared.clusters]
    if peaks != [event["peak"] for event in state["p0"]]:
        raise AssertionError("EXP6C_DESCRIPTOR_REPLAY_MISMATCH")
    state["features"], state["prepared"] = features, prepared
    return state


def gt_compatible(state: dict, gt: tuple[int, int]) -> list[dict]:
    return [item for item in state["p0"] if iou(item["interval"], gt) >= 0.5]


def representative(state: dict, gt: tuple[int, int]) -> dict | None:
    candidates = gt_compatible(state, gt)
    if not candidates:
        return None
    # Canonical temporal order completes the stated S / IoU tie-break.
    return sorted(candidates, key=lambda x: (-x["S"], -iou(x["interval"], gt), x["peak"], x["interval"]))[0]


def descriptor(state: dict, event: dict | None) -> dict:
    """Expose unnormalised global evidence and exact all-7 local correspondences."""
    base = {
        "peak": "", "onset": "", "offset": "", "raw_curve_peak_value": "",
        "reference_peak_value": "", "raw_prominence": "", "reference_response_max": "",
        "reference_response_min": "", "reference_dynamic_range": "", "H": "", "G": "",
        "L": "", "S": "", "local": [], "median_driver": "",
    }
    if event is None:
        return base
    features, prepared, peak = state["features"], state["prepared"], event["peak"]
    rpeaks, _, _, response, spread = features.scales[state["ref"]]
    rindex = int(np.flatnonzero(rpeaks == peak)[0])
    window, tolerance = max(1, round(state["rho"] * features.k)), max(1, round(0.5 * features.k))
    local = []
    for width, (other_peaks, _, _, _, _) in features.effective_scales.items():
        values = features.local_cache[width, window]
        choices = np.flatnonzero(np.abs(other_peaks - peak) <= tolerance)
        source_scales = [scale for scale in SCALES if min(len(features.curve), max(1, round(scale * features.k))) == width]
        if len(choices):
            chosen = min(choices, key=lambda idx: (abs(int(other_peaks[idx]) - peak), -values[idx]))
            matched_peak, q = int(other_peaks[chosen]), float(values[chosen])
            missing = "NO"
        else:
            matched_peak, q, missing = "", 0.0, "YES"
        local.append({"scale": "/".join(f"{scale:g}" for scale in source_scales), "width": width,
                      "matched_peak": matched_peak, "temporal_distance": "" if missing == "YES" else abs(matched_peak - peak),
                      "local_q": q, "missing": missing})
    median = float(np.median([item["local_q"] for item in local]))
    drivers = [item["scale"] for item in local if np.isclose(item["local_q"], median, rtol=0, atol=1e-12)]
    return {
        "peak": peak, "onset": event["interval"][0], "offset": event["interval"][1],
        "raw_curve_peak_value": float(features.curve[peak]), "reference_peak_value": float(response[peak]),
        "raw_prominence": float(event["G"] * spread), "reference_response_max": float(response.max()),
        "reference_response_min": float(response.min()), "reference_dynamic_range": float(spread),
        "H": event["H"], "G": event["G"], "L": event["L"], "S": event["S"],
        "local": local, "median_driver": "/".join(drivers),
    }


def transition_type(desc_a: dict, desc_c: dict, gt: tuple[int, int]) -> str:
    if not desc_a["peak"] or not desc_c["peak"]:
        return "NOT_APPLICABLE"
    pa, pc = desc_a["peak"], desc_c["peak"]
    inside_a, inside_c = gt[0] <= pa <= gt[1], gt[0] <= pc <= gt[1]
    if pa == pc:
        return "TYPE-SAME"
    if inside_a and inside_c:
        return "TYPE-SHIFT-IN"
    if inside_a and not inside_c:
        return "TYPE-SHIFT-OUT"
    return "NOT_APPLICABLE"


def g_source(delta_prom: float, delta_g: float) -> str:
    if delta_prom < 0 and delta_g < 0:
        return "G_DROP_PROMINENCE_DRIVEN"
    if delta_prom >= 0 and delta_g < 0:
        return "G_DROP_NORMALIZATION_DRIVEN"
    return "G_STABLE"


def score_component(delta_g: float, delta_l: float) -> str:
    if delta_g < 0 and abs(delta_g) > abs(delta_l):
        return "G_DOMINANT_DROP"
    if delta_l < 0 and abs(delta_l) > abs(delta_g):
        return "L_DOMINANT_DROP"
    if delta_g < 0 and delta_l < 0:
        return "JOINT_DROP"
    return "NO_NEGATIVE_DOMINANT_COMPONENT"


def alignment_rows(common: dict, desc_a: dict, desc_c: dict) -> tuple[list[dict], dict]:
    """Compare the exact effective all-7 scale correspondences without a new window."""
    rows, counts = [], Counter()
    a_by_scale = {item["scale"]: item for item in desc_a["local"]}
    c_by_scale = {item["scale"]: item for item in desc_c["local"]}
    for scale in sorted(set(a_by_scale) | set(c_by_scale), key=lambda value: float(value.split("/")[0])):
        a, c = a_by_scale.get(scale), c_by_scale.get(scale)
        if a is None or c is None:
            transition = "MATCH-CHANGED"
        elif a["missing"] == "NO" and c["missing"] == "NO" and a["matched_peak"] == c["matched_peak"]:
            transition = "MATCH-SAME"
        elif a["missing"] == "NO" and c["missing"] == "NO":
            transition = "MATCH-CHANGED"
        elif a["missing"] == "NO":
            transition = "MATCH-LOST"
        else:
            transition = "MATCH-GAINED"
        counts[transition] += 1
        rows.append({**common, "scale": scale, "width": "" if a is None else a["width"],
                     "matched_peak_A": "" if a is None else a["matched_peak"], "matched_peak_C": "" if c is None else c["matched_peak"],
                     "temporal_distance_A": "" if a is None else a["temporal_distance"], "temporal_distance_C": "" if c is None else c["temporal_distance"],
                     "local_q_A": "" if a is None else a["local_q"], "local_q_C": "" if c is None else c["local_q"],
                     "missing_A": "YES" if a is None else a["missing"], "missing_C": "YES" if c is None else c["missing"],
                     "alignment_transition": transition, "median_driver_A": desc_a["median_driver"], "median_driver_C": desc_c["median_driver"],
                     "median_driver_change": "YES" if desc_a["median_driver"] != desc_c["median_driver"] else "NO"})
    return rows, counts


def verify_parent_artifacts() -> list[dict]:
    """Fail closed on each EXP-6B fact that EXP-6C depends on."""
    checks = []
    def add(check, ok, observed, expected, detail):
        checks.append({"check": check, "status": "PASS" if ok else "FAIL", "observed": observed,
                       "expected": expected, "detail": detail})
        if not ok:
            raise RuntimeError("EXP6C_PARENT_REPLAY_FAILED")

    needed = ["EXP6B_ANALYSIS.md", "cross_replay_results.csv", "score_transition.csv",
              "gt_transition_detail.csv", "fp_lineage_detail.csv"]
    add("required_parent_artifacts", all((EXP6B / name).is_file() for name in needed),
        ",".join(name for name in needed if (EXP6B / name).is_file()), ",".join(needed), "all supplied parent artifacts readable")
    analysis = (EXP6B / "EXP6B_ANALYSIS.md").read_text(encoding="utf-8")
    add("parent_replay_statement", "Exact EXP-2A/2C count replay: **PASS**." in analysis, "PASS" if "Exact EXP-2A/2C count replay: **PASS**." in analysis else "missing", "PASS", "parent report")
    identity = read_csv(EXP6B / "identity_replay_check.csv")
    add("identity_cells", bool(identity) and all(row["prediction_identity"] == "PASS" for row in identity),
        sum(row["prediction_identity"] == "PASS" for row in identity), len(identity), "EXP-2A and EXP-2C saved predictions")
    isolation = read_csv(EXP6B / "evidence_isolation_check.csv")
    add("same_reference_isolation", bool(isolation) and all(row["candidate_identity"] == "PASS" and row["G_equal_within_1e12"] == "PASS" for row in isolation),
        sum(row["candidate_identity"] == "PASS" and row["G_equal_within_1e12"] == "PASS" for row in isolation), len(isolation), "candidate / geometry identity and G_7 == G_3")
    details = read_csv(EXP6B / "gt_transition_detail.csv")
    for anchor, expected in (("ANCHOR-A", 7), ("ANCHOR-C", 6)):
        states = {(x["subject"], x["video"], x["gt_index"], x["reference_condition"]): x["state"] for x in details
                  if x["setting"] == "BoostingVRME/SAMMLV" and x["anchor"] == anchor and x["evidence_condition"] == "E_7"}
        count = sum(value == "TP" and states.get((s, v, g, "R_C")) == "B"
                    for (s, v, g, r), value in states.items() if r == "R_A")
        add(f"bs_tp_to_b_{anchor}", count == expected, count, expected, "R_A+E_7 -> R_C+E_7")
    for anchor in ("ANCHOR-A", "ANCHOR-C"):
        for reference in ("R_A", "R_C"):
            states = {(x["subject"], x["video"], x["gt_index"], x["evidence_condition"]): x["state"] for x in details
                      if x["setting"] == "BoostingVRME/SAMMLV" and x["anchor"] == anchor and x["reference_condition"] == reference}
            count = sum(value == "TP" and states.get((s, v, g, "E_3")) == "B"
                        for (s, v, g, evidence), value in states.items() if evidence == "E_7")
            add(f"bs_fixed_reference_evidence_tp_to_b_{anchor}_{reference}", count == 0, count, 0,
                "fixed reference E_7 -> E_3")
    a_ms, c_ms = PARENT.selections("ME-TST/SAMMLV")
    add("ms_reference_factor_collapsed", all(a_ms[s]["ref"] == c_ms[s]["ref"] for s in a_ms),
        sum(a_ms[s]["ref"] == c_ms[s]["ref"] for s in a_ms), len(a_ms), "foldwise R_A == R_C")
    comparisons = read_csv(EXP6B / "candidate_set_comparison.csv")
    bc_deleted = {anchor: sum(int(x["R_A_candidate_count"]) - int(x["R_C_candidate_count"]) for x in comparisons
                              if x["setting"] == "BoostingVRME/CAS(ME)3" and x["evidence_condition"] == "E_7" and x["anchor"] == anchor)
                  for anchor in ("ANCHOR-A", "ANCHOR-C")}
    add("bc_reference_candidate_deletion", all(value == 195 for value in bc_deleted.values()),
        bc_deleted, {"ANCHOR-A": 195, "ANCHOR-C": 195}, "per-anchor pooled subject-video candidate events")
    fp = read_csv(EXP6B / "fp_lineage_detail.csv")
    for anchor, expected in (("ANCHOR-A", 29), ("ANCHOR-C", 46)):
        count = sum(x["setting"] == "BoostingVRME/CAS(ME)3" and x["anchor"] == anchor and
                    x["contrast"] == "R_A,E_7->R_C,E_7" and x["source"] == "R1" for x in fp)
        add(f"bc_r1_removed_fp_{anchor}", count == expected, count, expected, "fixed all-7 reference-only lineage")
    return checks


def bootstrap(rows: list[dict]) -> list[dict]:
    output = []
    for metric in ("delta_G", "delta_L", "delta_S"):
        by_subject = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["group"] in ("TP_TO_B", "TP_RETAINED"):
                by_subject[row["subject"]][row["group"]].append(float(row[metric]))
        paired = [(np.mean(parts["TP_TO_B"]), np.mean(parts["TP_RETAINED"])) for parts in by_subject.values()
                  if parts["TP_TO_B"] and parts["TP_RETAINED"]]
        n = len(paired)
        if n < 2:
            output.append({"comparison": "TP_TO_B_MINUS_TP_RETAINED", "metric": metric, "estimate": "",
                           "ci95_low": "", "ci95_high": "", "N_subjects": n, "resamples": BOOTSTRAPS,
                           "seed": SEED, "status": "INSUFFICIENT_SUBJECT_SUPPORT"})
            continue
        values = np.asarray(paired, dtype=float)
        rng = np.random.default_rng(SEED)
        draw = rng.integers(0, n, size=(BOOTSTRAPS, n))
        delta = values[draw, 0].mean(axis=1) - values[draw, 1].mean(axis=1)
        output.append({"comparison": "TP_TO_B_MINUS_TP_RETAINED", "metric": metric,
                       "estimate": float(values[:, 0].mean() - values[:, 1].mean()),
                       "ci95_low": float(np.quantile(delta, .025)), "ci95_high": float(np.quantile(delta, .975)),
                       "N_subjects": n, "resamples": BOOTSTRAPS, "seed": SEED, "status": "DESCRIPTIVE"})
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    identity_checks = verify_parent_artifacts()
    ENGINE.SCALES, ENGINE.REFERENCE_SCALES = SCALES, SCALES
    PARENT.fair.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    gt_rows, topology_rows, g_rows, l_rows = [], [], [], []
    bc_fp_rows, subject_rows, cross_rows = [], [], []
    summaries = []
    # Primary B/S and the two reference-collapse controls. B/C is separately handled below.
    settings = (("boostingvrme", "sammlv", "BoostingVRME/SAMMLV", "BoostingVRME/SAMMLV"),
                ("metst", "sammlv", "ME-TST+/SAMMLV", "ME-TST/SAMMLV"),
                ("metst", "casme3", "ME-TST+/CAS(ME)3", "ME-TST/CAS(ME)3"))
    for backbone, dataset, display, artifact_setting in settings:
        a_selection, c_selection = PARENT.selections(artifact_setting)
        records, subjects, _, _ = PARENT.fair.load_data(backbone, dataset)
        outer_k, _ = PARENT.fair.fold_priors(records, subjects, backbone)
        subject_index = {subject: idx for idx, subject in enumerate(subjects)}
        for anchor, selected in (("ANCHOR-A", a_selection), ("ANCHOR-C", c_selection)):
            current = []
            for record in records:
                subject, video = record["subject"], record["video"]
                k = int(outer_k[subject_index[subject]])
                common = {"setting": display, "backbone": backbone, "dataset": dataset, "anchor": anchor,
                          "subject": subject, "video": video, "k": k, "rho": selected[subject]["rho"], "tau": selected[subject]["tau"]}
                a_state = build_state(record, backbone, k, a_selection[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
                c_state = build_state(record, backbone, k, c_selection[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
                labels_a, labels_c = PARENT.gt_labels(record, a_state), PARENT.gt_labels(record, c_state)
                for gt_index, raw_gt in enumerate(record["gt"]):
                    gt = (int(raw_gt[0]), int(raw_gt[2]))
                    label_a, label_c = labels_a[gt_index], labels_c[gt_index]
                    group = {("TP", "B"): "TP_TO_B", ("TP", "TP"): "TP_RETAINED", ("B", "TP"): "B_TO_TP", ("B", "B"): "B_PERSISTENT"}.get((label_a, label_c), "OTHER")
                    event_a, event_c = representative(a_state, gt), representative(c_state, gt)
                    desc_a, desc_c = descriptor(a_state, event_a), descriptor(c_state, event_c)
                    delta = {key: (float(desc_c[key]) - float(desc_a[key])) if desc_a[key] != "" and desc_c[key] != "" else ""
                             for key in ("raw_prominence", "reference_dynamic_range", "G", "L", "S")}
                    topology = transition_type(desc_a, desc_c, gt)
                    mult_a, mult_c = len(gt_compatible(a_state, gt)), len(gt_compatible(c_state, gt))
                    component = score_component(delta["G"], delta["L"]) if delta["G"] != "" else "NOT_APPLICABLE"
                    source = g_source(delta["raw_prominence"], delta["G"]) if delta["G"] != "" else "NOT_APPLICABLE"
                    alignment, match_counts = alignment_rows({**common, "gt_index": gt_index, "group": group}, desc_a, desc_c)
                    row = {**common, "gt_index": gt_index, "gt_onset": gt[0], "gt_offset": gt[1], "gt_center": (gt[0] + gt[1]) / 2,
                           "state_A": label_a, "state_C": label_c, "group": group, "reference_A": a_state["ref"], "reference_C": c_state["ref"],
                           "candidate_count_compatible_A": mult_a, "candidate_count_compatible_C": mult_c,
                           "total_reference_candidates_in_gt_A": sum(gt[0] <= x["peak"] <= gt[1] for x in a_state["p0"]),
                           "total_reference_candidates_in_gt_C": sum(gt[0] <= x["peak"] <= gt[1] for x in c_state["p0"]),
                           "candidate_peak_inside_gt_A": "" if not desc_a["peak"] else ("YES" if gt[0] <= desc_a["peak"] <= gt[1] else "NO"),
                           "candidate_peak_inside_gt_C": "" if not desc_c["peak"] else ("YES" if gt[0] <= desc_c["peak"] <= gt[1] else "NO"),
                           "peak_to_gt_center_abs_distance_A": "" if not desc_a["peak"] else abs(desc_a["peak"] - (gt[0]+gt[1])/2),
                           "peak_to_gt_center_abs_distance_C": "" if not desc_c["peak"] else abs(desc_c["peak"] - (gt[0]+gt[1])/2),
                           "normalized_peak_distance_A": "" if not desc_a["peak"] else abs(desc_a["peak"] - (gt[0]+gt[1])/2) / max(k, 1),
                           "normalized_peak_distance_C": "" if not desc_c["peak"] else abs(desc_c["peak"] - (gt[0]+gt[1])/2) / max(k, 1),
                           "transition_type": topology, "multiplicity_drop": "YES" if mult_c < mult_a else "NO", "multiplicity_gain": "YES" if mult_c > mult_a else "NO",
                           "delta_raw_prominence": delta["raw_prominence"], "delta_range": delta["reference_dynamic_range"], "delta_G": delta["G"], "delta_L": delta["L"], "delta_S": delta["S"],
                           "g_drop_source": source, "tp_loss_component": component,
                           "num_match_same": match_counts["MATCH-SAME"], "num_match_changed": match_counts["MATCH-CHANGED"],
                           "num_match_lost": match_counts["MATCH-LOST"], "num_match_gained": match_counts["MATCH-GAINED"],
                           "median_driver_change": "YES" if desc_a["median_driver"] != desc_c["median_driver"] else "NO",
                           "distance_to_threshold_A": "" if desc_a["S"] == "" else float(desc_a["S"]) - a_state["tau"],
                           "distance_to_threshold_C": "" if desc_c["S"] == "" else float(desc_c["S"]) - c_state["tau"]}
                    for suffix, desc in (("A", desc_a), ("C", desc_c)):
                        for key in ("peak", "onset", "offset", "raw_curve_peak_value", "reference_peak_value", "raw_prominence", "reference_response_max", "reference_response_min", "reference_dynamic_range", "H", "G", "L", "S", "median_driver"):
                            row[f"{key}_{suffix}"] = desc[key]
                    gt_rows.append(row)
                    topology_rows.append({key: row[key] for key in ("setting", "anchor", "subject", "video", "gt_index", "group", "transition_type", "candidate_count_compatible_A", "candidate_count_compatible_C", "total_reference_candidates_in_gt_A", "total_reference_candidates_in_gt_C", "multiplicity_drop", "multiplicity_gain", "candidate_peak_inside_gt_A", "candidate_peak_inside_gt_C", "peak_to_gt_center_abs_distance_A", "peak_to_gt_center_abs_distance_C", "normalized_peak_distance_A", "normalized_peak_distance_C")})
                    for condition, desc in (("R_A", desc_a), ("R_C", desc_c)):
                        g_rows.append({**common, "gt_index": gt_index, "group": group, "reference_condition": condition,
                                       **{key: desc[key] for key in ("peak", "raw_curve_peak_value", "reference_peak_value", "raw_prominence", "reference_response_max", "reference_response_min", "reference_dynamic_range", "G")},
                                       "delta_raw_prominence": delta["raw_prominence"], "delta_range": delta["reference_dynamic_range"], "delta_G": delta["G"]})
                    l_rows.extend(alignment)
                    current.append(row)
            for subject in subjects:
                entries = [x for x in current if x["subject"] == subject]
                group_rows = {name: [x for x in entries if x["group"] == name] for name in ("TP_TO_B", "TP_RETAINED", "B_TO_TP")}
                subject_rows.append({"setting": display, "anchor": anchor, "subject": subject,
                                     "tp_to_b": len(group_rows["TP_TO_B"]), "tp_retained": len(group_rows["TP_RETAINED"]), "b_to_tp": len(group_rows["B_TO_TP"]),
                                     "mean_delta_G_tp_to_b": np.mean([x["delta_G"] for x in group_rows["TP_TO_B"]]) if group_rows["TP_TO_B"] else "",
                                     "mean_delta_L_tp_to_b": np.mean([x["delta_L"] for x in group_rows["TP_TO_B"]]) if group_rows["TP_TO_B"] else "",
                                     "mean_delta_S_tp_to_b": np.mean([x["delta_S"] for x in group_rows["TP_TO_B"]]) if group_rows["TP_TO_B"] else "",
                                     "removed_fp_reference": 0, "candidate_disappearance_fp": 0})
            for group in ("TP_TO_B", "TP_RETAINED", "B_TO_TP", "B_PERSISTENT"):
                entries = [x for x in current if x["group"] == group]
                summaries.append({"setting": display, "anchor": anchor, "group": group, "event_count": len(entries),
                                  "subject_count": len({x["subject"] for x in entries}),
                                  "mean_delta_raw_prominence": np.mean([x["delta_raw_prominence"] for x in entries]) if entries else "",
                                  "mean_delta_range": np.mean([x["delta_range"] for x in entries]) if entries else "",
                                  "mean_delta_G": np.mean([x["delta_G"] for x in entries]) if entries else "",
                                  "mean_delta_L": np.mean([x["delta_L"] for x in entries]) if entries else "",
                                  "mean_delta_S": np.mean([x["delta_S"] for x in entries]) if entries else ""})

    # Secondary B/C R1 final-FP lineage under the exact frozen reference-only contrast.
    backbone, dataset, display, artifact_setting = "boostingvrme", "casme3", "BoostingVRME/CAS(ME)3", "BoostingVRME/CAS(ME)3"
    a_selection, c_selection = PARENT.selections(artifact_setting)
    records, subjects, _, _ = PARENT.fair.load_data(backbone, dataset)
    outer_k, _ = PARENT.fair.fold_priors(records, subjects, backbone)
    subject_index = {subject: idx for idx, subject in enumerate(subjects)}
    for anchor, selected in (("ANCHOR-A", a_selection), ("ANCHOR-C", c_selection)):
        by_subject = defaultdict(list)
        for record in records:
            subject, video = record["subject"], record["video"]
            k = int(outer_k[subject_index[subject]])
            a_state = build_state(record, backbone, k, a_selection[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
            c_state = build_state(record, backbone, k, c_selection[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
            c_by_peak = {x["peak"]: x for x in c_state["p0"]}
            for event in a_state["p2"]:
                if event["matched_gt"] >= 0 or event["peak"] in c_by_peak:
                    continue
                desc = descriptor(a_state, event)
                overlap = [x for x in c_state["p0"] if iou(x["interval"], event["interval"]) > 0]
                closest = min(c_state["p0"], key=lambda x: (abs(x["peak"] - event["peak"]), x["peak"])) if c_state["p0"] else None
                classification = "FP-PEAK-SHIFTED" if overlap else "FP-REF-ABSENT"
                row = {"setting": display, "anchor": anchor, "subject": subject, "video": video, "k": k, "rho": selected[subject]["rho"], "tau": selected[subject]["tau"],
                       "source": "R1", "classification": classification, "peak_A": desc["peak"], "onset_A": desc["onset"], "offset_A": desc["offset"],
                       "raw_curve_peak_value_A": desc["raw_curve_peak_value"], "reference_peak_value_A": desc["reference_peak_value"], "raw_prominence_A": desc["raw_prominence"],
                       "G_A": desc["G"], "L_A": desc["L"], "S_A": desc["S"], "exact_peak_in_R_C": "NO", "overlapping_R_C_potential_event": "YES" if overlap else "NO",
                       "closest_peak_R_C": "" if closest is None else closest["peak"], "closest_peak_distance": "" if closest is None else abs(closest["peak"]-event["peak"]),
                       "potential_event_R_C": "YES" if overlap else "NO", "candidate_disappearance": "YES", "delta_raw_prominence": "", "delta_G": "", "delta_L": "", "delta_S": ""}
                bc_fp_rows.append(row); by_subject[subject].append(row)
                cross_rows.append({"comparison_group": "B_C_REMOVED_FP", **row})
        for subject in subjects:
            entries = by_subject[subject]
            subject_rows.append({"setting": display, "anchor": anchor, "subject": subject, "tp_to_b": 0, "tp_retained": 0, "b_to_tp": 0,
                                 "mean_delta_G_tp_to_b": "", "mean_delta_L_tp_to_b": "", "mean_delta_S_tp_to_b": "",
                                 "removed_fp_reference": len(entries), "candidate_disappearance_fp": sum(x["candidate_disappearance"] == "YES" for x in entries)})

    primary = [x for x in gt_rows if x["setting"] == "BoostingVRME/SAMMLV"]
    for row in primary:
        if row["group"] == "TP_TO_B":
            cross_rows.append({"comparison_group": "B_S_LOST_TP", "setting": row["setting"], "anchor": row["anchor"], "subject": row["subject"], "video": row["video"],
                               "raw_prominence_A": row["raw_prominence_A"], "G_A": row["G_A"], "L_A": row["L_A"], "S_A": row["S_A"],
                               "candidate_count_compatible_A": row["candidate_count_compatible_A"], "delta_raw_prominence": row["delta_raw_prominence"], "delta_G": row["delta_G"], "delta_L": row["delta_L"], "delta_S": row["delta_S"], "candidate_disappearance": "NO"})
    boot = bootstrap(primary)
    # Parent-level count guard makes the result fail closed even though descriptors are richer.
    for anchor, expected in (("ANCHOR-A", 7), ("ANCHOR-C", 6)):
        observed = sum(x["anchor"] == anchor and x["group"] == "TP_TO_B" for x in primary)
        identity_checks.append({"check": f"own_bs_tp_to_b_{anchor}", "status": "PASS" if observed == expected else "FAIL", "observed": observed, "expected": expected, "detail": "frozen E_7 replay"})
        if observed != expected:
            raise RuntimeError("EXP6C_PARENT_REPLAY_FAILED")

    write_csv("identity_check.csv", identity_checks)
    write_csv("reference_gt_detail.csv", gt_rows)
    write_csv("reference_gt_summary.csv", summaries)
    write_csv("tp_to_b_detail.csv", [x for x in primary if x["group"] == "TP_TO_B"], list(gt_rows[0]))
    write_csv("retained_tp_control.csv", [x for x in primary if x["group"] == "TP_RETAINED"], list(gt_rows[0]))
    write_csv("b_to_tp_control.csv", [x for x in primary if x["group"] == "B_TO_TP"], list(gt_rows[0]))
    write_csv("G_decomposition.csv", g_rows)
    write_csv("L_alignment_decomposition.csv", l_rows)
    write_csv("candidate_topology_transition.csv", topology_rows)
    write_csv("bc_removed_fp_reference.csv", bc_fp_rows)
    write_csv("cross_setting_tp_fp_comparison.csv", cross_rows)
    write_csv("subject_reference_diagnostic.csv", subject_rows)
    write_csv("mechanism_bootstrap.csv", boot)
    protocol = {"experiment": "EXP-6C Reference-Scale Candidate Diagnostic", "parent": "EXP-6B", "primary": "BoostingVRME/SAMMLV", "secondary": "BoostingVRME/CAS(ME)3", "controls": ["ME-TST+/SAMMLV", "ME-TST+/CAS(ME)3"], "evidence": "fixed E_7", "score": "S=(G+L)/2", "selection": "saved EXP-2A/EXP-2C foldwise rho/tau/reference only", "candidate_matching": "GT-centric; best S then IoU then canonical temporal order", "local_neighborhood": "GT closed interval only", "bootstrap": {"resamples": BOOTSTRAPS, "seed": SEED, "unit": "outer subject"}, "forbidden_changes": ["new method", "new scale", "new tau", "new fusion", "outer-test rule selection"], "canonical_GLSD_modified": False}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    manifest = {"source_sha256": {"EXP6B": digest(EXP6B / "run_exp6b.py"), "EXP2A": digest(EXP2A / "run_exp2a.py"), "EXP2C": digest(EXP2C / "run_exp2c.py"), "engine": digest(SIGNED / "unified_persistence.py"), "EXP6C": digest(Path(__file__))}, "parent_replay": "PASS", "identity_cells": "PASS", "fixed_evidence": "E_7", "new_method_executed": False, "outer_test_rule_selected": False, "canonical_GLSD_modified": False}
    (OUT / "replay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_analysis(primary, bc_fp_rows, boot, cross_rows)


def write_analysis(primary: list[dict], fp_rows: list[dict], boot: list[dict], cross_rows: list[dict]) -> None:
    lost = [x for x in primary if x["group"] == "TP_TO_B"]
    retained = [x for x in primary if x["group"] == "TP_RETAINED"]
    reverse = [x for x in primary if x["group"] == "B_TO_TP"]
    by_anchor = {anchor: [x for x in lost if x["anchor"] == anchor] for anchor in ("ANCHOR-A", "ANCHOR-C")}
    affected = sorted({x["subject"] for x in lost})
    g_dom = sum(x["tp_loss_component"] == "G_DOMINANT_DROP" for x in lost)
    l_dom = sum(x["tp_loss_component"] == "L_DOMINANT_DROP" for x in lost)
    shifts = sum(x["transition_type"] in ("TYPE-SHIFT-IN", "TYPE-SHIFT-OUT") for x in lost)
    multi_loss = sum(x["multiplicity_drop"] == "YES" for x in lost)
    mean = lambda rows, key: float(np.mean([x[key] for x in rows])) if rows else float("nan")
    cross_stable = len(affected) >= 2
    if lost and g_dom > len(lost) / 2 and mean(lost, "delta_G") < mean(retained, "delta_G") and cross_stable:
        verdict, next_exp = "REFERENCE_G_COLLAPSE_SUPPORTED", "REFERENCE_G_NORMALIZATION_DIAGNOSTIC"
    elif lost and l_dom > len(lost) / 2 and cross_stable:
        verdict, next_exp = "REFERENCE_L_ALIGNMENT_COLLAPSE_SUPPORTED", "REFERENCE_ANCHORED_L_DIAGNOSTIC"
    elif lost and (shifts > len(lost) / 2 or multi_loss > len(lost) / 2) and cross_stable:
        verdict, next_exp = "REFERENCE_CANDIDATE_TOPOLOGY_SUPPORTED", "REFERENCE_CANDIDATE_RETENTION_DIAGNOSTIC"
    elif lost and cross_stable:
        verdict, next_exp = "REFERENCE_EFFECT_MIXED", "STOP_REFERENCE_RULE_MINING"
    else:
        verdict, next_exp = "NO_STABLE_REFERENCE_FAILURE_SIGNATURE", "STOP_REFERENCE_RULE_MINING"
    sources = Counter(x["g_drop_source"] for x in lost)
    g_source = "RAW_PROMINENCE" if sources["G_DROP_PROMINENCE_DRIVEN"] > len(lost)/2 else "RANGE_NORMALIZATION" if sources["G_DROP_NORMALIZATION_DRIVEN"] > len(lost)/2 else "MIXED" if lost else "NOT_APPLICABLE"
    fp_mech = Counter(x["classification"] for x in fp_rows).most_common(1)[0][0] if fp_rows else "NOT_APPLICABLE"
    tp_values = {key: [float(x[key]) for x in lost] for key in ("raw_prominence_A", "G_A", "L_A", "S_A")}
    fp_values = {key: [float(x[key]) for x in fp_rows] for key in ("raw_prominence_A", "G_A", "L_A", "S_A")}
    overlap = bool(lost and fp_rows) and all(max(tp_values[key]) >= min(fp_values[key]) and max(fp_values[key]) >= min(tp_values[key]) for key in tp_values)
    separable = "NO" if overlap else "INCONCLUSIVE"
    text = ["# EXP-6C — Reference-Scale Candidate Diagnostic", "", "## Integrity", "", "- Parent EXP-6B replay, identity cells, same-reference isolation, B/S TP→B counts, and B/C R1 FP lineage were rechecked from retained artifacts and fresh frozen replays: **PASS**.", "- All primary comparisons use fixed all-7 evidence (`E_7`), saved foldwise `rho`/`tau`, and `S=(G+L)/2`. No rule, scale, threshold, fusion, or decoder was selected from outer-test labels.", "", "## Answers", "", f"1. B/S has {len(lost)} TP→B events (Anchor-A={len(by_anchor['ANCHOR-A'])}; Anchor-C={len(by_anchor['ANCHOR-C'])}) across {len(affected)} outer subjects. The gate verdict is **{verdict}**.", f"2. Mean TP→B changes are Δraw-prominence={mean(lost, 'delta_raw_prominence'):.6f}, Δrange={mean(lost, 'delta_range'):.6f}, ΔG={mean(lost, 'delta_G'):.6f}, ΔL={mean(lost, 'delta_L'):.6f}, ΔS={mean(lost, 'delta_S'):.6f}; the G-source label is **{g_source}**. Exact per-event values are retained in `tp_to_b_detail.csv` and `G_decomposition.csv`.", f"3. G-dominant/L-dominant/other TP losses are {g_dom}/{l_dom}/{len(lost)-g_dom-l_dom}; all scale-level correspondence and median drivers are in `L_alignment_decomposition.csv` (no alignment tolerance was searched).", f"4. Systematic representative peak shift: **{'YES' if shifts > len(lost)/2 else 'NO'}** ({shifts}/{len(lost)}); candidate multiplicity loss: **{'YES' if multi_loss > len(lost)/2 else 'NO'}** ({multi_loss}/{len(lost)}).", f"5. Retained TP ({len(retained)} events) has mean ΔG={mean(retained, 'delta_G'):.6f}, ΔL={mean(retained, 'delta_L'):.6f}, ΔS={mean(retained, 'delta_S'):.6f}; compare with the lost-TP values above rather than inferring from a handful of examples.", f"6. B→TP reverse control contains {len(reverse)} events; its event-level score decomposition is in `b_to_tp_control.csv`, which tests whether the reference effect is directionally mirrored without fitting a rule.", f"7. B/C has {len(fp_rows)} exact R1 removed final FPs. Its dominant topology label is **{fp_mech}**; each candidate's R_A raw/G/L/S evidence and R_C potential-event check are in `bc_removed_fp_reference.csv`.", f"8. Lost-TP versus removed-FP structural separability is **{separable}**. `{('All four R_A raw-prominence/G/L/S ranges overlap, so these diagnostics do not support a simple unified deterministic correction.' if separable == 'NO' else 'The small descriptive samples do not justify a deterministic separating rule.')}`", f"9. Subject-aware predeclared bootstrap rows are in `mechanism_bootstrap.csv`; rows marked `INSUFFICIENT_SUBJECT_SUPPORT` are intentionally not interpreted as significance.", f"10. A GLSD modification is **not justified by this diagnostic alone**. Recommended next experiment: **{next_exp}**.", "", "## Limits", "", "The GT-centric representative is diagnostic only: formal evaluation, prediction sets, Recognition F1, and STRS were not recomputed or changed. The B/C FP transition records R1 candidate disappearance; it does not test candidate union, dual references, ensembles, adaptive scale selection, new weights, calibration, or a new threshold.", ""]
    (OUT / "EXP6C_ANALYSIS.md").write_text("\n".join(text), encoding="utf-8")
    print("================================")
    print("EXP-6C REFERENCE DIAGNOSTIC VERDICT")
    print("================================")
    print("Parent replay: PASS")
    print(f"B/S TP→B count: Anchor-A = {len(by_anchor['ANCHOR-A'])}; Anchor-C = {len(by_anchor['ANCHOR-C'])}")
    print(f"Affected subjects: {len(affected)} ({', '.join(affected)})")
    print(f"Dominant TP-loss component: {'G' if verdict == 'REFERENCE_G_COLLAPSE_SUPPORTED' else 'L' if verdict == 'REFERENCE_L_ALIGNMENT_COLLAPSE_SUPPORTED' else 'TOPOLOGY' if verdict == 'REFERENCE_CANDIDATE_TOPOLOGY_SUPPORTED' else 'MIXED' if verdict == 'REFERENCE_EFFECT_MIXED' else 'NONE'}")
    print(f"G drop source: {g_source}")
    print("L drop source: NOT_APPLICABLE" if l_dom == 0 else "L drop source: ALIGNMENT_OR_MEDIAN_CHANGE")
    print(f"Systematic peak shift: {'YES' if shifts > len(lost)/2 else 'NO'}")
    print(f"Candidate multiplicity loss: {'YES' if multi_loss > len(lost)/2 else 'NO'}")
    print(f"B/C removed-FP dominant mechanism: {fp_mech}")
    print(f"Lost-TP vs removed-FP separable: {separable}")
    print(f"Cross-subject mechanism stable: {'YES' if cross_stable else 'NO'}")
    print(f"Recommended next experiment: {next_exp}")
    print("New method executed: NO")
    print("Outer-test rule selected: NO")
    print("Canonical GLSD modified: NO")
    print("================================")


if __name__ == "__main__":
    main()
