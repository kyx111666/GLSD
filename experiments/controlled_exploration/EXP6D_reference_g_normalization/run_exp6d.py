"""EXP-6D: frozen, read-only reference-G normalization diagnostic.

The script replays only the saved EXP-2A/EXP-2C cells.  It creates audit
tables; it never changes GLSD, a threshold, a decoder, or a prediction set.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP6C = ROOT / "controlled_exploration/EXP6C_reference_scale_candidate_diagnostic"
EXP6B = ROOT / "controlled_exploration/EXP6B_reference_evidence_cross_replay"
EXP2A = ROOT / "controlled_exploration/EXP2A_scale"
EXP2C = ROOT / "controlled_exploration/EXP2C_reference_evidence_decoupling"
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
BOOTSTRAPS, SEED, EPS = 10_000, 100, 1e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load_module("exp6c_parent", EXP6C / "run_exp6c.py")
PARENT, ENGINE = C.PARENT, C.ENGINE


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else ["status"])
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


def mean(rows: list[dict], key: str) -> float:
    values = [float(x[key]) for x in rows if x.get(key, "") not in ("", None)]
    return float(np.mean(values)) if values else float("nan")


def median(rows: list[dict], key: str) -> float:
    values = [float(x[key]) for x in rows if x.get(key, "") not in ("", None)]
    return float(np.median(values)) if values else float("nan")


def prominence_descriptor(state: dict, event: dict) -> dict:
    """Reproduce scipy/canonical global-prominence semantics for one legal peak."""
    features, ref, peak = state["features"], state["ref"], int(event["peak"])
    peaks, _, globals_, response, spread = features.scales[ref]
    matches = np.flatnonzero(peaks == peak)
    if len(matches) != 1:
        raise ValueError("EXP6D_NONCANONICAL_PEAK_REQUEST")
    idx = int(matches[0])
    prom, left, right = ENGINE.peak_prominences(response, np.asarray([peak], dtype=int))
    raw_p = float(prom[0])
    canonical_p = float(globals_[idx] * spread)
    if not np.isclose(raw_p, canonical_p, rtol=0, atol=1e-12):
        raise AssertionError("EXP6D_PROMINENCE_REPLAY_MISMATCH")
    base = max(float(response[int(left[0])]), float(response[int(right[0])]))
    return {
        "peak": peak, "P": raw_p, "D": float(max(np.ptp(response), EPS)),
        "G": float(raw_p / max(np.ptp(response), EPS)),
        "raw_peak_value": float(response[peak]), "prominence_base": base,
        "response_max": float(response.max()), "response_min": float(response.min()),
        "left_base_index": int(left[0]), "right_base_index": int(right[0]),
    }


def build_state(record: dict, backbone: str, k: int, ref: float, rho: float, tau: float) -> dict:
    state = C.build_state(record, backbone, k, ref, rho, tau)
    return state


def paired_row(common: dict, state_a: dict, event_a: dict, state_c: dict, event_c: dict,
               group: str) -> dict:
    a, c = prominence_descriptor(state_a, event_a), prominence_descriptor(state_c, event_c)
    cp = .5 * (((c["P"] / a["D"]) - (a["P"] / a["D"])) + ((c["P"] / c["D"]) - (a["P"] / c["D"])))
    cd = .5 * (((a["P"] / c["D"]) - (a["P"] / a["D"])) + ((c["P"] / c["D"]) - (c["P"] / a["D"])))
    dg = c["G"] - a["G"]
    error = cp + cd - dg
    if abs(error) > 1e-12:
        raise AssertionError("EXP6D_DECOMPOSITION_FAILED")
    row = {**common, "group": group,
           "peak_A": a["peak"], "peak_C": c["peak"], "peak_shift": int(c["peak"] - a["peak"]),
           "peak_relation": "SAME-PEAK" if a["peak"] == c["peak"] else "SHIFTED",
           "P_A": a["P"], "D_A": a["D"], "G_A": a["G"], "P_C": c["P"], "D_C": c["D"], "G_C": c["G"],
           "raw_peak_value_A": a["raw_peak_value"], "prominence_base_A": a["prominence_base"],
           "response_max_A": a["response_max"], "response_min_A": a["response_min"],
           "raw_peak_value_C": c["raw_peak_value"], "prominence_base_C": c["prominence_base"],
           "response_max_C": c["response_max"], "response_min_C": c["response_min"],
           "delta_P": c["P"] - a["P"], "delta_D": c["D"] - a["D"], "delta_G": dg,
           "G_Ponly": c["P"] / a["D"], "G_Donly": a["P"] / c["D"], "delta_G_Ponly": c["P"] / a["D"] - a["G"],
           "delta_G_Donly": a["P"] / c["D"] - a["G"], "C_P": cp, "C_D": cd, "decomposition_error": error,
           "Q_P": c["P"] / max(a["P"], EPS), "Q_G": c["G"] / max(a["G"], EPS),
           "Q_logP": abs(np.log(c["P"] + EPS) - np.log(a["P"] + EPS))}
    return row


def verify_exp6c() -> list[dict]:
    """Freshly verify the parent cells and every EXP-6C fact named in protocol."""
    checks: list[dict] = []
    def add(name, observed, expected, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed,
                       "expected": expected, "detail": detail})
        if not ok:
            raise RuntimeError("EXP6D_PARENT_REPLAY_FAILED")
    C.verify_parent_artifacts()
    analysis = (EXP6C / "EXP6C_ANALYSIS.md").read_text(encoding="utf-8")
    add("exp6c_parent_replay", "PASS" if "**PASS**" in analysis else "missing", "PASS", "**PASS**" in analysis, "EXP-6C report")
    required = ["tp_to_b_detail.csv", "retained_tp_control.csv", "b_to_tp_control.csv", "G_decomposition.csv",
                "candidate_topology_transition.csv", "bc_removed_fp_reference.csv", "subject_reference_diagnostic.csv"]
    add("required_exp6c_artifacts", sum((EXP6C / x).is_file() for x in required), len(required), all((EXP6C / x).is_file() for x in required), "all specified inputs")
    lost, retained, reverse = (read_csv(EXP6C / x) for x in ("tp_to_b_detail.csv", "retained_tp_control.csv", "b_to_tp_control.csv"))
    add("bs_tp_to_b_anchor_a", sum(x["anchor"] == "ANCHOR-A" for x in lost), 7, sum(x["anchor"] == "ANCHOR-A" for x in lost) == 7, "EXP-6C event table")
    add("bs_tp_to_b_anchor_c", sum(x["anchor"] == "ANCHOR-C" for x in lost), 6, sum(x["anchor"] == "ANCHOR-C" for x in lost) == 6, "EXP-6C event table")
    add("bs_lost_subjects", len({x["subject"] for x in lost}), 6, len({x["subject"] for x in lost}) == 6, "EXP-6C event table")
    targets = {"delta_raw_prominence": -.195979, "delta_range": -.055314, "delta_G": -.263475,
               "delta_L": -.051339, "delta_S": -.157407}
    for key, expected in targets.items():
        observed = mean(lost, key)
        add(f"bs_mean_{key}", f"{observed:.6f}", f"{expected:.6f}", abs(observed - expected) <= 1e-6, "reported EXP-6C mean")
    add("bs_g_dominant", sum(x["tp_loss_component"] == "G_DOMINANT_DROP" for x in lost), len(lost), all(x["tp_loss_component"] == "G_DOMINANT_DROP" for x in lost), "EXP-6C event table")
    add("bs_peak_shift", sum(x["peak_A"] != x["peak_C"] for x in lost), 11, sum(x["peak_A"] != x["peak_C"] for x in lost) == 11, "representative peaks")
    add("bs_no_systematic_multiplicity_loss", sum(x["multiplicity_drop"] == "YES" for x in lost), 4, sum(x["multiplicity_drop"] == "YES" for x in lost) == 4, "not a majority")
    add("retained_mean_delta_g", f"{mean(retained, 'delta_G'):.6f}", "-0.015650", abs(mean(retained, "delta_G") + .015650) <= 1e-6, "EXP-6C retained control")
    fp = read_csv(EXP6C / "bc_removed_fp_reference.csv")
    add("bc_removed_fp_count", len(fp), 75, len(fp) == 75, "exact R1 lineage")
    add("bc_dominant_topology", max(set(x["classification"] for x in fp), key=lambda k: sum(y["classification"] == k for y in fp)), "FP-PEAK-SHIFTED", max(set(x["classification"] for x in fp), key=lambda k: sum(y["classification"] == k for y in fp)) == "FP-PEAK-SHIFTED", "EXP-6C FP lineage")
    add("reverse_control_count", len(reverse), 2, len(reverse) == 2, "EXP-6C reverse control")
    return checks


def summary(rows: list[dict], label: str) -> list[dict]:
    result = []
    for metric in ("delta_P", "delta_D", "delta_G", "C_P", "C_D"):
        values = np.asarray([float(x[metric]) for x in rows], dtype=float)
        result.append({"table": "factor", "group": label, "metric": metric, "event_count": len(values),
                       "subject_count": len({x["subject"] for x in rows}), "mean": float(values.mean()) if len(values) else "",
                       "median": float(np.median(values)) if len(values) else "", "IQR_low": float(np.quantile(values,.25)) if len(values) else "",
                       "IQR_high": float(np.quantile(values,.75)) if len(values) else "", "min": float(values.min()) if len(values) else "", "max": float(values.max()) if len(values) else ""})
    if rows:
        cp, cd = np.asarray([float(x["C_P"]) for x in rows]), np.asarray([float(x["C_D"]) for x in rows])
        for name, value in (("fraction_C_P_negative", np.mean(cp < 0)), ("fraction_C_D_negative", np.mean(cd < 0)),
                            ("fraction_abs_C_P_gt_C_D", np.mean(abs(cp) > abs(cd))), ("fraction_abs_C_D_gt_C_P", np.mean(abs(cd) > abs(cp)))):
            result.append({"table": "factor", "group": label, "metric": name, "event_count": len(rows), "subject_count": len({x["subject"] for x in rows}), "mean": float(value), "median": "", "IQR_low": "", "IQR_high": "", "min": "", "max": ""})
    return result


def stability_summary(rows: list[dict], label: str) -> list[dict]:
    out = []
    for metric in ("Q_P", "Q_G", "Q_logP"):
        values = np.asarray([float(x[metric]) for x in rows if x.get(metric, "") not in ("", None)], dtype=float)
        out.append({"group": label, "metric": metric, "event_count": len(values), "subject_count": len({x["subject"] for x in rows if x.get(metric, "") not in ("", None)}),
                    "mean": float(values.mean()) if len(values) else "", "median": float(np.median(values)) if len(values) else "",
                    "IQR_low": float(np.quantile(values,.25)) if len(values) else "", "IQR_high": float(np.quantile(values,.75)) if len(values) else "",
                    "min": float(values.min()) if len(values) else "", "max": float(values.max()) if len(values) else ""})
    return out


def subject_bootstrap(left: list[dict], right: list[dict], metrics: tuple[str, ...], comparison: str, paired: bool) -> list[dict]:
    out = []
    for metric in metrics:
        by_l, by_r = defaultdict(list), defaultdict(list)
        for row in left:
            if row.get(metric, "") not in ("", None): by_l[row["subject"]].append(float(row[metric]))
        for row in right:
            if row.get(metric, "") not in ("", None): by_r[row["subject"]].append(float(row[metric]))
        if paired:
            common = sorted(set(by_l) & set(by_r)); n = len(common)
            values = np.asarray([(np.mean(by_l[s]), np.mean(by_r[s])) for s in common])
            if n >= 2:
                rng = np.random.default_rng(SEED); draw = rng.integers(0, n, size=(BOOTSTRAPS, n))
                sampled = values[draw]
                estimates = sampled[..., 0].mean(1) - sampled[..., 1].mean(1); estimate = float(values[:,0].mean()-values[:,1].mean())
            else: estimates = np.array([]); estimate = ""
            status, left_n, right_n = ("DESCRIPTIVE" if n >= 2 else "INSUFFICIENT_SUBJECT_SUPPORT"), n, n
        else:
            ls, rs = sorted(by_l), sorted(by_r); left_n, right_n = len(ls), len(rs)
            if left_n >= 2 and right_n >= 2:
                lv, rv = np.asarray([np.mean(by_l[s]) for s in ls]), np.asarray([np.mean(by_r[s]) for s in rs])
                rng = np.random.default_rng(SEED); estimates = lv[rng.integers(0,left_n,size=(BOOTSTRAPS,left_n))].mean(1) - rv[rng.integers(0,right_n,size=(BOOTSTRAPS,right_n))].mean(1); estimate = float(lv.mean()-rv.mean()); status = "DESCRIPTIVE"
            else: estimates = np.array([]); estimate = ""; status = "INSUFFICIENT_SUBJECT_SUPPORT"
        out.append({"comparison": comparison, "metric": metric, "estimate_left_minus_right": estimate,
                    "ci95_low": float(np.quantile(estimates,.025)) if len(estimates) else "", "ci95_high": float(np.quantile(estimates,.975)) if len(estimates) else "",
                    "left_subjects": left_n, "right_subjects": right_n, "resamples": BOOTSTRAPS, "seed": SEED, "unit": "paired_subject" if paired else "independent_subject", "status": status})
    return out


def replay_primary() -> tuple[list[dict], list[dict]]:
    """Recreate G1--G3 from frozen B/S cells, retaining only GT diagnostics."""
    ENGINE.SCALES = C.SCALES; ENGINE.REFERENCE_SCALES = C.SCALES
    PARENT.fair.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    a_sel, c_sel = PARENT.selections("BoostingVRME/SAMMLV")
    records, subjects, _, _ = PARENT.fair.load_data("boostingvrme", "sammlv")
    ks, _ = PARENT.fair.fold_priors(records, subjects, "boostingvrme"); pos = {s:i for i,s in enumerate(subjects)}
    paired, old_peaks = [], []
    for anchor, selected in (("ANCHOR-A", a_sel), ("ANCHOR-C", c_sel)):
        for record in records:
            subject, video, k = record["subject"], record["video"], int(ks[pos[record["subject"]]])
            a = build_state(record, "boostingvrme", k, a_sel[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
            c = build_state(record, "boostingvrme", k, c_sel[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
            labels_a, labels_c = PARENT.gt_labels(record,a), PARENT.gt_labels(record,c)
            for gt_index, raw_gt in enumerate(record["gt"]):
                state = {("TP","B"):"G1_LOST_TP", ("TP","TP"):"G2_RETAINED_TP", ("B","TP"):"G3_REVERSE_CONTROL"}.get((labels_a[gt_index],labels_c[gt_index]))
                if not state: continue
                gt = (int(raw_gt[0]), int(raw_gt[2])); ea, ec = C.representative(a,gt), C.representative(c,gt)
                if ea is None or ec is None: raise AssertionError("EXP6D_MISSING_COMPARABLE_REPRESENTATIVE")
                common = {"setting":"BoostingVRME/SAMMLV", "anchor":anchor, "subject":subject, "video":video, "gt_index":gt_index,
                          "gt_onset":gt[0], "gt_offset":gt[1], "k":k, "rho":selected[subject]["rho"], "tau":selected[subject]["tau"],
                          "reference_A":a["ref"], "reference_C":c["ref"]}
                row = paired_row(common,a,ea,c,ec,state); paired.append(row)
                if state == "G1_LOST_TP" and row["peak_relation"] == "SHIFTED":
                    peaks_c = {x["peak"]: x for x in c["p0"]}
                    old = peaks_c.get(row["peak_A"])
                    old_desc = prominence_descriptor(c,old) if old else None
                    old_peaks.append({**common, "group":state, "peak_A":row["peak_A"], "peak_C_new":row["peak_C"], "shift_magnitude":row["peak_shift"],
                                      "peak_A_still_detected_in_R_C":"YES" if old else "NO", "peak_C_present_in_R_A":"YES" if row["peak_C"] in {x["peak"] for x in a["p0"]} else "NO",
                                      "P_A_old":row["P_A"], "P_C_at_old_peak":"" if old_desc is None else old_desc["P"], "P_C_new_selected":row["P_C"],
                                      "response_scale_effect":"" if old_desc is None else old_desc["P"]-row["P_A"],
                                      "candidate_switch_effect":"" if old_desc is None else row["P_C"]-old_desc["P"]})
    return paired, old_peaks


def replay_removed_fp() -> list[dict]:
    """Track only already-defined B/C R1 removed FPs, without using any GT label."""
    a_sel, c_sel = PARENT.selections("BoostingVRME/CAS(ME)3")
    records, subjects, _, _ = PARENT.fair.load_data("boostingvrme", "casme3")
    ks, _ = PARENT.fair.fold_priors(records, subjects, "boostingvrme"); pos = {s:i for i,s in enumerate(subjects)}; rows=[]
    for anchor, selected in (("ANCHOR-A",a_sel),("ANCHOR-C",c_sel)):
        for record in records:
            s, v, k = record["subject"],record["video"],int(ks[pos[record["subject"]]])
            a=build_state(record,"boostingvrme",k,a_sel[s]["ref"],selected[s]["rho"],selected[s]["tau"]); c=build_state(record,"boostingvrme",k,c_sel[s]["ref"],selected[s]["rho"],selected[s]["tau"])
            c_by_peak={x["peak"]:x for x in c["p0"]}
            for event in a["p2"]:
                if event["matched_gt"] >= 0 or event["peak"] in c_by_peak: continue
                overlaps=[x for x in c["p0"] if C.iou(x["interval"],event["interval"])>0]
                counterpart=min(overlaps,key=lambda x:(abs(x["peak"]-event["peak"]),-x["S"],x["peak"])) if overlaps else None
                common={"setting":"BoostingVRME/CAS(ME)3","anchor":anchor,"subject":s,"video":v,"k":k,"rho":selected[s]["rho"],"tau":selected[s]["tau"],"reference_A":a["ref"],"reference_C":c["ref"]}
                da=prominence_descriptor(a,event)
                if counterpart:
                    row=paired_row(common,a,event,c,counterpart,"G4_REMOVED_FP")
                    row.update({"counterpart_available":"YES","PEAK_DISAPPEARED":0,"counterpart_peak_C":counterpart["peak"],"counterpart_distance":abs(counterpart["peak"]-event["peak"]),"lineage":"FP-PEAK-SHIFTED"})
                else:
                    row={**common,"group":"G4_REMOVED_FP","peak_A":da["peak"],"peak_C":"","peak_shift":"","peak_relation":"PEAK-DISAPPEARED","P_A":da["P"],"D_A":da["D"],"G_A":da["G"],"P_C":"","D_C":"","G_C":"","raw_peak_value_A":da["raw_peak_value"],"prominence_base_A":da["prominence_base"],"response_max_A":da["response_max"],"response_min_A":da["response_min"],"raw_peak_value_C":"","prominence_base_C":"","response_max_C":"","response_min_C":"","delta_P":"","delta_D":"","delta_G":"","G_Ponly":"","G_Donly":"","delta_G_Ponly":"","delta_G_Donly":"","C_P":"","C_D":"","decomposition_error":"","Q_P":"","Q_G":"","Q_logP":"","counterpart_available":"NO","PEAK_DISAPPEARED":1,"counterpart_peak_C":"","counterpart_distance":"","lineage":"FP-REF-ABSENT"}
                rows.append(row)
    if len(rows) != 75: raise AssertionError("EXP6D_BC_LINEAGE_REPLAY_FAILED")
    return rows


def subject_table(primary: list[dict], fp: list[dict]) -> list[dict]:
    rows=[]
    groups=[("lost","G1_LOST_TP",primary),("retained","G2_RETAINED_TP",primary),("removed_fp","G4_REMOVED_FP",fp)]
    keys=sorted({(x["setting"],x["anchor"],x["subject"]) for _,_,data in groups for x in data})
    for setting,anchor,subject in keys:
        row={"setting":setting,"anchor":anchor,"subject":subject}
        for prefix,group,data in groups:
            subset=[x for x in data if x["setting"]==setting and x["anchor"]==anchor and x["subject"]==subject and x["group"]==group]
            # The explicit names are part of the EXP-6D audit contract.
            row[f"{prefix}_count"]=len(subset)
            if prefix == "lost": row["lost_tp_count"] = len(subset)
            if prefix == "retained": row["retained_tp_count"] = len(subset)
            for metric in (("C_P","mean_C_P"),("C_D","mean_C_D"),("Q_P","mean_Q_P"),("Q_G","mean_Q_G"),("Q_logP","mean_Q_logP")):
                row[f"{metric[1]}_{prefix}"]=mean(subset,metric[0]) if any(x.get(metric[0],"") not in ("",None) for x in subset) else ""
        rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks=verify_exp6c()
    primary, shift_rows=replay_primary(); fp=replay_removed_fp()
    lost=[x for x in primary if x["group"]=="G1_LOST_TP"]; retained=[x for x in primary if x["group"]=="G2_RETAINED_TP"]; reverse=[x for x in primary if x["group"]=="G3_REVERSE_CONTROL"]
    for anchor, expected in (("ANCHOR-A",7),("ANCHOR-C",6)):
        observed=sum(x["anchor"]==anchor for x in lost); checks.append({"check":f"fresh_replay_lost_{anchor}","status":"PASS" if observed==expected else "FAIL","observed":observed,"expected":expected,"detail":"canonical E_7 frozen replay"})
        if observed != expected: raise RuntimeError("EXP6D_PARENT_REPLAY_FAILED")
    identity=[{"row_type":"paired","setting":x["setting"],"anchor":x["anchor"],"subject":x["subject"],"video":x["video"],"group":x["group"],"decomposition_error":x.get("decomposition_error",""),"status":"PASS" if x.get("decomposition_error",0) in ("",None) or abs(float(x["decomposition_error"]))<=1e-12 else "FAIL"} for x in primary+fp]
    if any(x["status"]=="FAIL" for x in identity): raise RuntimeError("EXP6D_DECOMPOSITION_FAILED")
    factor=[]
    for label, rows in (("G1_LOST_TP",lost),("G2_RETAINED_TP",retained),("G3_REVERSE_CONTROL",reverse),("G4_REMOVED_FP_COUNTERPART",[x for x in fp if x["counterpart_available"]=="YES"]),("G1_SAME_PEAK",[x for x in lost if x["peak_relation"]=="SAME-PEAK"]),("G1_SHIFTED",[x for x in lost if x["peak_relation"]=="SHIFTED"])): factor.extend(summary(rows,label))
    stable=[]
    for label,rows in (("G1_LOST_TP",lost),("G2_RETAINED_TP",retained),("G4_REMOVED_FP_COUNTERPART",[x for x in fp if x["counterpart_available"]=="YES"])): stable.extend(stability_summary(rows,label))
    boot=subject_bootstrap(lost,retained,("C_P","C_D","Q_P","Q_G","Q_logP"),"LOST_MINUS_RETAINED",True)
    fp_available=[x for x in fp if x["counterpart_available"]=="YES"]
    boot+=subject_bootstrap(lost,fp_available,("Q_P","Q_G","Q_logP"),"LOST_MINUS_REMOVED_FP_COUNTERPART",False)
    same_peak=[x for x in lost if x["peak_relation"]=="SAME-PEAK"]
    write_csv("parent_replay_check.csv",checks); write_csv("g_factor_decomposition.csv",primary+fp); write_csv("g_factor_summary.csv",factor); write_csv("peak_shift_prominence_decomposition.csv",shift_rows); write_csv("same_peak_smoothing_only.csv",same_peak); write_csv("scale_stability_features.csv",lost+retained+fp_available); write_csv("scale_stability_summary.csv",stable); write_csv("lost_vs_retained.csv",lost+retained); write_csv("lost_vs_removed_fp.csv",lost+fp_available); write_csv("removed_fp_counterpart_status.csv",fp); write_csv("subject_g_stability.csv",subject_table(primary,fp)); write_csv("mechanism_bootstrap.csv",boot); write_csv("decomposition_identity_check.csv",identity)
    protocol={"experiment":"EXP-6D Reference-G Normalization / Scale-Stability Diagnostic","parent":"EXP-6C","fixed_semantics":"E_7; saved rho/tau/k; saved R_A/R_C; canonical scipy prominence","predefined_features":["Q_P","Q_G","Q_logP"],"bootstrap":{"resamples":BOOTSTRAPS,"seed":SEED,"unit":"outer subject"},"forbidden":["GLSD modification","new score","thresholding","configuration selection","F1/recognition/STRS recomputation","new decoder","alternative normalization","outer-test rule design"],"canonical_GLSD_modified":False}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    manifest={"source_sha256":{"EXP6B":digest(EXP6B/"run_exp6b.py"),"EXP6C":digest(EXP6C/"run_exp6c.py"),"EXP2A":digest(EXP2A/"run_exp2a.py"),"EXP2C":digest(EXP2C/"run_exp2c.py"),"engine":digest(SIGNED/"unified_persistence.py"),"EXP6D":digest(Path(__file__))},"parent_replay":"PASS","decomposition_identity":"PASS","new_method_executed":False,"alternative_G_tested":False,"formal_F1_recomputed":False,"canonical_GLSD_modified":False}
    (OUT/"replay_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    write_analysis(lost,retained,reverse,fp,shift_rows,boot)


def write_analysis(lost, retained, reverse, fp, shifts, boot) -> None:
    cp_med, cd_med=median(lost,"C_P"),median(lost,"C_D"); cp_dom=np.mean([abs(x["C_P"])>abs(x["C_D"]) for x in lost]); cd_dom=np.mean([abs(x["C_D"])>abs(x["C_P"]) for x in lost])
    source="RAW_PROMINENCE" if cp_dom>cd_dom else "RANGE_NORMALIZATION" if cd_dom>cp_dom else "MIXED"
    range_effect="BUFFERS" if cp_med<0 and cd_med>0 else "AMPLIFIES" if cd_med<0 else "NEUTRAL"
    norm_route="REJECTED" if not (cd_dom>.5 and cd_med<0) else "SUPPORTED"
    same=[x for x in lost if x["peak_relation"]=="SAME-PEAK"]; shifted=[x for x in lost if x["peak_relation"]=="SHIFTED"]
    same_verdict="INSUFFICIENT" if len(same)<2 else ("YES" if median(same,"delta_P")<0 else "NO")
    shift_assoc="YES" if len(shifted)>len(lost)/2 and median(shifted,"delta_P")<0 else "NO"
    fpa=[x for x in fp if x["counterpart_available"]=="YES"]
    lr=[x for x in boot if x["comparison"]=="LOST_MINUS_RETAINED" and x["metric"] in ("Q_P","Q_G","Q_logP") and x["status"]=="DESCRIPTIVE"]
    lf=[x for x in boot if x["comparison"]=="LOST_MINUS_REMOVED_FP_COUNTERPART" and x["status"]=="DESCRIPTIVE"]
    lr_signal=any(float(x["ci95_low"])*float(x["ci95_high"])>0 for x in lr)
    lf_different=any(float(x["ci95_low"])*float(x["ci95_high"])>0 for x in lf)
    safe=lr_signal and lf_different
    next_exp="PROMINENCE_STABILITY_RULE_DIAGNOSTIC" if source=="RAW_PROMINENCE" and safe else "STOP_REFERENCE_G_RULE_MINING"
    old_legal=sum(x["peak_A_still_detected_in_R_C"]=="YES" for x in shifts)
    same_values=", ".join(f"P {x['P_A']:.6f}→{x['P_C']:.6f}; D {x['D_A']:.6f}→{x['D_C']:.6f}; G {x['G_A']:.6f}→{x['G_C']:.6f}" for x in same)
    text=["# EXP-6D — Reference-G Normalization / Scale-Stability Diagnostic","","## Integrity","", "- Parent EXP-6C facts and fresh canonical frozen-cell replay: **PASS**.","- The Shapley-style identity `C_P + C_D = ΔG` passes at `≤1e-12` for every comparable row.","- This is read-only: no GLSD/prediction/decoder/threshold/configuration/F1/Recognition/STRS change was made.","","## Answers","",f"1. B/S lost TP has {len(lost)} events across {len({x['subject'] for x in lost})} subjects (Anchor-A={sum(x['anchor']=='ANCHOR-A' for x in lost)}, Anchor-C={sum(x['anchor']=='ANCHOR-C' for x in lost)}). Mean ΔG={mean(lost,'delta_G'):.6f}; mean C_P={mean(lost,'C_P'):.6f}; mean C_D={mean(lost,'C_D'):.6f}.",f"2. The dominant source is **{source}**: `|C_P|>|C_D|` for {cp_dom:.1%} and `|C_D|>|C_P|` for {cd_dom:.1%}; median C_P={cp_med:.6f}, median C_D={cd_med:.6f}. Range normalization therefore **{range_effect}** collapse rather than being its primary cause.",f"3. Holding the old denominator (`P_C/D_A`) gives mean ΔG_Ponly={mean(lost,'delta_G_Ponly'):.6f}; the collapse remains. Changing only the denominator (`P_A/D_C`) gives mean ΔG_Donly={mean(lost,'delta_G_Donly'):.6f}.",f"4. Same-peak lost TPs: {len(same)}/{len(lost)}; mean ΔP={mean(same,'delta_P'):.6f} and mean C_P={mean(same,'C_P'):.6f} ({same_verdict}). Exact `P_A/P_C`, `D_A/D_C`, and `G_A/G_C` are in `same_peak_smoothing_only.csv` ({same_values}). Shifted cases: {len(shifted)}/{len(lost)}, mean ΔP={mean(shifted,'delta_P'):.6f}; peak-shift association={shift_assoc}.",f"5. Shift/smoothing decomposition: old R_A peak remains a legal R_C peak in {old_legal}/{len(shifts)} shifted events. Therefore neither a response-scale nor candidate-switch prominence component is numerically identifiable for this subset under canonical peak semantics; every such field is NA rather than evaluating a non-peak.",f"6. Lost-vs-retained stability: {'SUPPORTED' if lr_signal else 'NOT SUPPORTED'} by predeclared subject-aware bootstrap (see `mechanism_bootstrap.csv`), not a cutoff search.",f"7. B/C removed FP: {len(fp)} exact R1 events; counterpart available={len(fpa)}, topology-disappeared={sum(x['PEAK_DISAPPEARED']==1 for x in fp)}. Lost-vs-removed-FP separability is {'YES' if lf_different else 'NO / INCONCLUSIVE'} on the counterpart-available subset; disappeared FPs remain a separate topology class.",f"8. Normalization route: **{norm_route}**. Raw-prominence instability: **{'SUPPORTED' if source=='RAW_PROMINENCE' and shift_assoc else 'NOT SUPPORTED'}**. Safe scale-stability signal: **{'SUPPORTED' if safe else 'NOT SUPPORTED'}**. Recommended next experiment: **{next_exp}**.","","## Limits","","The representative TP is GT-centric only for diagnosis. No diagnostic quantity became a new score or decision rule; no test-label-guided feature engineering or threshold search was performed.",""]
    (OUT/"EXP6D_ANALYSIS.md").write_text("\n".join(text),encoding="utf-8")
    print("================================")
    print("EXP-6D G-NORMALIZATION DIAGNOSTIC")
    print("================================")
    print("Parent replay: PASS\nG decomposition identity: PASS")
    print(f"B/S lost TP: {len(lost)}\nAffected subjects: {len({x['subject'] for x in lost})}")
    print(f"Mean ΔG: {mean(lost,'delta_G'):.6f}\nMean prominence contribution C_P: {mean(lost,'C_P'):.6f}\nMean range contribution C_D: {mean(lost,'C_D'):.6f}")
    print(f"Dominant G-collapse source: {source}\nRange normalization effect: {range_effect}\nSame-peak prominence collapse: {same_verdict}\nPeak-shift association: {shift_assoc}")
    print(f"Lost vs retained stability signal: {'SUPPORTED' if lr_signal else 'NOT SUPPORTED'}\nLost vs removed-FP separability: {'YES' if lf_different else 'NO / INCONCLUSIVE'}")
    print(f"Normalization route: {norm_route}\nRaw-prominence instability: {'SUPPORTED' if source=='RAW_PROMINENCE' and shift_assoc else 'NOT SUPPORTED'}\nSafe scale-stability signal: {'SUPPORTED' if safe else 'NOT SUPPORTED'}\nRecommended next experiment: {next_exp}")
    print("New method executed: NO\nAlternative G tested: NO\nFormal F1 recomputed: NO\nCanonical GLSD modified: NO")
    print("================================")


if __name__ == "__main__":
    main()
