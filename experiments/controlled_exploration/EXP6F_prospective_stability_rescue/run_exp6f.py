"""EXP-6F: frozen nested Q_P-only selective reference rescue validation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import types
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP2A = ROOT / "controlled_exploration/EXP2A_scale"
EXP2C = ROOT / "controlled_exploration/EXP2C_reference_evidence_decoupling"
EXP6A = ROOT / "controlled_exploration/EXP6A_bottleneck_localization"
EXP6E = ROOT / "controlled_exploration/EXP6E_prominence_stability_rule_diagnostic"
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
BASE = ROOT / "controlled_exploration/baseline_snapshot/results/main_results.csv"
BOOTSTRAPS, SEED, EPS = 10_000, 100, 1e-12
SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SETTINGS = (
    ("metst", "sammlv", "ME-TST/SAMMLV"),
    ("metst", "casme3", "ME-TST/CAS(ME)3"),
    ("boostingvrme", "sammlv", "BoostingVRME/SAMMLV"),
    ("boostingvrme", "casme3", "BoostingVRME/CAS(ME)3"),
)
STATE_CACHE: dict[tuple, dict] = {}
PROMINENCE_CACHE: dict[tuple, float] = {}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PARENT = load_module("exp6f_parent", ROOT / "controlled_exploration/EXP6B_reference_evidence_cross_replay/run_exp6b.py")
EXP6A_MODULE = load_module("exp6f_exp6a", EXP6A / "run_exp6a.py")
ENGINE, FAIR = PARENT.engine, PARENT.fair
BASE_FEATURES = PARENT.EXP2C_MODULE.BaseCurveFeatures


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


def f1(counts: np.ndarray | tuple[int, int, int] | list[int]) -> float:
    tp, fp, fn = (int(x) for x in counts)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def metrics(counts: np.ndarray | tuple[int, int, int] | list[int]) -> dict:
    tp, fp, fn = (int(x) for x in counts)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": tp / (tp + fp) if tp + fp else 0.0, "F1": f1((tp, fp, fn))}


def config_grid() -> list:
    return [ENGINE.Config("unified", ref, 0.0, tau, rho)
            for ref in SCALES for rho in ENGINE.LOCAL_RADII for tau in ENGINE.THRESHOLDS]


def raw_prominence(record: dict, k: int, ref: float, peak: int) -> float:
    cache_key = (id(record), k, ref, peak)
    if cache_key in PROMINENCE_CACHE:
        return PROMINENCE_CACHE[cache_key]
    features = BASE_FEATURES(record["curve"], k)
    peaks, _, _, response, _ = features.scales[ref]
    if not np.any(peaks == peak):
        raise AssertionError("EXP6F_NONCANONICAL_PEAK")
    value = float(ENGINE.peak_prominences(response, np.asarray([peak], dtype=int))[0][0])
    PROMINENCE_CACHE[cache_key] = value
    return value


def frozen_state(record: dict, backbone: str, k: int, ref: float, rho: float, tau: float, evidence: str) -> dict:
    """Cache label-free frozen branch replay; eta affects neither branch."""
    key = (id(record), backbone, k, ref, rho, tau, evidence)
    if key not in STATE_CACHE:
        STATE_CACHE[key] = PARENT.replay(record, backbone, k, ref, rho, tau, evidence)
    return STATE_CACHE[key]


def event_key(event: dict) -> tuple[int, int, int]:
    return int(event["peak"]), int(event["interval"][0]), int(event["interval"][1])


def overlaps(a: dict, b: dict, k: int, backbone: str) -> bool:
    if backbone != "boostingvrme":
        return False
    left, right = max(a["interval"][0], b["interval"][0]), min(a["interval"][1], b["interval"][1])
    span = max(a["interval"][1], b["interval"][1]) - min(a["interval"][0], b["interval"][0])
    ratio = max(0, right - left) / span if span else 0.0
    return ratio >= .2 or abs(int(a["peak"]) - int(b["peak"])) <= k


def evaluate_final(record: dict, events: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """The original chronological greedy one-to-one evaluator, after conflict filtering."""
    matched, annotated = set(), []
    for event in events:
        best, best_iou = -1, 0.0
        for index, gt in enumerate(record["gt"]):
            value = PARENT.iou(event["interval"], (int(gt[0]), int(gt[2])))
            if value > best_iou:
                best, best_iou = index, value
        final = dict(event, matched_gt=-1)
        if best_iou >= .5 and best not in matched:
            final["matched_gt"] = best
            matched.add(best)
        annotated.append(final)
    return np.asarray((len(matched), len(events) - len(matched), len(record["gt"]) - len(matched)), dtype=int), annotated


def run_rescue(record: dict, backbone: str, k: int, a_cfg, c_cfg, eta: float | None,
               setting: str, subject: str, stage: str) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    """One frozen two-branch decoder application.  eta=None is formal OFF."""
    a = frozen_state(record, backbone, k, a_cfg.reference, a_cfg.radius, a_cfg.threshold, "E_7")
    c = frozen_state(record, backbone, k, c_cfg.reference, c_cfg.radius, c_cfg.threshold, "E_3")
    delta = max(1, round(.5 * k))
    c_p2 = list(c["p2"])
    eligible_rows, inserted = [], []
    for event in a["p1"]:
        counterparts = [x for x in c["p0"] if abs(int(x["peak"]) - int(event["peak"])) <= delta]
        if not counterparts:
            continue
        cp = min(counterparts, key=lambda x: (abs(int(x["peak"]) - int(event["peak"])), -raw_prominence(record, k, c_cfg.reference, int(x["peak"])), int(x["peak"])))
        equivalent = any(abs(int(x["peak"]) - int(event["peak"])) <= delta for x in c_p2)
        pa, pc = raw_prominence(record, k, a_cfg.reference, int(event["peak"])), raw_prominence(record, k, c_cfg.reference, int(cp["peak"]))
        qp = pc / max(pa, EPS)
        base = {
            "setting": setting, "subject": subject, "video": record["video"], "stage": stage,
            "peak_A": event["peak"], "onset_A": event["interval"][0], "offset_A": event["interval"][1],
            "peak_C": cp["peak"], "P_A": pa, "P_C": pc, "Q_P": qp, "delta": delta,
            "S_A": event["S"], "tau_A": a_cfg.threshold, "S_C": cp["S"], "tau_C": c_cfg.threshold,
            "equivalent_2C_final": "YES" if equivalent else "NO",
            "eligible": "YES" if not equivalent else "NO", "eta": "OFF" if eta is None else eta,
            "selected_by_eta": "NO", "survived_conflict": "NO", "formal_outcome": "NOT_SELECTED",
        }
        eligible_rows.append(base)
        if not equivalent and cp["S"] < c_cfg.threshold and eta is not None and qp <= eta:
            rescue = dict(event, source="RESCUE_A", source_index=event["index"], q_p=qp, paired_peak_C=cp["peak"])
            inserted.append(rescue)
            base["selected_by_eta"] = "YES"
    merged = [dict(event, source="BASE_C", source_index=event["index"]) for event in c_p2] + inserted
    merged.sort(key=lambda x: (int(x["interval"][0]), int(x["peak"]), 0 if x["source"] == "BASE_C" else 1, int(x["source_index"])))
    keep, suppressed = [], {}
    for event in merged:
        blocker = next((prior for prior in keep if overlaps(prior, event, k, backbone)), None)
        if blocker is None:
            keep.append(event)
        else:
            suppressed[event_key(event)] = blocker
    for row in eligible_rows:
        key = (int(row["peak_A"]), int(row["onset_A"]), int(row["offset_A"]))
        if row["selected_by_eta"] == "YES":
            if key in suppressed:
                row["formal_outcome"] = "SUPPRESSED_BY_CONFLICT"
            elif any(event_key(x) == key and x["source"] == "RESCUE_A" for x in keep):
                row["survived_conflict"], row["formal_outcome"] = "YES", "SURVIVED"
    counts, final = evaluate_final(record, keep)
    final_by_key = {event_key(x): x for x in final}
    for row in eligible_rows:
        key = (int(row["peak_A"]), int(row["onset_A"]), int(row["offset_A"]))
        if row["survived_conflict"] == "YES":
            row["formal_outcome"] = "TP" if final_by_key[key]["matched_gt"] >= 0 else "FP"
    mechanism = {"eligible": sum(x["eligible"] == "YES" for x in eligible_rows), "rescued_before_conflict": len(inserted),
                 "survived_conflict": sum(x["survived_conflict"] == "YES" for x in eligible_rows),
                 "rescued_tp": sum(x["formal_outcome"] == "TP" for x in eligible_rows), "rescued_fp": sum(x["formal_outcome"] == "FP" for x in eligible_rows),
                 "suppressed_base_tp": 0, "suppressed_base_fp": 0}
    base_by_key = {event_key(x): x for x in c_p2}
    for key, blocker in suppressed.items():
        gone = base_by_key.get(key)
        if gone is not None and blocker["source"] == "RESCUE_A":
            mechanism["suppressed_base_tp"] += int(gone["matched_gt"] >= 0)
            mechanism["suppressed_base_fp"] += int(gone["matched_gt"] < 0)
    return counts, final, eligible_rows, mechanism


def select_index(counts: np.ndarray) -> int:
    return ENGINE.choose(counts, list(range(len(counts))))


def validate_exp6e() -> list[dict]:
    report = (EXP6E / "EXP6E_ANALYSIS.md").read_text(encoding="utf-8")
    events = read_csv(EXP6E / "group_event_features.csv")
    directions = read_csv(EXP6E / "lost_retained_direction_consistency.csv")
    comparisons = read_csv(EXP6E / "comparison_diagnostics.csv")
    checks = []
    def add(name, ok, observed, expected):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})
    lost = [r for r in events if r["group"] == "G1_LOST_TP"]
    retained = [r for r in events if r["group"] == "G2_RETAINED_TP"]
    fp = [r for r in read_csv(EXP6E / "all_fp_counterpart_audit.csv") if r.get("group") == "GROUP_FA_ALL_TRACEABLE_FP"]
    qp = next(r for r in comparisons if r["metric"] == "Q_P")
    add("parent_replay", "**PASS**" in report, "PASS" if "**PASS**" in report else "missing", "PASS")
    add("lost_events", len(lost) == 13, len(lost), 13); add("lost_subjects", len({r["subject"] for r in lost}) == 6, len({r["subject"] for r in lost}), 6)
    add("retained_events", len(retained) == 78, len(retained), 78); add("retained_subjects", len({r["subject"] for r in retained}) == 23, len({r["subject"] for r in retained}), 23)
    add("traceable_fp", len(fp) == 1802 and len({r["subject"] for r in fp}) == 91, f"{len(fp)}/{len({r['subject'] for r in fp})}", "1802/91")
    add("qp_auc", abs(float(qp["auc_lost_vs_retained"]) - .890) < .002, qp["auc_lost_vs_retained"], "≈0.890")
    add("qp_direction", "direction consistency=5/5" in report, "5/5" if "direction consistency=5/5" in report else "missing", "5/5")
    for token in ("SAFE_STABILITY_SIGNAL_SUPPORTED", "TRAINING_ONLY_RULE_FEASIBLE", "RESCUE_RISK = MODERATE", "No threshold was searched"):
        add(token, token in report, token in report, True)
    if any(x["status"] == "FAIL" for x in checks):
        raise RuntimeError("EXP6F_PARENT_CHECK_FAILED")
    return checks


def nested_stats(records, subjects, backbone: str, feature_class, extra_k: set[int]) -> dict[int, np.ndarray]:
    ENGINE.SCALES = SCALES; ENGINE.REFERENCE_SCALES = SCALES; ENGINE.CurveFeatures = feature_class
    outer, inner = FAIR.fold_priors(records, subjects, backbone)
    configs, result = config_grid(), {}
    for k in sorted(set(outer.tolist()) | set(inner[inner > 0].tolist()) | extra_k):
        result[int(k)] = ENGINE.evaluate_k(records, subjects, backbone, int(k), np.ones(len(subjects), dtype=bool), configs, ("native" if backbone == "boostingvrme" else "fixed",))["native" if backbone == "boostingvrme" else "fixed"]
    return result


def triple_duration_priors(records: list[dict], subjects: list[str], backbone: str) -> np.ndarray:
    """Exact duration_k for every unordered three-subject exclusion, without rescanning records."""
    position = {subject: index for index, subject in enumerate(subjects)}
    durations = [[int(gt[2]) - int(gt[0]) for record in records if record["subject"] == subject for gt in record["gt"]] for subject in subjects]
    minimum = min(value for values in durations for value in values)
    maximum = max(value for values in durations for value in values)
    histograms = np.zeros((len(subjects), maximum - minimum + 1), dtype=int)
    sums = np.zeros(len(subjects), dtype=int); counts = np.zeros(len(subjects), dtype=int)
    for index, values in enumerate(durations):
        histograms[index] = np.bincount(np.asarray(values) - minimum, minlength=maximum - minimum + 1)
        sums[index], counts[index] = sum(values), len(values)
    total_hist, total_sum, total_count = histograms.sum(axis=0), int(sums.sum()), int(counts.sum())
    @lru_cache(maxsize=None)
    def prior(a: int, b: int, c: int) -> int:
        selected = (a, b, c)
        remain = total_count - int(counts[list(selected)].sum())
        if backbone == "metst":
            return max(1, int(((total_sum - int(sums[list(selected)].sum())) / remain + 1) / 2))
        residual = total_hist - histograms[list(selected)].sum(axis=0)
        median_duration = int(np.searchsorted(np.cumsum(residual), remain // 2 + 1, side="left") + minimum)
        return max(1, int((median_duration + 1) / 2))
    output = np.zeros((len(subjects), len(subjects), len(subjects)), dtype=int)
    for held in range(len(subjects)):
        for validation in range(len(subjects)):
            if validation == held: continue
            for train in range(len(subjects)):
                if train not in (held, validation):
                    output[held, validation, train] = prior(*sorted((held, validation, train)))
    return output


def inner_configs(stats_a, stats_c, triple_k: np.ndarray, held: int, validation: int) -> tuple[int, int]:
    ids = [i for i in range(len(triple_k)) if i not in (held, validation)]
    initial = np.zeros((len(config_grid()), 3), dtype=int)
    a_total = sum((stats_a[int(triple_k[held, validation, train])][:, train] for train in ids), initial.copy())
    c_total = sum((stats_c[int(triple_k[held, validation, train])][:, train] for train in ids), initial.copy())
    return select_index(a_total), select_index(c_total)


def pair_bootstrap(candidate: np.ndarray, base: np.ndarray, comparison: str, setting: str) -> dict:
    rng, n = np.random.default_rng(SEED), len(candidate)
    draw = rng.integers(0, n, size=(BOOTSTRAPS, n))
    totals_c, totals_b = candidate[draw].sum(axis=1), base[draw].sum(axis=1)
    delta = np.asarray([f1(x) for x in totals_c]) - np.asarray([f1(x) for x in totals_b])
    return {"setting": setting, "comparison": comparison, "delta_F1": f1(candidate.sum(axis=0)) - f1(base.sum(axis=0)),
            "ci95_low": float(np.quantile(delta, .025)), "ci95_high": float(np.quantile(delta, .975)),
            "positive_resample_fraction": float(np.mean(delta > 0)), "resamples": BOOTSTRAPS, "seed": SEED}


def me_strs(dataset: str, prediction_rows: list[dict]) -> tuple[dict, dict]:
    """Call the archived paper evaluator on the exact frozen Strategy-1 emotion cache."""
    # The archived evaluator imports only sklearn.metrics.confusion_matrix.
    # Supply its exact small-count behavior when this desktop Python lacks sklearn.
    try:
        import sklearn.metrics  # noqa: F401
    except ModuleNotFoundError:
        sklearn = types.ModuleType("sklearn")
        sklearn_metrics = types.ModuleType("sklearn.metrics")
        def confusion_matrix(y_true, y_pred):
            labels = sorted(set(y_true) | set(y_pred))
            table = np.zeros((len(labels), len(labels)), dtype=int)
            index = {value: i for i, value in enumerate(labels)}
            for truth, prediction in zip(y_true, y_pred): table[index[truth], index[prediction]] += 1
            return table
        sklearn_metrics.confusion_matrix = confusion_matrix
        sklearn.metrics = sklearn_metrics
        sys.modules["sklearn"], sys.modules["sklearn.metrics"] = sklearn, sklearn_metrics
    import pandas as pd
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)  # type: ignore[attr-defined]
    evaluator = load_module("exp6f_me_evaluator", ROOT / "RethinkFuse_reproduction/results/glsd_str_evaluation_v2/run_metst_evaluation.py")
    add, empty, finish, gt_from_samples = evaluator.import_repository_evaluator()
    cache_path = ROOT / "RethinkFuse_reproduction/caches/me_tst" / f"{dataset}_strategy1_outputs.pkl"
    with cache_path.open("rb") as handle: cache = pickle.load(handle)
    lookup = {(str(x["subject"]), str(x["video"])): x for x in cache["records"]}
    total = empty()
    for row in prediction_rows:
        record = lookup[row["subject"], row["video"]]
        preds = evaluator.prediction_rows([{"onset":x["interval"][0], "offset":x["interval"][1], "peak":x["peak"]} for x in row["events"]])
        add(total, preds, gt_from_samples(record["samples"]), record["emotion"], record["gt_emotions"], iou_threshold=.5)
    value = finish(total); value["spotting_f1_exact"] = f1((value["tp"], value["fp"], value["fn"]))
    manifest = {"cache": str(cache_path), "cache_sha256": digest(cache_path), "videos": len(prediction_rows), "evaluator": str(ROOT / "RethinkFuse_reproduction/senior_original/me_tst_video/me-tst-video/paper_metrics.py")}
    return value, manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FAIR.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    parent_checks = validate_exp6e()
    configs = config_grid(); all_results=[]; subjects_out=[]; eta_rows=[]; inner_rows=[]; rescue_rows=[]; mechanism_rows=[]; bootstrap=[]; lost_audit=[]; bc_fp=[]; runtime_rows=[]; replay=[]; me_predictions=defaultdict(list)
    historical = [r for r in read_csv(EXP6E / "group_event_features.csv") if r["group"] == "G1_LOST_TP"]
    frozen_predictions = json.loads((EXP6A / "replay_predictions.json").read_text())
    canonical_events = {(r["backbone"], r["dataset"], r["subject"], r["video"]): r["predictions"] for r in frozen_predictions if r["variant"] == "canonical"}
    exp2c_events = {(r["backbone"], r["dataset"], r["subject"], r["video"]): r["predictions"] for r in frozen_predictions if r["variant"] == "EXP-2C"}
    baseline_rows = {(r["Backbone"], r["Dataset"]): r for r in read_csv(BASE)}
    for backbone, dataset, setting in SETTINGS:
        started=time.perf_counter(); records, subjects, _, source = FAIR.load_data(backbone, dataset); outer, inner = FAIR.fold_priors(records, subjects, backbone); pos={s:i for i,s in enumerate(subjects)}
        a_saved, c_saved = PARENT.selections(setting)
        canonical_selection, _ = EXP6A_MODULE.selection_rows("canonical", backbone, dataset, setting)
        triple_k = triple_duration_priors(records, subjects, backbone)
        extra_k = set(int(x) for x in triple_k.flatten() if x > 0)
        stats_a = nested_stats(records, subjects, backbone, BASE_FEATURES, extra_k)
        stats_c = nested_stats(records, subjects, backbone, PARENT.EXP2C_MODULE.DecoupledCurveFeatures, extra_k)
        ENGINE.CurveFeatures = BASE_FEATURES
        # Reconstruct outer 2A/2C choices, and fail if the full nested selector differs from the archived fold choices.
        for held, subject in enumerate(subjects):
            oa=select_index(ENGINE.inner_counts(stats_a, held, inner)); oc=select_index(ENGINE.inner_counts(stats_c, held, inner))
            sa, sc=configs[oa], configs[oc]
            expected_a, expected_c=a_saved[subject],c_saved[subject]
            ok=(sa.reference,sa.radius,sa.threshold)==(expected_a["ref"],expected_a["rho"],expected_a["tau"]) and (sc.reference,sc.radius,sc.threshold)==(expected_c["ref"],expected_c["rho"],expected_c["tau"])
            replay.append({"setting":setting,"subject":subject,"check":"nested_outer_config_replay","status":"PASS" if ok else "FAIL","observed_A":sa.identifier,"observed_C":sc.identifier,"saved_A":f"{expected_a['ref']}/{expected_a['rho']}/{expected_a['tau']}","saved_C":f"{expected_c['ref']}/{expected_c['rho']}/{expected_c['tau']}"})
            if not ok: raise RuntimeError("EXP6F_BASELINE_REPLAY_FAILED")
        # Exact baseline identities are re-evaluated below per video while holding the saved outer configurations.
        subject_counts=[]; canonical_subject_counts=[]
        for held, subject in enumerate(subjects):
            pair_cache={}
            q_pool=[]
            for val in range(len(subjects)):
                if val == held: continue
                ia,ic=inner_configs(stats_a,stats_c,triple_k,held,val); pair_cache[val]=(configs[ia],configs[ic])
                rows=[]
                for record in records:
                    if record["subject"] != subjects[val]: continue
                    _,_,eligible,_=run_rescue(record,backbone,int(inner[held,val]),configs[ia],configs[ic],None,setting,subjects[val],"inner_grid")
                    q_pool += [float(x["Q_P"]) for x in eligible if x["eligible"]=="YES"]
            if not q_pool: candidates=[None]
            else: candidates=[None]+[float(np.quantile(np.asarray(q_pool),q)) for q in QUANTILES]
            # Each eta is assessed only on every inner-validation subject using configurations selected without held/validation.
            scores=[]
            for option_index,eta in enumerate(candidates):
                total=np.zeros(3,dtype=int); rescues=0
                for val in range(len(subjects)):
                    if val == held: continue
                    a_cfg,c_cfg=pair_cache[val]
                    for record in records:
                        if record["subject"] != subjects[val]: continue
                        count,_,_,mechanism=run_rescue(record,backbone,int(inner[held,val]),a_cfg,c_cfg,eta,setting,subjects[val],"inner_validation")
                        total += count; rescues += mechanism["rescued_before_conflict"]
                m=metrics(total); scores.append({"eta":eta,"order":option_index,"counts":total,"rescues":rescues,**m})
                inner_rows.append({"setting":setting,"outer_subject":subject,"eta":"OFF" if eta is None else eta,"eta_source":"OFF" if eta is None else "training_eligible_QP_quantile","TP":m["TP"],"FP":m["FP"],"FN":m["FN"],"F1":m["F1"],"precision":m["precision"],"rescued_candidates":rescues,"inner_validation_subjects":len(subjects)-1})
            selected=sorted(scores,key=lambda x:(-x["F1"],-x["precision"],x["FP"],x["rescues"],0 if x["eta"] is None else 1,float("-inf") if x["eta"] is None else x["eta"],x["order"]))[0]
            eta=selected["eta"]
            eta_rows.append({"setting":setting,"subject":subject,"selected_eta":"OFF" if eta is None else eta,"selected_source":"OFF" if eta is None else QUANTILES[candidates[1:].index(eta)],"eligible_QP_count":len(q_pool),"candidate_grid":"|".join("OFF" if x is None else f"{x:.12g}" for x in candidates),"inner_F1":selected["F1"],"inner_TP":selected["TP"],"inner_FP":selected["FP"],"inner_FN":selected["FN"]})
            final_total=np.zeros(3,dtype=int); base_total=np.zeros(3,dtype=int); canonical_total=np.zeros(3,dtype=int); mech=Counter(); subject_events=[]
            a_cfg,c_cfg=configs[select_index(ENGINE.inner_counts(stats_a, held, inner))],configs[select_index(ENGINE.inner_counts(stats_c, held, inner))]
            for record in records:
                if record["subject"] != subject: continue
                final_count, final_events, rows, one_mech=run_rescue(record,backbone,int(outer[held]),a_cfg,c_cfg,eta,setting,subject,"outer")
                base=PARENT.replay(record,backbone,int(outer[held]),c_cfg.reference,c_cfg.radius,c_cfg.threshold,"E_3")
                base_count=np.asarray(base["counts"],dtype=int); final_total += final_count; base_total += base_count; mech.update(one_mech); rescue_rows += rows; subject_events.append({"subject":subject,"video":record["video"],"events":final_events})
                # Exact 2C and canonical GLSD-v1 saved-event identity checks.
                exp_key=(backbone,dataset,subject,record["video"])
                expected_c = exp2c_events[exp_key]
                actual_c=[PARENT.saved_event(x) for x in base["p2"]]
                if actual_c != expected_c: raise RuntimeError("EXP6F_BASELINE_REPLAY_FAILED")
                canonical_cfg = EXP6A_MODULE.config_from_row("canonical", canonical_selection[subject])
                canonical_state = EXP6A_MODULE.replay_video(record, canonical_cfg, backbone, int(outer[held]), "canonical")
                if [EXP6A_MODULE.event_json(x) for x in canonical_state["p2"]] != canonical_events[exp_key]:
                    raise RuntimeError("EXP6F_BASELINE_REPLAY_FAILED")
                canonical_total += np.asarray(canonical_state["counts"], dtype=int)
            if tuple(base_total)!=(c_saved[subject]["TP"],c_saved[subject]["FP"],c_saved[subject]["FN"]): raise RuntimeError("EXP6F_BASELINE_REPLAY_FAILED")
            subject_counts.append((final_total,base_total)); canonical_subject_counts.append(canonical_total); fm,bm=metrics(final_total),metrics(base_total)
            subjects_out.append({"setting":setting,"backbone":backbone,"dataset":dataset,"subject":subject,"selected_eta":"OFF" if eta is None else eta,"eta_quantile_source":next(x["selected_source"] for x in eta_rows if x["setting"]==setting and x["subject"]==subject),"eligible_count":mech["eligible"],"rescued_count":mech["rescued_before_conflict"],"TP":fm["TP"],"FP":fm["FP"],"FN":fm["FN"],"F1":fm["F1"],"2C_TP":bm["TP"],"2C_FP":bm["FP"],"2C_FN":bm["FN"],"2C_F1":bm["F1"],"delta_TP":fm["TP"]-bm["TP"],"delta_FP":fm["FP"]-bm["FP"],"delta_FN":fm["FN"]-bm["FN"],"delta_F1":fm["F1"]-bm["F1"]})
            mechanism_rows.append({"setting":setting,"subject":subject,**mech})
            if backbone=="metst": me_predictions[dataset] += subject_events
        candidate=np.asarray([x[0] for x in subject_counts]); base=np.asarray([x[1] for x in subject_counts]); cm,bm=metrics(candidate.sum(axis=0)),metrics(base.sum(axis=0)); canonical=baseline_rows[backbone,dataset]
        # Canonical locked event list comes from the independently exact EXP-6A replay artifact; compare every pooled count too.
        can_counts=np.asarray([int(canonical[f"GL_Skill_{x}"]) for x in ("TP","FP","FN")])
        all_results.append({"setting":setting,"backbone":backbone,"dataset":dataset,"EXP2C_TP":bm["TP"],"EXP2C_FP":bm["FP"],"EXP2C_FN":bm["FN"],"EXP2C_F1":bm["F1"],"EXP6F_TP":cm["TP"],"EXP6F_FP":cm["FP"],"EXP6F_FN":cm["FN"],"EXP6F_F1":cm["F1"],"delta_TP":cm["TP"]-bm["TP"],"delta_FP":cm["FP"]-bm["FP"],"delta_FN":cm["FN"]-bm["FN"],"delta_F1":cm["F1"]-bm["F1"],"canonical_TP":int(can_counts[0]),"canonical_FP":int(can_counts[1]),"canonical_FN":int(can_counts[2]),"canonical_F1":f1(can_counts),"delta_vs_canonical":cm["F1"]-f1(can_counts)})
        bootstrap += [pair_bootstrap(candidate,base,"EXP6F-minus-EXP2C",setting),pair_bootstrap(candidate,np.asarray(canonical_subject_counts),"EXP6F-minus-canonical",setting)]
        runtime_rows.append({"setting":setting,"decoder_seconds":time.perf_counter()-started,"videos":len(records),"seconds_per_video":(time.perf_counter()-started)/len(records),"excludes":"backbone forward/cache IO/configuration search"})
    # Historical audit joins B/S event identity only after eta is frozen.
    for row in historical:
        candidates=[x for x in rescue_rows if x["setting"]=="BoostingVRME/SAMMLV" and x["subject"]==row["subject"] and x["video"]==row["video"] and int(x["peak_A"])==int(row["peak_A"])]
        observed=candidates[0] if candidates else {}
        lost_audit.append({"subject":row["subject"],"video":row["video"],"gt_index":row["gt_index"],"historical_peak_A":row["peak_A"],"eligible":observed.get("eligible","NO"),"selected_by_training_eta":observed.get("selected_by_eta","NO"),"survived_conflict":observed.get("survived_conflict","NO"),"recovered_formal_TP":"YES" if observed.get("formal_outcome")=="TP" else "NO"})
    # B/C FP safety uses final rescued FPs; lineage definition is the fixed parent R1 list, audited post hoc only.
    r1={(r["subject"],r["video"],int(r["peak_A"])) for r in read_csv(ROOT/"controlled_exploration/EXP6D_reference_g_normalization/removed_fp_counterpart_status.csv") if r.get("setting")=="BoostingVRME/CAS(ME)3"}
    for row in rescue_rows:
        if row["setting"]=="BoostingVRME/CAS(ME)3" and row["formal_outcome"]=="FP":
            bc_fp.append({"subject":row["subject"],"video":row["video"],"peak_A":row["peak_A"],"reintroduced_removed_FP":"YES" if (row["subject"],row["video"],int(row["peak_A"])) in r1 else "NO"})
    strs_rows=[]; strs_manifest={}
    for dataset,label in (("sammlv","ME-TST/SAMMLV"),("casme3","ME-TST/CAS(ME)3")):
        value,manifest=me_strs(dataset,me_predictions[dataset]); strs_manifest[dataset]=manifest
        base_str=next(r for r in read_csv(ROOT/"RethinkFuse_reproduction/results/glsd_str_evaluation_v2/metst_summary.csv") if r["dataset"]==("SAMMLV" if dataset=="sammlv" else "CAS(ME)3"))
        strs_rows.append({"setting":label,"method":"EXP6F","spotting_f1":value["spotting_f1_exact"],"recognition_f1":value["recognition_f1_score_3emo_wo_others"],"STRS":value["strs_3emo_wo_others"],"TP":value["tp"],"FP":value["fp"],"FN":value["fn"]})
        strs_rows.append({"setting":label,"method":"canonical_replay","spotting_f1":base_str["glsd_spot_f1"],"recognition_f1":base_str["glsd_recognition_f1"],"STRS":base_str["glsd_strs"],"TP":base_str["glsd_spot_tp"],"FP":base_str["glsd_spot_fp"],"FN":base_str["glsd_spot_fn"]})
    write_csv("parent_check.csv",parent_checks); write_csv("main_results.csv",all_results); write_csv("subject_results.csv",subjects_out); write_csv("eta_selection.csv",eta_rows); write_csv("eta_inner_validation.csv",inner_rows); write_csv("rescue_events.csv",rescue_rows); write_csv("rescue_mechanism_summary.csv",mechanism_rows); write_csv("historical_lost_tp_recovery.csv",lost_audit); write_csv("bc_fp_reintroduction.csv",bc_fp, ["subject","video","peak_A","reintroduced_removed_FP"]); write_csv("bootstrap_results.csv",bootstrap); write_csv("strs_results.csv",strs_rows); write_csv("runtime.csv",runtime_rows); write_csv("replay_manifest.csv",replay)
    (OUT/"strs_replay_manifest.json").write_text(json.dumps(strs_manifest,indent=2)+"\n",encoding="utf-8")
    protocol={"experiment":"EXP-6F Training-Only Prominence-Stability Rescue Validation","status":"POST_HOC_DEVELOPED_NESTED_EVALUATED_VARIANT","feature":"Q_P=P_C/max(P_A,1e-12) ONLY","eta_quantiles":QUANTILES,"eta_OFF_allowed":True,"trainable_parameters":0,"outer_test_used_for_tuning":False,"matching":"max(1,round(0.5*k)); nearest, then higher P_C, then temporal order","canonical_GLSD_modified":False,"sources":{str(x):digest(x) for x in (EXP2A/"run_exp2a.py",EXP2C/"run_exp2c.py",EXP6E/"run_exp6e.py",Path(__file__))}}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    manifest={"baseline_replay":"PASS","nested_eta_selection":"PASS","parent_check":"PASS","source_sha256":protocol["sources"],"new_prediction_set":True,"outer_test_threshold_tuning":False,"canonical_GLSD_modified":False}
    (OUT/"replay_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    make_report(all_results,eta_rows,mechanism_rows,lost_audit,bc_fp,bootstrap,strs_rows,runtime_rows)
    print_terminal(all_results,eta_rows,lost_audit,bc_fp,mechanism_rows,bootstrap,runtime_rows,strs_rows)


def make_report(results, etas, mechanism, lost, bc, bootstrap, strs, runtime):
    by={r["setting"]:r for r in results}; ci={(r["setting"],r["comparison"]):r for r in bootstrap}; rec=sum(r["recovered_formal_TP"]=="YES" for r in lost); all_mech=Counter(); [all_mech.update(r) for r in mechanism]; off=sum(r["selected_eta"]=="OFF" for r in etas); active=len(etas)-off; b_s=by["BoostingVRME/SAMMLV"]; b_c=by["BoostingVRME/CAS(ME)3"]
    no_clear_regression = all(ci[r["setting"], "EXP6F-minus-EXP2C"]["ci95_high"] >= 0 for r in results)
    validated=b_s["delta_F1"]>0 and b_s["delta_TP"]>0 and b_c["delta_FP"]<=max(5,.1*max(1,b_c["EXP2C_FP"])) and no_clear_regression and sum(r["delta_F1"]>=0 for r in results)>=2
    strong=validated and ci["BoostingVRME/SAMMLV","EXP6F-minus-EXP2C"]["ci95_low"]>=0 and b_c["delta_F1"]>=0 and all(by[s]["delta_F1"]>=-.005 for s in ("ME-TST/SAMMLV","ME-TST/CAS(ME)3"))
    status="UPGRADE" if strong else "EXPLORATORY_ONLY" if validated else "REJECT"
    lines=["# EXP-6F — Training-Only Prominence-Stability Rescue Validation","","**Status:** `POST_HOC_DEVELOPED_NESTED_EVALUATED_VARIANT`; this is not independent prospective or untouched external validation.","","## Integrity","", "- EXP-6E parent facts: PASS.","- Canonical and EXP-2C frozen baseline replay: PASS.","- Nested eta selection: PASS; Q_P only; no trainable parameters; outer test labels or F1 were never used to tune eta.","","## Primary results vs EXP-2C","", "| Setting | 2C F1 | EXP6F F1 | ΔF1 | TP/FP/FN | 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        c=ci[r["setting"],"EXP6F-minus-EXP2C"]; lines.append(f"| {r['setting']} | {r['EXP2C_F1']:.6f} | {r['EXP6F_F1']:.6f} | {r['delta_F1']:+.6f} | {r['EXP6F_TP']}/{r['EXP6F_FP']}/{r['EXP6F_FN']} | [{c['ci95_low']:+.6f}, {c['ci95_high']:+.6f}] |")
    lines += ["","## Rescue accounting","",f"- Historical B/S lost TP recovered: {rec}/13.",f"- New rescued TP/FP: {all_mech['rescued_tp']}/{all_mech['rescued_fp']}; rescue precision={all_mech['rescued_tp']/max(1,all_mech['rescued_tp']+all_mech['rescued_fp']):.3f}.",f"- B/C removed FP reintroduced: {sum(r['reintroduced_removed_FP']=='YES' for r in bc)}; total B/C rescued FP={len(bc)}.",f"- eta OFF folds: {off}; active folds: {active}.","","## Recognition / STRS","", "| Setting | Method | Spot F1 | Recognition F1 | STRS |", "|---|---|---:|---:|---:|"]
    for r in strs: lines.append(f"| {r['setting']} | {r['method']} | {float(r['spotting_f1']):.6f} | {float(r['recognition_f1']):.4f} | {float(r['STRS']):.4f} |")
    lines += ["","## Verdict","",f"- `PROMINENCE_RESCUE_VALIDATED`: **{'YES' if validated else 'NO'}**.",f"- `STRONG_VALIDATION`: **{'YES' if strong else 'NO'}**.",f"- Recommended status: **{status}**.","- Additional method cost: second reference branch YES; learned parameters 0; new scalar eta selected training-only; additional candidate matching; decoder-only runtime in `runtime.csv`."]
    (OUT/"EXP6F_ANALYSIS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def print_terminal(results, etas, lost, bc, mechanism, bootstrap, runtime, strs):
    by={r["setting"]:r for r in results}; ci={(r["setting"],r["comparison"]):r for r in bootstrap}; all_mech=Counter(); [all_mech.update(r) for r in mechanism]
    print("================================\nEXP-6F STABILITY RESCUE VALIDATION\n================================\nBaseline replay: PASS\nNested eta selection: PASS\nFeature: Q_P ONLY\nTrainable parameters: 0\nOuter-test threshold used for tuning: NO")
    for name in ("ME-TST/SAMMLV","ME-TST/CAS(ME)3","BoostingVRME/SAMMLV","BoostingVRME/CAS(ME)3"):
        r=by[name]; c=ci[name,"EXP6F-minus-EXP2C"]; print(f"\n{name}:\n2C F1 = {r['EXP2C_F1']:.6f}\nEXP6F F1 = {r['EXP6F_F1']:.6f}\nΔF1 = {r['delta_F1']:+.6f}\nTP/FP/FN = {r['EXP6F_TP']}/{r['EXP6F_FP']}/{r['EXP6F_FN']}\n95% CI = [{c['ci95_low']:+.6f}, {c['ci95_high']:+.6f}]")
    print(f"\nHistorical B/S lost TP recovered: {sum(r['recovered_formal_TP']=='YES' for r in lost)} / 13\nNew rescued FP: {all_mech['rescued_fp']}\nB/C removed FP reintroduced: {sum(r['reintroduced_removed_FP']=='YES' for r in bc)}\neta OFF folds: {sum(r['selected_eta']=='OFF' for r in etas)}\neta active folds: {sum(r['selected_eta']!='OFF' for r in etas)}\nRescue precision: {all_mech['rescued_tp']/max(1,all_mech['rescued_tp']+all_mech['rescued_fp']):.3f}\nBoosting STRS: NA\nRuntime overhead: see runtime.csv\nPost-hoc developed: YES\nIndependent untouched validation: NO\nCanonical GLSD modified: NO\n================================")


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize"]:
        def numbers(rows):
            categorical = {"setting", "backbone", "dataset", "subject", "video", "method", "comparison", "selected_eta", "selected_source", "eta_quantile_source", "eta_source", "recovered_formal_TP", "eligible", "selected_by_training_eta", "survived_conflict", "reintroduced_removed_FP"}
            for row in rows:
                for key, value in list(row.items()):
                    if key in categorical or value in ("", "OFF", "YES", "NO"): continue
                    try: row[key] = float(value)
                    except ValueError: pass
            return rows
        results = numbers(read_csv(OUT / "main_results.csv"))
        etas = read_csv(OUT / "eta_selection.csv")
        mechanism = numbers(read_csv(OUT / "rescue_mechanism_summary.csv"))
        lost = read_csv(OUT / "historical_lost_tp_recovery.csv")
        bc = read_csv(OUT / "bc_fp_reintroduction.csv")
        bootstrap = numbers(read_csv(OUT / "bootstrap_results.csv"))
        strs = numbers(read_csv(OUT / "strs_results.csv"))
        runtime = read_csv(OUT / "runtime.csv")
        sources = {str(x): digest(x) for x in (EXP2A / "run_exp2a.py", EXP2C / "run_exp2c.py", EXP6E / "run_exp6e.py", Path(__file__))}
        (OUT / "protocol.json").write_text(json.dumps({"experiment":"EXP-6F Training-Only Prominence-Stability Rescue Validation","status":"POST_HOC_DEVELOPED_NESTED_EVALUATED_VARIANT","feature":"Q_P ONLY","eta_quantiles":QUANTILES,"eta_OFF_allowed":True,"trainable_parameters":0,"outer_test_used_for_tuning":False,"canonical_GLSD_modified":False,"sources":sources}, indent=2) + "\n", encoding="utf-8")
        (OUT / "replay_manifest.json").write_text(json.dumps({"baseline_replay":"PASS","nested_eta_selection":"PASS","parent_check":"PASS","source_sha256":sources,"new_prediction_set":True,"outer_test_threshold_tuning":False,"canonical_GLSD_modified":False}, indent=2) + "\n", encoding="utf-8")
        make_report(results, etas, mechanism, lost, bc, bootstrap, strs, runtime)
        print_terminal(results, etas, lost, bc, mechanism, bootstrap, runtime, strs)
    else:
        main()
