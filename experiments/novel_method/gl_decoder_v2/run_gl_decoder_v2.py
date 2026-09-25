"""Run the frozen GL-Decoder-v2 reconstruction and its controlled audits."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from .adapters import load_boostingvrme, load_me_tst
from .audit import compare_predictions, set_jaccard, neighborhood_peak_jaccard
from .bootstrap import paired_bootstrap
from .decoder import GLConfig, GLDecoder
from .evaluation import add_counts, evaluate_events, native_events
from .selection import enumerate_configs, select_config


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
RESULT_ROOT = PROJECT / "results" / "gl_decoder_v2"
PHASE_A = RESULT_ROOT / "phase_a_metst_historical_audit"
PHASE_B = RESULT_ROOT / "phase_b_current_cache"
HISTORICAL_METST = WORKSPACE / "pure_persistence_matched_v1 2"
HISTORICAL_BOOSTING = WORKSPACE / "pure_persistence_matched_v1"
CONTROL_REPORT = PROJECT / "results" / "persistence_operator_controlled_v1" / "combined_report.json"

DISCLAIMER = {
    "implementation_identity": "THIS IS A RECONSTRUCTED REPRODUCIBLE IMPLEMENTATION",
    "historical_claim": "NOT A CLAIM OF BYTE-IDENTICAL HISTORICAL ENGINE REPRODUCTION",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def historical_protocols() -> list[tuple[Path, dict]]:
    paths = [
        HISTORICAL_METST / dataset / "PROTOCOL.json" for dataset in ("sammlv", "casme3")
    ] + [
        HISTORICAL_BOOSTING / dataset / "PROTOCOL.json" for dataset in ("sammlv", "casme3")
    ]
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def freeze_spec() -> tuple[list[GLConfig], dict]:
    protocols = historical_protocols()
    grids = [item[1]["protocol"] for item in protocols]
    first = grids[0]
    for grid in grids:
        assert grid["primary_score"] == "(G+L)/2"
        assert grid["scales"] == first["scales"]
        assert grid["radii"] == first["radii"]
        assert grid["thresholds"] == first["thresholds"]
    configs = enumerate_configs(first["scales"], [int(x) for x in first["radii"]], first["thresholds"])
    assert len(configs) == 90
    assumptions = {
        **DISCLAIMER,
        "frozen_before_result_inspection": True,
        "unresolved_historical_engine_details": [
            "exact historical G normalization",
            "exact historical L construction and alignment",
            "exact historical candidate generator",
            "exact historical outer-fold temporal-scale estimator",
            "exact historical tie-break implementation",
        ],
        "v2_fixed_choices": {
            "temporal_scale": "read k_p from each frozen cache record; no GT-derived test-fold scale",
            "smoothing": "centered NumPy same-mode boxcar with width 2*round_half_up(scale*k_p)",
            "normalization": "full-video mean-to-max: (x-mean(x))/(max(x)-mean(x)); constant curves map to zero",
            "candidate_pool": "all scipy.find_peaks maxima of the reference-scale G curve; distance=round_half_up(reference_scale*k_p); no height gate",
            "G": "mean-to-max normalized reference-scale smoothed curve at fixed G candidates",
            "L": "mean of normalized saliency at the nearest peak from each of scales [1,1.5,2] within radius*k_p; missing scale contributes zero",
            "local_alignment_tie_break": "smaller distance, then higher normalized evidence, then earlier peak",
            "fusion": "S=(G+L)/2 exactly",
            "decoder_interval": "[peak-k_p, peak+k_p], inclusive, no clipping",
            "matching": "chronological predictions; best unmatched IoU; one-to-one; IoU>=0.5",
            "selection": "outer-subject LOSO; pooled training TP/FP/FN Raw F1",
            "selection_tie_break": "higher exact F1, fewer FP, fewer predictions, lexicographically smaller config_id",
            "ablation": "reuse each outer fold's GL-selected config; substitute G or L; no reselection",
            "robustness": "reuse selected config at threshold-0.05, threshold, threshold+0.05; no selection",
        },
        "forbidden_components_absent": [
            "persistence", "window", "recovery", "boundary module", "new evidence source", "weight search"
        ],
    }
    historical_targets = {}
    for dataset in ("sammlv", "casme3"):
        report_path = HISTORICAL_METST / dataset / "fixed" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        historical_targets[f"ME-TST/{dataset.upper()}"] = {
            "Native": report["metrics"]["original_native"],
            "G+L": report["metrics"]["pure"],
            "source": str(report_path.resolve()),
        }
    spec = {
        **DISCLAIMER,
        "version": "GL-Decoder-v2",
        "primary_formula": "S(c)=(G(c)+L(c))/2",
        "primary_score": "(G+L)/2",
        "scale_values": first["scales"],
        "radius_values": [int(x) for x in first["radii"]],
        "threshold_values": first["thresholds"],
        "outer_split": "leave one subject out; held-out subject excluded from selection counts",
        "selection_metric": "pooled training-subject Raw Spotting F1",
        "tie_break": "V2_RECONSTRUCTION_TIE_BREAK: higher exact F1 -> fewer FP -> fewer predictions -> lexicographically smaller config_id",
        "candidate_convention": "UNRESOLVED_FROM_HISTORICAL_ARTIFACTS; v2 fixed choice recorded in candidate_policy",
        "interval_convention": "[peak-k_p, peak+k_p], inclusive, no clipping",
        "matching_convention": "chronological predictions, best unmatched IoU, one-to-one, IoU>=0.5",
        "historical_definition_recovery": {
            "directly_recovered": [
                "arithmetic-mean formula (G+L)/2",
                "G-anchored fixed candidate-pool requirement",
                "three scale values, three radius values, ten threshold values",
                "outer-subject LOSO and pooled-training Raw Spotting F1",
                "inclusive IoU>=0.5 evaluation and historical target metrics",
            ],
            "G_exact_definition": "UNRESOLVED_FROM_HISTORICAL_ARTIFACTS",
            "L_exact_definition": "UNRESOLVED_FROM_HISTORICAL_ARTIFACTS",
            "normalization": "UNRESOLVED_FROM_HISTORICAL_ARTIFACTS",
            "tie_break_exact_implementation": "UNRESOLVED_FROM_HISTORICAL_ARTIFACTS",
        },
        "candidate_policy": assumptions["v2_fixed_choices"]["candidate_pool"],
        "G_definition": assumptions["v2_fixed_choices"]["G"],
        "L_definition": assumptions["v2_fixed_choices"]["L"],
        "search_space": {
            "scales": first["scales"],
            "radii": [int(x) for x in first["radii"]],
            "thresholds": first["thresholds"],
            "total": len(configs),
        },
        "phase_a_grade_policy": {
            "A": "both ME-TST GL counts exactly equal history, config agreement=1, exact interval Jaccard=1",
            "B": "both ME-TST |v2 F1-historical F1|<=0.01 and temporal-neighborhood Jaccard>=0.75",
            "C": "ME-TST native gate passes and v2 GL F1 exceeds Native on both datasets",
            "D": "otherwise; Phase B must not run",
        },
        "protocol_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path), "signature": payload["signature"]}
            for path, payload in protocols
        ],
        "historical_cache_sha256": {
            f"{payload['backbone']}/{payload['dataset']}": payload["input_sha256"]
            for _, payload in protocols
        },
        "historical_target_metrics": historical_targets,
        "source_boundary": {
            "historical_artifacts": "grid, formula, reports, selections and predictions are constraints/audit references only",
            "historical_executable_engine": "not recovered; no fresh historical execution claimed",
            "v2_code": str(Path(__file__).resolve().parent),
            "current_cache_experiments": "Phase A uses current ME-TST caches; Phase B uses current BoostingVRME caches",
        },
    }
    write_json(RESULT_ROOT / "reconstruction_spec.json", spec)
    write_json(RESULT_ROOT / "reconstruction_assumptions.json", assumptions)
    write_json(RESULT_ROOT / "gl_v2_configs.json", [config.as_dict() for config in configs])
    write_json(RESULT_ROOT / "candidate_generation_spec.json", {
        **DISCLAIMER,
        "historical_status": "UNRESOLVED_FROM_HISTORICAL_ARTIFACTS",
        "smoothing": assumptions["v2_fixed_choices"]["smoothing"],
        "peak_detector": "scipy.signal.find_peaks",
        "height_gate": "none during proposal; selected config threshold is applied to candidate evidence",
        "minimum_distance": "round_half_up(reference_scale*k_p)",
        "boundary_handling": "NumPy mode=same smoothing; scipy excludes endpoints; decoded intervals are not clipped",
    })
    return configs, spec


def sources_from_control_report() -> dict[tuple[str, str], dict]:
    payload = json.loads(CONTROL_REPORT.read_text(encoding="utf-8"))
    result = {}
    for report in payload["reports"]:
        normalized_dataset = report["dataset"].lower().replace("_", "")
        result[(report["backbone"].lower(), normalized_dataset)] = report
    return result


def empty_counter() -> dict:
    return {"TP": 0, "FP": 0, "FN": 0}


def accumulate(target: dict, source: dict) -> None:
    for key in ("TP", "FP", "FN"):
        target[key] += int(source[key])


def prepare_config_counts(records: list[dict], configs: list[GLConfig], decoder: GLDecoder):
    subjects = sorted({record["subject"] for record in records})
    banks: list[dict[tuple[float, int], object]] = []
    for record in records:
        bank = {}
        for scale in sorted({config.reference_scale for config in configs}):
            for radius in sorted({config.radius for config in configs}):
                prototype = GLConfig(scale, radius, 0.0)
                bank[(scale, radius)] = decoder.evidence(
                    record["score"], record["temporal_scale"], prototype
                )
        banks.append(bank)
    per_config = {}
    for config in configs:
        subject_counts = {subject: empty_counter() for subject in subjects}
        for index, record in enumerate(records):
            evidence = banks[index][(config.reference_scale, config.radius)]
            events = decoder.events_from_evidence(
                evidence, record["temporal_scale"], config.threshold, "GL"
            )
            accumulate(subject_counts[record["subject"]], evaluate_events(events, record["gt"]))
        per_config[config.config_id] = subject_counts
    return banks, per_config


def run_dataset(backbone: str, dataset: str, records: list[dict], configs: list[GLConfig]) -> dict:
    decoder = GLDecoder()
    subjects = sorted({record["subject"] for record in records})
    banks, per_config = prepare_config_counts(records, configs, decoder)
    selections = []
    selected_by_subject = {}
    for subject in subjects:
        config, inner = select_config(configs, per_config, subject)
        selected_by_subject[subject] = config
        selections.append({
            "backbone": backbone, "dataset": dataset, "outer_subject": subject,
            **config.as_dict(), "inner_TP": inner["TP"], "inner_FP": inner["FP"],
            "inner_FN": inner["FN"], "inner_F1": inner["F1"],
        })

    predictions = []
    subject_methods = {subject: {mode: empty_counter() for mode in ("Native", "G", "L", "GL")} for subject in subjects}
    robustness_subject = {
        offset: {subject: empty_counter() for subject in subjects} for offset in (-0.05, 0.0, 0.05)
    }
    mechanism = Counter()
    for index, record in enumerate(records):
        subject = record["subject"]
        config = selected_by_subject[subject]
        evidence = banks[index][(config.reference_scale, config.radius)]
        events_by_mode = {"Native": native_events(record["score"], record["temporal_scale"])}
        for mode in ("G", "L", "GL"):
            events_by_mode[mode] = decoder.events_from_evidence(
                evidence, record["temporal_scale"], config.threshold, mode
            )
        evaluations = {}
        for mode, events in events_by_mode.items():
            evaluations[mode] = evaluate_events(events, record["gt"])
            accumulate(subject_methods[subject][mode], evaluations[mode])
        native_gt = set(evaluations["Native"]["matched_gt"])
        gl_gt = set(evaluations["GL"]["matched_gt"])
        mechanism["retained_TP"] += len(native_gt & gl_gt)
        mechanism["lost_TP"] += len(native_gt - gl_gt)
        mechanism["rescued_TP"] += len(gl_gt - native_gt)
        native_fp = {
            (item["onset"], item["offset"], item["peak"])
            for idx, item in enumerate(events_by_mode["Native"])
            if idx not in set(evaluations["Native"]["matched_predictions"])
        }
        gl_fp = {
            (item["onset"], item["offset"], item["peak"])
            for idx, item in enumerate(events_by_mode["GL"])
            if idx not in set(evaluations["GL"]["matched_predictions"])
        }
        mechanism["removed_FP"] += len(native_fp - gl_fp)
        mechanism["new_FP"] += len(gl_fp - native_fp)
        for global_score, local_score in zip(evidence.global_scores, evidence.local_scores, strict=True):
            quadrant = f"G{'hi' if global_score >= config.threshold else 'lo'}_L{'hi' if local_score >= config.threshold else 'lo'}"
            mechanism[quadrant] += 1
        predictions.append({
            "backbone": backbone, "dataset": dataset, "subject": subject,
            "video": record["video"], "temporal_scale": record["temporal_scale"],
            "selected_config": config.as_dict(), "gt": record["gt"],
            "events": events_by_mode,
            "evaluation": evaluations,
        })
        for offset in (-0.05, 0.0, 0.05):
            threshold = max(0.0, round(config.threshold + offset, 10))
            events = decoder.events_from_evidence(evidence, record["temporal_scale"], threshold, "GL")
            accumulate(robustness_subject[offset][subject], evaluate_events(events, record["gt"]))

    subject_rows = []
    for subject in subjects:
        row = {"backbone": backbone, "dataset": dataset, "subject": subject}
        for mode in ("Native", "G", "L", "GL"):
            for key, value in subject_methods[subject][mode].items():
                row[f"{mode}_{key}"] = value
        subject_rows.append(row)
    metrics = {
        mode: add_counts([subject_methods[subject][mode] for subject in subjects])
        for mode in ("Native", "G", "L", "GL")
    }
    robustness = []
    for offset in (-0.05, 0.0, 0.05):
        total = add_counts(list(robustness_subject[offset].values()))
        robustness.append({
            "backbone": backbone, "dataset": dataset, "threshold_offset": offset, **total
        })
    return {
        "backbone": backbone, "dataset": dataset, "metrics": metrics,
        "selections": selections, "predictions": predictions,
        "subject_rows": subject_rows, "bootstrap": paired_bootstrap(subject_rows),
        "robustness": robustness, "mechanism": dict(mechanism),
        "selection_frequency": dict(Counter(row["config_id"] for row in selections)),
    }


def native_gate(records: list[dict], expected: list[int]) -> dict:
    observed = add_counts([
        evaluate_events(native_events(record["score"], record["temporal_scale"]), record["gt"])
        for record in records
    ])
    triplet = [observed[key] for key in ("TP", "FP", "FN")]
    return {"expected": expected, "observed": triplet, "F1": observed["F1"], "status": "PASS" if triplet == expected else "FAIL"}


def historical_reference(dataset: str) -> dict:
    base = HISTORICAL_METST / dataset / "fixed"
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))
    predictions = json.loads((base / "selected_predictions.json").read_text(encoding="utf-8"))
    selections = [row for row in read_csv(base / "outer_loso_selections.csv") if row["family"] == "pure"]
    return {"base": base, "report": report, "predictions": predictions, "selections": selections}


def parse_historical_config(value: str) -> tuple[float, int, float]:
    fields = dict(re.findall(r"(scale|radius|tau)=([0-9.]+)", value))
    return float(fields["scale"]), int(float(fields["radius"])), float(fields["tau"])


def phase_a_agreement(result: dict, historical: dict) -> tuple[list[dict], dict]:
    historical_predictions = {
        (str(row["subject"]), str(row["video"])): row["predictions"]["pure"]
        for row in historical["predictions"]
    }
    detail = []
    current_all_intervals, historical_all_intervals = set(), set()
    current_all_peaks, historical_all_peaks = [], []
    neighborhood_numerator = []
    for row in result["predictions"]:
        key = (row["subject"], row["video"])
        old = historical_predictions[key]
        current = row["events"]["GL"]
        comparison = compare_predictions(current, old, row["temporal_scale"])
        detail.append({"subject": key[0], "video": key[1], **comparison})
        current_all_intervals |= {(key, item["onset"], item["offset"]) for item in current}
        historical_all_intervals |= {(key, item["onset"], item["offset"]) for item in old}
        current_all_peaks.extend([(key, int(item["peak"])) for item in current])
        historical_all_peaks.extend([(key, int(item["peak"])) for item in old])
        neighborhood_numerator.append(comparison["temporal_neighborhood_jaccard"])
    historical_selection = {str(row["subject"]): parse_historical_config(row["config"]) for row in historical["selections"]}
    config_matches = 0
    config_detail = []
    for row in result["selections"]:
        current = (float(row["reference_scale"]), int(row["radius"]), float(row["threshold"]))
        old = historical_selection[row["outer_subject"]]
        match = current == old
        config_matches += int(match)
        config_detail.append({
            "subject": row["outer_subject"], "v2_config_id": row["config_id"],
            "historical_scale": old[0], "historical_radius": old[1], "historical_threshold": old[2],
            "exact_match": match,
        })
    aggregate = {
        "exact_interval_jaccard": set_jaccard(current_all_intervals, historical_all_intervals),
        "exact_peak_jaccard": set_jaccard(set(current_all_peaks), set(historical_all_peaks)),
        "mean_per_video_temporal_neighborhood_jaccard": sum(neighborhood_numerator) / len(neighborhood_numerator),
        "config_exact_agreement": config_matches / len(result["selections"]),
        "config_detail": config_detail,
    }
    historical_subjects = {
        str(row["subject"]): {key: int(row[key]) for key in ("TP", "FP", "FN")}
        for row in read_csv(historical["base"] / "subject_counts.csv")
        if row["family"] == "pure"
    }
    subject_detail = []
    exact_subjects = 0
    for row in result["subject_rows"]:
        current = {key: int(row[f"GL_{key}"]) for key in ("TP", "FP", "FN")}
        old = historical_subjects[row["subject"]]
        exact = current == old
        exact_subjects += int(exact)
        subject_detail.append({
            "subject": row["subject"],
            **{f"v2_{key}": current[key] for key in ("TP", "FP", "FN")},
            **{f"historical_{key}": old[key] for key in ("TP", "FP", "FN")},
            "exact_match": exact,
        })
    aggregate["subject_count_exact_agreement"] = exact_subjects / len(subject_detail)
    aggregate["subject_count_detail"] = subject_detail
    return detail, aggregate


def grade_phase_a(results: list[dict], agreements: dict[str, dict], gates: dict[str, dict]) -> str:
    if not all(gate["status"] == "PASS" for gate in gates.values()):
        return "D"
    exact = True
    close = True
    improved = True
    for result in results:
        dataset = result["dataset"].lower()
        historical = historical_reference(dataset)["report"]["metrics"]["pure"]
        current = result["metrics"]["GL"]
        exact &= all(current[key] == historical[key] for key in ("TP", "FP", "FN"))
        exact &= agreements[dataset]["config_exact_agreement"] == 1.0
        exact &= agreements[dataset]["exact_interval_jaccard"] == 1.0
        close &= abs(current["F1"] - historical["F1"]) <= 0.01
        close &= agreements[dataset]["mean_per_video_temporal_neighborhood_jaccard"] >= 0.75
        improved &= current["F1"] > result["metrics"]["Native"]["F1"]
    if exact:
        return "A"
    if close:
        return "B"
    if improved:
        return "C"
    return "D"


def metric_rows(results: list[dict]) -> list[dict]:
    return [
        {"backbone": result["backbone"], "dataset": result["dataset"], "method": method, **metrics}
        for result in results for method, metrics in result["metrics"].items()
    ]


def write_phase_b(results: list[dict], sources: dict) -> None:
    write_csv(PHASE_B / "main_results.csv", metric_rows(results))
    write_csv(PHASE_B / "ablation_results.csv", metric_rows(results))
    write_csv(PHASE_B / "bootstrap_ci.csv", [
        {"backbone": item["backbone"], "dataset": item["dataset"], **item["bootstrap"]}
        for item in results
    ])
    write_csv(PHASE_B / "robustness_results.csv", [row for item in results for row in item["robustness"]])
    write_json(PHASE_B / "mechanism_audit.json", {
        "definition": "Native-versus-GL matched-GT transitions, exact FP geometry transitions, and candidate G/L threshold quadrants",
        "groups": [{"backbone": item["backbone"], "dataset": item["dataset"], **item["mechanism"]} for item in results],
    })
    write_csv(PHASE_B / "mechanism_audit.csv", [
        {"backbone": item["backbone"], "dataset": item["dataset"], **item["mechanism"]}
        for item in results
    ])
    write_json(PHASE_B / "selected_predictions.json", [row for item in results for row in item["predictions"]])
    write_csv(PHASE_B / "outer_loso_selections.csv", [row for item in results for row in item["selections"]])
    write_csv(PHASE_B / "subject_counts.csv", [row for item in results for row in item["subject_rows"]])
    report = {
        **DISCLAIMER,
        "groups": [
            {
                "backbone": item["backbone"], "dataset": item["dataset"],
                "cache": sources[(item["backbone"].lower(), item["dataset"].lower())]["input"],
                "metrics": item["metrics"], "bootstrap": item["bootstrap"],
                "selection_frequency": item["selection_frequency"],
            }
            for item in results
        ],
    }
    write_json(PHASE_B / "main_report.json", report)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    configs, spec = freeze_spec()
    source_reports = sources_from_control_report()
    cache_specs = [
        ("ME-TST", "SAMMLV", PROJECT / "caches/me_tst/sammlv_strategy1_outputs.pkl", load_me_tst),
        ("ME-TST", "CASME3", PROJECT / "caches/me_tst/casme3_strategy1_outputs.pkl", load_me_tst),
        ("BoostingVRME", "SAMMLV", PROJECT / "caches/boostingvrme/sammlv_curves.pkl", load_boostingvrme),
        ("BoostingVRME", "CASME3", PROJECT / "caches/boostingvrme/casme3_curves.pkl", load_boostingvrme),
    ]
    loaded = {}
    provenance = []
    gates = {}
    for backbone, dataset, cache, loader in cache_specs:
        records = loader(cache)
        loaded[(backbone, dataset)] = records
        source = source_reports[(backbone.lower(), dataset.lower())]
        expected = source["baseline_verification"]["expected_TP_FP_FN"]
        gate = native_gate(records, expected)
        gate["cache_sha256"] = sha256(cache)
        gate["verified_source_sha256"] = source["input"]["sha256"]
        gate["cache_identity_pass"] = gate["cache_sha256"] == gate["verified_source_sha256"]
        if backbone == "ME-TST" and not gate["cache_identity_pass"]:
            gate["status"] = "FAIL"
        gates[f"{backbone}/{dataset}"] = gate
        provenance.append({
            "backbone": backbone, "dataset": dataset, "cache": str(cache.resolve()),
            "cache_sha256": sha256(cache), "verified_source": str(CONTROL_REPORT.resolve()),
            "verified_native_counts": expected,
        })
    write_json(RESULT_ROOT / "input_provenance.json", provenance)
    write_json(PHASE_A / "native_gate.json", {key: value for key, value in gates.items() if key.startswith("ME-TST/")})
    if not all(value["status"] == "PASS" for key, value in gates.items() if key.startswith("ME-TST/")):
        write_json(RESULT_ROOT / "combined_report.json", {
            **DISCLAIMER, "phase_a_status": "NATIVE_GATE_FAILED", "phase_b_status": "NOT_RUN", "gates": gates
        })
        raise SystemExit("ME-TST native hash/count gate failed; Phase A GL and Phase B were not run")

    phase_a_results = []
    agreements = {}
    for dataset in ("SAMMLV", "CASME3"):
        result = run_dataset("ME-TST", dataset, loaded[("ME-TST", dataset)], configs)
        phase_a_results.append(result)
        historical = historical_reference(dataset.lower())
        detail, aggregate = phase_a_agreement(result, historical)
        agreements[dataset.lower()] = aggregate
        write_csv(PHASE_A / f"{dataset.lower()}_prediction_agreement.csv", detail)
        write_json(PHASE_A / f"{dataset.lower()}_config_agreement.json", aggregate)
    config_agreement_rows = []
    prediction_agreement_rows = []
    for item in phase_a_results:
        dataset_key = item["dataset"].lower()
        for row in agreements[dataset_key]["config_detail"]:
            config_agreement_rows.append({"dataset": item["dataset"], **row})
        for row in read_csv(PHASE_A / f"{dataset_key}_prediction_agreement.csv"):
            prediction_agreement_rows.append({"dataset": item["dataset"], **row})
    write_csv(PHASE_A / "config_agreement.csv", config_agreement_rows)
    write_csv(PHASE_A / "prediction_agreement.csv", prediction_agreement_rows)
    grade = grade_phase_a(phase_a_results, agreements, {
        key.split("/")[1].lower(): value for key, value in gates.items() if key.startswith("ME-TST/")
    })
    write_csv(PHASE_A / "outer_loso_selections.csv", [row for item in phase_a_results for row in item["selections"]])
    write_json(PHASE_A / "selected_predictions.json", [row for item in phase_a_results for row in item["predictions"]])
    write_csv(PHASE_A / "subject_counts.csv", [row for item in phase_a_results for row in item["subject_rows"]])
    phase_a_report = {
        **DISCLAIMER, "grade": grade,
        "level_status": {
            "A": "GL_V2_EXACT_HISTORICAL_MATCH",
            "B": "GL_V2_SEMANTIC_RECONSTRUCTION_MATCH",
            "C": "GL_V2_DIRECTIONAL_ONLY",
            "D": "GL_V2_RECONSTRUCTION_FAILED",
        }[grade],
        "grade_policy": spec["phase_a_grade_policy"],
        "native_gates": {key: value for key, value in gates.items() if key.startswith("ME-TST/")},
        "datasets": [
            {
                "dataset": item["dataset"], "metrics": item["metrics"],
                "historical_GL": historical_reference(item["dataset"].lower())["report"]["metrics"]["pure"],
                "GL_minus_historical_F1": item["metrics"]["GL"]["F1"] - historical_reference(item["dataset"].lower())["report"]["metrics"]["pure"]["F1"],
                "GL_minus_historical_counts": {
                    key: item["metrics"]["GL"][key] - historical_reference(item["dataset"].lower())["report"]["metrics"]["pure"][key]
                    for key in ("TP", "FP", "FN")
                },
                "agreement": agreements[item["dataset"].lower()],
                "selection_frequency": item["selection_frequency"],
            }
            for item in phase_a_results
        ],
    }
    write_json(PHASE_A / "reconstruction_report.json", phase_a_report)
    write_json(PHASE_A / "report.json", phase_a_report)

    if grade == "D":
        write_json(RESULT_ROOT / "combined_report.json", {
            **DISCLAIMER, "phase_a_status": "GRADE_D", "phase_a_grade": grade,
            "phase_b_status": "NOT_RUN_PER_PROTOCOL", "phase_a_report": phase_a_report,
        })
        print(json.dumps({"phase_a_grade": grade, "phase_b": "NOT_RUN_PER_PROTOCOL"}, indent=2))
        return

    boosting_results = [
        run_dataset("BoostingVRME", dataset, loaded[("BoostingVRME", dataset)], configs)
        for dataset in ("SAMMLV", "CASME3")
    ]
    all_results = phase_a_results + boosting_results
    write_phase_b(all_results, source_reports)
    combined = {
        **DISCLAIMER,
        "phase_a_status": "PASSED_FOR_PHASE_B", "phase_a_grade": grade,
        "phase_b_status": "COMPLETED_ON_CURRENT_CACHES",
        "fresh_historical_engine_reproduction": "BLOCKED_ENGINE_NOT_RECOVERED",
        "main_results": metric_rows(all_results),
        "bootstrap": [
            {"backbone": item["backbone"], "dataset": item["dataset"], **item["bootstrap"]}
            for item in all_results
        ],
        "answers": {
            "1_v2_identity": DISCLAIMER["implementation_identity"],
            "2_historical_identity": DISCLAIMER["historical_claim"],
            "3_primary_formula": "(G+L)/2",
            "4_config_count": 90,
            "5_candidate_policy": spec["candidate_policy"],
            "6_G": spec["G_definition"],
            "7_L": spec["L_definition"],
            "8_selection": "outer-subject LOSO pooled-training Raw F1",
            "9_tie_break": "higher F1, fewer FP, fewer predictions, lexicographically smaller ID",
            "10_phase_a_grade": grade,
            "11_phase_b_cache_scope": "current BoostingVRME caches only",
            "12_decoder_sharing": "one GLDecoder class; adapters contain schema conversion only",
            "13_ablation_search": "none; selected GL configs frozen",
            "14_bootstrap": "subject-paired, 10000 resamples, seed 100, recomputed_v2",
            "15_robustness": "selected threshold +/-0.05 only",
            "16_historical_engine": "missing; no fresh historical algorithm reproduction claimed",
            "17_stop_condition": "Phase B completed; no further method exploration performed",
        },
    }
    write_json(RESULT_ROOT / "combined_report.json", combined)
    print(json.dumps({
        "phase_a_grade": grade, "phase_b": "COMPLETED_ON_CURRENT_CACHES",
        "results": str(RESULT_ROOT.resolve())
    }, indent=2))


if __name__ == "__main__":
    main()
