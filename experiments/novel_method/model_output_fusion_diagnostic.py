"""Decoder-aware Native/GLSD output-fusion diagnostic.

This is a local diagnostic on the historical frozen caches.  It deliberately
separates deployable routing rules from the label-using per-video oracle.
It does not modify GLSD, the official runners, or paper results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from my_method.gl_saliency_skill.benchmark import GROUPS, load_bundle, signed_context
from my_method.gl_saliency_skill.evaluation import evaluate, metrics
from my_method.gl_saliency_skill.evidence import configuration_grid
from my_method.gl_saliency_skill.selection import inner_counts, select_all
from my_method.gl_saliency_skill.skill import GLSaliencySkill


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_output_fusion_diagnostic"
COUNT_KEYS = ("TP", "FP", "FN")


def f1(counts: dict[str, int]) -> float:
    return float(metrics(counts)["F1"])


def add_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left[key]) + int(right[key]) for key in COUNT_KEYS}


def event_key(event: dict) -> tuple[int, int, int]:
    return int(event["onset"]), int(event["peak"]), int(event["offset"])


def interval_iou(a: dict, b: dict) -> float:
    left = max(int(a["onset"]), int(b["onset"]))
    right = min(int(a["offset"]), int(b["offset"]))
    inter = max(0, right - left + 1)
    union = (int(a["offset"]) - int(a["onset"]) + 1) + (
        int(b["offset"]) - int(b["onset"]) + 1
    ) - inter
    return inter / union if union else 0.0


def conflict(a: dict, b: dict, k: int) -> bool:
    # Output-level duplicate suppression needs a declared rule.  IoU>=.5 is
    # the evaluator's event identity threshold; peak distance<=k handles two
    # nearby intervals that do not overlap enough after different decoders.
    return interval_iou(a, b) >= 0.5 or abs(int(a["peak"]) - int(b["peak"])) <= k


def union_max_nms(native: list[dict], gl: list[dict], k: int) -> list[dict]:
    candidates = []
    for source, events in (("Native", native), ("GLSD", gl)):
        for index, event in enumerate(events):
            candidates.append({**event, "_source": source, "_index": index})
    # Higher decoder confidence wins duplicate clusters.  Source/index are
    # deterministic tie-breaks; no labels are used.
    candidates.sort(key=lambda e: (-float(e.get("confidence", 0.0)), e["_source"], e["_index"]))
    kept: list[dict] = []
    for event in candidates:
        if any(conflict(event, previous, k) for previous in kept):
            continue
        kept.append(event)
    kept.sort(key=lambda e: (int(e["onset"]), int(e["peak"]), e["_source"], e["_index"]))
    return [{key: value for key, value in event.items() if not key.startswith("_")} for event in kept]


def agreement_consensus(native: list[dict], gl: list[dict], k: int) -> list[dict]:
    # Keep only events supported by both decoders.  Native and GLSD events are
    # generated from the same response, so this is a conservative agreement
    # decoder rather than a second score-level fusion rule.
    selected = []
    for event in native:
        if any(conflict(event, other, k) for other in gl):
            selected.append(event)
    return selected


def aggregate(rows: list[dict]) -> dict[str, int]:
    return {key: int(sum(int(row[key]) for row in rows)) for key in COUNT_KEYS}


def choose_oracle(native_counts: dict[str, int], gl_counts: dict[str, int]) -> str:
    # Diagnostic only: this accesses the held-out video's GT through counts.
    n = (f1(native_counts), -native_counts["FP"], native_counts["TP"], 0)
    g = (f1(gl_counts), -gl_counts["FP"], gl_counts["TP"], -1)
    return "Native" if n >= g else "GLSD"


def run_group(backbone: str, dataset: str, configs: list):
    bundle = load_bundle(backbone, dataset)
    signed_root, protocol, outer, inner, stats, mode = signed_context(bundle, configs)
    selected, _ = select_all(configs, stats, bundle.subjects, outer, inner)
    subject_index = {subject: index for index, subject in enumerate(bundle.subjects)}
    subject_routes = {}
    for held, subject in enumerate(bundle.subjects):
        training = inner_counts(stats, held, inner)
        native = {key: int(value) for key, value in zip(COUNT_KEYS, training[0])}
        gl_id = selected[subject]["GL"]
        gl = {key: int(value) for key, value in zip(COUNT_KEYS, training[gl_id])}
        subject_routes[subject] = "Native" if (f1(native), -native["FP"], native["TP"]) >= (f1(gl), -gl["FP"], gl["TP"]) else "GLSD"

    video_rows = []
    per_subject = {name: {subject: [] for subject in bundle.subjects} for name in ("Native", "GLSD", "UnionMaxNMS", "Agreement", "SubjectRoute", "VideoOracle")}
    for record in bundle.records:
        sid = subject_index[record.subject]
        native_config = configs[0]
        native_events = GLSaliencySkill(native_config).decode(record.score, bundle.legacy_temporal_scale, {"interval_adapter": bundle.interval_adapter})
        gl_config = configs[selected[record.subject]["GL"]]
        gl_events = GLSaliencySkill(gl_config).decode(record.score, int(outer[sid]), {"interval_adapter": bundle.interval_adapter})
        predictions = {
            "Native": native_events,
            "GLSD": gl_events,
            "UnionMaxNMS": union_max_nms(native_events, gl_events, int(outer[sid])),
            "Agreement": agreement_consensus(native_events, gl_events, int(outer[sid])),
            "SubjectRoute": native_events if subject_routes[record.subject] == "Native" else gl_events,
        }
        counts = {}
        for name, events in predictions.items():
            counts[name], _ = evaluate(events, record.ground_truth)
            per_subject[name][record.subject].append(counts[name])
        oracle = choose_oracle(counts["Native"], counts["GLSD"])
        predictions["VideoOracle"] = predictions[oracle]
        counts["VideoOracle"] = counts[oracle]
        per_subject["VideoOracle"][record.subject].append(counts["VideoOracle"])
        video_rows.append({
            "backbone": backbone,
            "dataset": dataset,
            "subject": record.subject,
            "video": record.video,
            "native_f1": f1(counts["Native"]),
            "glsd_f1": f1(counts["GLSD"]),
            "oracle_choice": oracle,
            "native_tp": counts["Native"]["TP"], "native_fp": counts["Native"]["FP"], "native_fn": counts["Native"]["FN"],
            "glsd_tp": counts["GLSD"]["TP"], "glsd_fp": counts["GLSD"]["FP"], "glsd_fn": counts["GLSD"]["FN"],
            "union_tp": counts["UnionMaxNMS"]["TP"], "union_fp": counts["UnionMaxNMS"]["FP"], "union_fn": counts["UnionMaxNMS"]["FN"],
            "agreement_tp": counts["Agreement"]["TP"], "agreement_fp": counts["Agreement"]["FP"], "agreement_fn": counts["Agreement"]["FN"],
            "subject_route": subject_routes[record.subject],
            "route_tp": counts["SubjectRoute"]["TP"], "route_fp": counts["SubjectRoute"]["FP"], "route_fn": counts["SubjectRoute"]["FN"],
            "oracle_tp": counts["VideoOracle"]["TP"], "oracle_fp": counts["VideoOracle"]["FP"], "oracle_fn": counts["VideoOracle"]["FN"],
        })

    summary = []
    for method, subject_rows in per_subject.items():
        rows = [row for values in subject_rows.values() for row in values]
        counts = aggregate(rows)
        summary.append({"backbone": backbone, "dataset": dataset, "method": method, **counts, **metrics(counts), "videos": len(rows)})
    return summary, video_rows, subject_routes, str(signed_root), protocol["signature"], mode


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configs = configuration_grid()
    all_summary, all_videos, routes, signatures = [], [], {}, {}
    for backbone, dataset in GROUPS:
        summary, videos, subject_routes, signed_root, signature, mode = run_group(backbone, dataset, configs)
        all_summary.extend(summary)
        all_videos.extend(videos)
        routes[f"{backbone}/{dataset}"] = subject_routes
        signatures[f"{backbone}/{dataset}"] = {"signed_root": signed_root, "protocol_signature": signature, "interval_mode": mode}
    write_csv(OUT / "summary.csv", all_summary)
    write_csv(OUT / "per_video.csv", all_videos)
    (OUT / "protocol.json").write_text(json.dumps({
        "status": "DIAGNOSTIC_ONLY",
        "historical_cache": True,
        "datasets": "SAMMLV and CAS(ME)3 local frozen caches; CAS cache uses the historical 858-event protocol",
        "methods": ["Native", "GLSD", "UnionMaxNMS", "Agreement", "SubjectRoute", "VideoOracle"],
        "video_oracle_uses_ground_truth": True,
        "subject_route_uses_inner_training_counts_only": True,
        "union_rule": "greedy confidence-max duplicate suppression, IoU>=0.5 or peak distance<=k",
        "signatures": signatures,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DIAGNOSTIC_ONLY", "output": str(OUT.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
