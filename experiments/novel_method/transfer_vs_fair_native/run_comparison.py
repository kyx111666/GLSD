"""Paired comparison of locked transferred GLSD and locked Fair Tuned Native.

This script is read-only with respect to all locked experiment artifacts.  It
independently re-evaluates both prediction sets against the same per-video GT,
aggregates subject counts, and performs subject-paired bootstrap resampling.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from my_method.gl_saliency_skill.benchmark import load_bundle
from my_method.gl_saliency_skill.evaluation import evaluate, metrics


PROJECT = Path(__file__).resolve().parents[2]
TRANSFER_ROOT = PROJECT / "results" / "final_gl_skill_transfer"
FAIR_ROOT = PROJECT / "results" / "fair_tuned_native_v2"
OUTPUT_ROOT = PROJECT / "results" / "transfer_vs_fair_native"
SEED = 100
RESAMPLES = 10_000
COUNT_KEYS = ("TP", "FP", "FN")
SETTINGS = (
    ("metst", "boostingvrme", "sammlv"),
    ("metst", "boostingvrme", "casme3"),
    ("boostingvrme", "metst", "sammlv"),
    ("boostingvrme", "metst", "casme3"),
)
LOCKED_F1 = {
    ("metst", "boostingvrme", "sammlv"): (0.28923076923076924, 0.28486646884273),
    ("metst", "boostingvrme", "casme3"): (0.1232741617357002, 0.11559009227780476),
    ("boostingvrme", "metst", "sammlv"): (0.24920127795527156, 0.2561307901907357),
    ("boostingvrme", "metst", "casme3"): (0.09702209414024976, 0.09970089730807577),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_events(events: list[dict]) -> list[dict]:
    """Remove stored matching labels before independent evaluation."""
    return [
        {
            "onset": int(event["onset"]),
            "peak": int(event["peak"]),
            "offset": int(event["offset"]),
            **(
                {"confidence": float(event["confidence"])}
                if "confidence" in event
                else {}
            ),
        }
        for event in events
    ]


def add_counts(target: np.ndarray, counts: dict) -> None:
    target += np.array([counts[key] for key in COUNT_KEYS], dtype=np.int64)


def aggregate(values: np.ndarray) -> dict:
    totals = values.sum(axis=0)
    return metrics(dict(zip(COUNT_KEYS, totals)))


def paired_bootstrap(transferred: np.ndarray, fair: np.ndarray) -> dict:
    if transferred.shape != fair.shape or transferred.ndim != 2 or transferred.shape[1] != 3:
        raise ValueError("Paired subject count arrays must have identical N x 3 shape")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(transferred), size=(RESAMPLES, len(transferred)))

    def sampled_f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0],
            denominator,
            out=np.zeros(RESAMPLES, dtype=float),
            where=denominator > 0,
        )

    deltas = sampled_f1(transferred) - sampled_f1(fair)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "point_delta_F1": float(aggregate(transferred)["F1"] - aggregate(fair)["F1"]),
        "bootstrap_mean_delta_F1": float(deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_delta_gt_0": float(np.mean(deltas > 0.0)),
        "crosses_zero": bool(low <= 0.0 <= high),
        "resamples": RESAMPLES,
        "seed": SEED,
        "paired_unit": "outer subject",
    }


def compare_setting(source: str, target: str, dataset: str, transfer_rows: list[dict]):
    fair_dir = FAIR_ROOT / target / dataset
    fair_path = fair_dir / "selected_predictions.json"
    protocol_path = fair_dir / "PROTOCOL.json"
    fair_rows = read_json(fair_path)
    fair_protocol = read_json(protocol_path)
    bundle = load_bundle(target, dataset)

    fair_by_video = {(row["subject"], row["video"]): row for row in fair_rows}
    bundle_by_video = {
        (record.subject, record.video): record for record in bundle.records
    }
    selected_transfer = [
        row
        for row in transfer_rows
        if (row["source"], row["target"], row["dataset"])
        == (source, target, dataset)
    ]
    transfer_by_video = {
        (row["subject"], row["video"]): row for row in selected_transfer
    }

    fair_keys = set(fair_by_video)
    transfer_keys = set(transfer_by_video)
    bundle_keys = set(bundle_by_video)
    if len(fair_by_video) != len(fair_rows):
        raise AssertionError("Duplicate Fair Tuned Native subject/video key")
    if len(transfer_by_video) != len(selected_transfer):
        raise AssertionError("Duplicate transferred GLSD subject/video key")
    if fair_keys != transfer_keys or fair_keys != bundle_keys:
        raise AssertionError("Prediction/GT video identities are not exactly aligned")

    cache_digest = sha256(Path(bundle.cache_path))
    if cache_digest != fair_protocol["input_sha256"]:
        raise AssertionError("Fair Native and target GLSD bundle cache SHA differ")

    subjects = list(bundle.subjects)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    if set(subjects) != {subject for subject, _ in fair_keys}:
        raise AssertionError("Outer-subject identities differ")

    transferred_counts = np.zeros((len(subjects), 3), dtype=np.int64)
    fair_counts = np.zeros((len(subjects), 3), dtype=np.int64)
    stored_transfer_mismatches = 0
    stored_fair_match_mismatches = 0
    gt_mismatches = 0

    for key in sorted(fair_keys, key=lambda item: (int(item[0]), item[0], item[1])):
        subject, _ = key
        fair_row = fair_by_video[key]
        transfer_row = transfer_by_video[key]
        record = bundle_by_video[key]
        gt = [[int(value) for value in event] for event in fair_row["gt"]]
        if gt != record.ground_truth:
            gt_mismatches += 1

        transfer_result, _ = evaluate(
            clean_events(transfer_row["predictions"]), gt
        )
        fair_result, fair_details = evaluate(
            clean_events(fair_row["predictions"]["single_tuned"]), gt
        )
        if transfer_result != transfer_row["counts"]:
            stored_transfer_mismatches += 1
        stored_fair = [int(item["matched_gt"]) for item in fair_row["predictions"]["single_tuned"]]
        recounted_fair = [int(item["matched_gt"]) for item in fair_details]
        if stored_fair != recounted_fair:
            stored_fair_match_mismatches += 1
        add_counts(transferred_counts[subject_index[subject]], transfer_result)
        add_counts(fair_counts[subject_index[subject]], fair_result)

    if gt_mismatches or stored_transfer_mismatches or stored_fair_match_mismatches:
        raise AssertionError(
            "Independent prediction/GT recount failed: "
            f"gt={gt_mismatches}, transfer={stored_transfer_mismatches}, "
            f"fair={stored_fair_match_mismatches}"
        )

    transfer_metrics = aggregate(transferred_counts)
    fair_metrics = aggregate(fair_counts)
    expected_transfer, expected_fair = LOCKED_F1[(source, target, dataset)]
    if abs(transfer_metrics["F1"] - expected_transfer) > 1e-15:
        raise AssertionError("Transferred GLSD does not replay locked aggregate F1")
    if abs(fair_metrics["F1"] - expected_fair) > 1e-15:
        raise AssertionError("Fair Tuned Native does not replay locked aggregate F1")

    outer_rows = []
    for index, subject in enumerate(subjects):
        transfer_subject = metrics(dict(zip(COUNT_KEYS, transferred_counts[index])))
        fair_subject = metrics(dict(zip(COUNT_KEYS, fair_counts[index])))
        outer_rows.append(
            {
                "source": source,
                "target": target,
                "dataset": dataset,
                "outer_subject": subject,
                "transferred_tp": transfer_subject["TP"],
                "transferred_fp": transfer_subject["FP"],
                "transferred_fn": transfer_subject["FN"],
                "transferred_f1": transfer_subject["F1"],
                "fair_native_tp": fair_subject["TP"],
                "fair_native_fp": fair_subject["FP"],
                "fair_native_fn": fair_subject["FN"],
                "fair_native_f1": fair_subject["F1"],
                "subject_f1_delta": transfer_subject["F1"] - fair_subject["F1"],
                "gt_count_transfer": transfer_subject["TP"] + transfer_subject["FN"],
                "gt_count_fair_native": fair_subject["TP"] + fair_subject["FN"],
            }
        )

    bootstrap = paired_bootstrap(transferred_counts, fair_counts)
    alignment = {
        "source": source,
        "target": target,
        "dataset": dataset,
        "outer_subjects_exact": True,
        "video_identities_exact": True,
        "ground_truth_exact": True,
        "cache_sha_exact": True,
        "cache_sha256": cache_digest,
        "evaluation": "inclusive-frame IoU >= 0.5; chronological highest-IoU greedy; no second-best rematch",
        "evaluation_implementation": "my_method/gl_saliency_skill/evaluation.py:evaluate",
        "transferred_stored_counts_exact": True,
        "fair_stored_matches_exact": True,
        "subjects": len(subjects),
        "videos": len(fair_keys),
        "ground_truth_events": int((transferred_counts[:, 0] + transferred_counts[:, 2]).sum()),
        "fair_protocol_signature": fair_protocol["signature"],
    }
    return transfer_metrics, fair_metrics, bootstrap, outer_rows, alignment


def render_report(summary: list[dict], bootstraps: list[dict], alignments: list[dict]) -> str:
    labels = {
        ("metst", "boostingvrme", "sammlv"): "ME-TST+→BoostingVRME / SAMMLV",
        ("metst", "boostingvrme", "casme3"): "ME-TST+→BoostingVRME / CAS(ME)3",
        ("boostingvrme", "metst", "sammlv"): "BoostingVRME→ME-TST+ / SAMMLV",
        ("boostingvrme", "metst", "casme3"): "BoostingVRME→ME-TST+ / CAS(ME)3",
    }
    lines = [
        "# Transferred GLSD vs Target Fair Tuned Native",
        "",
        "## 结论",
        "",
        "四组比较均严格按 outer subject 配对重采样。M→B/S 与 M→B/C 的点估计为正，"
        "但是否具有统计支持以如下 95% CI 与 P(Δ>0) 为准。‘competitive’只能解释为点估计接近且"
        "配对区间未显示稳定劣势，不能替代显著性结论。",
        "",
        "## 主结果",
        "",
        "| Setting | Transferred GLSD F1 | Fair Tuned Native F1 | Δ | 95% CI | P(Δ>0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["source"], row["target"], row["dataset"]): row for row in bootstraps}
    for row in summary:
        key = (row["source"], row["target"], row["dataset"])
        boot = by_key[key]
        lines.append(
            f"| {labels[key]} | {row['transferred_glsd_f1']:.6f} | "
            f"{row['fair_tuned_native_f1']:.6f} | {row['delta']:+.6f} | "
            f"[{boot['ci95_low']:+.6f}, {boot['ci95_high']:+.6f}] | "
            f"{boot['p_delta_gt_0']:.4f} |"
        )

    significant = [
        labels[(row["source"], row["target"], row["dataset"])]
        for row in bootstraps
        if row["ci95_low"] > 0.0
    ]
    ms = by_key[("metst", "boostingvrme", "sammlv")]
    mc = by_key[("metst", "boostingvrme", "casme3")]
    lines.extend(
        [
            "",
            "## Reviewer-facing 判断",
            "",
            f"- **M→B/S 的 +0.004365：**95% CI "
            f"[{ms['ci95_low']:+.6f}, {ms['ci95_high']:+.6f}]，"
            + ("不跨 0，具有统计支持。" if ms["ci95_low"] > 0 else "跨 0，不具有常用 95% CI 标准下的统计支持。"),
            f"- **M→B/C 的 +0.007684：**95% CI "
            f"[{mc['ci95_low']:+.6f}, {mc['ci95_high']:+.6f}]，"
            + ("不跨 0，具有统计支持。" if mc["ci95_low"] > 0 else "跨 0，不具有常用 95% CI 标准下的统计支持。"),
            "- **可否称为 competitive：**可以谨慎表述为 ‘cross-backbone transferred GLSD is competitive with a target-specifically tuned Native decoder’，"
            "但必须同时展示四组原始 Δ 与 CI；该表述表示整体量级可比，不表示四组均非劣或显著更优。",
            "- **显著优于 Tuned Native：**"
            + ("、".join(significant) + " 的配对 95% CI 全部高于 0。" if significant else "没有 setting 的配对 95% CI 全部高于 0，因此不能声称显著优于。"),
            "",
            "## 配对与协议审计",
            "",
            "每组均通过：outer-subject 集合完全一致、逐视频 identity 完全一致、Fair 文件中的 GT 与目标 backbone cache 中 GT 完全一致、"
            "cache SHA 与 Fair PROTOCOL 一致、两种预测均删除已有 matched_gt 后用同一 evaluator 独立重算。",
            "",
        ]
    )
    for audit in alignments:
        key = (audit["source"], audit["target"], audit["dataset"])
        lines.append(
            f"- {labels[key]}：{audit['subjects']} subjects / {audit['videos']} videos / "
            f"{audit['ground_truth_events']} GT；cache `{audit['cache_sha256']}`。"
        )
    lines.extend(
        [
            "",
            "## 统计定义",
            "",
            "使用 NumPy `default_rng(100)`，每组独立生成 10,000 个 N×N outer-subject 索引；同一索引同时用于"
            "Transferred GLSD 和 Fair Tuned Native。每次先累计抽中被试的 TP/FP/FN，再计算 pooled Raw Spotting F1，"
            "最后形成 Δ=F1(Transferred GLSD)−F1(Fair Tuned Native)。CI 为 percentile 2.5%–97.5%，"
            "P(Δ>0) 为 10,000 个配对重采样中 Δ>0 的比例。",
            "",
            "## 边界",
            "",
            "本分析复用已有锁定逐视频预测，不重新选择 transferred GLSD 或 Fair Native 配置，不修改 decoder，"
            "也不重新训练 backbone。subject bootstrap 反映在当前 frozen-cache/outer-LOSO 结果条件下的被试变异。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    transfer_path = TRANSFER_ROOT / "per_video_transferred_predictions.json"
    transfer_rows = read_json(transfer_path)
    summaries: list[dict] = []
    bootstraps: list[dict] = []
    outer_rows: list[dict] = []
    alignments: list[dict] = []

    for source, target, dataset in SETTINGS:
        transferred, fair, bootstrap, subjects, alignment = compare_setting(
            source, target, dataset, transfer_rows
        )
        summaries.append(
            {
                "source": source,
                "target": target,
                "dataset": dataset,
                "transferred_glsd_tp": transferred["TP"],
                "transferred_glsd_fp": transferred["FP"],
                "transferred_glsd_fn": transferred["FN"],
                "transferred_glsd_f1": transferred["F1"],
                "fair_tuned_native_tp": fair["TP"],
                "fair_tuned_native_fp": fair["FP"],
                "fair_tuned_native_fn": fair["FN"],
                "fair_tuned_native_f1": fair["F1"],
                "delta": transferred["F1"] - fair["F1"],
            }
        )
        bootstraps.append({"source": source, "target": target, "dataset": dataset, **bootstrap})
        outer_rows.extend(subjects)
        alignments.append(alignment)

    write_csv(OUTPUT_ROOT / "summary.csv", summaries)
    write_csv(OUTPUT_ROOT / "outer_subject_deltas.csv", outer_rows)
    write_json(
        OUTPUT_ROOT / "bootstrap.json",
        {
            "protocol": {
                "resamples": RESAMPLES,
                "seed": SEED,
                "paired_unit": "outer subject",
                "aggregation": "sum sampled TP/FP/FN then compute pooled Raw Spotting F1",
                "ci": "percentile 2.5% and 97.5%",
                "p_delta_gt_0": "fraction of paired bootstrap deltas strictly greater than zero",
            },
            "input_artifacts": {
                "transferred_predictions": str(transfer_path.resolve()),
                "transferred_predictions_sha256": sha256(transfer_path),
                "fair_tuned_native_root": str(FAIR_ROOT.resolve()),
            },
            "alignment_audit": alignments,
            "results": bootstraps,
        },
    )
    (OUTPUT_ROOT / "TRANSFER_VS_FAIR_NATIVE_REPORT_CN.md").write_text(
        render_report(summaries, bootstraps, alignments), encoding="utf-8"
    )
    print(json.dumps({"summary": summaries, "bootstrap": bootstraps}, indent=2))


if __name__ == "__main__":
    main()
