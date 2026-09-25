"""Range audit and candidate-level alignment-reliability diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from my_method.gl_saliency_skill.benchmark import GROUPS, fold_priors, load_bundle
from my_method.gl_saliency_skill.evaluation import overlap_matrix
from my_method.gl_saliency_skill.evidence import LOCAL_RADII, REFERENCE_SCALES
from my_method.glsd_v2_reliability.reliability import (
    evidence_and_reliability,
    v2_score,
)


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT / "results" / "glsd_v2_reliability"
CONTROLLED_REFERENCE = 2.0
CONTROLLED_RADIUS = 1.0
EPSILON = 1e-12
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


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


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "std": None, "quantiles": {}}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "zero_fraction": float(np.mean(array == 0.0)),
        "one_fraction": float(np.mean(array == 1.0)),
        "quantiles": {
            f"q{int(round(q * 100)):02d}": float(np.quantile(array, q))
            for q in QUANTILES
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    range_rows = []
    diagnostic_rows = []
    distributions = []
    global_min = {"G": float("inf"), "L": float("inf"), "A": float("inf"), "S_v2": float("inf")}
    global_max = {"G": float("-inf"), "L": float("-inf"), "A": float("-inf"), "S_v2": float("-inf")}

    for backbone, dataset in GROUPS:
        bundle = load_bundle(backbone, dataset)
        outer, _ = fold_priors(bundle)
        subject_index = {subject: index for index, subject in enumerate(bundle.subjects)}
        controlled_values = {"all": [], "positive": [], "negative": []}
        group_range = {key: [] for key in global_min}

        for record in bundle.records:
            k = int(outer[subject_index[record.subject]])
            for reference in REFERENCE_SCALES:
                for radius in LOCAL_RADII:
                    _, evidence, reliability = evidence_and_reliability(
                        record.score, k, reference, radius
                    )
                    scores = v2_score(evidence, reliability)
                    arrays = {
                        "G": evidence[:, 1],
                        "L": evidence[:, 2],
                        "A": reliability,
                        "S_v2": scores,
                    }
                    for name, values in arrays.items():
                        if len(values):
                            group_range[name].extend(map(float, values))

            peaks, evidence, reliability = evidence_and_reliability(
                record.score, k, CONTROLLED_REFERENCE, CONTROLLED_RADIUS
            )
            geometry = bundle.interval_adapter.geometry(record.score, k, peaks)
            ordered_reliability = reliability[geometry.order]
            overlaps = overlap_matrix(geometry.intervals, record.ground_truth)
            positive = (
                np.max(overlaps, axis=1) >= 0.5
                if len(record.ground_truth) and len(geometry.intervals)
                else np.zeros(len(geometry.intervals), dtype=bool)
            )
            for index, value in enumerate(ordered_reliability):
                label = "positive" if positive[index] else "negative"
                controlled_values["all"].append(float(value))
                controlled_values[label].append(float(value))
                diagnostic_rows.append(
                    {
                        "backbone": backbone,
                        "dataset": dataset,
                        "subject": record.subject,
                        "video": record.video,
                        "peak": int(peaks[geometry.order][index]),
                        "onset": int(geometry.intervals[index, 0]),
                        "offset": int(geometry.intervals[index, 1]),
                        "A": float(value),
                        "posthoc_candidate_label": label,
                    }
                )

        for name, values in group_range.items():
            array = np.asarray(values, dtype=float)
            if not len(array):
                raise AssertionError(f"No {name} values for {backbone}/{dataset}")
            minimum, maximum = float(array.min()), float(array.max())
            global_min[name] = min(global_min[name], minimum)
            global_max[name] = max(global_max[name], maximum)
            range_rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "quantity": name,
                    "count": len(array),
                    "minimum": minimum,
                    "maximum": maximum,
                    "below_zero": int(np.sum(array < -EPSILON)),
                    "above_one": int(np.sum(array > 1.0 + EPSILON)),
                }
            )
        distributions.append(
            {
                "backbone": backbone,
                "dataset": dataset,
                "scope": "controlled a0=2.0, rho=1.0 candidate pool",
                "all": describe(controlled_values["all"]),
                "positive": describe(controlled_values["positive"]),
                "negative": describe(controlled_values["negative"]),
            }
        )

    valid = all(
        global_min[name] >= -EPSILON and global_max[name] <= 1.0 + EPSILON
        for name in global_min
    )
    write_csv(OUTPUT_ROOT / "range_values.csv", range_rows)
    write_csv(OUTPUT_ROOT / "alignment_reliability_candidates.csv", diagnostic_rows)
    write_json(
        OUTPUT_ROOT / "alignment_reliability_distribution.json",
        {"formula_uses_gt": False, "posthoc_label_uses_gt": True, "groups": distributions},
    )

    range_lines = [
        "# GLSD-v2 Range Audit",
        "",
        "本审计在四组 frozen cache、所有 outer-subject training-only `k_out(s)`、"
        "全部历史 reference scales `{1,1.5,2}` 与 radii `{1,2,3}` 上枚举 v1 的真实 candidate evidence。",
        "没有对 G、L、A 或 S_v2 做 clipping。",
        "",
        "| Quantity | Global observed minimum | Global observed maximum |",
        "|---|---:|---:|",
    ]
    for name in ("G", "L", "A", "S_v2"):
        range_lines.append(f"| {name} | {global_min[name]:.12g} | {global_max[name]:.12g} |")
    range_lines.extend(
        [
            "",
            "理论上，v1 的 `G=prominence/ptp(smoothed)` 与非负 local relief `L/ptp(smoothed)` "
            "均位于 `[0,1]`；定义直接保证 `A∈[0,1]`。因此 `A*L∈[0,1]`，"
            "`S_v2=1-(1-G)(1-A*L)∈[0,1]`。实际枚举与该范围一致。"
            if valid
            else "至少一个量超出 `[0,1]`，固定公式的概率并集解释不再成立，实验必须停止。",
            "",
            f"Final range gate: `{'PASS' if valid else 'FAIL_STOP'}`",
        ]
    )
    (OUTPUT_ROOT / "RANGE_AUDIT.md").write_text("\n".join(range_lines) + "\n", encoding="utf-8")

    diagnostic_lines = [
        "# Alignment Reliability Diagnostic",
        "",
        "A(c) 完全由已有多尺度峰、`delta=max(1,round(0.5*k))` 和 v1 的确定性 aligned-peak "
        "选择得到，不读取 GT。下表使用 controlled candidate pool (`a0=2.0`, `rho=1.0`)。"
        "positive/negative 仅为计算 A 后的 post-hoc 分析标签：candidate interval 对任一 GT 的 IoU≥0.5 为 positive。",
        "",
        "| Setting / label | N | Mean | Median | Std | Q05 | Q25 | Q75 | Q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in distributions:
        for label in ("all", "positive", "negative"):
            values = group[label]
            q = values["quantiles"]
            diagnostic_lines.append(
                f"| {group['backbone']}/{group['dataset']} / {label} | {values['count']} | "
                f"{values['mean']:.6f} | {values['median']:.6f} | {values['std']:.6f} | "
                f"{q['q05']:.6f} | {q['q25']:.6f} | {q['q75']:.6f} | {q['q95']:.6f} |"
            )
    (OUTPUT_ROOT / "ALIGNMENT_RELIABILITY_DIAGNOSTIC.md").write_text(
        "\n".join(diagnostic_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"range_valid": valid, "minimum": global_min, "maximum": global_max}, indent=2))
    if not valid:
        raise SystemExit("GLSD-v2 range gate failed; do not run controlled study")


if __name__ == "__main__":
    main()
