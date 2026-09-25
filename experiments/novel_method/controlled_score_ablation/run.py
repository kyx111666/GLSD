"""Controlled H/G/L/GL score ablation on one fixed candidate pool and geometry."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from my_method.gl_saliency_skill.benchmark import GROUPS, load_bundle, signed_context
from my_method.gl_saliency_skill.evaluation import evaluate, metrics
from my_method.gl_saliency_skill.evidence import (
    Config,
    THRESHOLDS,
    configuration_grid,
)
from my_method.gl_saliency_skill.selection import choose, inner_counts
from my_method.gl_saliency_skill.skill import GLSaliencySkill


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT / "results" / "controlled_score_ablation"
REFERENCE = 2.0
RADIUS = 1.0
METHODS = ("H", "G", "L", "GL")
FAMILY = {"H": "height", "G": "global", "L": "local", "GL": "unified"}
SEED = 100
RESAMPLES = 10_000
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


def pools(configs: list[Config]) -> dict[str, list[int]]:
    result = {}
    for method in METHODS:
        family = FAMILY[method]
        indexes = [
            index
            for index, config in enumerate(configs)
            if config.family == family
            and config.reference == REFERENCE
            and (family not in ("local", "unified") or config.radius == RADIUS)
            and (family != "unified" or config.height_weight == 0.0)
        ]
        if len(indexes) != len(THRESHOLDS):
            raise AssertionError(f"{method} does not have exactly the common threshold grid")
        if tuple(configs[index].threshold for index in indexes) != THRESHOLDS:
            raise AssertionError(f"{method} threshold order differs from the frozen grid")
        result[method] = indexes
    return result


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
        "crosses_zero": bool(low <= 0.0 <= high),
        "p_delta_gt_0": float(np.mean(deltas > 0.0)),
        "resamples": RESAMPLES,
        "seed": SEED,
        "paired_unit": "outer subject",
    }


def candidate_signature(audits) -> list[tuple[int, int, int]]:
    return [(item.peak, item.onset, item.offset) for item in audits]


def run_group(backbone: str, dataset: str, configs: list[Config], method_pools: dict):
    bundle = load_bundle(backbone, dataset)
    signed_root, protocol, outer, inner, stats, mode = signed_context(bundle, configs)
    subjects = bundle.subjects
    selected: dict[str, dict[str, int]] = {}
    selection_rows = []
    subject_counts = {
        method: np.zeros((len(subjects), 3), dtype=np.int64) for method in METHODS
    }

    for held, subject in enumerate(subjects):
        training = inner_counts(stats, held, inner)
        selected[subject] = {}
        for method in METHODS:
            config_id = choose(training, method_pools[method])
            selected[subject][method] = config_id
            inner_metric = metrics(dict(zip(COUNT_KEYS, training[config_id])))
            outer_values = stats[int(outer[held])][config_id, held].astype(np.int64)
            subject_counts[method][held] = outer_values
            outer_metric = metrics(dict(zip(COUNT_KEYS, outer_values)))
            config = configs[config_id]
            selection_rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "outer_subject": subject,
                    "method": method,
                    "config_id": config_id,
                    "tau": config.threshold,
                    "reference": config.reference,
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

    replay = {method: np.zeros_like(subject_counts[method]) for method in METHODS}
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    candidate_audit_rows = []
    prediction_rows = []
    for record in bundle.records:
        sid = subject_index[record.subject]
        k = int(outer[sid])
        signatures = {}
        for method in METHODS:
            config = configs[selected[record.subject][method]]
            events, audits = GLSaliencySkill(config).decode_with_audit(
                record.score,
                k,
                {"interval_adapter": bundle.interval_adapter},
            )
            counts, details = evaluate(events, record.ground_truth)
            replay[method][sid] += np.asarray([counts[key] for key in COUNT_KEYS], dtype=np.int64)
            signatures[method] = candidate_signature(audits)
            prediction_rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "subject": record.subject,
                    "video": record.video,
                    "method": method,
                    "tau": config.threshold,
                    "k_out": k,
                    "counts": counts,
                    "predictions": details,
                }
            )
        reference_signature = signatures["H"]
        candidate_equal = all(signatures[method] == reference_signature for method in METHODS)
        if not candidate_equal:
            raise AssertionError(
                f"Controlled candidate/geometry mismatch: {backbone}/{dataset}/"
                f"{record.subject}/{record.video}"
            )
        candidate_audit_rows.append(
            {
                "backbone": backbone,
                "dataset": dataset,
                "subject": record.subject,
                "video": record.video,
                "k_out": k,
                "adapter": bundle.interval_adapter.name,
                "candidate_count": len(reference_signature),
                "H_G_L_GL_candidate_and_geometry_exact": True,
            }
        )

    for method in METHODS:
        if not np.array_equal(replay[method], subject_counts[method]):
            raise AssertionError(f"Signed stats and fresh {method} replay differ")

    aggregate_metrics = {method: aggregate(subject_counts[method]) for method in METHODS}
    bootstraps = []
    for comparator in ("H", "G", "L"):
        bootstraps.append(
            {
                "backbone": backbone,
                "dataset": dataset,
                "comparison": f"GL-{comparator}",
                **paired_bootstrap(subject_counts["GL"], subject_counts[comparator]),
            }
        )
    return {
        "backbone": backbone,
        "dataset": dataset,
        "subjects": len(subjects),
        "videos": len(bundle.records),
        "cache_path": bundle.cache_path,
        "signed_root": str(signed_root.resolve()),
        "protocol_signature": protocol["signature"],
        "interval_mode": mode,
        "metrics": aggregate_metrics,
        "bootstraps": bootstraps,
        "selection_rows": selection_rows,
        "candidate_audit_rows": candidate_audit_rows,
        "prediction_rows": prediction_rows,
    }


def render_report(runs: list[dict]) -> str:
    labels = {
        ("metst", "sammlv"): "ME-TST+ / SAMMLV",
        ("metst", "casme3"): "ME-TST+ / CAS(ME)3",
        ("boostingvrme", "sammlv"): "BoostingVRME / SAMMLV",
        ("boostingvrme", "casme3"): "BoostingVRME / CAS(ME)3",
    }
    lines = [
        "# Controlled Candidate-Score Ablation",
        "",
        "## 协议结论",
        "",
        "本实验固定 `a0=2.0`、`rho=1.0`。H/G/L/GL 对每个视频使用相同 response、"
        "reference candidates、candidate order、target adapter geometry、duration prior、冲突逻辑和 evaluator。"
        "四种 score 仅允许在共同的 10 个历史 tau 上，使用 inner training subjects 独立选择工作点。",
        "",
        "## Pooled Raw Spotting F1",
        "",
        "| Setting | H | G | L | GL | GL-H | GL-G | GL-L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        m = run["metrics"]
        lines.append(
            f"| {labels[(run['backbone'], run['dataset'])]} | {m['H']['F1']:.6f} | "
            f"{m['G']['F1']:.6f} | {m['L']['F1']:.6f} | {m['GL']['F1']:.6f} | "
            f"{m['GL']['F1']-m['H']['F1']:+.6f} | {m['GL']['F1']-m['G']['F1']:+.6f} | "
            f"{m['GL']['F1']-m['L']['F1']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Subject-paired bootstrap",
            "",
            "| Setting | Comparison | Point Δ | 95% CI | P(Δ>0) | CI crosses 0 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        label = labels[(run["backbone"], run["dataset"])]
        for row in run["bootstraps"]:
            lines.append(
                f"| {label} | {row['comparison']} | {row['point_delta_F1']:+.6f} | "
                f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | "
                f"{row['p_delta_gt_0']:.4f} | {'yes' if row['crosses_zero'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Equality gate",
            "",
            "四组所有视频均逐一验证 H/G/L/GL 的 `(peak,onset,offset)` candidate signature 完全一致；"
            "fresh prediction recount 与 signed `stats_k*.npz` 的每个 outer-subject TP/FP/FN 完全一致。"
            "运行还复用了 `signed_context` 对历史配置 grid、cache SHA、fold prior、source SHA 和 stats signature 的 gate。",
            "",
            "## 解释边界",
            "",
            "该实验回答的是固定 candidate generation 和 geometry 后，不同 evidence score 的条件效果。"
            "它不替代论文主实验中允许完整 GLSD configuration selection 的结果。CI 跨 0 只表示当前样本下"
            "未获得方向稳定的 95% bootstrap 支持，不构成等效或非劣证明。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    protocol_path = PROJECT / "CONTROLLED_SCORE_PROTOCOL.md"
    if not protocol_path.exists():
        raise AssertionError("Controlled protocol must be frozen before execution")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = configuration_grid()
    method_pools = pools(configs)
    runs = [run_group(backbone, dataset, configs, method_pools) for backbone, dataset in GROUPS]

    summary_rows = []
    bootstrap_rows = []
    for run in runs:
        row = {"backbone": run["backbone"], "dataset": run["dataset"]}
        for method in METHODS:
            values = run["metrics"][method]
            row.update(
                {
                    f"{method}_tp": values["TP"],
                    f"{method}_fp": values["FP"],
                    f"{method}_fn": values["FN"],
                    f"{method}_f1": values["F1"],
                }
            )
        row.update(
            {
                "GL_minus_H": row["GL_f1"] - row["H_f1"],
                "GL_minus_G": row["GL_f1"] - row["G_f1"],
                "GL_minus_L": row["GL_f1"] - row["L_f1"],
            }
        )
        summary_rows.append(row)
        bootstrap_rows.extend(run["bootstraps"])

    write_csv(OUTPUT_ROOT / "summary.csv", summary_rows)
    write_csv(
        OUTPUT_ROOT / "outer_selections.csv",
        [row for run in runs for row in run["selection_rows"]],
    )
    write_csv(
        OUTPUT_ROOT / "candidate_geometry_equivalence.csv",
        [row for run in runs for row in run["candidate_audit_rows"]],
    )
    write_json(OUTPUT_ROOT / "per_video_predictions.json", [row for run in runs for row in run["prediction_rows"]])
    write_json(
        OUTPUT_ROOT / "bootstrap.json",
        {
            "protocol": {
                "resamples": RESAMPLES,
                "seed": SEED,
                "paired_unit": "outer subject",
                "reference": REFERENCE,
                "rho": RADIUS,
                "thresholds": list(THRESHOLDS),
            },
            "results": bootstrap_rows,
        },
    )
    write_json(
        OUTPUT_ROOT / "combined_report.json",
        {
            "status": "CONTROLLED_SCORE_ABLATION_COMPLETE",
            "protocol_path": str(protocol_path.resolve()),
            "candidate_geometry_exact_all_videos": True,
            "signed_stats_fresh_replay_exact": True,
            "groups": [
                {
                    key: run[key]
                    for key in (
                        "backbone",
                        "dataset",
                        "subjects",
                        "videos",
                        "cache_path",
                        "signed_root",
                        "protocol_signature",
                        "interval_mode",
                        "metrics",
                        "bootstraps",
                    )
                }
                for run in runs
            ],
        },
    )
    (OUTPUT_ROOT / "CONTROLLED_SCORE_ABLATION_REPORT_CN.md").write_text(
        render_report(runs), encoding="utf-8"
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
