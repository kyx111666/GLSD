#!/usr/bin/env python3
"""Paired nested Native tuning: original vs uniform overlap mean on SAMMLV."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_SOURCE = HERE / "run_overlap_aggregation_feasibility.py"
DEFAULT_RAW = Path(
    os.environ.get("SAMMLV_RAW_ROOT", "data/raw_prestitch_sammlv_full")
)
DEFAULT_OUT = ROOT / "results/paired_nested_overlap_aggregation_sammlv"
HISTORICAL_REPORT = ROOT / "results/rgr1_sammlv_nested/report.json"
EXPECTED_B0 = (49, 143, 110)
EXPECTED_SUBJECTS = 29
EXPECTED_VIDEOS = 79
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260905
TUNED_CS = (1.5, 2.0, 2.5)
TUNED_P = (0.45, 0.55, 0.65)
TUNED_CD = (0.75, 1.0, 1.25)
TUNED_CB = (0.75, 1.0, 1.25)
METHODS = ("B0 Original Nested", "B1 Uniform Mean Nested")


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = import_module("paired_nested_overlap_base", BASE_SOURCE)
native = base.native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grid() -> list[dict]:
    configs = [
        dict(zip(("c_s", "p", "c_d", "c_b"), values))
        for values in itertools.product(TUNED_CS, TUNED_P, TUNED_CD, TUNED_CB)
    ]
    if len(configs) != 81:
        raise RuntimeError("locked Native grid cardinality changed")
    return configs


def config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def counts_from_metrics(row: dict) -> dict:
    return {key: int(row[key]) for key in ("TP", "FP", "FN", "event_count")}


def sum_counts(rows) -> dict:
    total = native.counts_empty()
    for row in rows:
        native.add_counts(total, counts_from_metrics(row))
    return total


def exact_f1(counts: dict) -> Fraction:
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return Fraction(2 * counts["TP"], denominator) if denominator else Fraction(0)


def make_records(gated: dict) -> tuple[list[dict], list[dict]]:
    original_records = []
    uniform_records = []
    for identity, item in gated["loaded"].items():
        original = dict(item["record"])
        original["score"] = item["original"]
        uniform, _center, _overlap = base.aggregate_raw(
            {
                "raw_window_score": item["raw"],
                "global_indices": item["global_indices"],
                "original_fresh_result_video": item["original"],
            }
        )
        mean_record = dict(item["record"])
        mean_record["score"] = uniform
        if (str(original["subject"]), str(original["video"])) != identity:
            raise RuntimeError(f"record identity mismatch: {identity}")
        original_records.append(original)
        uniform_records.append(mean_record)
    identity = lambda row: (str(row["subject"]), str(row["video"]))
    original_records.sort(key=identity)
    uniform_records.sort(key=identity)
    if [identity(row) for row in original_records] != [identity(row) for row in uniform_records]:
        raise RuntimeError("paired record order mismatch")
    if len(original_records) != EXPECTED_VIDEOS:
        raise RuntimeError("paired record cardinality mismatch")
    return original_records, uniform_records


def precompute(records: list[dict], configs: list[dict], subjects: list[str]):
    bank = []
    for config in configs:
        per_subject = {subject: native.counts_empty() for subject in subjects}
        for record in records:
            subject = str(record["subject"])
            events = native.tuned_decode(record, config, base.K_P)
            native.add_counts(per_subject[subject], native.evaluate(record, events))
        bank.append(per_subject)
    return bank


def nested_select(method: str, subjects: list[str], configs: list[dict], bank):
    outer_rows = []
    selected_rows = []
    inner_rows = []
    selected_counts = {}
    for held in subjects:
        ranked = []
        for index, config in enumerate(configs):
            train = native.counts_empty()
            for subject in subjects:
                if subject != held:
                    native.add_counts(train, bank[index][subject])
            rank = (-exact_f1(train), train["FP"], 0, index)
            metrics = native.metrics(train)
            ranked.append((rank, index, config, metrics))
            inner_rows.append(
                {
                    "method": method,
                    "outer_subject": held,
                    "config_index": index,
                    **config,
                    "inner_TP": metrics["TP"],
                    "inner_FP": metrics["FP"],
                    "inner_FN": metrics["FN"],
                    "inner_F1": metrics["F1"],
                }
            )
        _rank, index, config, inner = min(ranked, key=lambda row: row[0])
        selected_counts[held] = bank[index][held]
        outer = native.metrics(bank[index][held])
        outer_rows.append(
            {
                "method": method,
                "outer_subject": held,
                **outer,
                "selected_config_index": index,
                **config,
                "inner_TP": inner["TP"],
                "inner_FP": inner["FP"],
                "inner_FN": inner["FN"],
                "inner_F1": inner["F1"],
            }
        )
        selected_rows.append(
            {
                "method": method,
                "outer_subject": held,
                "selected_config_index": index,
                **config,
                "inner_TP": inner["TP"],
                "inner_FP": inner["FP"],
                "inner_FN": inner["FN"],
                "inner_F1": inner["F1"],
            }
        )
    aggregate = native.metrics(sum_counts(selected_counts.values()))
    return {
        "outer_rows": outer_rows,
        "selected_rows": selected_rows,
        "inner_rows": inner_rows,
        "subject_counts": selected_counts,
        "aggregate": aggregate,
    }


def historical_b0_gate(b0: dict, configs: list[dict]) -> None:
    counts = tuple(b0["aggregate"][key] for key in ("TP", "FP", "FN"))
    if counts != EXPECTED_B0:
        raise RuntimeError(f"B0 aggregate={counts}, expected={EXPECTED_B0}")
    report = json.loads(HISTORICAL_REPORT.read_text(encoding="utf-8"))
    expected = {}
    for fold in report["outer_folds"]:
        source = fold["selected_strong_config"]
        expected[str(fold["subject"])] = {
            "index": int(fold["selected_strong_config_index"]),
            "config": {
                "c_s": float(source["c_s"]),
                "p": float(source["p_s"]),
                "c_d": float(source["c_d"]),
                "c_b": float(source["c_b"]),
            },
        }
    if len(expected) != EXPECTED_SUBJECTS:
        raise RuntimeError("historical B0 fold count mismatch")
    for row in b0["selected_rows"]:
        subject = row["outer_subject"]
        observed_config = {key: float(row[key]) for key in ("c_s", "p", "c_d", "c_b")}
        if subject not in expected:
            raise RuntimeError(f"B0 unknown historical subject: {subject}")
        if row["selected_config_index"] != expected[subject]["index"]:
            raise RuntimeError(f"B0 config index mismatch for {subject}")
        if config_key(observed_config) != config_key(expected[subject]["config"]):
            raise RuntimeError(f"B0 config mismatch for {subject}")
        if config_key(configs[int(row["selected_config_index"])]) != config_key(observed_config):
            raise RuntimeError(f"B0 grid/index mismatch for {subject}")


def comparison_label(candidate: dict, baseline: dict) -> str:
    delta = native.f1(candidate) - native.f1(baseline)
    if delta > 1e-15:
        return "improved"
    if delta < -1e-15:
        return "worse"
    return "equal"


def stability_and_sensitivity(b0_counts: dict, b1_counts: dict, subjects: list[str]):
    labels = Counter(comparison_label(b1_counts[s], b0_counts[s]) for s in subjects)
    deltas = []
    for held in subjects:
        b0_leave = sum_counts(b0_counts[s] for s in subjects if s != held)
        b1_leave = sum_counts(b1_counts[s] for s in subjects if s != held)
        deltas.append(native.f1(b1_leave) - native.f1(b0_leave))
    return (
        {key: int(labels.get(key, 0)) for key in ("improved", "equal", "worse")},
        {"min": float(np.min(deltas)), "median": float(np.median(deltas)), "max": float(np.max(deltas))},
    )


def paired_bootstrap(baseline: dict, candidate: dict, subjects: list[str], label: str) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPEATS):
        indices = rng.integers(0, len(subjects), size=len(subjects))
        b0 = native.counts_empty()
        b1 = native.counts_empty()
        for index in indices:
            subject = subjects[int(index)]
            native.add_counts(b0, baseline[subject])
            native.add_counts(b1, candidate[subject])
        draws.append(native.f1(b1) - native.f1(b0))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    observed_b0 = sum_counts(baseline.values())
    observed_b1 = sum_counts(candidate.values())
    return {
        "comparison": label,
        "unit": "subject",
        "paired": True,
        "repeats": BOOTSTRAP_REPEATS,
        "seed": BOOTSTRAP_SEED,
        "observed_delta_F1": native.f1(observed_b1) - native.f1(observed_b0),
        "bootstrap_mean_delta_F1": float(np.mean(draws)),
        "CI95": [float(lower), float(upper)],
    }


def parameter_frequencies(selected_rows: list[dict]) -> list[dict]:
    values = {"c_s": TUNED_CS, "p": TUNED_P, "c_d": TUNED_CD, "c_b": TUNED_CB}
    output = []
    for method in METHODS:
        method_rows = [row for row in selected_rows if row["method"] == method]
        for parameter, candidates in values.items():
            counts = Counter(float(row[parameter]) for row in method_rows)
            for value in candidates:
                output.append(
                    {
                        "method": method,
                        "parameter": parameter,
                        "value": value,
                        "count": int(counts.get(float(value), 0)),
                        "proportion": counts.get(float(value), 0) / EXPECTED_SUBJECTS,
                    }
                )
    return output


def fixed_aggregation_counts(gated: dict, subjects: list[str]):
    b0 = {subject: native.counts_empty() for subject in subjects}
    b1 = {subject: native.counts_empty() for subject in subjects}
    for (subject, _video), item in gated["loaded"].items():
        config = gated["configs"][subject]
        original_record = dict(item["record"])
        original_record["score"] = item["original"]
        uniform, _center, _overlap = base.aggregate_raw(
            {
                "raw_window_score": item["raw"],
                "global_indices": item["global_indices"],
                "original_fresh_result_video": item["original"],
            }
        )
        mean_record = dict(item["record"])
        mean_record["score"] = uniform
        native.add_counts(b0[subject], native.evaluate(original_record, native.tuned_decode(original_record, config, base.K_P)))
        native.add_counts(b1[subject], native.evaluate(mean_record, native.tuned_decode(mean_record, config, base.K_P)))
    if tuple(sum_counts(b0.values())[key] for key in ("TP", "FP", "FN")) != EXPECTED_B0:
        raise RuntimeError("fixed O0 bootstrap anchor mismatch")
    return b0, b1


def report_markdown(summary: dict) -> str:
    b0 = summary["aggregate"][METHODS[0]]
    b1 = summary["aggregate"][METHODS[1]]
    fixed_boot = summary["bootstrap"]["frozen_decoder_O1_vs_O0"]
    nested_boot = summary["bootstrap"]["paired_nested_B1_vs_B0"]
    stability = summary["subject_stability"]
    sensitivity = summary["leave_one_subject_delta_F1"]
    lines = [
        "# Original vs Uniform Mean — Paired Nested Decoder Tuning (SAMMLV)",
        "",
        "## 协议与 gate",
        "",
        "- Raw pre-stitch source：29 subjects / 79 videos；全部文件通过既有 exact reconstruction 与 SHA gate。",
        "- 两个分支使用相同的冻结 81-config Native grid：`c_s={1.5,2.0,2.5}`, `p={0.45,0.55,0.65}`, `c_d={0.75,1.0,1.25}`, `c_b={0.75,1.0,1.25}`。",
        "- 每个 outer fold 只在其余 28 subjects 的 pooled Spotting F1 上选参。Native decoder 没有需要拟合的权重，因此每个固定 config 在各 inner subject 上各评估一次再汇总，严格等价于 subject-disjoint inner LOSO pooling。",
        "- Tie-break：更高 exact pooled F1、较少 FP、历史固定 Cartesian config index。B0 的 29 个历史 config index 与配置值全部一致。",
        f"- B0 anchor: {b0['TP']}/{b0['FP']}/{b0['FN']}, F1={b0['F1']:.6f}, PASS.",
        "- Uniform Mean 未修改；未使用 recognition、weighting、disagreement、Window-Consensus feature、backbone forward 或 grid expansion。",
        "",
        "## Paired nested outer 结果",
        "",
        "| Method | TP | FP | FN | Precision | Recall | F1 | Events | ΔF1 vs B0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {METHODS[0]} | {b0['TP']} | {b0['FP']} | {b0['FN']} | {b0['Precision']:.6f} | {b0['Recall']:.6f} | {b0['F1']:.6f} | {b0['event_count']} | +0.000000 |",
        f"| {METHODS[1]} | {b1['TP']} | {b1['FP']} | {b1['FN']} | {b1['Precision']:.6f} | {b1['Recall']:.6f} | {b1['F1']:.6f} | {b1['event_count']} | {b1['F1']-b0['F1']:+.6f} |",
        "",
        "## 参数选择频率",
        "",
    ]
    for method in METHODS:
        lines.append(f"### {method}")
        lines.append("")
        for parameter, frequencies in summary["selected_parameter_frequencies"][method].items():
            lines.append(f"- `{parameter}`: `{json.dumps(frequencies, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    lines += [
        "## Subject stability",
        "",
        f"- B1 vs B0 improved/equal/worse: {stability['improved']}/{stability['equal']}/{stability['worse']}.",
        f"- Leave-one-subject aggregate ΔF1: min={sensitivity['min']:+.6f}, median={sensitivity['median']:+.6f}, max={sensitivity['max']:+.6f}.",
        "",
        "## 1000 subject bootstrap",
        "",
        "| Comparison | Observed ΔF1 | Bootstrap mean | 95% CI |",
        "|---|---:|---:|---:|",
        f"| Frozen decoder O1 vs O0 | {fixed_boot['observed_delta_F1']:+.6f} | {fixed_boot['bootstrap_mean_delta_F1']:+.6f} | [{fixed_boot['CI95'][0]:+.6f}, {fixed_boot['CI95'][1]:+.6f}] |",
        f"| Paired nested B1 vs B0 | {nested_boot['observed_delta_F1']:+.6f} | {nested_boot['bootstrap_mean_delta_F1']:+.6f} | [{nested_boot['CI95'][0]:+.6f}, {nested_boot['CI95'][1]:+.6f}] |",
        "",
        "## 解释边界",
        "",
        "Paired nested 结果只检验 Uniform Mean 与 Original 获得相同冻结 decoder search budget 后是否仍有增益；不涉及 weighting、disagreement、recognition 或 Window-Consensus features。",
        "",
        "实验到此停止；未设计或启动下一阶段方法。",
    ]
    return "\n".join(lines) + "\n"


def blocked_report(output_root: Path, reason: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "PAIRED_NESTED_OVERLAP_AGGREGATION_SAMMLV_CN.md").write_text(
        "# Paired Nested Decoder Tuning — SAMMLV\n\n"
        "`BLOCKED-PAIRED-NESTED-B0-ANCHOR`\n\n"
        f"{reason}\n\nB1 was not run.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    configs = grid()
    try:
        gated = base.provenance_gate(argparse.Namespace(raw_dir=args.raw_dir, output_root=args.output_root))
        subjects = list(gated["subjects"])
        if len(subjects) != EXPECTED_SUBJECTS:
            raise RuntimeError("subject cardinality mismatch")
        original_records, uniform_records = make_records(gated)
        b0_bank = precompute(original_records, configs, subjects)
        b0 = nested_select(METHODS[0], subjects, configs, b0_bank)
        historical_b0_gate(b0, configs)
    except Exception as error:
        blocked_report(args.output_root, f"{type(error).__name__}: {error}")
        print("BLOCKED-PAIRED-NESTED-B0-ANCHOR")
        print(f"{type(error).__name__}: {error}")
        return 2

    # B1 is deliberately deferred until every B0 provenance/anchor/fold check passes.
    b1_bank = precompute(uniform_records, configs, subjects)
    b1 = nested_select(METHODS[1], subjects, configs, b1_bank)
    stability, sensitivity = stability_and_sensitivity(
        b0["subject_counts"], b1["subject_counts"], subjects
    )
    fixed_b0, fixed_b1 = fixed_aggregation_counts(gated, subjects)
    bootstraps = {
        "frozen_decoder_O1_vs_O0": paired_bootstrap(
            fixed_b0, fixed_b1, subjects, "O1 Uniform Mean vs O0 Original, frozen fold-wise decoder"
        ),
        "paired_nested_B1_vs_B0": paired_bootstrap(
            b0["subject_counts"], b1["subject_counts"], subjects, "B1 Uniform Mean Nested vs B0 Original Nested"
        ),
    }
    aggregate = {METHODS[0]: b0["aggregate"], METHODS[1]: b1["aggregate"]}
    selected_rows = b0["selected_rows"] + b1["selected_rows"]
    frequency_rows = parameter_frequencies(selected_rows)
    frequency_json = {}
    for method in METHODS:
        frequency_json[method] = {}
        for parameter in ("c_s", "p", "c_d", "c_b"):
            frequency_json[method][parameter] = dict(
                sorted(
                    Counter(
                        str(row[parameter])
                        for row in selected_rows
                        if row["method"] == method
                    ).items()
                )
            )
    summary_rows = []
    baseline = aggregate[METHODS[0]]
    for method in METHODS:
        current = aggregate[method]
        summary_rows.append(
            {
                "method": method,
                **current,
                "Delta_TP_vs_B0": current["TP"] - baseline["TP"],
                "Delta_FP_vs_B0": current["FP"] - baseline["FP"],
                "Delta_FN_vs_B0": current["FN"] - baseline["FN"],
                "Delta_F1_vs_B0": current["F1"] - baseline["F1"],
            }
        )
    outer_rows = b0["outer_rows"] + b1["outer_rows"]
    subject_lookup = {
        (row["method"], row["outer_subject"]): row for row in outer_rows
    }
    for subject in subjects:
        b0_row = subject_lookup[(METHODS[0], subject)]
        b1_row = subject_lookup[(METHODS[1], subject)]
        b1_row["Delta_F1_vs_B0"] = b1_row["F1"] - b0_row["F1"]
        b1_row["comparison_vs_B0"] = comparison_label(
            b1["subject_counts"][subject], b0["subject_counts"][subject]
        )
        b0_row["Delta_F1_vs_B0"] = 0.0
        b0_row["comparison_vs_B0"] = "baseline"
    summary = {
        "experiment": "paired_nested_original_vs_uniform_mean_sammlv",
        "provenance": {
            "raw_manifest": str(gated["manifest_path"]),
            "raw_manifest_sha256": gated["manifest_sha"],
            "compact_cache_sha256": base.EXPECTED_CACHE_SHA,
            "historical_config_report": str(HISTORICAL_REPORT),
            "historical_config_report_sha256": base.sha256(HISTORICAL_REPORT),
            "subjects": len(subjects),
            "videos": len(original_records),
            "GT": sum(len(row["samples"]) for row in original_records),
            "k_p": base.K_P,
        },
        "protocol": {
            "outer": "29-fold subject LOSO",
            "selection": "outer-train 28 subjects pooled Spotting F1",
            "subject_disjoint_inner_equivalence": "fixed decoder configs have no fitted state; each config is evaluated once per train subject and pooled",
            "tie_break": ["higher exact Fraction F1", "fewer FP", "fixed Cartesian config index"],
            "grid_count_per_method": len(configs),
            "same_search_budget": True,
            "outer_test_GT_used_for_selection": False,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "grid": {"c_s": TUNED_CS, "p": TUNED_P, "c_d": TUNED_CD, "c_b": TUNED_CB},
        "gates": {
            "raw_reconstruction_exact": True,
            "B0_anchor_exact": True,
            "B0_all_29_historical_config_indices_exact": True,
            "B0_all_29_historical_configs_exact": True,
        },
        "aggregate": aggregate,
        "Delta_F1_B1_minus_B0": aggregate[METHODS[1]]["F1"] - aggregate[METHODS[0]]["F1"],
        "selected_parameter_frequencies": frequency_json,
        "subject_stability": stability,
        "leave_one_subject_delta_F1": sensitivity,
        "bootstrap": bootstraps,
        "integrity": {
            "uniform_mean_modified": False,
            "native_grid_expanded": False,
            "aggregation_parameter_added": False,
            "recognition_used": False,
            "window_consensus_feature_used": False,
            "weighting_used": False,
            "backbone_forward": False,
        },
    }
    output_dir = args.output_root / "outputs"
    write_csv(output_dir / "paired_nested_summary.csv", summary_rows)
    write_csv(output_dir / "paired_nested_outer_metrics.csv", outer_rows)
    write_csv(output_dir / "paired_nested_selected_configs.csv", selected_rows)
    write_csv(output_dir / "paired_nested_parameter_frequency.csv", frequency_rows)
    write_csv(output_dir / "paired_nested_inner_scores.csv", b0["inner_rows"] + b1["inner_rows"])
    (output_dir / "paired_nested_bootstrap.json").write_text(
        json.dumps(bootstraps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "paired_nested_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "PAIRED_NESTED_OVERLAP_AGGREGATION_SAMMLV_CN.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"B0": aggregate[METHODS[0]], "B1": aggregate[METHODS[1]], "Delta_F1": summary["Delta_F1_B1_minus_B0"], "bootstrap": bootstraps}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
