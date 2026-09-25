"""Compare locked transferred GLSD with locked transferred Native predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from my_method.gl_saliency_skill.evaluation import evaluate, metrics


PROJECT = Path(__file__).resolve().parents[2]
GL_PATH = PROJECT / "results" / "final_gl_skill_transfer" / "per_video_transferred_predictions.json"
NATIVE_PATH = PROJECT / "results" / "native_config_transfer" / "per_video_transferred_predictions.json"
OUTPUT_ROOT = PROJECT / "results" / "transfer_glsd_vs_native"
SETTINGS = (
    ("metst", "boostingvrme", "sammlv"),
    ("metst", "boostingvrme", "casme3"),
    ("boostingvrme", "metst", "sammlv"),
    ("boostingvrme", "metst", "casme3"),
)
COUNT_KEYS = ("TP", "FP", "FN")
SEED = 100
RESAMPLES = 10_000


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


def clean(events: list[dict]) -> list[dict]:
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


def aggregate(values: np.ndarray) -> dict:
    total = values.sum(axis=0)
    return metrics(dict(zip(COUNT_KEYS, total)))


def paired_bootstrap(gl_values: np.ndarray, native_values: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(gl_values), size=(RESAMPLES, len(gl_values)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0], denominator, out=np.zeros(RESAMPLES), where=denominator > 0
        )

    deltas = f1(gl_values) - f1(native_values)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "point_delta_F1": float(aggregate(gl_values)["F1"] - aggregate(native_values)["F1"]),
        "bootstrap_mean_delta_F1": float(deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_delta_gt_0": float(np.mean(deltas > 0.0)),
        "crosses_zero": bool(low <= 0.0 <= high),
        "resamples": RESAMPLES,
        "seed": SEED,
        "paired_unit": "outer subject",
    }


def compare_setting(source: str, target: str, dataset: str, gl_rows: list[dict], native_rows: list[dict]):
    match = lambda row: (row["source"], row["target"], row["dataset"]) == (source, target, dataset)
    gl = [row for row in gl_rows if match(row)]
    native = [row for row in native_rows if match(row)]
    gl_by_video = {(row["subject"], row["video"]): row for row in gl}
    native_by_video = {(row["subject"], row["video"]): row for row in native}
    if len(gl_by_video) != len(gl) or len(native_by_video) != len(native):
        raise AssertionError("Duplicate subject/video keys")
    if set(gl_by_video) != set(native_by_video):
        raise AssertionError("Transferred GLSD and Native video identities differ")

    subjects = sorted({subject for subject, _ in gl_by_video}, key=lambda value: (int(value), value))
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    gl_counts = np.zeros((len(subjects), 3), dtype=np.int64)
    native_counts = np.zeros_like(gl_counts)
    for key in sorted(gl_by_video, key=lambda item: (int(item[0]), item[0], item[1])):
        subject, _ = key
        gl_row = gl_by_video[key]
        native_row = native_by_video[key]
        gt = native_row["gt"]
        gl_result, _ = evaluate(clean(gl_row["predictions"]), gt)
        native_result, _ = evaluate(clean(native_row["predictions"]), gt)
        if gl_result != gl_row["counts"]:
            raise AssertionError("Transferred GLSD stored count mismatch")
        if native_result != native_row["counts"]:
            raise AssertionError("Transferred Native stored count mismatch")
        if gl_result["TP"] + gl_result["FN"] != native_result["TP"] + native_result["FN"]:
            raise AssertionError("Ground-truth count mismatch")
        sid = subject_index[subject]
        gl_counts[sid] += np.asarray([gl_result[key] for key in COUNT_KEYS], dtype=np.int64)
        native_counts[sid] += np.asarray([native_result[key] for key in COUNT_KEYS], dtype=np.int64)

    gl_metric = aggregate(gl_counts)
    native_metric = aggregate(native_counts)
    bootstrap = paired_bootstrap(gl_counts, native_counts)
    subject_rows = []
    for index, subject in enumerate(subjects):
        gm = metrics(dict(zip(COUNT_KEYS, gl_counts[index])))
        nm = metrics(dict(zip(COUNT_KEYS, native_counts[index])))
        subject_rows.append(
            {
                "source": source,
                "target": target,
                "dataset": dataset,
                "outer_subject": subject,
                "glsd_tp": gm["TP"],
                "glsd_fp": gm["FP"],
                "glsd_fn": gm["FN"],
                "glsd_f1": gm["F1"],
                "native_tp": nm["TP"],
                "native_fp": nm["FP"],
                "native_fn": nm["FN"],
                "native_f1": nm["F1"],
                "subject_delta": gm["F1"] - nm["F1"],
            }
        )
    return gl_metric, native_metric, bootstrap, subject_rows, len(gl)


def render_report(summary: list[dict], bootstrap: list[dict]) -> str:
    labels = {
        ("metst", "boostingvrme", "sammlv"): "M→B / SAMMLV",
        ("metst", "boostingvrme", "casme3"): "M→B / CAS(ME)3",
        ("boostingvrme", "metst", "sammlv"): "B→M / SAMMLV",
        ("boostingvrme", "metst", "casme3"): "B→M / CAS(ME)3",
    }
    boot_by_key = {(row["source"], row["target"], row["dataset"]): row for row in bootstrap}
    lines = [
        "# Transferred GLSD vs Transferred Native",
        "",
        "本分析只读取两套已锁定逐 outer-subject predictions；没有重新运行配置选择，也没有 target-side 调参。"
        "两种方法删除保存的 `matched_gt` 后，均使用 Native transfer 文件中的同一逐视频 GT 和锁定 evaluator 独立重算。",
        "",
        "| Setting | Transferred GLSD F1 | Transferred Native F1 | Δ GLSD−Native | 95% CI | P(Δ>0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        key = (row["source"], row["target"], row["dataset"])
        boot = boot_by_key[key]
        lines.append(
            f"| {labels[key]} | {row['transferred_glsd_f1']:.6f} | "
            f"{row['transferred_native_f1']:.6f} | {row['delta']:+.6f} | "
            f"[{boot['ci95_low']:+.6f}, {boot['ci95_high']:+.6f}] | "
            f"{boot['p_delta_gt_0']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "点估计只描述当前锁定样本：正值表示 transferred GLSD 较高，负值表示 transferred Native 较高。"
            "95% CI 跨 0 时，只能说未获得稳定方向的统计支持，禁止解释为两者等效或 GLSD 非劣。",
            "",
            "统计使用 10,000 次 outer-subject paired bootstrap，seed=100；每次以同一 subject 索引联合重采样两种方法，"
            "先累计 TP/FP/FN，再计算 pooled Raw Spotting F1。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    gl_rows = read_json(GL_PATH)
    native_rows = read_json(NATIVE_PATH)
    summaries = []
    bootstraps = []
    subject_rows = []
    for source, target, dataset in SETTINGS:
        gl, native, bootstrap, subjects, videos = compare_setting(
            source, target, dataset, gl_rows, native_rows
        )
        summaries.append(
            {
                "source": source,
                "target": target,
                "dataset": dataset,
                "transferred_glsd_tp": gl["TP"],
                "transferred_glsd_fp": gl["FP"],
                "transferred_glsd_fn": gl["FN"],
                "transferred_glsd_f1": gl["F1"],
                "transferred_native_tp": native["TP"],
                "transferred_native_fp": native["FP"],
                "transferred_native_fn": native["FN"],
                "transferred_native_f1": native["F1"],
                "delta": gl["F1"] - native["F1"],
                "videos": videos,
            }
        )
        bootstraps.append({"source": source, "target": target, "dataset": dataset, **bootstrap})
        subject_rows.extend(subjects)
    write_csv(OUTPUT_ROOT / "summary.csv", summaries)
    write_csv(OUTPUT_ROOT / "outer_subject_deltas.csv", subject_rows)
    write_json(
        OUTPUT_ROOT / "bootstrap.json",
        {
            "protocol": {
                "resamples": RESAMPLES,
                "seed": SEED,
                "paired_unit": "outer subject",
                "selection_rerun": False,
                "matching": "independent locked evaluator replay on shared per-video GT",
            },
            "results": bootstraps,
        },
    )
    (OUTPUT_ROOT / "TRANSFER_GLSD_VS_NATIVE_REPORT_CN.md").write_text(
        render_report(summaries, bootstraps), encoding="utf-8"
    )
    print(json.dumps({"summary": summaries, "bootstrap": bootstraps}, indent=2))


if __name__ == "__main__":
    main()
