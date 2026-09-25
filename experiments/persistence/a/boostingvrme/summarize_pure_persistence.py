"""校验已保存的结果，生成中文纯显著性验证报告与图表。"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import equiscale_fair_validation as fair
import pure_persistence_validation as pure


def verify_saved_report(target):
    report = pure.load_json(target / "report.json")
    manifest = pure.load_json(target.parent / "PROTOCOL.json")
    signed = {key: value for key, value in manifest.items() if key != "signature"}
    assert hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest() == manifest["signature"]
    assert manifest["engine_sha256"] == fair.digest(Path(pure.__file__))
    assert report["manifest_signature"] == manifest["signature"]
    with (target / "subject_counts.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for family in pure.METHODS:
        values = np.array([[int(row[key]) for key in ("TP", "FP", "FN")] for row in rows if row["family"] == family])
        assert len(values)
        assert fair.metrics(values.sum(axis=0)) == report["metrics"][family]
    with (target / "threshold_sensitivity.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for offset in pure.OFFSETS:
        values = np.array([[int(row[key]) for key in ("TP", "FP", "FN")] for row in rows if float(row["offset"]) == offset])
        assert fair.metrics(values.sum(axis=0)) == report["threshold_sensitivity"][str(offset)]["metrics"]
    errors = pure.load_json(target / "error_audit.json")
    assert dict(Counter(row["kind"] + ":" + row["reason"] for row in errors)) == report["error_audit_counts"]
    change = report["comparisons"]["pure_vs_original_native"]
    assert sum(row["kind"] == "lost_gt" for row in errors) == change["GT_lost"]
    assert sum(row["kind"] == "new_fp" for row in errors) == change["FP_added_exact_interval"]
    return report


def title(report):
    backbone = "ME-TST" if report["backbone"] == "metst" else "BoostingVRME"
    dataset = "SAMMLV" if report["dataset"] == "sammlv" else "CASME_3"
    return f"{backbone} / {dataset}"


def main():
    root = fair.WORKSPACE
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    reports = []
    all_reports = []
    for backbone in ("metst", "boostingvrme"):
        for dataset in ("sammlv", "casme3"):
            modes = ("native", "fixed") if backbone == "boostingvrme" else ("fixed",)
            for mode in modes:
                target = root / backbone / "results" / pure.RESULT_NAME / dataset / mode
                report = verify_saved_report(target)
                all_reports.append(report)
                if report["primary_intervals"]:
                    reports.append(report)
    target = root / "boostingvrme" / "results" / pure.RESULT_NAME
    fair.json_write(target / "combined_report.json", all_reports)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    for ax, report in zip(axes.flat, reports):
        values = [report["threshold_sensitivity"][str(offset)]["metrics"]["F1"] for offset in pure.OFFSETS]
        baseline = report["metrics"]["original_native"]["F1"]
        ax.axhline(baseline, color="#6b6b6b", linestyle="--", linewidth=1.2, label="原始基线")
        ax.plot(pure.OFFSETS, values, marker="o", color="#237a55", linewidth=1.6, label="固定配置的 G+L 方案")
        ax.scatter([0], [values[2]], s=45, color="#272727", zorder=3, label="训练折选中的阈值")
        ax.set_title(title(report), fontsize=12)
        ax.set_xlabel("阈值偏移量（不重新选参）", fontsize=10)
        ax.set_ylabel("原始事件定位 F1", fontsize=10)
        ax.set_xticks(pure.OFFSETS)
        ax.set_ylim((0.23, 0.31) if report["dataset"] == "sammlv" else (0.08, 0.125))
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if report is reports[0]:
            ax.legend(fontsize=8, loc="lower left", frameon=False)
    plot_path = target / "threshold_sensitivity.png"
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)
    lines = ["# 纯显著性方案：同候选验证报告", "",
             "主方案固定为 `S=(G+L)/2`，其中 G 为归一化的全局显著性，L 为归一化的局部多尺度显著性。"
             "不使用高度项，也不根据 P 的高低进行路由。四组实验共用同一个包含 90 个配置的搜索空间，"
             "各折仅用训练被试选择参数，并保留各项目原有的区间解码方式。", "",
             "## 主方案结果", "",
             "下表指标均为原始事件定位 F1（Raw Spotting F1）。TP、FP、FN 分别表示正确检测、误检和漏检数量。"
             "“原始基线”保留原实验时长参数；“训练折时长基线”仅用训练被试估计 k。"
             "增量和被试配对 95% 置信区间均相对原始基线计算。", "",
             "| 项目／数据集 | 原始基线 | 训练折时长基线 | 纯 G+L | TP / FP / FN | 相对原始基线增量 | 配对 95% 置信区间 |",
             "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for report in reports:
        metrics = report["metrics"]
        m, c = metrics["pure"], report["comparisons"]["pure_vs_original_native"]
        lines.append(f"| {title(report)} | {metrics['original_native']['F1']:.6f} | {metrics['native']['F1']:.6f} | "
                     f"{m['F1']:.6f} | {m['TP']} / {m['FP']} / {m['FN']} | {c['delta_F1']:+.6f} | "
                     f"[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] |")
    lines += ["", "四组主方案的 F1 点估计均有提升，但相对原始基线的置信区间都包含零，尚不能认定为稳定的显著提升。"
              "本轮复现了此前纯显著性消融的结果，没有追加搜索以挑选更高分的方案。", "",
              "## 固定全部设置的逐项消融", "",
              "每组固定主方案选中的参考尺度、局部半径、阈值、候选点、预测区间和非极大值抑制（NMS）设置，"
              "只将评分公式分别替换为 G、L 或 `(G+L)/2`。数值相同的阈值不代表相同的候选通过率，"
              "因此本节用于诊断评分项的作用，不能单独证明局部项具有独立预测收益。", "",
              "| 项目／数据集 | 相同设置下仅用 G | 相同设置下仅用 L | 相同设置下使用 G+L |",
              "| --- | ---: | ---: | ---: |"]
    for report in reports:
        m = report["metrics"]
        lines.append(f"| {title(report)} | {m['global_at_pure_settings']['F1']:.6f} | "
                     f"{m['local_at_pure_settings']['F1']:.6f} | {m['pure']['F1']:.6f} |")
    lines += ["", "## 由 G 选择候选池的同候选对照", "",
              "先由纯 G 方案在内层被试上选择参考尺度和阈值，然后固定该参考尺度，允许 L 和 G+L "
              "仅使用内层被试重新校准局部半径与阈值。三种方法使用完全相同的候选点和区间解码输入。"
              "由于参考尺度由 G 优化选择，这种设置有利于 G，不是对三种方法完全对称的尺度选择。", "",
              "| 项目／数据集 | 仅用 G | 同候选下仅用 L | 同候选下使用 G+L | G+L 相对 G 的增量 | 配对 95% 置信区间 |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for report in reports:
        m, c = report["metrics"], report["comparisons"]["fusion_at_global_candidates_vs_global_tuned"]
        lines.append(f"| {title(report)} | {m['global_tuned']['F1']:.6f} | {m['local_at_global_candidates']['F1']:.6f} | "
                     f"{m['fusion_at_global_candidates']['F1']:.6f} | {c['delta_F1']:+.6f} | "
                     f"[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] |")
    lines += ["", "在当前冻结曲线的对照条件下，两组 SAMMLV 都显示局部项的正向增量证据。"
              "ME-TST/CASME_3 没有正向增量；BoostingVRME/CASME_3 的点估计为正，但置信区间跨过零。"
              "同候选对照用于分析机制，没有在查看结果后替换主方案。置信区间以已有曲线和已选配置为条件，"
              "未进行多重比较校正。", "",
              "## 阈值敏感性", "",
              "固定每折已经选中的参考尺度和局部半径，只改变阈值。不根据测试结果挑选最优偏移量。"
              "偏移后的阈值保留十位小数，并将下限设为零。", "",
              "| 项目／数据集 | -0.10 | -0.05 | 已选阈值 | +0.05 | +0.10 | 原始基线 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        values = " | ".join(f"{report['threshold_sensitivity'][str(offset)]['metrics']['F1']:.6f}" for offset in pure.OFFSETS)
        lines.append(f"| {title(report)} | {values} | {report['metrics']['original_native']['F1']:.6f} |")
    lines += ["", f"![阈值敏感性分析]({plot_path.as_posix()})", "",
              "阈值偏移 -0.05 和 +0.05 时，四组结果仍高于各自原始基线。"
              "BoostingVRME/SAMMLV 在 -0.10 和 +0.10 时均低于基线，其中 +0.10 会造成较明显的召回损失。"
              "这说明方案在较小的阈值邻域内仍有提升，但不能认为它在较大范围内对参数不敏感。", "",
              "## 错误归因", "",
              "GT 表示真实标注事件。“救回”指原始基线未检出而新方案检出的事件；"
              "“丢失”指原始基线已检出而新方案未检出的事件。IoU 为预测区间与标注区间的交并比。", "",
              "| 项目／数据集 | 救回 GT | 丢失 GT | 缺少合格候选区间 | 评分未通过 | 抑制或匹配冲突 | 新增 FP：IoU <0.1 | 新增 FP：0.1≤IoU<0.5 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        c, e = report["comparisons"]["pure_vs_original_native"], report["error_audit_counts"]
        values = [c["GT_rescued"], c["GT_lost"], e.get("lost_gt:candidate_or_interval_miss", 0),
                  e.get("lost_gt:score_rejected", 0), e.get("lost_gt:suppression_or_greedy_matching", 0),
                  e.get("new_fp:low_overlap", 0), e.get("new_fp:partial_overlap", 0)]
        lines.append(f"| {title(report)} | " + " | ".join(str(value) for value in values) + " |")
    lines += ["", "新增 FP 按预测区间是否与原始基线完全一致来统计。即使覆盖同一段实际活动，"
              "位置发生偏移的区间也可能被计为新增。IoU 分类仅用于数值诊断，不代表视频内容的语义类别。"
              "ME-TST/CASME_3 丢失的事件中，有 13 个在当前候选区间池里不存在 IoU≥0.5 的匹配；"
              "在该候选池固定的条件下，只调整评分阈值无法救回这些事件。", "",
              "## 复现与验证", "",
              "此前带签名的缓存及算法保持不变，新验证协议已在本轮复算前记录。"
              "所有方法及阈值扰动后的预测均使用原始逐条匹配实现重新计数，并核对每条预测对应的真实事件；"
              "所有同候选对照均检查了候选峰、区间和 NMS 冲突是否一致。"
              "逐被试计数、选参结果、预测列表和错误归因 CSV/JSON 保存在 "
              "`<project>/results/pure_persistence_matched_v1/<dataset>/<intervals>/`.", "",
              "本轮属于对已查看过的冻结折外预测曲线（OOF）的探索性分析。固定 w=0 的决定参考了此前结果，"
              "骨干网络也没有针对每个内层折重新训练。因此，本报告不能证明方法在独立外部数据上的泛化能力。", "",
              "以下命令依次运行四组验证、执行本轮测试并重新生成中文报告：", "",
              "```powershell",
              "& 'D:\\Anaconda3\\envs\\ME-TST\\python.exe' -B 'D:\\workspace\\a\\boostingvrme\\pure_persistence_validation.py' --backbone both --dataset both",
              "& 'D:\\Anaconda3\\envs\\ME-TST\\python.exe' -B -m unittest discover -s 'D:\\workspace\\a\\boostingvrme' -p 'test_pure_persistence_validation.py' -v",
              "& 'D:\\Anaconda3\\envs\\ME-TST\\python.exe' -B 'D:\\workspace\\a\\boostingvrme\\summarize_pure_persistence.py'",
              "```", ""]
    (root / "PURE_PERSISTENCE_MATCHED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("已校验六套区间协议，并生成中文报告 PURE_PERSISTENCE_MATCHED_RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
