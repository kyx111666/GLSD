# Self-Calibrated Threshold Decoder — Stage A

本实现仅执行用户批准的 ME-TST+ × SAMMLV / CASME3 frozen-cache 可行性实验。

- 在任何 q 计算前检查 cache SHA、subjects/videos/GT/k_p，复现四个 Native 锚点。
- SAMMLV 冻结配置取 `results/rgr1_sammlv_nested/report.json` 中既有 `selected_strong_config`；CASME3 取 `results/final_native_closure_strong_anchor_morphology/final_strong_native_selected_configs.csv`。保留每个 outer fold 自己的 c_s/c_d/c_b 及 fallback p。
- 对 outer fold h，所有 outer-train videos 和 outer-test videos 均使用配置 C_h。不能使用训练 subject 自己的 outer-fold 配置，因为该配置的历史选择可能看过 h 的 GT。本轮不重新选择 c_s/c_d/c_b。
- 复用 Native-only `run_expanded_native_fairness_audit.py` 中 moving_average、threshold、event、evaluate 和 matching。round 为 Python ties-to-even，边界不额外裁剪，inclusive IoU≥0.5，按时间顺序 greedy one-to-one matching。
- P=`scipy.signal.find_peaks(g, distance=distance)[0]`，不指定 height；合法 local peaks 已满足原有 distance 约束。τ_q=`numpy.quantile(g[P], q, method='linear')`；保留 height≥τ_q 的 P 子集，原 fixed boundary 构造区间。
- q 只取 0.70/0.75/0.80/0.85/0.90/0.95。若 |P|<2，使用 C_h 中原 p 的 Native threshold。fallback 仅由 score / peak count 决定。
- 每个 outer fold 仅在 outer-train subjects pooled counts 上选 q；精确 Fraction F1 排序，同 F1 取较少 FP，再按 q grid 原顺序（小 q 优先）。选好后评估一次 held-out subject。
- 1000 次 paired subject bootstrap，seed=20260905，同一 replicate 对两方法使用相同抽样 subject；CI 为 2.5/97.5 percentile。bootstrap 不重新选择 q。
- 增益的单 subject 依赖：固定 outer predictions，逐一删去 subject；若任一次 ΔF1≤0，则不满足 STRONG-GO 的单 subject 稳定要求。
- q regime 描述：输出完整频数、mode、median、range、分布重叠；两个 dataset 的 selected-q median 差≥0.15 定义为 grossly contradictory（用户例子 .90 vs .70 满足）。此数值仅是判读约定，不是统计检验。
- effectively tied 定义为 |ΔF1|≤0.001 且 paired bootstrap 95% CI 包含 0。clearly degrades 定义为 ΔF1<−0.001 或 CI 上界<0。
- STRONG-GO：两侧 observed ΔF1>0、两侧所有 leave-one-subject-out aggregate ΔF1>0、q regime 不矛盾。WEAK-GO：一侧改善另一侧 effectively tied，且无 clearly-degraded dataset 或矛盾 q regime。其余未满足 GO 条件者标 NO-GO 并解释。
- 此阶段为从既有两 dataset 诊断出发的 feasibility 实验；outer q 选择排除了当次 held-out GT，但不能把整个方法开发过程描述为全新未见数据验证。
- 不运行 backbone、recognition、Morphology、MSCR、LR、parameter predictor、adaptive smoothing、boundary/NMS redesign 或 Stage B。
