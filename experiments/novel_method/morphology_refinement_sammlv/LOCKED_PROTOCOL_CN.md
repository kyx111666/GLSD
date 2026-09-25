# Morphology Rescue Local Refinement — SAMMLV

本文件在 refinement 结果产生前锁定，依据用户 2026-09-05 请求。它只允许这一轮 SAMMLV local refinement；运行结束后 `SAMMLV-METHOD-DEVELOPMENT-FROZEN`。

- 输入、GT、正式 evaluator、outer subject LOSO、pooled inner-validation Spotting F1、subject-disjoint inner LOSO、weak candidate 定义、candidate label、LR/scaler、balanced/liblinear、seed、high-event preservation、rescue interval、single-class handling、无额外 NMS 全部沿用 Morphology v1。
- 首先验证并复现已有 v1 artifact：54/152/105，F1=0.2958904109589041；任何不一致均为 `BLOCKED-V1-REPRODUCTION`。
- 正式方法只用 peak-relative patch，输入严格为 `[H(p), x_1,...,x_31]`。`Delta_p=(.15,.20,.25,.30)`，requested `L/k_p=(.25,.50,.75,1.00)`，`C=(.1,1,10)`，`gamma=(.60,.70,.80,.90)`，共 192 configs；patch 固定 31 点。k_p=5，Python round 映射 actual radius=(1,2,4,5)，无重复 radius。
- weak pool 由现有 `decode_low` 的同源 low-minus-high 差集逻辑生成，仅传 score 和零占位 R2；不读取 recognition 作为 feature，不读取 GT。每个 weak interval 使用该 outer fold Tuned Native 的 c_b boundary，保留原 integer rounding / clipping 语义。
- high Tuned Native events 完整原样保留。每个 outer fold 对每个 `(Delta_p,L,C)` 做 inner subject LOSO：scaler 与 LR 只在 inner-training subjects fit，聚合 OOF probabilities 后以 pooled F1 选择 gamma/config；单类别 inner-training 时固定输出训练中唯一类别概率，并记录，不使用 validation/test label。
- 若 inner-validation subject 没有 weak candidate，则没有 OOF 样本，不对空矩阵调用 sklearn；记录 `no_validation_candidates`，该 subject 对 pooled validation counts 贡献零。这是由 candidate pool 结构决定的固定情况。
- 若 outer-test subject 没有 weak candidate，则最终 refit 不调用 sklearn 的空 transform，直接返回空概率数组；该折只有完整保留的 high events。
- 选择后在全部 outer-train subjects refit 一次，再评价 held-out outer subject；outer-test GT 不影响 Delta/L/C/gamma、scaler、权重。使用 exact rational pooled F1，再 fewer FP、fewer rescue、smaller Delta、higher gamma、确定性字段序列 tie-break。
- v1 comparison 读取已有冻结 `morphology_outer_metrics.csv`；不重跑 PSED/Recognition，不扩大 v1 grid。bootstrap 使用 1000 次相同 subject indices、seed=20260902，报告 refined−TunedNative 与 refined−v1。
- endpoint 频率超过半数标记 `BOUNDARY-SELECTION-REMAINS`，不再扩大 grid。最终只输出 `MORPHOLOGY-REFINED-WINNER` 或 `MORPHOLOGY-V1-RETAINED`，并固定写入 `SAMMLV-METHOD-DEVELOPMENT-FROZEN`。不运行 CASME3。
