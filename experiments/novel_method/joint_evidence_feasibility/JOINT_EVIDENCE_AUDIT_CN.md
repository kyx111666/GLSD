# Joint Evidence Audit / Spotting–Recognition Complementarity Test

日期：2026-09-02

## 最终结论

**NO-GO：recognition 没有提供足够独立增量，不建议继续 TST-Dec / SCED。**

ME-TST+ 上 equal-weight Joint 的 ROC-AUC 在 SAMMLV 和 CASME3 均有小幅上升，说明 recognition stream 不是完全无信息；但它没有通过本实验预先规定的关键门槛：

- 主标签和次标签的 PR-AUC 均未稳定超过 Spotting-only；
- matched-FP 下，CASME3 的 missed-GT recovery 明显下降；
- SAMMLV 选阈值后锁定迁移到 CASME3，Joint F1 低于 Spotting-only；
- BoostingVRME 的可用 hard recognition evidence 整体呈反向或无增益。

因此不能因为局部 ROC-AUC 上升而继续实现 Viterbi、finite-state decoder 或 duration prior。

## 1. Cache recognition 结构审计

### ME-TST+

两份 cache 均保存：

- `score`: frame/time-step level spotting curve；
- `logits`: shape 为 `[T, 5]` 的连续 recognition logits；
- `emotion`: shape 为 `[T]` 的 hard class ID；
- `mirror_score` 和 `mirror_logits`；
- `score`、`logits`、`emotion` 的时间长度在全部视频上一一对应。

类别语义由项目源码确定为：

| ID | 语义 |
|---:|---|
| 0 | negative |
| 1 | positive |
| 2 | surprise |
| 3 | others |
| 4 | neutral |

Cache 没有直接保存 softmax probability，但可由 logits 确定性计算。本审计使用原图 logits，不使用 mirror，避免引入 TTA 因素。

两份 cache 的 strategy 均为 `1: recognition_mainly_with_neutral_synergy`。已保存的 hard `emotion` 是项目 strategy 逻辑处理后的 argmax；本实验的 soft evidence 直接由原始 5 类 logits 计算。

### BoostingVRME

两份 curve cache 均保存：

- frame/time-step level spotting `score`；
- frame/time-step level hard `emotion_pred`；
- `score` 和 `emotion_pred` 在全部视频上等长；
- observed class IDs 为 0–4，类别 4 对应 neutral。

但 cache **没有保存 recognition logits 或 probabilities**，因此不能审计：

- `1-P(neutral)`；
- entropy；
- top1–top2 margin；
- calibrated confidence；
- emotion-class soft temporal contrast。

BoostingVRME 仅能使用 hard non-neutral fraction 做受限审计，不能与 ME-TST+ 的 soft evidence 完全等价。没有人为构造不存在的连续概率。

## 2. 固定实验协议

- 输入：四份真实 frozen-output cache；
- 不训练或修改 backbone，不训练 fusion，不修改 cache；
- candidate pool：native moving average（宽度 `2*k_p`）后所有局部峰；
- Spotting-only 与 Recognition/Joint 使用完全相同的 candidate peaks；
- 主标签：peak 落在 GT onset–offset 内；
- 次标签：peak-centered native `±k_p` interval 达到 IoU ≥ 0.5；
- normalization：deterministic per-video median/MAD robust z-score；
- bootstrap：subject-level，2,000 repeats，seed 20260902；
- matched-FP budget：对应 cache 的 native decoder FP 数；
- transfer：只在 SAMMLV 选择 threshold，原样应用于 CASME3。

这是研究方向筛查，不是可直接进入论文主表的无偏正式实验。完整 cache GT 已参与候选验证。

## 3. Evidence 定义

### Spotting-only

`S_i` 为 smoothed spotting curve 在 candidate peak 的逐视频 robust z-score。

### Recognition

ME-TST+：

```text
frame evidence = 1 - softmax(logits)[neutral]
R_i = mean(frame evidence over peak ± k_p)
```

BoostingVRME：

```text
frame evidence = 1[hard class ID != neutral]
R_i = non-neutral fraction over peak ± k_p
```

随后对 temporal recognition evidence 做逐视频 robust z-score。

### Joint

主组合固定为：

```text
J_i = (Z(S_i) + Z(R_i)) / 2
```

另外只作敏感性检查：recognition contrast、within-video rank average，以及 spotting 权重 0.25/0.75。没有按数据集选择最优权重。

## 4. 全部候选峰：主标签结果

表内格式为 `ROC-AUC / PR-AUC`。

| Backbone | Dataset | Spotting | Recognition | Joint |
|---|---|---:|---:|---:|
| ME-TST+ | SAMMLV | 0.7327 / 0.2370 | 0.7177 / 0.1701 | 0.7540 / 0.2337 |
| ME-TST+ | CASME3 | 0.6580 / 0.0622 | 0.6580 / 0.0517 | 0.6788 / 0.0579 |
| BoostingVRME | SAMMLV | 0.7444 / 0.2033 | 0.6358 / 0.0760 | 0.7413 / 0.2030 |
| BoostingVRME | CASME3 | 0.6580 / 0.0634 | 0.5970 / 0.0296 | 0.6510 / 0.0631 |

ME-TST+ 的 Joint ROC-AUC 同方向提高，但 PR-AUC 在两个数据集都低于 Spotting-only。BoostingVRME 的 Joint 没有提高。

## 5. 次标签结果

| Backbone | Dataset | Spotting | Recognition | Joint |
|---|---|---:|---:|---:|
| ME-TST+ | SAMMLV | 0.7800 / 0.2481 | 0.7511 / 0.1673 | 0.7957 / 0.2396 |
| ME-TST+ | CASME3 | 0.7248 / 0.0438 | 0.7298 / 0.0317 | 0.7535 / 0.0371 |
| BoostingVRME | SAMMLV | 0.7556 / 0.1935 | 0.6506 / 0.0675 | 0.7506 / 0.1934 |
| BoostingVRME | CASME3 | 0.7877 / 0.0558 | 0.6623 / 0.0132 | 0.7701 / 0.0554 |

次标签同样表现为：ME-TST+ 的 ROC-AUC 上升，但 PR-AUC 下降；BoostingVRME 无增益。

## 6. Native missed GT vs background

| Backbone | Dataset | Spotting | Recognition | Joint |
|---|---|---:|---:|---:|
| ME-TST+ | SAMMLV | 0.6217 / 0.0443 | 0.6331 / 0.0474 | 0.6555 / 0.0491 |
| ME-TST+ | CASME3 | 0.6388 / 0.0421 | 0.6415 / 0.0399 | 0.6617 / 0.0432 |
| BoostingVRME | SAMMLV | 0.6492 / 0.0541 | 0.5583 / 0.0375 | 0.6444 / 0.0538 |
| BoostingVRME | CASME3 | 0.6449 / 0.0440 | 0.5848 / 0.0262 | 0.6374 / 0.0438 |

ME-TST+ 的 missed-GT subset 存在弱 complementarity，但幅度小且不能稳定转化为 fixed-FP recovery。BoostingVRME 的 hard recognition evidence 明显更差。

## 7. Subject-level bootstrap

以下为 `Joint − Spotting` 的平均受试者差值及 95% CI。

### 全部候选峰，主标签

| Backbone | Dataset | Δ ROC-AUC 95% CI | Δ PR-AUC 95% CI |
|---|---|---:|---:|
| ME-TST+ | SAMMLV | +0.0364 [0.0048, 0.0807] | +0.0077 [-0.0139, 0.0324] |
| ME-TST+ | CASME3 | +0.0310 [0.0111, 0.0515] | +0.0070 [-0.0244, 0.0384] |
| BoostingVRME | SAMMLV | -0.0026 [-0.0226, 0.0155] | +0.0039 [-0.0016, 0.0118] |
| BoostingVRME | CASME3 | -0.0159 [-0.0246, -0.0076] | -0.0006 [-0.0012, -0.0002] |

ME-TST+ 的 ROC 排序改善是可重复的，但 PR 增益区间跨零。CASME3 BoostingVRME 出现显著反向。

### Missed-GT subset

ME-TST+：

- SAMMLV ΔROC CI：[0.0089, 0.1334]；ΔPR CI：[-0.0017, 0.0328]；
- CASME3 ΔROC CI：[0.0119, 0.0550]；ΔPR CI：[0.0036, 0.0620]。

BoostingVRME：

- SAMMLV ΔROC CI：[-0.0334, 0.0206]；ΔPR CI：[-0.0018, 0.0126]；
- CASME3 ΔROC CI：[-0.0270, -0.0083]；ΔPR CI：[-0.0012, -0.0002]。

## 8. Matched-FP missed-GT recovery

FP budget 固定为同一 cache 的 native decoder FP 数。

| Backbone | Dataset | Evidence | Recovered missed GT | FP | Recovery precision | Recovery recall |
|---|---|---|---:|---:|---:|---:|
| ME-TST+ | SAMMLV | Spotting | 7 | 184 | 0.0366 | 0.0660 |
|  |  | Recognition | 9 | 184 | 0.0466 | 0.0849 |
|  |  | Joint | 8 | 184 | 0.0417 | 0.0755 |
| ME-TST+ | CASME3 | Spotting | 66 | 912 | 0.0675 | 0.0849 |
|  |  | Recognition | 49 | 912 | 0.0510 | 0.0631 |
|  |  | Joint | 56 | 912 | 0.0579 | 0.0721 |
| BoostingVRME | SAMMLV | Spotting | 7 | 158 | 0.0424 | 0.0667 |
|  |  | Recognition | 4 | 158 | 0.0247 | 0.0381 |
|  |  | Joint | 7 | 158 | 0.0424 | 0.0667 |
| BoostingVRME | CASME3 | Spotting | 73 | 830 | 0.0808 | 0.0938 |
|  |  | Recognition | 22 | 830 | 0.0258 | 0.0283 |
|  |  | Joint | 73 | 830 | 0.0808 | 0.0938 |

这是 NO-GO 的关键证据。ME-TST+ Joint 只在 SAMMLV 多恢复 1 个事件，却在 CASME3 少恢复 10 个；不存在稳定方向。

## 9. SAMMLV → CASME3 locked threshold transfer

| Backbone | Evidence | Source peak F1 | Target precision | Target recall | Target F1 |
|---|---|---:|---:|---:|---:|
| ME-TST+ | Spotting | 0.2605 | 0.0775 | 0.2528 | 0.1187 |
|  | Recognition | 0.2310 | 0.0524 | 0.3271 | 0.0903 |
|  | Joint | 0.2717 | 0.0645 | 0.3180 | 0.1072 |
| BoostingVRME | Spotting | 0.2636 | 0.0778 | 0.2273 | 0.1159 |
|  | Recognition | 0.1401 | 0.0293 | 0.4370 | 0.0550 |
|  | Joint | 0.2567 | 0.0796 | 0.2231 | 0.1173 |

ME-TST+ Joint 在 source 上略好，但迁移到 CASME3 后比 Spotting-only 更差，说明增益不能稳定迁移。BoostingVRME Joint 的目标 F1 只提高约 0.0014，source 又更差，不能构成支持证据。

## 10. Sensitivity 与 cross-backbone 解释

- α=0.25/0.75 没有改变总体判断；较高 spotting 权重只会让 Joint 逐渐退化为 Spotting-only。
- Rank average 没有在四组结果中形成稳定优势。
- Recognition temporal contrast 普遍较弱，不值得继续增加 handcrafted variants。
- ME-TST+ 的 soft logits 包含一定排序补充信息，但更像 dataset-dependent weak cue，而非足以支撑 structured decoder 的稳定证据。
- BoostingVRME cache 缺少 soft logits是证据上限；当前 hard sequence 不支持 backbone-independent complementarity。若未来重新生成包含 logits 的 cache，可作为一个全新的审计问题，但本次不能因此改写结论。

## 11. GO / NO-GO 判定

预设 GO 条件要求：两数据集方向一致、PR-AUC或 matched-FP 有实质增益、locked transfer 保留优势、第二 backbone 不系统性反向。

实际结果：

- ME-TST+ ROC-AUC：通过；
- ME-TST+ PR-AUC：未通过；
- ME-TST+ matched-FP：未通过；
- ME-TST+ locked transfer：未通过；
- BoostingVRME：未通过，并在 CASME3 系统性反向。

因此停止：

- TST-Dec；
- SCED；
- Background/Onset/Apex/Offset Viterbi；
- finite-state structured decoder；
- duration prior；
- joint-evidence Skill packaging。

## 12. 可复现命令

```bash
RethinkFuse_reproduction/.venv/bin/python \
  RethinkFuse_reproduction/my_method/joint_evidence_feasibility/run_joint_evidence_audit.py
```

机器可读输出：`outputs/joint_evidence_results.json`。文件包含全部 cache path、SHA-256、参数、随机种子、subject/video/GT counts 和完整敏感性结果。

**NO-GO：recognition 没有提供足够独立增量，不建议继续 TST-Dec / SCED。**

