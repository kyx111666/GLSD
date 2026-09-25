# Decoder-Equivalence Alignment Audit

> **最终状态：PASS-DECODER-EQUIVALENT**  
> **范围：ME-TST+ × SAMMLV，subject `006`，video `006_1`**  
> **性质：engineering alignment gate；不是 Context Agreement 科学实验**

## 1. 结论

fresh frozen-forward score 与 historical compact score 仍然满足：

```text
np.allclose(default) = False
```

这个事实没有被修改，也没有通过校准、缩放、调阈值或改 decoder 参数规避。

但是，两条原始 score 分别独立进入相同的 ME-TST+ native spotting decoder 后：

- 6 个 peak 的 temporal index 完全相同；
- 6 个 `[onset, peak, offset]` 事件完全相同；
- event IoU 全部为 1；
- 相同 GT evaluator 下，两者均为 `TP=4, FP=2, FN=1`；
- Precision、Recall、F1 完全相同；
- 不存在 threshold-side classification disagreement，也不存在需要分析的不一致 candidate。

因此本次预先固定的 spotting engineering gate 判定为：

> **PASS-DECODER-EQUIVALENT：fresh frozen forward 与 historical compact score 虽未达到默认 `np.allclose`，但在 native ME-TST event-decoding 层面具备 spotting-decision equivalence。当前数值误差没有改变 peak、event boundary 或 TP/FP/FN，可以使用 fresh forward 作为后续 pre-stitch scientific audit 的自洽来源。**

## 2. 固定输入与数据来源

| 项目 | 值 |
|---|---|
| historical cache | `caches/me_tst/sammlv_strategy1_outputs.pkl` |
| historical cache SHA-256 | `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569` |
| fresh Drive 文件 | `SAMMLV_FULL/subject_006/006_1_frozen_hidden_score.npz` |
| fresh NPZ SHA-256 | `d940498053ecddee9915ee7f0930ad7f50d751fd39090ff3f463a3644e1d1b2f` |
| subject / video | `006 / 006_1` |
| `T_old / T_fresh` | `1245 / 1245` |
| score 修改、校准或缩放 | 无 |

fresh NPZ 实际字段为：`hidden`, `score`, `valid_mask`, `subject`, `video_id`。

## 3. Native decoder 固定规则

审计逐字对应恢复代码中的 native 规则：

1. `smooth(score, 2*k_p)`：宽度为 `10` 的 box convolution，`mode='same'`；
2. `threshold = mean + 0.55 * (max - mean)`；
3. `find_peaks(height=threshold, distance=k_p)`；
4. `k_p=5`；
5. event 为 `[peak-k_p, peak, peak+k_p]`；
6. spotting GT 匹配采用 interval IoU `>=0.5` 的一对一匹配。

原始依据位于 `training_utils.py:9-13, 15-37, 52-57`。两条 score 的 smoothing 和 threshold 均独立计算，没有共享任一条曲线的中间结果。

## 4. Score numerical comparison

| 指标 | 数值 |
|---|---:|
| max absolute error | `0.0018059015274047852` |
| mean absolute error | `0.00004485700401821385` |
| RMSE | `0.00016291876412085786` |
| Pearson | `0.999999700555799` |
| Spearman | `0.999998090973982` |
| default `np.allclose` | `False` |

误差分位数：

| 分位数 | absolute error |
|---|---:|
| p50 | `0.000001982756657525897` |
| p90 | `0.0000823542475700379` |
| p95 | `0.0002559781074523922` |
| p99 | `0.0008516472578048694` |
| p99.9 | `0.0016453759670257595` |

## 5. Native smoothing equivalence

| 指标 | 数值 |
|---|---:|
| smoothed MAE | `0.00003254902868207534` |
| smoothed RMSE | `0.0000659384674838196` |
| smoothed max error | `0.00040743239223958727` |
| smoothed Pearson | `0.999999918162371` |

## 6. Native threshold equivalence

| 指标 | 数值 |
|---|---:|
| threshold old | `0.572681804399491` |
| threshold fresh | `0.5726839166026682` |
| absolute delta | `0.000002112203177140337` |
| relative delta | `0.0000036882666096841936` |

阈值分别由各自 smoothed score 独立产生，没有强制共用 threshold。

## 7. Peak candidate equivalence

```text
peaks_old   = [91, 278, 485, 515, 743, 1236]
peaks_fresh = [91, 278, 485, 515, 743, 1236]
```

| 判据 | 结果 |
|---|---:|
| old peak count | 6 |
| fresh peak count | 6 |
| exact common peaks | 6 |
| exact Jaccard | 1.0 |
| ±1 one-to-one matches | 6 / 6 |
| ±2 one-to-one matches | 6 / 6 |

Primary exact 判据通过；tolerance 结果只作补充。

## 8. Native event-decoding equivalence

两条 score 都得到以下事件：

| onset | peak | offset |
|---:|---:|---:|
| 86 | 91 | 96 |
| 273 | 278 | 283 |
| 480 | 485 | 490 |
| 510 | 515 | 520 |
| 738 | 743 | 748 |
| 1231 | 1236 | 1241 |

| 指标 | 结果 |
|---|---:|
| old / fresh event count | 6 / 6 |
| exact event matches | 6 |
| mean matched IoU | 1.0 |
| minimum matched IoU | 1.0 |
| fraction IoU >= 0.5 | 1.0 |
| fraction IoU >= 0.9 | 1.0 |
| fraction IoU == 1 | 1.0 |

### Emotion 证据边界

fresh NPZ 没有保存 fresh recognition logits 或 fresh emotion sequence。为了隔离“score 数值差异是否改变 spotting decoder 决策”，本审计对两条 score 使用同一条 historical archived emotion sequence；在相同事件边界下，受控 emotion 输出均为 `negative (id=0)`。

这证明了 **score-only decoder equivalence**，但不等价于“fresh recognition logits 已被独立验证”。本 Gate 明确不要求新增 recognition 分析，因此该限制不阻断 spotting PASS，也不会被写成独立 recognition PASS。

## 9. GT evaluation equivalence

相同的 5 个 GT 与相同的 interval-IoU evaluator 得到：

| 输入 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| historical compact | 4 | 2 | 1 | 0.6666667 | 0.8 | 0.7272727 |
| fresh frozen forward | 4 | 2 | 1 | 0.6666667 | 0.8 | 0.7272727 |

TP/FP/FN 完全一致。

## 10. Margin sensitivity

- 不一致 peak candidate：`0`；
- threshold-side classification disagreement：`0`；
- old 最小已选 peak margin：`0.0756633941`；
- fresh 最小已选 peak margin：`0.0757862529`；
- threshold delta：约 `2.11e-6`。

因此没有任何微小误差造成 threshold crossing、peak creation/deletion 或 event boundary 变化。最靠近 threshold 的最终 peak 仍比各自阈值高约 `7.6e-2`，远大于两阈值之间的差值。

## 11. 与原 BLOCKED-ALIGNMENT 的关系

原报告及原始事实保持不变：

```text
np.allclose = False
BLOCKED-ALIGNMENT（数值逐点强对齐门槛）
```

本文件新增的是另一项、预先固定的 engineering gate：

```text
PASS-DECODER-EQUIVALENT（native spotting decision 门槛）
```

二者回答的问题不同，不互相覆盖，也不把 `np.allclose=False` 改写为 True。

## 12. 后续数据协议锁定

若后续单独授权 Context Agreement Audit，只允许：

```text
same frozen inference
        ↓
raw window predictions
        ↓
same fresh stitching
        ↓
fresh stitched score
        ↓
native candidate generation
        ↓
Agreement analysis
```

historical compact 只保留为 reproduction/alignment reference，不与 fresh raw contexts 混用。

本次未运行 Agreement、height-conditioned、missed-GT、matched-FP、bootstrap、CASME3、BoostingVRME、Skill 或新 decoder。

