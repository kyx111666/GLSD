# PSED 48 小时可行性审计

日期：2026-09-02

## 结论

**NO-GO：不建议把 PSED（Persistence-Stable Event Decoding）继续作为论文主方法。**

图片中的总结构——冻结 backbone、通过 adapter 形成标准化输出、由统一 inference skill 解码事件——是合理且比原 SkillME + EquiScale 更完整的研究结构。但本次真实缓存测试不支持 PSED 的核心假设：**一维 persistence/prominence 没有比 robust peak height 更好地区分事件相关峰与背景峰。**

因此应区分两个判断：

1. **“跨 backbone 的确定性 inference skill”结构保留；**
2. **“以 persistence 为核心事件发现算法”停止投入。**

下一主候选应回到更能利用 ME-TST/+ 特有信息的 structured joint decoding（TST-Dec/SCED），EquiScale 保留为已经有正向结果的 baseline/保底方案。Peak prominence 最多作为一个普通对照或辅助特征，不能承担主要创新。

## 本次测试回答的问题

PSED 文本提出的首个证伪问题是：

> TP/event-related peak 的 persistence 是否明显高于 FP/noise peak，并且比 peak score 更有判别力？

测试使用四份真实 frozen-output cache：

- ME-TST+ × SAMMLV：79 videos，29 subjects，159 GT；
- ME-TST+ × CASME3：462 videos，94 subjects，858 GT；
- BoostingVRME × SAMMLV：79 videos，29 subjects，159 GT；
- BoostingVRME × CASME3：462 videos，94 subjects，858 GT。

每份输入均记录 SHA-256，见 `outputs/feasibility_results.json`。未训练或修改任何 backbone。

## 固定协议

- 主分析：复用 native moving average，宽度为 `2*k_p`；
- 反证分析：直接使用未平滑 frozen score；
- 标准化：逐视频 `median/MAD` robust z-score；
- persistence：一维 superlevel signal 的 peak prominence；
- 主标签：局部峰是否落在 GT onset–offset 内；
- 次标签：以峰为中心的 `±k_p` 区间是否达到 IoU ≥ 0.5；
- 指标：ROC-AUC、PR-AUC、受试者级 paired bootstrap；
- 迁移：只在 SAMMLV 选峰筛选阈值，原样应用于 CASME3；
- 随机种子：20260902。

注意：这是一项**候选方法筛查**，不是可用于论文主表的无偏正式实验。完整 GT cache 已经参与研究方向筛选。

## 核心结果

### 1. 平滑曲线上的局部峰判别

| Backbone | Dataset | Peak height AUC | Persistence AUC | Δ(Persistence−Height) |
|---|---:|---:|---:|---:|
| ME-TST+ | SAMMLV | 0.7327 | 0.7023 | -0.0305 |
| ME-TST+ | CASME3 | 0.6580 | 0.5552 | -0.1028 |
| BoostingVRME | SAMMLV | 0.7444 | 0.6697 | -0.0747 |
| BoostingVRME | CASME3 | 0.6580 | 0.5995 | -0.0585 |

四个组合全部是 persistence 更差。

受试者级 bootstrap 的 ΔAUC 95% CI：

- ME-TST+ / SAMMLV：[-0.0681, 0.0037]；
- ME-TST+ / CASME3：[-0.1493, -0.0957]；
- BoostingVRME / SAMMLV：[-0.1108, -0.0159]；
- BoostingVRME / CASME3：[-0.1001, -0.0606]。

除 ME-TST+ / SAMMLV 的区间轻微跨零外，其余三组都稳定支持 peak height 优于 persistence。

### 2. 未平滑曲线反证检查

| Backbone | Dataset | Raw height AUC | Raw persistence AUC | Δ |
|---|---:|---:|---:|---:|
| ME-TST+ | SAMMLV | 0.6823 | 0.6410 | -0.0413 |
| ME-TST+ | CASME3 | 0.6177 | 0.6021 | -0.0157 |
| BoostingVRME | SAMMLV | 0.7176 | 0.6733 | -0.0443 |
| BoostingVRME | CASME3 | 0.6444 | 0.6250 | -0.0195 |

移除 native smoothing 后结论不变，因此不能把失败归因于平滑设置。

### 3. Native TP 与 FP

仅在 native 已选出的峰中，ME-TST+ 上 persistence 相对 height 有很小的 AUC 增益：SAMMLV +0.0198、CASME3 +0.0158；但 BoostingVRME 上分别为 -0.0029、-0.0166。

这最多说明 prominence 可能作为 **ME-TST+ 已有候选的弱过滤特征**，并不支持“persistent event discovery”作为跨 backbone 核心算法，也不能解决大量 FN。

### 4. 漏检事件与迁移

Native 漏检 GT 中，74.5%–82.9% 的区间内部仍至少存在一个局部峰，说明 frozen curve 中确实还有可利用信息；但 persistence 没有比 height 更好地给这些峰排序。这更支持后续尝试：

- 联合 spotting 与 recognition logits；
- 利用 rise–apex–fall 顺序或其它结构证据；
- 而不是只换一个峰显著性度量。

SAMMLV 锁定阈值迁移到 CASME3 时，persistence 的峰级 F1 也低于 height：

- ME-TST+：0.0932 vs 0.1187；
- BoostingVRME：0.1096 vs 0.1159。

未平滑版本同样没有逆转结论。

## 对三个候选方向的修正评价

| 方向 | 可行性 | 新颖性风险 | 当前建议 |
|---|---|---|---|
| EquiScale + SkillME | 已有双数据集正向 PoC | mirror、多尺度投票和聚合较增量 | 强 baseline / 保底，不宜主创新 |
| PSED | 本次核心假设未通过 | 1D persistence 与经典 peak prominence 接近 | 停止作为主方法；最多保留普通对照 |
| TST-Dec / SCED | 尚未完成最小 PoC | 参数化 unary/transition 可能变成 heuristic；但结构层创新更强 | 下一主候选，必须先做最小可证伪测试 |

## 对图片结构的必要修正

建议保留总框架，但不要只写 `standardized output`，而应明确 Skill Contract：

```text
Backbone-specific Adapter
  -> eventness[T]
  -> optional class probabilities[T, C]
  -> fps / frame mapping / neutral-class semantics
  -> provenance + input hash

Backbone-invariant Deterministic Decoder
  -> onset / apex / offset / class / confidence

Verifier
  -> schema / protocol / leakage / reproducibility checks
```

所谓“迁移成功”必须满足：核心 decoder 和参数冻结，只允许 adapter 做张量/语义映射；不能在每个 backbone 或数据集重新搜索一套规则。ME-TST 与 ME-TST+ 同家族迁移可作为最低证据，BoostingVRME 这类不同输出分布的 frozen backbone 更有说服力。

## 投稿判断

改成“算法是 Method，Skill 是可审计、可迁移的 realization”确实比原先把 SkillME 当主创新更适合 ICASSP，也更容易形成统一的信号处理故事。但 **Skill 包装和跨 backbone 表格不能挽救一个核心假设未通过的方法**。

本次结果触发了 PSED 文本自己设定的止损条件。继续为 PSED 设计 stable-boundary、reliability 或 Skill 包装，会把时间投入到一个未通过事件发现门槛的方向上，不建议继续。

## 可复现命令

```bash
RethinkFuse_reproduction/.venv/bin/python \
  RethinkFuse_reproduction/my_method/psed_feasibility/run_feasibility.py
```

输出：`outputs/feasibility_results.json`。

## 下一道门

下一步不应立即写完整 TST-Dec 或 Skill，而应给 TST-Dec 做同等级的最小门槛测试：比较 spotting-only evidence 与 spotting+recognition joint evidence，检查后者是否能对 native missed GT 提供独立增量；只有 joint evidence 在两个数据集或锁定迁移设置下产生稳定增益，才实现四状态/结构解码。

