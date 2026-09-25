# PCBR-1 锁定协议

来源：用户 2026-09-05 粘贴的 PCBR-1 请求。本文在 PCBR 结果产生前保存。

- 只读指定 SAMMLV frozen cache，先精确复现 49/143/110、F1=0.2792022792022792、Rec F1=0.699、STRS=0.19516239316239314。
- 复用 RGR-1 report 保存的每个 outer fold Native 配置，不重新选择 Native 参数。`k=max(1,round(c_b*5))`，本任务 k 通常为 6，subject 037 为 5；不是统一使用 cache k_p=5。
- 每折将该 outer fold 配置用于全部 outer-training subjects。沿用现有 `evaluate_decoding`、`add_counts`、`f1` 的 pooled train-subject F1 选择流程；28 个 subject 分别作为 inner validation subject，各评价一次后汇总。PCBR 无拟合步骤，其余 27 个 inner-training subjects 不执行训练。不得使用训练 subject 各自的 outer-fold 配置，因为这些配置的历史选择可能见过当前 outer-test subject。
- 唯一 alpha grid 为 0.30、0.40、0.50、0.60、0.70，来自本次请求。按更高 pooled inner F1、再更高 alpha 选择；以整数分数比较 F1，避免浮点 tie 偏差。
- inference 仅接受 curve、candidate 字典、k 和 alpha。curve 完全复用 Native smoothing；背景为闭区间 [p-2k,p+2k] 的最小值；从 peak 连续搜索到首个低于阈值的位置，停止在最后一个合格 index；仅向外扩展，每侧最多 k。
- Native 区间按请求裁剪到有效 index。已发现 026/026_1/pred_0 的 onset=-1，裁剪为 0；独立核查裁剪不改变 aggregate anchor。保留 raw 和 clipped Native 坐标，emotion 按原始 Native 区间计算一次后固定。
- 使用既有 formal spotting evaluator、既有 paper recognition summary。新边界不改变 candidate emotion/confidence/order；recognition F1 可以因 matched cohort 改变而改变。
- 先单独运行 006 smoke，成功后 full 必须校验 smoke 的脚本、协议、输入哈希及断言。固定 alpha 全缓存指标在所有 outer alpha 选择冻结后生成，仅为 sensitivity。
- Oracle identities、GT-pair cohort 文件只在全部正式预测与选择完成后读取，用于 retrospective tracking，不参与 inference/selection。报告同一 Native matched GT 的 IoU 变化及匹配对象变化，避免混淆。
- 只报告事实，不在程序中判定 GO/NO-GO；不做 bootstrap，不修改规则、不追加实验。
