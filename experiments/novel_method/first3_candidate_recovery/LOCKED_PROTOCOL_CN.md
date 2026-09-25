# FIRST3 Candidate Recovery 锁定协议

本实验是在此前 fixed-configuration audits 之后，根据导师建议重新开放预定义、有限、subject-disjoint 的 hyperparameter tuning；outer test subject 从未用于参数选择。

本文件在本轮结果产生前锁定。授权来源为本次用户附件；此前 NO-GO 作为历史背景，不中止本轮三个 Idea。仅 SAMMLV frozen cache；允许本次明确指定的 LR 拟合，禁止 backbone forward / retraining。

## 共用定义

- 首先通过 Author 53/184/106 与 Tuned 49/143/110 exact anchor gate，否则 BLOCKED-BASELINE-MISMATCH。
- 复用 results/rgr1_sammlv_nested/report.json 的 selected_strong_config。每个 outer 的该配置固定应用于本折全部 train/validation/test records；不引用其他 subject 各自的 OOF 配置来构建训练数据。
- 直接调用现有 RGR 的 decode_low 作为共同候选生成器。其 recognition 数组传全零占位，仅供现有函数形式参数使用，gamma=-inf 接受全部 weak；不读取 recognition 或 GT 来生成 pool。保留其 low-minus-high、rounding、event geometry 和按 peak 排序的 union。原 evaluator 不 clip interval，故沿用原有坐标（包括已有 Native onset=-1），不引入 PCBR 的 clipping。
- 所有方法保留 high event 字典原样。共同 weak ID 为 subject/video/weak_peak_p。Delta_p={.05,.10,.15,.20}。H=clip((g[p]-tau_L)/(tau_H-tau_L+1e-8),0,1)。
- PSED：复用 run_feasibility.robust_z(g)=(g-median(g))/(1.4826*MAD(g)+1e-8)，复用 scipy peak_prominences 在该曲线上所有局部峰计算的 prominence_z；P=sigmoid(prominence_z)。不对 prominence 集合再次 z-score，不新增归一化选项。lambda={.25,.50,.75}、gamma={.25,.40,.55,.70,.85}，共60组。
- Recognition：复用 separability audit 的 softmax(axis=1) 与 aggregate mean，class order=[negative,positive,surprise,others,neutral]，neutral=4；R2=max(prob[:,:4])，窗口固定 ±k_p=±5，合法 frame clipping。alpha={0,.25,.50,.75,1}、gamma 同 PSED，共100组。本轮不运行可选 R1 control。
- Morphology：L multiplier={.5,1,1.5,2}；整数半径=max(1,round(L*k_p))，Python round 与 Native convention 一致（本次为2、5、8、10）。复用旧 extract_patch edge padding 与 resample_patch 到31点。raw-z 在重采样31点内减均值除(std+1e-8)；peak-relative 为减中心后除(L2 norm+1e-8)，复用旧 relative_l2 数值语义。输入为 [H,31点]，不加入其他特征。
- 沿用旧 fit_probe 的 StandardScaler，仅在每个 inner-training 集及最后 outer-training refit 集拟合，无全局 scaler。LR solver=liblinear、balanced、max_iter=2000、seed=20260902；C={.1,1,10}、gamma={.3,.5,.7}。4×4×2×3×3=288组。
- 标签按独立 weak event 能否用正式 match_events 匹配 high 未匹配 GT 定义；每个 weak 都独立判断。重复 weak 的竞争只在最终正式 sorted-union evaluator 中处理，不以 peak-inside-GT 标注。
- 正式运行前的训练可用性检查发现：Delta=.05正例只来自007/013，两者同时被outer/inner排除时训练集全负。对此固定处理：单类别inner-training不拟合不可定义的LR，validation概率固定为该训练类别(0或1)，空训练集固定为0；记录single_class_constant与训练标签哈希。仅由inner-training标签确定，不使用validation/test标签，不改变任何grid或正常折LR。此处理在正式29折结果产生前加入并重新smoke；早先smoke保留作审计。

## Selection / evaluation

- LowRescue、PSED、Recognition：对 outer-training subjects pooled F1 选择；outer test 不进入 selection 函数。
- Morphology：28个 outer-training subjects 做 inner subject LOSO；每个 inner-validation subject 的概率只能来自其余27个 subject 训练的 LR/scaler。聚合 OOF events 的 pooled F1 选288组，然后用全部28个 subjects refit 最佳 LR，再评价一次 outer test。
- 三方法与 LowRescue 统一 tie：高 pooled F1 → 少 FP → 少 rescue → 小 Delta_p → 高 gamma（LowRescue不适用）→ config 按排序字段形成数值/字符串 lexicographic order。使用精确整数分数比较 F1。
- CPU 进程并行仅用于相互独立 outer folds。所有 grids 在 smoke 前锁定。相同 feature / C 的3个 gamma 共用同一次拟合结果，不改变语义。
- 顺序：baseline → common pool → LowRescue → PSED → Recognition → Morphology → comparison → bootstrap/stability。先006 smoke完成整个流程，再正式29折。full校验smoke输入、代码、协议哈希。
- 保存全部候选/最终预测/inner配置得分/inner分割与fit摘要/最终LR参数，以供复核。程序不因单个方法NO-GO中止后续方法。
- Bootstrap：1000次 subject-with-replacement，seed=20260902，三方法共用同一1000×29索引矩阵；逐次从TP/FP/FN重算pooled F1差值。输出均值与percentile95%CI。
- 单subject驱动：删除任一subject后重算现有OOF计数的aggregate delta，不重训；若有删除导致delta<=0则不满足“非单subject驱动”的强GO条件。参数数值端点选择达到15/29折时标记BOUNDARY-SELECTION；报告全部频数，不扩大grid。
- NO-GO：不超过精确Tuned F1；STRONG-GO：delta>=.005、rescueTP>0、rescue precision超过LowRescue且所有leave-one-subject-out aggregate delta>0；正增益但不满足全部强GO要求记WEAK-GO，并明确原因（也涵盖用户未单独指定的“delta>=.005但附加条件失败”情况）。
- BEST_OF_FIRST3 仅作事后描述，最大outer F1；同分按PSED、Recognition、Morphology顺序。不是独立于模型选择的无偏获胜方法评估。历史全缓存审计已影响研究方向，本轮subject-disjoint只约束本次参数/权重选择，不抹去历史选择偏差。
- 报告仅给结果与决定，运行后STOP。
