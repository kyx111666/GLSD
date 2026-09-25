# CAS(ME)3 固定 beta=1 验证

本轮将 SAMMLV 上得到的简单候选 `S=max(G,L)` 带到 ME-TST+ / CAS(ME)3；CAS(ME)3 自己在每个训练折内选择 a0、rho、tau。beta=1 固定，不把 SAMMLV 的 (2,4,.8) 直接搬过来。

在已配置好的 ME-TST+ Colab 中，新建 cell，粘贴 `CASME3_FIXED_MAX_COLAB_CELL.py`，运行后上传 `metst_casme3_fixed_max.zip`。不用运行旧 cell。输出目录是 `GLSD_ONE_TO_ONE_FULL_TUNING/oto_casme3_max_<UTC>/`，完成后回传同名 zip。

搜索：a0={0.5,1,1.5,2,2.5,3,4}，rho={0.5,1,1.5,2,3,4}，tau=0.05:0.05:0.95；L 支持尺度和 gamma 保持原设置；一对一评价协议保持 `event_one_to_one_v1_prediction_order`。固定 beta=1 的方法网格为 798 个配置，合计约 7315 个方法配置（G=133、L=798、EqualMean=798、WeightedMean=4788、DominantEvidence=798）。

所有方法用 CAS(ME)3 其他被试 pooled raw TP/FP/FN 选择，最终用留出被试 full 评价。会先做原 Native/GLSD 回放和一对一评价检查。`PASS` 只表示代码执行完成，不表示融合胜出。

重点看 `one_to_one_summary_full.csv`：CAS(ME)3 的 Native、G、L、EqualMean、WeightedMean、固定 beta=1。只有固定 beta=1 在 CAS(ME)3 也超过同范围 G、L 及均值，才有理由继续做正式跨数据集结果；不要根据 SAMMLV 的结果事先假设会成功。
