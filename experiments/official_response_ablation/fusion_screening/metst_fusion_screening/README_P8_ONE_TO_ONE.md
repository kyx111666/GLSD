# P8 第二阶段评价诊断与重新运行

先在现有 Colab 环境运行 probe，只检查 SAMMLV 出错被试 037，
并统一重评 SAMMLV Native；不会进行参数搜索。probe 先做旧结果精确重放，
再记录 L-only / a0=1 / rho=2 / tau=0.05 的旧 raw 计数，
在独立子进程内启用 `event_one_to_one_v1_prediction_order`，
验证该例 raw/full 计数非负、TP+FN=GT、预测事件未改变。

probe PASS 后设 P8_OTO_MODE='full' 运行两个 ME-TST+ 设置。
仍然固定 p=8；G/L/Mean/GM_p8 搜索和 inner full F1 选择与前一版一致。
Native 和所有搜索配置、外层评价都使用统一的一对一协议。
该协议每个预测最多匹配一个 GT，每视频同一 GT 最多用一次；
保留评价器原预测遍历顺序和 IoU 排序/阈值，不是最大二分匹配。
只在当前进程内替换函数，不写 locked evidence 或 ME-TST 源码。

新输出：MyDrive/GLSD_P8_ONE_TO_ONE/。旧计数不再参与新搜索或汇总。
历史结果仅写入 legacy_preflight，不与新结果混用。
旧结果无负 FN 也不代表完全没有重复匹配，需统一重评。

新协议的中断可以通过 P8_OTO_RESUME=新的输出目录 续跑。
不能填 GLSD_P8_PHASE2 的旧目录，也不能用 probe 目录续跑 full。
新输入/代码/runtime/评价器校验不一致时会拒绝续跑。

本地合成回归、兼容 core 测试不能代替真实 Colab 的 037/recognition 回放。
若 probe 未通过，停止 full 搜索并保留错误日志；不得裁剪负 FN 或跳过配置。
完成后主要看 summary_full.csv、paired_comparisons.csv、各数据集 selected_configs.csv。
此轮仍是开发实验，PASS 是程序检查通过，不是证明融合最好。
