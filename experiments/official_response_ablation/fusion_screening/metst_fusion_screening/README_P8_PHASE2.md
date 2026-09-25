# 在现有 Colab notebook 接着做第二阶段

环境仍在时，无须重新执行安装依赖、生成 response 或第一阶段实验。
在原 notebook 最后追加 `Colab_P8_第二阶段接续.ipynb` 中的两个代码单元格。
第一格上传 `metst_p8_phase2.zip` 并解压；第二格自动测试、检查输入、
复现 sealed Native/旧 GLSD，再搜索两个 ME-TST+ 数据集。

## 本轮协议

- 仅 ME-TST+ / SAMMLV 和 ME-TST+ / CAS(ME)3。
- G-only：a0 × tau，共 57 配置；rho 对 G 无效，不重复搜索。
- L-only / Mean / GM_p8：a0 × rho × tau，各 171 配置。
- a0=[1,1.5,2]，rho=[0.5,1,2]，tau=[0.05,0.10,...,0.95]。
- GM_p8 固定 p=8；Mean 固定 p=1，作为融合形式对照。
- 每个 outer subject 的全部选择指标都只用其他 subjects 的计数。
- 主结果以内层 pooled official full F1 选择，依次按 precision、较少 FP、固定网格序号打破并列。
- 同一个新生成的 raw/full 计数张量也生成 inner raw 选择对照，不另调方法。
- 外层汇总用 official full counts；输出 raw/full 均保留。
- fixed responses 和 k 继承官方协议；这里验证的是后处理选择隔离，不是 backbone 训练隔离。
- p=8 来自已看过的第一阶段结果，所以本轮仍是开发实验。不得称独立确认。
- 有效搜索空间不同，不称 matched-budget；先冻结本轮网格，不在运行中动态扩展。

## 输出和续跑

新结果在 `/content/drive/MyDrive/GLSD_P8_PHASE2/p8_时间戳/`。
`summary_full.csv` 是主比较；每个数据集的 `summary_all.csv` 包含 raw 选择对照。
还有 paired comparisons、per-subject counts、selected configs、inner search、
selection frequency（含边界）、selected predictions、candidate trace、input/code hashes。
candidate trace 使用 subject/response hash/peak 标识，不冒充事件匹配诊断。

每完成一个 subject 保存一次该方法的 raw/full 张量。若中断，把第二个单元格的
`P8_RESUME = None` 改成之前的完整第二阶段输出路径后重跑；环境重置后需恢复
原 ME-TST 环境并重新上传同一个 zip。续跑要求脚本、输入和 runtime 版本一致，
第一阶段张量不能拿来续跑第二阶段。已完成的 subject 计数不会重新搜索。

程序 PASS 只表示完成；融合是否超过单模块看 `decision.json` 和 F1 表。
更换 inner 选择指标、结构网格、tau 网格都可能影响结果，所以不能把本轮与
第一阶段的差值全部归因于融合公式。以本轮四方法间的比较为主。

本地只验证算子、隔离、缓存/续跑和 synthetic adapter 集成；真实 locked
ME-TST/Drive 回放在 Colab 启动实验时执行，不将本地测试当成性能证据。
