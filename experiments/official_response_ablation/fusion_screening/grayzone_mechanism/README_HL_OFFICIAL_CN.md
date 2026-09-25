# H+L 官方响应评估入口

`official_response_hl_component_ablation.py` 是独立入口，复用 sealed official response、官方 decoder 和官方 evaluator；原有的 `official_response_component_ablation.py` 及其 SHA-256 锁定包保持不变。

新增分数模式：

- `HL`: `(H+L)/2`
- `HL75`: `0.25H+0.75L`

两者与 `H/G/L/GL` 共享三参考尺度和十个锁定阈值，`rho=2` 固定，因此每个模式都是 30 个配置。`HL75` 是固定公式候选，不是根据 CASME3 外层结果事后挑选的权重。

在已挂载官方 Drive 的环境中运行：

```bash
python official_response_hl_component_ablation.py \
  --setting metst_sammlv \
  --output-root /content/drive/MyDrive/GLSD_HL_OFFICIAL
```

Colab 请直接使用 `HL_OFFICIAL_CASME3_COLAB_CELL.py`。该 cell 会在当前
`/content` runtime 写入一次性 `sitecustomize.py`，兼容官方 evaluator 依赖的
`pandas.DataFrame.append`，并在正式评估前打印 `DataFrame.append available = True`。
这不会修改 Drive 中的 sealed 源码，也不需要重新训练。
上传时即使 Colab 返回带路径或重复后缀的文件名，cell 也会按
`official_response_hl_component_ablation*.py` 自动识别，并打印实际收到的文件名。

正式主表只使用当前 sealed 853-GT CAS(ME)3 结果。SAMM-LV 本地冻结缓存和旧 858-GT CASME3 cache 只能作为候选筛选或迁移诊断，不能替代该官方运行。
