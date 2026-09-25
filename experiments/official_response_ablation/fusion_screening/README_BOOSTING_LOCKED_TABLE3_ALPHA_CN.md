# BoostingVRME Table 3 同协议消融

这个 runner 复用 BoostingVRME Table 1 的 GLSD-90 `outer_selected_configs.csv`，只改变

`S_alpha = alpha * G + (1 - alpha) * L`。

`alpha=0.5` 会先按被试与 Table 1 的 `per_subject_counts.csv` 做完整对齐；只要有一个被试不一致，运行立即失败，不会生成可用于 Table 3 的 full 结果。

## Colab 运行

1. 使用原 BoostingVRME Colab 环境，确保 `/content/BoostingVRME` 已初始化，且 Drive 已挂载。
2. 运行 `BOOSTING_LOCKED_TABLE3_ALPHA_COLAB_CELL.py`，第一次保持 `PROBE_ONLY = True`。
3. 先运行 `SETTING = "boosting_sammlv"`，通过后再运行 `SETTING = "boosting_casme3"`。
4. 两次 probe 都显示 `alpha_0p5_alignment.json` 的 `mismatches: []` 后，将 `PROBE_ONLY = False`，使用相同 setting 完整运行。
5. 只读取对应输出目录的 `alpha_summary.csv`、`alpha_per_subject_counts.csv` 和 `protocol.json`。

验收条件：

- `completed=true`；
- `fixed_source` 为 `Table 1 GLSD-90 outer_selected_configs.csv`；
- `selection` 明确为每个被试复用 Table 1 外层配置；
- `alpha=0.5` 的逐被试 `TP/FP/FN` 与 Table 1 完全一致；
- raw/full 的 `TP+FN` 等于数据集 GT 总数；
- Table 3 的 `Full` 才能填 alpha=0.5，`w/o L` 填 alpha=1，`w/o G` 填 alpha=0。

BoostingVRME 的 Table 1 与 ME-TST+ 的响应缓存、锁定配置和 runner 不同，不能把 ME-TST+ 的 cell 或旧 `EqualMean` 结果直接套用。当前 ZIP 已经包含两个数据集专用的官方 Table 1 runner：`boosting_official_glds_full.py` 和 `boosting_official_glds_full_casme3.py`，因此不要求把这两个文件额外放到 `/content/BoostingVRME` 根目录。

上传时使用带版本后缀的 `boosting_locked_table3_alpha_v3.zip`。ZIP 内应恰好有四个文件：两个 alpha runner/依赖文件，以及两个数据集专用官方 runner。`BOOSTING_LOCKED_TABLE3_ALPHA_COLAB_CELL.py` 只复制到 Colab 代码单元，不要再放回 ZIP。

runner 启动时会自动恢复官方代码需要的 `pandas.DataFrame.append` 兼容接口，因此不需要手动修改 `training_utils.py` 或官方评估器。

成功启动时日志第一行必须是 `LOCKED_ALPHA_BOOSTING_RUNNER = v3_pandas_append_compat`。如果没有这行，说明 Colab 仍执行的是旧代码单元或旧 ZIP。
