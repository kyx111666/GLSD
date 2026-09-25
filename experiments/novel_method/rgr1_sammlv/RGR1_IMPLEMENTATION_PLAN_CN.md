# RGR-1 SAMMLV 实现计划（运行前冻结）

## Code audit

| 对象 | 已确认实现/资产 | 结论 |
|---|---|---|
| Author Native | `multi_scale_candidate_rescue/run_mscr_nested_loso.py:108-114` | 既有实现可复用；已验证输出 53/184/106 |
| Tuned Native | 同文件 `tuned_native_decode`、81-config grid | 复用，不重写定义 |
| Nested LOSO | `run_mscr_posthoc_diagnostic.py:select_pooled_config` | outer-training-only pooled F1 选择可复用 |
| Frozen cache | `caches/me_tst/sammlv_strategy1_outputs.pkl` | 29 subjects / 79 videos / 159 GT；score/logits/emotion 对齐 |
| Recognition | 每条 logits 为 `[T,5]` | softmax axis=1；class 4=neutral；顺序 negative/positive/surprise/others/neutral |
| Evaluation | 既有 `evaluate_decoding` 与 `paper_metrics.py` recognition 定义 | spotting 使用已复现 baseline 的同一匹配器；recognition 按原项目 majority emotion 与 3emo-without-others F1 |
| R2-mean | `run_native_parameter_recognition_audit.py` | 固定 `max(non-neutral softmax)` 后在 candidate interval 内取 mean |
| gamma grid | 上一轮 `run_rgr_nested_loso.py` | 已存在 `{0.25,0.35,0.45,0.55,0.65}`，按任务书原样复用 |
| Candidate schema | 既有 event 为 onset/peak/offset/source | 输出 adapter 附加 emotion、emotion_id、R2 confidence；不修改解码几何 |

当前目录未检测到可用 Git repository，因此 provenance 中 `git_commit=null`，不能伪造 commit。

## 冻结实现选择

1. 每折 Stage A 先在另外 28 subjects 上以 pooled Spotting F1 选择 81 个 Tuned Native 配置之一；held-out subject 不参与。
2. Stage B 固定 Stage-A 配置，仅搜索 4×5 个 `(Delta_p, gamma_R)`；tie-break 严格为 higher F1、smaller Delta_p、higher gamma_R。
3. `W=P_low-P_high` 使用 peak identity；这是现有 pipeline 的 candidate identity，boundary 继续使用该折 Stage-A `c_b`。
4. Low-threshold Native 使用该折 RGR 选中的同一个 Delta_p，直接接收完整 low-threshold decoder，保证与 RGR 具有相同 weak-threshold source。
5. Recognition F1 主列固定为项目的 `recognition_f1_score_3emo_wo_others`；STRS=`raw Spotting F1 × Recognition F1`。同时在 JSON 保存 4-emotion recognition，避免口径丢失。
6. 先运行 subject 006 smoke test到独立目录；所有 assertions 通过后才允许运行 29 folds。
7. 正式输出仅写入 `results/rgr1_sammlv_nested/`，不覆盖任何旧 baseline 或上一轮 RGR 产物。

## 预期文件

- `run_rgr1_nested_loso.py`
- `results/rgr1_sammlv_nested/report.json`
- `results/rgr1_sammlv_nested/summary.csv`
- `results/rgr1_sammlv_nested/outer_fold_metrics.csv`
- `results/rgr1_sammlv_nested/selected_config_frequency.csv`
- `results/rgr1_sammlv_nested/rescued_candidates.csv`
- `results/rgr1_sammlv_nested/weak_candidate_diagnostics.csv`

