# Frozen Feature Phase-0 Python Code Review

> **Status**: passed_with_blocker  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-02  
> **Script reviewed**: `my_method/feature_relation_feasibility/phase0_feature_tap_audit.py`

## Pass Items

1. ✅ `phase0_feature_tap_audit.py:20-27` 使用项目相对根路径定位四个指定 cache，不依赖用户机器的硬编码绝对路径。
2. ✅ `phase0_feature_tap_audit.py:31-36` 对四个输入逐块计算 SHA-256；结果与前三个 audit 使用的 hash 一致。
3. ✅ `phase0_feature_tap_audit.py:50-84` 分别识别 ME-TST+ `records` 和 BoostingVRME `subject_curves` schema，并实际统计 subjects、videos、GT 与 score time points。
4. ✅ `phase0_feature_tap_audit.py:39-47` 同时检查直接 ndarray 和 list 内第一条 ndarray，确认 BoostingVRME 的 score/emotion 数组字段但没有 hidden tensor。
5. ✅ `phase0_feature_tap_audit.py:106-119` 对本地模型源码、输入 cache 和 weights 执行只读 inventory；五类必需资产均为空。
6. ✅ `phase0_feature_tap_audit.py:91-100` 实际检查 PyTorch、CUDA 与 MPS runtime，而非仅根据包名推断；输出记录 CUDA/MPS 均不可用。
7. ✅ `phase0_feature_tap_audit.py:128-157` 明确写入 `blocked_phase0`、`not_evaluated` 和三项 forbidden substitution，避免将 logits 或 scalar score 伪装成 hidden feature。
8. ✅ `phase0_feature_tap_audit.py:166-174` 将完整 schema、hash、资产状态和 blockers 保存为 JSON，脚本已从项目根目录成功运行。

## Failed / Repaired Items

| # | File:line | Issue | Action | Status |
|---|---|---|---|---|
| 1 | `phase0_feature_tap_audit.py:39-47` | 初版只展示直接 ndarray，BoostingVRME 的 list-of-array 字段未显示 shape | 增加 list 第一条 ndarray 的 shape/dtype 检查 | fixed |
| 2 | `phase0_feature_tap_audit.py:91-100` | 初版只检查 torch module 是否存在 | 增加 torch version、CUDA 和 MPS runtime 检查 | fixed |

## Remaining Risks / Blocker

- 当前没有 aligned hidden feature，因此不能评审或运行 self-similarity probe。
- 官方网络 tap 来自 GitHub 源码结构审计，不是本地 checkpoint smoke forward。
- `pickle.load` 只应读取当前受信任的本地项目 cache；不要对未知来源 pickle 运行此脚本。

## Constraint Direction Review

本脚本不包含优化约束或物理不等式。

## Run Instructions

```bash
cd <workspace-root>/RethinkFuse_reproduction
.venv/bin/python my_method/feature_relation_feasibility/phase0_feature_tap_audit.py
```

## Expected Output

- `my_method/feature_relation_feasibility/outputs/feature_relation_results.json`

## Recommended Next Action

先恢复官方输入 cache、全部 checkpoint 和 Linux/CUDA `mamba_ssm` 环境，并通过单视频 hidden-score alignment smoke test；在此之前不要生成 relation probe 结果。
