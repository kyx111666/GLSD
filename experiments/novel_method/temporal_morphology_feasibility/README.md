# Temporal Morphology Feasibility Audit

这是 frozen spotting curve 上的诊断实验，不是正式 decoder。

运行：

```bash
cd <workspace-root>/RethinkFuse_reproduction
.venv/bin/python my_method/temporal_morphology_feasibility/run_morphology_audit.py
```

输入为 `caches/me_tst/` 和 `caches/boostingvrme/` 下的四个只读 cache。结果写入：

```text
my_method/temporal_morphology_feasibility/outputs/temporal_morphology_results.json
```

依赖版本见项目 `environment/requirements-cache-reproduction.txt`。
