# Frozen Temporal Feature Relation Audit

当前目录只完成 Phase-0 readiness audit。关系实验因缺少 aligned hidden feature、输入 cache、checkpoint 与兼容 runtime 而未运行。

复现 Phase 0：

```bash
cd <workspace-root>/RethinkFuse_reproduction
.venv/bin/python my_method/feature_relation_feasibility/phase0_feature_tap_audit.py
```

输出：

- `outputs/feature_relation_results.json`
- `FEATURE_TAP_AUDIT_CN.md`
- `TEMPORAL_FEATURE_RELATION_AUDIT_CN.md`

不要把 `scientific_verdict: not_evaluated` 改写成 relation hypothesis 的 NO-GO。

