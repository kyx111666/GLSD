# ME-TST+ GLSD 3.5 generalized-mean phase-1 screen

这是第一阶段风险筛查包。它只读取 Drive 中已锁定的 ME-TST+ official
response，不运行 backbone、不重新生成 response，也不写入 locked evidence。

实验固定 `a0=2.0, rho=1.0`，在 nested LOSO 内分别为 `p=1,2,4,8` 选择
`tau=0.1,...,0.9`；`p=inf` 只做诊断，不参与正式联合选择。正式联合选择
只在 `(p,tau)` 上进行，使用内层 subject 的 raw counts；外层 subject 只用于
最终 full counts（包含原官方 recognition/result-synergy）。

`p=1` 会在运行开始时逐阈值复现 sealed arithmetic-mean scorer。结果目录会
包含 screening summary、per-subject counts、selection/search tables、p 频率、
score/ranking/mask/disagreement diagnostics、paired bootstrap，以及压缩的
selected predictions/candidate trace。由于 sealed ME-TST hook 没有暴露可验证的
video/event ID，`event_mechanism.csv` 会明确标记不可用，不会伪造 event 级结论。

Colab 中上传本包后运行：

```python
subprocess.run([sys.executable, str(task_dir / "run_generalized_mean_screening.py"),
                "--setting", "both", "--check-inputs"], check=True)
```
