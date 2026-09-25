# ME-TST+ GLSD 3.5 generalized-mean phase-1 screen

This package is for the first Colab risk screen only. It reads the locked
ME-TST+ official responses from Drive and never runs a backbone or writes to a
locked evidence directory.

## Protocol

- Settings: ME-TST+ / SAMMLV and ME-TST+ / CAS(ME)3.
- Frozen structure: `a0=2.0`, `rho=1.0`.
- Fusion: `S_p=((G^p+L^p)/2)^(1/p)` for `p=1,2,4,8`; `p=inf` is diagnostic only.
- Tau: `0.1, ..., 0.9`; each fixed p selects its own tau inside nested LOSO.
- Formal phase-1 joint selection: `(p,tau)` with `p in {1,2,4,8}` only.
- Selection uses inner-subject raw counts; reporting uses held-subject full counts
  after the unchanged official recognition/result-synergy path.
- `p=1` is checked against the sealed arithmetic-mean scorer before screening.

## Colab commands

```python
subprocess.run([sys.executable, str(task_dir / "run_generalized_mean_screening.py"),
                "--setting", "both", "--check-inputs"], check=True)
```

The full run writes to a new timestamped directory under
`/content/drive/MyDrive/GLSD_GENERALIZED_MEAN_SCREENING/`.

## Interpretation gate

Do not promote this to a four-setting or full `(a0,rho,p,tau)` experiment until
the two settings have been checked for: FP-dominated gains, systematic loss to
G-only/L-only, repeated p=1 mask equivalence after tau retuning, and whether
the joint winner is stable across outer subjects. `event_mechanism.csv` is
explicitly marked unavailable because the sealed ME-TST hook does not expose a
verified video/event identifier; `candidate_trace.jsonl.gz` contains the
response hash and peak-level trace for audit.
