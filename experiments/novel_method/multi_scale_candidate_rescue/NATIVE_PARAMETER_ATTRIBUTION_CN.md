# Tuned Native Parameter Attribution

## 结论

四个单因素中贡献最大的是 **Threshold-only**，其相对 Author Native 的 pooled outer F1 变化为 +0.009944。这是归因结论，不构成新方法。

## 协议

- SAMMLV，29-fold outer subject LOSO；每折仅用其余 subjects 的 pooled F1 选参。
- tie-break：fewer FP → fewer rescue events → 固定 lexicographic config index。
- 原 grid、候选生成和 evaluator 均未改变；outer test GT 未参与选参。

## Pooled outer 结果

| 方法 | TP | FP | FN | Precision | Recall | F1 | ΔF1 vs Author |
|---|---:|---:|---:|---:|---:|---:|---:|
| Author Native | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | +0.000000 |
| Smooth-only | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | +0.000000 |
| Threshold-only | 49 | 145 | 110 | 0.252577 | 0.308176 | 0.277620 | +0.009944 |
| Distance-only | 53 | 180 | 106 | 0.227468 | 0.333333 | 0.270408 | +0.002731 |
| Boundary-only | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | +0.000000 |
| Full Tuned Native | 49 | 143 | 110 | 0.255208 | 0.308176 | 0.279202 | +0.011526 |

## 参数选择频率

- Smooth-only: `{"c_b": {"1.0": 29}, "c_d": {"1.0": 29}, "c_s": {"2.0": 29}, "p": {"0.55": 29}}`
- Threshold-only: `{"c_b": {"1.0": 29}, "c_d": {"1.0": 29}, "c_s": {"2.0": 29}, "p": {"0.65": 29}}`
- Distance-only: `{"c_b": {"1.0": 29}, "c_d": {"1.25": 29}, "c_s": {"2.0": 29}, "p": {"0.55": 29}}`
- Boundary-only: `{"c_b": {"1.0": 1, "1.25": 28}, "c_d": {"1.0": 29}, "c_s": {"2.0": 29}, "p": {"0.55": 29}}`
- Full Tuned Native: `{"c_b": {"1.0": 1, "1.25": 28}, "c_d": {"1.25": 29}, "c_s": {"2.0": 29}, "p": {"0.65": 29}}`

## Full Tuned Native 逐折配置

完整 29-fold 配置及 outer TP/FP/FN/F1 见 `outputs/tuned_native_selected_configs.csv`。
