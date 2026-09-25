# Recognition Weak-Candidate Separability Audit

最终状态：**GO-RECOGNITION-SEPARABILITY**

## 数据与语义核验

- LowRescue weak pool：45 条；true=2，weak FP=43。
- class order：`['negative', 'positive', 'surprise', 'others', 'neutral']`；neutral id=4。
- 类别语义由现有 ME-TST 评估代码与 cache strategy 元数据交叉确认；logits 仅做 softmax，没有 model forward。
- GO 预设规则：ROC-AUC ≥ 0.60 且 PR-AUC ≥ 1.5 × prevalence。

## Evidence 结果

| Evidence | ROC-AUC | PR-AUC | Prevalence | true mean/median | FP mean/median | pass |
|---|---:|---:|---:|---:|---:|---|
| R1_center | 0.732558 | 0.118056 | 0.044444 | 0.424167/0.424167 | 0.244833/0.050731 | True |
| R1_mean | 0.813953 | 0.191667 | 0.044444 | 0.466945/0.466945 | 0.219908/0.140487 | True |
| R1_max | 0.569767 | 0.073935 | 0.044444 | 0.815996/0.815996 | 0.563913/0.665714 | False |
| R2_center | 0.755814 | 0.130252 | 0.044444 | 0.423384/0.423384 | 0.226925/0.037752 | True |
| R2_mean | 0.825581 | 0.233333 | 0.044444 | 0.450344/0.450344 | 0.204763/0.132012 | True |
| R2_max | 0.581395 | 0.075397 | 0.044444 | 0.737239/0.737239 | 0.550729/0.662434 | False |
| R3_center | 0.197674 | 0.041022 | 0.044444 | 0.451615/0.451615 | 0.745438/0.914423 | False |
| R3_mean | 0.069767 | 0.036005 | 0.044444 | 0.476267/0.476267 | 0.780687/0.841660 | False |
| R3_max | 0.220930 | 0.041632 | 0.044444 | 0.976239/0.976239 | 0.987407/0.998419 | False |

## True rescue cases

- 007/007_6 peak=180，matched_gt=1：`{"R1_center": 0.1987087164021355, "R1_max": 0.8522422633080702, "R1_mean": 0.3887106166326888, "R2_center": 0.19722754144053958, "R2_max": 0.695433270189783, "R2_mean": 0.35589993127944647, "R3_center": 0.604063742157325, "R3_max": 0.9567326200387133, "R3_mean": 0.48176281458674564}`
- 011/011_6 peak=379，matched_gt=3：`{"R1_center": 0.6496252872804038, "R1_max": 0.7797487566152734, "R1_mean": 0.5451796561455824, "R2_center": 0.6495410456195284, "R2_max": 0.7790446609160083, "R2_mean": 0.5447889155485801, "R3_center": 0.29916633289993216, "R3_max": 0.9957452195742261, "R3_mean": 0.47077175414785005}`

## 解释边界

该结果只回答现有 recognition logits 对固定 weak pool 是否具有排序分离性；未调 rescue threshold、未训练分类器、未实现联合 decoder，也未运行 CASME3。

特别注意：正例仅 2 条，因此 AUC/AP 方差可能很大。本次 GO 是 feasibility gate 通过，不是已证明可泛化的方法收益，也不是论文结论。
