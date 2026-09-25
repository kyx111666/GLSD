# CASME3 Raw Pre-Stitch Source Preparation

## 当前阶段

本目录只准备 CASME3 raw pre-stitch instrumentation 和后续 Selective Veto validation 代码。当前静态检查不会导入 CUDA、实例化模型或运行 backbone。

冻结 geometry 来自原始源码：

- `main.py`: `dataset_name="CASME_3"`, `k=50`, `batch_size=256`, `frame_skip=1`。
- `train.py`: stitching 使用 `k//2`，故由源码推导 `stride=25`。
- `train.py`: `i==0` 是 DataLoader batch 内索引；batch 首 window 写完整 50，其余写后 25。
- checkpoint mapping: `weights/CASME_3_4emo/subject_{subject}.pkl`。
- 原始输入由 `CASME_3_dataset_1.pkl` 至 `CASME_3_dataset_5.pkl` 顺序拼接。

Exporter 额外使用 verified compact CASME3 cache 作为已经完成 463→462 skip 后的唯一有身份 alignment manifest。它要求 SHA-256：

`9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda`

该 cache 提供 94 subjects / 462 videos 的原始 `subject`, `subject_index`, `video`, `video_index` 顺序；feature-cache 每个视频的 window 数还必须满足：

`(num_windows + 1) * 25 == verified stitched length`

原 `skipped_manifest.csv` 也必须显式提供并写入最终 provenance。缺少任一输入时，preflight 在 CUDA 初始化前停止。

## 文件

- `export_raw_prestitch_casme3_full.py`: smoke/full GPU exporter，逐 subject 单次 checkpoint load、一次完整 subject traversal。
- `validate_selective_veto_casme3.py`: 后续 CPU validation；完整 raw manifest 未达到 462/462 exact PASS 时拒绝运行。

## 1. 静态 preflight（不运行 GPU）

```bash
.venv/bin/python my_method/window_consensus/export_raw_prestitch_casme3_full.py \
  --project-root /path/to/ME-TST-main \
  --input-cache-dir /path/to/ME-TST-main/cache/ME-TST \
  --alignment-cache /path/to/casme3_strategy1_outputs.pkl \
  --skip-manifest /path/to/ME-TST-main/skipped_manifest.csv \
  --weights-dir /path/to/ME-TST-main/weights/CASME_3_4emo \
  --output-dir /path/to/raw_prestitch_casme3 \
  --static-preflight
```

必须得到 `STATIC-PREFLIGHT-PASS` 才能进入 smoke。

## 2. One-video smoke

Smoke 仍完整 forward 目标 subject，以保留 subject-level flatten order 和 batch-local `index_within_batch`；只保存指定视频。

```bash
.venv/bin/python my_method/window_consensus/export_raw_prestitch_casme3_full.py \
  --project-root /path/to/ME-TST-main \
  --input-cache-dir /path/to/ME-TST-main/cache/ME-TST \
  --alignment-cache /path/to/casme3_strategy1_outputs.pkl \
  --skip-manifest /path/to/ME-TST-main/skipped_manifest.csv \
  --weights-dir /path/to/ME-TST-main/weights/CASME_3_4emo \
  --output-dir /path/to/raw_prestitch_casme3_smoke \
  --subject 1 --video a --smoke-only
```

只有 score/logits 均 `array_equal=true` 才输出：

`CASME3-RAW-WINDOW-SMOKE-PASS`

Smoke 不通过时禁止 full run。

## 3. Full GPU export

Full run 强制要求上一节生成且 exact PASS 的 `smoke_manifest.json`。没有该文件、状态不是
`CASME3-RAW-WINDOW-SMOKE-PASS`、source/alignment hash 改变，exporter 都会在 CUDA 初始化前拒绝 full run。

```bash
.venv/bin/python my_method/window_consensus/export_raw_prestitch_casme3_full.py \
  --project-root /path/to/ME-TST-main \
  --input-cache-dir /path/to/ME-TST-main/cache/ME-TST \
  --alignment-cache /path/to/casme3_strategy1_outputs.pkl \
  --skip-manifest /path/to/ME-TST-main/skipped_manifest.csv \
  --weights-dir /path/to/ME-TST-main/weights/CASME_3_4emo \
  --smoke-manifest /path/to/raw_prestitch_casme3_smoke/smoke_manifest.json \
  --output-dir /path/to/raw_prestitch_casme3
```

预期输出：462 个 video NPZ 加 `full_manifest.json`。只有 462/462 score/logits exact reconstruction PASS 才输出：

`CASME3-RAW-WINDOW-SOURCE-READY`

每个 NPZ 保存：subject/video、global/subject video index、subject flat window index、batch index、batch-local index、video window id、write 前 framecount、global mapping、raw score/logits 以及 same-run fresh stitched score/logits。

## 4. Selective Veto validation（本阶段禁止运行）

只有 full manifest 已为 `CASME3-RAW-WINDOW-SOURCE-READY` 后才能运行：

```bash
.venv/bin/python my_method/window_consensus/validate_selective_veto_casme3.py \
  --raw-dir /path/to/raw_prestitch_casme3
```

脚本包含且只包含：

- Frozen Cross-Dataset: `delta=0.20`, `tau=0.80`，CASME3 零调参。
- Nested Same-Grid Calibration: `delta={0.05,0.10,0.15,0.20}` × `tau={0.20,0.40,0.60,0.80}`，共 16 configs。
- Final Strong Native anchor gate: `124/1148/734`, F1=`0.116432`。

不含 recognition、LR、aggregation replacement、rescue、Native decoder 修改或 grid expansion。

## 本机 2026-09-06 静态发现

- 原源码存在于 `<source-root>/ME-TST-main`，`train.py/main.py/network_sf.py` 可读。
- `train.py` SHA-256：`fca74a096f1e8268005990e4dce443ddaeb0736f84b21b4448dffe3a9c2c6852`；实际测试分支确认 `with torch.no_grad()`、`i==0` 以及两处 `k//2` 写入。
- `main.py` SHA-256：`030671c84556a894b38185f7caeee1d5e75d321e11e8dcb028758bc354553374`；实际 CASME3 分支确认 `k=50`、`batch_size=256`、`frame_skip=1`。
- Verified compact alignment cache：94 subjects / 462 videos，顺序与 per-subject `video_index` 全部一致。
- Static cache inspection 逐 part 加载并立即释放，不把全量 tensor 同时保留在内存。
- 当前原 feature-cache 目录只有 part 1–4：90+82+85+92=349 videos、39,622 windows；part 5（剩余 113 videos）缺失。现有四段的每个 video shape 均与 compact alignment 中 `(expected_windows,10,50)` 一致。
- `weights/CASME_3_4emo/subject_*.pkl`：0/94。
- `skipped_manifest.csv`：未找到。
- Final Strong Native fold artifacts：94 configs / 94 outer subjects，汇总 `124/1148/734`、F1=`0.1164319249`，subject mapping exact。

因此当前本机不能执行 smoke/full GPU dump。需要在 GPU 环境补齐原 part 5、94 个 checkpoint 和原 skip manifest 后重新运行 static preflight。

当前准备状态：

`CASME3-EXPORTER-PREPARATION-BLOCKED`

该状态只表示执行资产不完整；exporter 与 validation 脚本已经完成静态准备。没有运行 GPU、backbone 或 Selective Veto validation。
