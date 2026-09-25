# SAMMLV Raw Pre-Stitch Window Dump Instrumentation

## 当前状态

`READY-FOR-GPU-EXECUTION`

本目录只增加 frozen inference instrumentation 和原 stitching 重建检查，不实现 Window Consensus，不改变 backbone、权重、batch size、window length、stride、window order 或 decoder。

原 `train.py` 没有被修改，因此不需要备份或 patch diff。

## 文件

- `export_raw_prestitch_sammlv.py`：在 `model(x)` 之后、任何 stitching/overwrite/argmax 之前复制 `yhat` 和 `yhat1`，同时执行并保存原 batch-sensitive stitching 结果。
- `reconstruct_original_stitch.py`：仅根据 raw dump 中的窗口输出和 batch metadata 重建原 `result_video`，并检查逐元素完全一致性。

## 冻结约束

```text
dataset       = SAMMLV
train         = False
batch_size    = 32
window_length = 30
stride        = 15
model.eval()
torch.no_grad()
```

Exporter 总是先按目标 subject 的全部视频原顺序 flatten 和分 batch，再从中保存指定视频。因此 `batch_index` 和 `index_within_batch` 保留的是原 subject-level DataLoader 语义；不会因为只导出一个视频而人为重置 batch。

每个 window 保存：

```text
subject
video
batch_index
index_within_batch
video_window_id
framecount_before_write
global_start
global_indices[30]
raw_window_score[30]
raw_window_logits[30,5]
```

另外保存同一次 forward 产生的：

```text
original_fresh_result_video[T]
original_fresh_logits_video[T,5]
```

## 本机静态检查

此命令不加载 torch、不 forward backbone：

```bash
python my_method/window_consensus/reconstruct_original_stitch.py --self-test
```

预期状态：

```text
RAW-WINDOW-INSTRUMENTATION-READY
```

该状态只表示 reconstruction 代码在合成输入上通过，不代表真实 GPU smoke 已执行。

## Colab / GPU one-video smoke

在此前通过验证的原 ME-TST+ CUDA/Mamba 环境中，从项目根目录运行：

```bash
python my_method/window_consensus/export_raw_prestitch_sammlv.py \
  --project-root /content/metst_01984_exact \
  --input-cache /content/metst_01984_exact/cache/ME-TST+/SAMMLV_dataset.pkl \
  --weights-dir "/content/drive/MyDrive/ME-TST_复现结果备份/weights/SAMMLV_4emo" \
  --output-dir "/content/drive/MyDrive/ME-TST_复现结果备份/raw_prestitch_smoke" \
  --subject 006 \
  --video 006_1
```

然后仅用 dump 重建：

```bash
python my_method/window_consensus/reconstruct_original_stitch.py \
  --input "/content/drive/MyDrive/ME-TST_复现结果备份/raw_prestitch_smoke/006_006_1_raw_prestitch.npz" \
  --output-json "/content/drive/MyDrive/ME-TST_复现结果备份/raw_prestitch_smoke/006_006_1_reconstruction.json"
```

通过条件同时要求 score 和 logits：

```text
shape_match = true
max_abs_error = 0
mean_abs_error = 0
np_allclose(rtol=0, atol=0) = true
array_equal = true
Pearson = 1（浮点显示应为 1 或机器精度意义下等价于 1）
```

通过时输出：

`RAW-WINDOW-INSTRUMENTATION-READY`

否则输出非零退出码和：

`RAW-WINDOW-DUMP-EQUIVALENCE-FAIL`

失败后应立即停止，不得运行 Window Consensus 或其它聚合实验。

## 输出内容

Exporter 生成：

```text
{subject}_{video}_raw_prestitch.npz
{subject}_{video}_raw_prestitch_manifest.json
```

Manifest 记录 input cache、checkpoint、输出 dump 的 SHA-256，以及 source commit、CUDA/Python/PyTorch 环境和冻结几何参数。

## 当前阶段禁止项

- 不在无 CUDA/Mamba 的机器上伪装完成真实 smoke；
- 不运行 full SAMMLV dump；
- 不运行 Window Consensus、Mean aggregation 或 center weighting；
- 不修改原 stitching；
- 不训练或更新权重。
