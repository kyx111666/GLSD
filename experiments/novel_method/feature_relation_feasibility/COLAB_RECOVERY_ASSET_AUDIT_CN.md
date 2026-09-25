# ME-TST+ × SAMMLV Colab 恢复资产清单

> 审计日期：2026-09-02  
> 范围：只恢复 frozen ME-TST+ × SAMMLV one-video inference；不训练、不处理 BoostingVRME、不跑 CASME3。

## 修正后的结论

上一阶段实验不是“科学上不可行”，而是**没有在错误的 Mac 环境中继续执行**。Google Drive 检索已经证明历史 Colab 资产大量存在；Mac 缺 CUDA、`mamba_ssm` 等不能用于判断历史资产是否存在。

当前尚未宣称 `RECOVERED`，因为 one-video hidden→score 数值对齐还没有在 Colab GPU 中实际执行。也没有给出 scientific NO-GO。

## 资产清单

| 资产 | 状态 | 证据或路径 |
|---|---|---|
| ME-TST 项目代码 | KNOWN PATH FROM CODE/LOG + NEEDS USER/COLAB DRIVE ACCESS | Drive：[`MyDrive/ME-TST_CASME3_SelfTrain/ME-TST_code_backup`](https://drive.google.com/drive/folders/1JTYtQkIDb9gHEKo7bjuzeHu57EiFbrzZ)，含 `network_sf.py`、`main.py`、`train.py` 等；历史日志还记录 `/content/metst_verify/ME-TST` 与 `/content/metst_verify/backup_01984/content/ME-TST`。 |
| 成功复现完整备份 | KNOWN PATH FROM CODE/LOG + NEEDS USER/COLAB DRIVE ACCESS | Drive 根目录：[`ME-TST_0.1984_Success_Backup.zip`](https://drive.google.com/file/d/10Iq7TPy-DXKRY9jQPifWnaBgD6stZDw8/view)，6,452,972,458 bytes；另有 0.1115 备份。Connector 不能直接列 zip 内部成员，因此不能声称其中没有 input cache。 |
| SAMMLV pretrained weights | KNOWN PATH FROM CODE/LOG + NEEDS USER/COLAB DRIVE ACCESS | Drive：[`SAMMLV_4emo`](https://drive.google.com/drive/folders/1GaXZ_6o5lBSoHuv_6z7lKwiHi5WcXrJi)，已列出 29 个 subject checkpoint；目标 [`subject_006.pkl`](https://drive.google.com/file/d/1bL8PlZPqEvrB06unx_G4-mhXMg6zvQwz/view)，67,013,402 bytes。原脚本相对路径：`weights/SAMMLV_4emo/subject_{subject}.pkl`。 |
| SAMMLV input feature cache | KNOWN PATH FROM CODE/LOG；NEEDS USER/COLAB DRIVE ACCESS | 原代码固定为 `cache/ME-TST+/SAMMLV_dataset.pkl`。Drive 文件名搜索未直接命中，但 0.1984 成功备份内部尚未在 Colab 用 `zipfile` 核对。恢复脚本只选择性提取这一成员，不展开整个备份。若备份内也没有，再报告官方公开 cache 作为最小缺件。 |
| compact reference score | PRESENT IN CURRENT WORKSPACE；NEEDS USER/COLAB DRIVE ACCESS | 本机：`caches/me_tst/sammlv_strategy1_outputs.pkl`，2,232,150 bytes，SHA-256 `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569`。Drive 搜索未命中该文件名；运行前只需把这一文件同步/上传到 Drive，或通过参数指定 Colab 可见路径。 |
| 目标 subject/video | PRESENT IN CURRENT WORKSPACE | compact cache 第 1 条为 subject `006`、video `006_1`，reference `score.shape == (1245,)`；subject 006 的 compact 顺序为 `006_1, 006_2, 006_3, 006_5`。 |
| 原 batch/order/stitching | PRESENT IN CURRENT WORKSPACE | `compare_paper_aligned_strategies.py`：SAMMLV `k=30`、`batch_size=32`；先把一个 subject 的全部视频窗口按原顺序 flatten，DataLoader batch 内局部 `i` 每批重置，并用该 `i` 决定 full/half-window stitching。新脚本逐行复刻这一分支。 |
| frozen inference branch | PRESENT IN CURRENT WORKSPACE | `model.eval()` + `torch.no_grad()`；hook `model.spot_pathway`；`fc_spot + sigmoid`。脚本中没有 `model.train()`、`backward()` 或 optimizer。 |
| 已验证环境记录 | KNOWN PATH FROM CODE/LOG + NEEDS USER/COLAB DRIVE ACCESS | [`pip_freeze_metst310_cu128.txt`](https://drive.google.com/file/d/1jRwJVeFLLMR3170GGmcEgvprVK7Wp7us/view) 与 [`conda_list_metst310_cu128.txt`](https://drive.google.com/file/d/1vsqegFd2FJdNDTRTPDgnvVdP61kthYMu/view)：Python 3.10.20、torch 2.8.0+cu128、mamba-ssm 2.3.2.post1、causal-conv1d 1.6.2.post1、timm 1.0.7、einops 0.8.0。 |
| 环境 archive | KNOWN PATH FROM CODE/LOG + NEEDS USER/COLAB DRIVE ACCESS | `MyDrive/ME-TST_CASME3_SelfTrain/runtime_snapshot/metst310_cu128_blackwell_env.tar.gz`（4,831,165,535 bytes）及 `metst_blackwell_cu128_verified.tar.gz`（1,795,806,923 bytes）。脚本优先恢复现有 archive，不在 Mac 补环境，也不主动升级。 |
| notebook | UNKNOWN | Drive 搜索未命中明确的历史 ME-TST notebook；这不影响使用现有代码/环境/备份恢复最小 forward。 |
| `train=False` 命令 | UNKNOWN | 当前检索到的 SAMMLV 日志仍打印 `Training` 并在进入 LOSO 前失败，不能把它当成成功 inference 命令证据。恢复脚本绕开 `train.py`，直接做 frozen forward。 |
| stage2 / inference logs | KNOWN PATH FROM CODE/LOG + NEEDS USER/COLAB DRIVE ACCESS | Drive 根目录存在 `me_tst_01984_infer_log.txt` 和 backuprepo 版本；CASME3 目录还有大量 stage2/inference 日志。本阶段不执行 CASME3。 |

## one-video 恢复实现

生成文件：`ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py`。

它在 Colab 中执行以下顺序：

1. mount `MyDrive`；
2. 定位代码、成功备份、环境 archive、`subject_006.pkl`、SAMMLV input cache 和 compact cache；
3. 若 input cache 只存在于 0.1984 成功 zip，仅选择性提取 `SAMMLV_dataset.pkl`；
4. 解包并使用已有 `metst310_cu128` 环境，不执行 pip 升级；
5. 取 compact cache 中 subject 006 的全部视频顺序，保持 batch=32 的 subject-level flatten；
6. `model.eval()`、`torch.no_grad()` frozen forward；
7. hook `spot_pathway` 的 `[B,30,384]`；
8. 对 hidden 使用与 score 完全相同的 batch-sensitive stitching，得到目标视频 `[T,384]`；
9. 经同一个 `fc_spot + sigmoid` 重建 score，与 `006_1` 的 compact `score[T]` 比较；
10. 保存 hidden `.npy` 和包含要求指标的 JSON。

## 当前最小 blocker

运行 Colab 脚本前仍需完成两项运行时确认：

- 在 0.1984 success zip 中确认并选择性提取 `cache/ME-TST+/SAMMLV_dataset.pkl`；若不在其中，提供官方同名 cache。
- 将当前本机 2.23 MB 的 `sammlv_strategy1_outputs.pkl` 同步到 Google Drive，或在 Colab 中通过 `--compact-cache` 指定可见路径。

这两个问题属于资产定位/可见性，不是 scientific NO-GO，也不是要求重新下载原视频、重提光流或重训练。
