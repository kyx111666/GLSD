# 服务器运行说明

这轮是冻结响应上的后处理筛查。服务器不需要重新训练，也不需要原始视频；需要 Python 3.8、封存 ME-TST runner/core、官方响应 dump 和 ME-TST 的评价器代码。

建议配置：Linux x86_64，Python 3.8，4-8 vCPU，16-32 GB RAM，至少 50 GB 可用磁盘。GPU 对这轮不是必需的；如果服务器已经有 NVIDIA GPU，可以使用，但不要为这轮专门提高 GPU 规格。

## 目录布局

将文件准备成如下结构。`GLSD_DATA_ROOT` 对应原来的 `/content/drive/MyDrive`：

```text
/data/glsd/
  GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE/
  ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1/
  GLSD_EVALUATOR_AUDIT/sammlv_20260922T091343_286158Z/
/data/ME-TST/
  training_utils.py
  Utils/
```

其中 `GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE/` 至少需要：

```text
metst_official_glds_full.py
boosting_official_glds_full_LOCKED.py
evidence_bundle_sha256.txt
results_v1/run_manifest.json
results_v1/outer_selected_configs.csv
```

响应 dump 目录需要 29 个 `subject_*.pkl`。评价核验目录需要解压后的 `completion.json`、`contracts.json`、`run_manifest.json`；`completion.json` 必须是 `status=PASS`。

## 环境

优先使用与原 Colab 相同的环境：Python 3.8、NumPy 1.24.4、pandas 1.1.3、SciPy 1.5.4、scikit-learn 0.23.2、PyTorch 2.1.0、timm 1.0.7。后处理本身主要使用 NumPy/pandas/scipy；ME-TST 的 `Utils` 和 `training_utils.py` 必须来自已核验版本。

```bash
python3.8 -m venv /data/venvs/metst38
source /data/venvs/metst38/bin/activate
pip install 'numpy==1.24.4' 'pandas==1.1.3' 'scipy==1.5.4' 'scikit-learn==0.23.2'
```

如果服务器无法安装这些旧版本，先不要开始正式筛查。代码会把运行环境和输入哈希写入 `run_manifest.json`，环境差异应在结果解释中明确记录。

## 运行

把本目录和 `metst_fusion_screening` 目录中的代码放在同一个项目 checkout 后：

```bash
python3.8 official_response_ablation/fusion_screening/reference_residual/run_server_reference_residual.py \
  --data-root /data/glsd \
  --metst-root /data/ME-TST \
  --audit /data/glsd/GLSD_EVALUATOR_AUDIT/sammlv_20260922T091343_286158Z \
  --output /data/glsd/GLSD_REFERENCE_RESIDUAL/screen_sammlv_20260922T100000Z
```

输出目录必须是新的目录。中断后，使用同一个目录加 `--resume`，并保持代码、输入和环境不变。程序会先跑本地关键路径测试，然后运行 29 个被试、5 种评分、19 个阈值以及锁参消融。

## 数据搬运

推荐用 rclone/Google Drive 下载到 `/data/glsd`，或者先在本地整理后用 `rsync` 上传。不要把整个 Drive 复制到服务器；只需要上面的封存证据、29 个 SAMMLV dump、评价核验目录和 ME-TST 代码。搬运后先按阶段0 `run_manifest.json` 中的 SHA256 核对所有文件，再启动筛查。

结果包含 `features.jsonl.gz`、`feature_summary.csv`、`search_counts_*.npz`、`summary_full.csv`、`selected_configs.csv`、`per_subject_counts.csv`、`event_records.jsonl.gz` 和 `paired_comparisons.csv`。完成后把整个输出目录压缩带回，不要只发送 summary，因为需要用事件账本核对 GT 身份。
