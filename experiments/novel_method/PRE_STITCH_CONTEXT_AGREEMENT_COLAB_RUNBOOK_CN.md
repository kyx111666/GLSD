# Pre-Stitch Context Agreement：Colab 运行说明

## 当前状态

`PASS-DECODER-EQUIVALENT` 已确认，但真实 Context Agreement 科学审计尚未运行。

现有 `SAMMLV_FULL/subject_xxx/*_frozen_hidden_score.npz` 只有 stitched score、hidden 和 valid mask，不能反推出 overlapping raw window observations。Google Drive 搜索也没有发现现成的 `SAMMLV_PRE_STITCH_CONTEXT_CACHE`。因此必须在已验证的 Colab 环境中重新做一次 **frozen inference-only raw-score export**。

本流程不会训练模型，也不会覆盖 historical compact cache。

## 需要先上传到 Google Drive 根目录

从本机项目上传以下两个文件：

1. `pre_stitch_context_feasibility/export_sammlv_pre_stitch_context.py`
2. `pre_stitch_context_feasibility/run_context_agreement_scientific_audit.py`

上传后的预期路径：

```text
/content/drive/MyDrive/export_sammlv_pre_stitch_context.py
/content/drive/MyDrive/run_context_agreement_scientific_audit.py
```

## Cell 1：确认 GPU 并挂载 Drive

```python
import subprocess
from google.colab import drive

subprocess.run(['nvidia-smi'], check=True)
drive.mount('/content/drive', force_remount=True)
```

## Cell 2：使用已确认的 Drive 资产

```python
from pathlib import Path

PROJECT = Path('/content/drive/MyDrive/ME-TST_CASME3_SelfTrain/ME-TST_code_backup')
INPUT_CACHE = Path('/content/drive/MyDrive/SAMMLV_dataset.pkl')
WEIGHTS = Path('/content/drive/MyDrive/ME-TST_复现结果备份/weights/SAMMLV_4emo')
GT_REFERENCE = Path('/content/drive/MyDrive/sammlv_strategy1_outputs.pkl')
EXPORT_SCRIPT = Path('/content/drive/MyDrive/export_sammlv_pre_stitch_context.py')
AUDIT_SCRIPT = Path('/content/drive/MyDrive/run_context_agreement_scientific_audit.py')
OUTPUT = Path('/content/drive/MyDrive/ME_TST_FRESH_FROZEN_OUTPUT/SAMMLV_PRE_STITCH_CONTEXT')

required = [
    PROJECT / 'network_sf.py',
    INPUT_CACHE,
    WEIGHTS,
    GT_REFERENCE,
    EXPORT_SCRIPT,
    AUDIT_SCRIPT,
]

for path in required:
    print(path, '->', path.exists())

assert all(path.exists() for path in required), '上面至少一个路径不存在；只修正对应路径，不重建环境。'
```

## Cell 3：建立独立 Python 3.10 环境

> 仅在原环境 archive 确认不存在时运行。不使用 Colab 当前的 Python 3.13。

```python
import os
import shutil
import subprocess
import sys
from pathlib import Path

ENV_ROOT = Path('/content/metst310_cu128')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
UV = shutil.which('uv')
assert UV is not None, 'uv 安装后未找到可执行文件'
subprocess.run([UV, 'python', 'install', '3.10'], check=True)
subprocess.run([
    UV, 'venv', '--python', '3.10', '--seed', str(ENV_ROOT)
], check=True)

PYTHON = ENV_ROOT / 'bin/python'
print('PYTHON =', PYTHON)
subprocess.run([str(PYTHON), '--version'], check=True)
```

## Cell 4：安装原已验证版本

```python
import os
import subprocess

# 先移除已经与错误 PyTorch ABI 绑定的 CUDA 扩展。
subprocess.run([
    str(PYTHON), '-m', 'pip', 'uninstall', '-y',
    'mamba-ssm', 'causal-conv1d',
], check=False)

# 在任何 CUDA 扩展之前锁定 torch/vision/audio，并强制替换错误的新版本。
subprocess.run([
    str(PYTHON), '-m', 'pip', 'install',
    '--force-reinstall', '--no-cache-dir',
    'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0',
    '--index-url', 'https://download.pytorch.org/whl/cu128',
], check=True)

version_check = subprocess.run([
    str(PYTHON), '-c',
    "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.__version__.startswith('2.8.0+cu128')",
], text=True, capture_output=True)
print(version_check.stdout)
print(version_check.stderr)
assert version_check.returncode == 0, 'PyTorch 未成功锁定到 2.8.0+cu128'

subprocess.run([
    str(PYTHON), '-m', 'pip', 'install',
    'packaging', 'wheel', 'setuptools', 'ninja',
    'numpy', 'scipy', 'scikit-learn',
    'timm==1.0.7', 'einops==0.8.0',
], check=True)

build_env = os.environ.copy()
build_env['MAX_JOBS'] = '4'
subprocess.run([
    str(PYTHON), '-m', 'pip', 'install',
    '--force-reinstall', '--no-cache-dir', '--no-deps',
    'causal-conv1d==1.6.2.post1', '--no-build-isolation',
], check=True, env=build_env)
subprocess.run([
    str(PYTHON), '-m', 'pip', 'install',
    '--force-reinstall', '--no-cache-dir', '--no-deps',
    'mamba-ssm==2.3.2.post1', '--no-build-isolation',
], check=True, env=build_env)
```

## Cell 5：环境与模型 import smoke test

```python
import subprocess

checks = [
    ('python', "import sys; print(sys.version)"),
    ('torch', "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"),
    ('timm', "import timm; print(timm.__version__)"),
    ('einops', "import einops; print(einops.__version__)"),
    ('causal_conv1d', "import causal_conv1d; print(causal_conv1d.__version__)"),
    ('mamba_ssm', "import mamba_ssm; print(mamba_ssm.__version__)"),
    ('network_sf', f"import sys; sys.path.insert(0, {str(PROJECT)!r}); from network_sf import METST_SF; print('METST_SF: PASS')"),
]

all_passed = True
for name, code in checks:
    completed = subprocess.run(
        [str(PYTHON), '-c', code],
        text=True,
        capture_output=True,
    )
    print(f'===== {name}: return code {completed.returncode} =====')
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    all_passed &= completed.returncode == 0

assert all_passed, '上面至少有一项 import 失败；不要进入全量导出。'
print('imports: PASS')
```

## Cell 6：全量 frozen raw pre-stitch 导出

```python
import subprocess

command = [
    str(PYTHON), '-u', str(EXPORT_SCRIPT),
    '--project-root', str(PROJECT),
    '--input-cache', str(INPUT_CACHE),
    '--weights-dir', str(WEIGHTS),
    '--output-dir', str(OUTPUT),
]

print(' '.join(command))
result = subprocess.run(command)
print('return code:', result.returncode)
assert result.returncode == 0
```

首次运行不要添加 `--overwrite`。只有确认要主动替换一次失败或不完整的本次新导出时，才在 command 末尾加入 `--overwrite`。它不会修改 historical compact cache。

成功后必须出现：

```text
SAMMLV_PRE_STITCH_CONTEXT_CACHE.pkl
sammlv_pre_stitch_context_manifest.json
```

## Cell 7：检查 fresh cache manifest

```python
import json

manifest_path = OUTPUT / 'sammlv_pre_stitch_context_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

for key in [
    'status', 'num_subjects', 'num_videos', 'raw_observations',
    'total_stitched_T', 'input_feature_cache_sha256',
    'output_cache_sha256', 'batch_size', 'window_length', 'stride',
    'environment', 'training_calls',
]:
    print(key, ':', manifest[key])

assert manifest['status'] == 'COMPLETE'
assert manifest['num_subjects'] == 29
assert manifest['num_videos'] == 79
assert manifest['training_calls'] == {
    'model_train': 0,
    'backward': 0,
    'optimizer_step': 0,
}
```

## Cell 8：运行固定规则 scientific audit

```python
import subprocess

context_cache = OUTPUT / 'SAMMLV_PRE_STITCH_CONTEXT_CACHE.pkl'
command = [
    str(PYTHON), '-u', str(AUDIT_SCRIPT),
    '--fresh-context-cache', str(context_cache),
    '--gt-reference-cache', str(GT_REFERENCE),
    '--output-dir', str(OUTPUT / 'outputs'),
]

print(' '.join(command))
result = subprocess.run(command)
print('return code:', result.returncode)
```

返回码含义：

```text
0 = full reproduction 通过，科学审计已完成并给出 GO/NO-GO
2 = BLOCKED-FULL-REPRODUCTION，Agreement 未运行
3 = NO-GO-INSUFFICIENT-MULTIPLICITY，Agreement 主实验未运行
```

## Cell 9：只打印最终摘要

```python
result_path = OUTPUT / 'outputs/context_agreement_results.json'
result = json.loads(result_path.read_text(encoding='utf-8'))

print('status:', result['status'])
print('engineering:', result['engineering_equivalence'])
print('full reproduction:', result['full_reproduction_sanity'])

if result.get('scientific_audit_executed'):
    print('multiplicity:', result['context_multiplicity'])
    print('candidate metrics:', result['candidate_discrimination']['primary_GT_related'])
    print('height-conditioned:', result['height_conditioned'])
    print('bootstrap:', result['subject_bootstrap'])
    print('matched-FP:', result['matched_FP_recovery'])
    print('GO checks:', result['go_checks'])
```

## 预期输出

```text
SAMMLV_PRE_STITCH_CONTEXT_CACHE.pkl
sammlv_pre_stitch_context_manifest.json
outputs/PRE_STITCH_CONTEXT_AGREEMENT_AUDIT_CN.md
outputs/context_agreement_results.json
outputs/sammlv_context_candidates.csv
outputs/sammlv_context_subject_metrics.csv
```

在 Cell 8 完成前，不能从现有 stitched NPZ 推断或填写任何 Agreement 数值，也不能判科学 GO/NO-GO。
