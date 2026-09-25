# ================================================================
# H+L 官方响应实验：直接接在原 ME-TST+ 官方响应 notebook 的最后
#
# 运行前提：
# 1. 当前 notebook 已经完成 CAS(ME)3 官方响应生成；
# 2. Drive 中已经存在：
#    GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/results_v1/
#    GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/source/
#    GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/evidence_bundle_sha256.txt
# 3. 不要在旧的 Grayzone/BoostingVRME notebook 中运行本 cell。
# ================================================================

from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import os
import subprocess
import sys


drive.mount('/content/drive', force_remount=False)

print('请在弹出的文件选择器中选择：official_response_hl_component_ablation.py')
uploaded = files.upload()
runner_name = 'official_response_hl_component_ablation.py'
uploaded_names = list(uploaded.keys())
selected_name = next(
    (name for name in uploaded_names if Path(name).name == runner_name),
    None,
)
if selected_name is None and len(uploaded_names) == 1:
    only_name = uploaded_names[0]
    only_basename = Path(only_name).name
    if (
        only_basename.startswith('official_response_hl_component_ablation')
        and Path(only_basename).suffix == '.py'
    ):
        selected_name = only_name
if selected_name is None:
    raise RuntimeError(
        f'请上传 {runner_name}。Colab 实际收到的文件：{uploaded_names}'
    )
runner_path = Path('/content') / runner_name
runner_path.write_bytes(uploaded[selected_name])
print('H+L runner upload = PASS:', runner_path, '(received:', selected_name, ')')

# Colab 当前 pandas 已移除 DataFrame.append，而官方 ME-TST evaluator
# 仍调用该旧接口。写入 /content/sitecustomize.py 只影响本次 runtime；
# 由于官方评估通过 subprocess 启动，必须通过 PYTHONPATH 让子进程自动加载。
compat_path = Path('/content') / 'sitecustomize.py'
compat_path.write_text(
    "import pandas as pd\n\n"
    "if not hasattr(pd.DataFrame, 'append') and hasattr(pd.DataFrame, '_append'):\n"
    "    def _compat_dataframe_append(self, other, ignore_index=False, "
    "verify_integrity=False, sort=False):\n"
    "        return pd.concat(\n"
    "            [self, other],\n"
    "            ignore_index=ignore_index,\n"
    "            verify_integrity=verify_integrity,\n"
    "            sort=sort,\n"
    "        )\n\n"
    "    pd.DataFrame.append = _compat_dataframe_append\n",
    encoding='utf-8',
)
print('pandas compatibility patch = PASS:', compat_path)

assert Path('/content/ME-TST').is_dir(), (
    '没有找到 /content/ME-TST。请回到原 ME-TST+ 官方响应 notebook，'
    '不要新建一个空白 Colab 环境。'
)

drive_root = Path('/content/drive/MyDrive')
required = [
    drive_root / 'GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/results_v1/run_manifest.json',
    drive_root / 'GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/results_v1/per_subject_counts.csv',
    drive_root / 'GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/results_v1/outer_selected_configs.csv',
    drive_root / 'GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/evidence_bundle_sha256.txt',
    drive_root / 'GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/source/metst_casme3_official_glds_full.py',
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        'CASME3 官方响应尚未完成，缺少以下文件：\n' + '\n'.join(missing) +
        '\n请先等待当前 CASME3 响应生成 cell 完成，再运行本 cell。'
    )

tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
output_root = drive_root / 'GLSD_HL_OFFICIAL'
output_root.mkdir(parents=True, exist_ok=True)
log_path = output_root / f'launcher_metst_casme3_{tag}.log'

environment = dict(os.environ)
environment['PYTHONPATH'] = os.pathsep.join(
    ['/content/ME-TST', '/content'] +
    ([environment['PYTHONPATH']] if environment.get('PYTHONPATH') else [])
)

compat_probe = subprocess.run(
    [
        sys.executable,
        '-c',
        'import pandas as pd; '
        'print("DataFrame.append available =", hasattr(pd.DataFrame, "append"))',
    ],
    cwd='/content/ME-TST',
    env=environment,
    capture_output=True,
    text=True,
)
print(compat_probe.stdout.strip())
if compat_probe.returncode or 'DataFrame.append available = True' not in compat_probe.stdout:
    raise RuntimeError(
        'pandas 兼容补丁未生效。子进程输出：\n' +
        (compat_probe.stdout + compat_probe.stderr)
    )

command = [
    sys.executable,
    '-u',
    str(runner_path),
    '--setting',
    'metst_casme3',
    '--output-root',
    str(output_root),
]

print('开始运行 H/G/L/GL/HL/HL75 官方 CASME3 评估。')
print('结果目录：', output_root)
print('日志文件：', log_path)
print('这一步只读取 sealed official response，不训练模型，也不生成响应。')

with log_path.open('w', encoding='utf-8') as log:
    process = subprocess.Popen(
        command,
        cwd='/content/ME-TST',
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end='')
        log.write(line)
        log.flush()
    return_code = process.wait()

if return_code:
    raise RuntimeError(f'H+L 官方评估失败，完整日志在：{log_path}')

print('H+L_OFFICIAL_CASME3 = PASS')
print('请下载或查看 Drive 目录：', output_root)
