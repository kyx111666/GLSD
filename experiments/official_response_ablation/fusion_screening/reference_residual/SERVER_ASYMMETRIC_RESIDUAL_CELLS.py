# %% CELL 1：服务器路径和本轮配置
from pathlib import Path
import os, sys

SERVER_WORKSPACE = Path.home() / 'workspace'
DATA_ROOT = Path(os.environ.get('GLSD_DATA_ROOT', SERVER_WORKSPACE / 'glsd'))
ME_TST_ROOT = Path(os.environ.get('ME_TST_ROOT', SERVER_WORKSPACE / 'ME-TST'))
CODE_ZIP = Path(os.environ.get(
    'ASYM_PACKAGE_ZIP', SERVER_WORKSPACE / 'metst_asymmetric_residual.zip'))
AUDIT = DATA_ROOT / 'GLSD_EVALUATOR_AUDIT/sammlv_20260922T091343_286158Z'
CODE_ROOT = SERVER_WORKSPACE / 'metst_asymmetric_residual'
OUTPUT_ROOT = DATA_ROOT / 'GLSD_REFERENCE_RESIDUAL'

for path in (DATA_ROOT, ME_TST_ROOT, CODE_ZIP, AUDIT):
    assert path.exists(), path
print('DATA_ROOT =', DATA_ROOT)
print('ME_TST_ROOT =', ME_TST_ROOT)
print('CODE_ZIP =', CODE_ZIP)
print('AUDIT =', AUDIT)


# %% CELL 2：解压并核验新实验包
import hashlib, json, zipfile

with zipfile.ZipFile(CODE_ZIP) as z:
    manifest = json.loads(z.read('package_manifest.json'))
    assert set(z.namelist()) == set(manifest) | {'package_manifest.json'}
    for name, digest in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    z.extractall(CODE_ROOT)
print('PACKAGE = PASS')
print('CODE_ROOT =', CODE_ROOT)


# %% CELL 3：输入、审计目录和运行环境核验
completion = json.loads((AUDIT / 'completion.json').read_text())
assert completion.get('status') == 'PASS' and completion.get('completed')
dump = DATA_ROOT / 'ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1'
assert len(list(dump.glob('subject_*.pkl'))) == 29
assert (ME_TST_ROOT / 'training_utils.py').exists()
assert (ME_TST_ROOT / 'Utils').exists()

os.environ['GLSD_DATA_ROOT'] = str(DATA_ROOT.resolve())
os.environ['ME_TST_ROOT'] = str(ME_TST_ROOT.resolve())
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONPATH'] = os.pathsep.join([
    str(CODE_ROOT), str(ME_TST_ROOT), os.environ.get('PYTHONPATH', '')])
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(ME_TST_ROOT))

import numpy as np, pandas as pd
print('Python:', sys.version)
print('NumPy:', np.__version__, 'pandas:', pd.__version__)
print('INPUTS = PASS')


# %% CELL 4：先跑关键路径测试
import subprocess

test_cmd = [sys.executable, '-m', 'unittest', 'discover', '-s',
            str(CODE_ROOT), '-p', 'test_reference_residual.py', '-v']
subprocess.run(test_cmd, cwd=ME_TST_ROOT, check=True)
print('TESTS = PASS')


# %% CELL 5：后台启动正式实验
from datetime import datetime, timezone

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUTPUT = OUTPUT_ROOT / ('asym_negative_strength_sammlv_' + stamp)
LOG = OUTPUT_ROOT / ('asym_negative_strength_sammlv_' + stamp + '.log')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

command = [sys.executable, '-u',
           str(CODE_ROOT/'run_server_asymmetric_residual.py'),
           '--data-root', str(DATA_ROOT), '--metst-root', str(ME_TST_ROOT),
           '--audit', str(AUDIT), '--output', str(OUTPUT)]
log_stream = LOG.open('w', encoding='utf-8')
process = subprocess.Popen(command, cwd=ME_TST_ROOT,
                           stdout=log_stream, stderr=subprocess.STDOUT,
                           start_new_session=True, env=os.environ.copy())
(OUTPUT_ROOT / (OUTPUT.name + '.pid')).write_text(str(process.pid))
print('PID =', process.pid)
print('OUTPUT =', OUTPUT)
print('LOG =', LOG)


# %% CELL 6：查看运行状态（可重复运行）
pid_file = OUTPUT_ROOT / (OUTPUT.name + '.pid')
pid = int(pid_file.read_text()) if pid_file.exists() else None
status = subprocess.run(['bash', '-lc',
                         f'ps -p {pid} -o pid=,stat=,etime=,cmd='],
                        capture_output=True, text=True)
print(status.stdout.strip() or '进程已结束')
if LOG.exists():
    print(LOG.read_text(encoding='utf-8', errors='replace')[-8000:])


# %% CELL 6B：中断后用同一输出目录续跑（只在需要时运行）
resume_command = [sys.executable, '-u',
                  str(CODE_ROOT/'run_server_asymmetric_residual.py'),
                  '--data-root', str(DATA_ROOT), '--metst-root', str(ME_TST_ROOT),
                  '--audit', str(AUDIT), '--output', str(OUTPUT), '--resume']
subprocess.run(resume_command, cwd=ME_TST_ROOT, check=True,
               env=os.environ.copy())


# %% CELL 7：完成后核对结果并打包
import csv

completion_path = OUTPUT / 'completion.json'
assert completion_path.exists(), '实验尚未完成；先运行 CELL 6 查看日志'
done = json.loads(completion_path.read_text())
assert done.get('completed') is True
summary = pd.read_csv(OUTPUT / 'summary_full.csv')
display(summary.sort_values('F1', ascending=False))
assert len(summary) == 8
assert set(summary['method']) >= {'G', 'PositiveResidual', 'RefResidual'}

RESULT_ZIP = Path(str(OUTPUT) + '.zip')
with zipfile.ZipFile(RESULT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in OUTPUT.rglob('*'):
        if path.is_file():
            z.write(path, path.relative_to(OUTPUT.parent))
print('RESULT_ZIP =', RESULT_ZIP)
print('SHA256 =', hashlib.sha256(RESULT_ZIP.read_bytes()).hexdigest())
