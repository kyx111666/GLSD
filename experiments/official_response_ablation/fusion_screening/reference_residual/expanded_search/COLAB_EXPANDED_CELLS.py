# %% CELL 1：挂载Drive并上传本轮扩展搜索代码包
from google.colab import drive, files
from pathlib import Path
import io, json, os, subprocess, sys, tempfile, zipfile

drive.mount('/content/drive')
if sys.version_info < (3, 10):
    raise RuntimeError('请使用Colab默认Python内核，不要切换旧Python 3.8环境')
EXP_DUMP = Path('/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1')
if len(list(EXP_DUMP.glob('subject_*.pkl'))) != 29:
    raise RuntimeError(f'需要29份SAMMLV响应，请检查：{EXP_DUMP}')
uploaded_expanded = files.upload()  # 选择 glsd_expanded_structure_ready.zip
if len(uploaded_expanded) != 1:
    raise RuntimeError('请只上传一个代码ZIP')
expanded_unpack = Path(tempfile.mkdtemp(prefix='glsd_expanded_', dir='/content'))
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded_expanded.values())))) as archive:
    for name in archive.namelist():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('压缩包包含不支持的路径')
    if 'glsd_expanded_structure_ready/run_expanded.py' not in archive.namelist():
        raise RuntimeError('请上传 glsd_expanded_structure_ready.zip，不是上一轮结果ZIP')
    archive.extractall(expanded_unpack)
EXP_CODE = expanded_unpack / 'glsd_expanded_structure_ready'
probe = subprocess.run([sys.executable, '-c', 'import numpy,pandas,scipy,sklearn'],
                       capture_output=True, text=True)
if probe.returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(EXP_CODE / 'requirements.txt')], check=True)
print('EXPANDED_SETUP = PASS')
print('物理尺度 = {1,1.5,2,2.5,3,4}; rho = {0.5,1,2,3,4,6}; 默认tau = 0.01:0.01:0.99')
print('所有方法共用扩展尺度集合；不训练模型。')


# %% CELL 2：运行扩展结构搜索
from datetime import datetime, timezone

EXP_RESUME = None  # 中断后填入本轮打印的OUTPUT路径
exp_base = Path('/content/drive/MyDrive/GLSD_RESIDUAL_EXPANDED')
exp_base.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
EXP_OUTPUT = Path(EXP_RESUME) if EXP_RESUME else exp_base / ('sammlv_' + stamp)
EXP_LOG = Path(str(EXP_OUTPUT) + '.log')
# 默认复现细阈值正向结果。粗网格对照用run_expanded.py；旧尺度细阈值用run_original_fine.py。
EXP_DRIVER = 'run_expanded_fine.py'
cmd = [sys.executable, '-u', str(EXP_CODE / EXP_DRIVER),
       '--dump', str(EXP_DUMP), '--output', str(EXP_OUTPUT)]
if EXP_RESUME:
    cmd.append('--resume')
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env.pop('PYTHONPATH', None)
print('OUTPUT =', EXP_OUTPUT, flush=True)
print('LOG =', EXP_LOG, flush=True)
with EXP_LOG.open('a', encoding='utf-8') as log:
    process = subprocess.Popen(cmd, cwd=EXP_CODE, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        exp_exit = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        print('已停止，续跑时将EXP_RESUME设为：', str(EXP_OUTPUT))
        raise

EXP_ZIP = Path(str(EXP_OUTPUT) + '.zip')
with zipfile.ZipFile(EXP_ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(EXP_OUTPUT.rglob('*')):
        if path.is_file():
            archive.write(path, str(Path(EXP_OUTPUT.name) / path.relative_to(EXP_OUTPUT)))
    archive.write(EXP_LOG, EXP_OUTPUT.name + '/colab.log')
print('结果ZIP =', EXP_ZIP)
if exp_exit == 0:
    completed = json.loads((EXP_OUTPUT / 'completion.json').read_text())
    if not completed.get('completed'):
        raise RuntimeError('没有完成标记，请保留结果ZIP和日志')
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(EXP_OUTPUT / 'summary_full.csv'))
    print('EXPANDED_SEARCH = PASS（执行完成，不代表方法有效）')
files.download(str(EXP_ZIP))
if exp_exit != 0:
    raise RuntimeError(f'退出码={exp_exit}，错误日志已打包：{EXP_ZIP}')
