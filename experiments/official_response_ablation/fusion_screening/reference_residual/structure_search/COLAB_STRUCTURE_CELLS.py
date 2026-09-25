# %% CELL 1：挂载Drive并上传 glsd_structure_ready.zip
from google.colab import drive, files
from pathlib import Path
import io, json, os, subprocess, sys, tempfile, zipfile

drive.mount('/content/drive')
if sys.version_info < (3, 10):
    raise RuntimeError('请使用Colab默认Python内核，不要切换旧Python 3.8环境')
STRUCT_DUMP = Path('/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1')
if len(list(STRUCT_DUMP.glob('subject_*.pkl'))) != 29:
    raise RuntimeError(f'需要29份SAMMLV响应，请检查：{STRUCT_DUMP}')
uploaded_structure = files.upload()  # 选择本轮的 glsd_structure_ready.zip
if len(uploaded_structure) != 1:
    raise RuntimeError('请只上传一个代码ZIP')
structure_unpack = Path(tempfile.mkdtemp(prefix='glsd_structure_', dir='/content'))
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded_structure.values())))) as archive:
    for name in archive.namelist():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('压缩包包含不支持的路径')
    if 'glsd_structure_ready/run_structure.py' not in archive.namelist():
        raise RuntimeError('请上传 glsd_structure_ready.zip，不是上一轮代码或结果ZIP')
    archive.extractall(structure_unpack)
STRUCT_CODE = structure_unpack / 'glsd_structure_ready'
structure_probe = subprocess.run([sys.executable, '-c', 'import numpy,pandas,scipy,sklearn'],
                                 capture_output=True, text=True)
if structure_probe.returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(STRUCT_CODE / 'requirements.txt')], check=True)
print('STRUCTURE_SETUP = PASS')
print('本轮：a0={1,1.5,2}，rho={1,2,3}；5种方法，共741个配置；无需GPU训练。')


# %% CELL 2：运行共同结构搜索，保存Drive并下载结果
from datetime import datetime, timezone

# 首次保持None。中断后重跑此格：填入本轮打印的OUTPUT完整路径。
# 不要填上一轮 GLSD_NEGATIVE_STRENGTH 的目录。
STRUCT_RESUME = None
structure_base = Path('/content/drive/MyDrive/GLSD_RESIDUAL_STRUCTURE')
structure_base.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
STRUCT_OUTPUT = Path(STRUCT_RESUME) if STRUCT_RESUME else structure_base / ('sammlv_' + stamp)
STRUCT_LOG = Path(str(STRUCT_OUTPUT) + '.log')
structure_command = [sys.executable, '-u', str(STRUCT_CODE / 'run_structure.py'),
                     '--dump', str(STRUCT_DUMP), '--output', str(STRUCT_OUTPUT)]
if STRUCT_RESUME:
    structure_command.append('--resume')
structure_env = os.environ.copy()
structure_env['PYTHONUNBUFFERED'] = '1'
structure_env.pop('PYTHONPATH', None)
print('OUTPUT =', STRUCT_OUTPUT, flush=True)
print('LOG =', STRUCT_LOG, flush=True)
with STRUCT_LOG.open('a', encoding='utf-8') as log:
    structure_process = subprocess.Popen(structure_command, cwd=STRUCT_CODE, env=structure_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in structure_process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        structure_exit = structure_process.wait()
    except KeyboardInterrupt:
        structure_process.terminate()
        try:
            structure_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            structure_process.kill()
            structure_process.wait()
        print('已停止。续跑时将STRUCT_RESUME设为：', str(STRUCT_OUTPUT))
        raise

STRUCT_ZIP = Path(str(STRUCT_OUTPUT) + '.zip')
with zipfile.ZipFile(STRUCT_ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(STRUCT_OUTPUT.rglob('*')):
        if path.is_file():
            archive.write(path, str(Path(STRUCT_OUTPUT.name) / path.relative_to(STRUCT_OUTPUT)))
    archive.write(STRUCT_LOG, STRUCT_OUTPUT.name + '/colab.log')
print('结果ZIP =', STRUCT_ZIP)
if structure_exit == 0:
    completed = json.loads((STRUCT_OUTPUT / 'completion.json').read_text())
    if not completed.get('completed') or completed.get('configs') != 741:
        raise RuntimeError('完成标记或配置数量不符，请保留结果ZIP')
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(STRUCT_OUTPUT / 'summary_full.csv'))
    print('STRUCTURE_SEARCH = PASS（执行完成，不代表方法有效）')
files.download(str(STRUCT_ZIP))
if structure_exit != 0:
    raise RuntimeError(f'退出码={structure_exit}。错误日志已打包：{STRUCT_ZIP}')
