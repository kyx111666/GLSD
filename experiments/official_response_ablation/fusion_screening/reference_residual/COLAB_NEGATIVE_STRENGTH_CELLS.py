# %% CELL 1：挂载Drive，上传已验证的 glsd_server_ready.zip
from google.colab import drive, files
from pathlib import Path
import io, json, os, subprocess, sys, tempfile, zipfile

drive.mount('/content/drive')
if sys.version_info < (3, 10):
    raise RuntimeError('请使用Colab默认Python内核，不要使用micromamba的Python 3.8')
NEG_DUMP = Path('/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1')
if len(list(NEG_DUMP.glob('subject_*.pkl'))) != 29:
    raise RuntimeError(f'需要29份SAMMLV响应，请检查Drive路径：{NEG_DUMP}')

uploaded_negative = files.upload()  # 只选择 glsd_server_ready.zip；重名后的(1)后缀不影响
if len(uploaded_negative) != 1:
    raise RuntimeError('请只上传一个实验ZIP')
negative_bytes = next(iter(uploaded_negative.values()))
negative_unpack = Path(tempfile.mkdtemp(prefix='glsd_negative_', dir='/content'))
with zipfile.ZipFile(io.BytesIO(negative_bytes)) as archive:
    for name in archive.namelist():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('压缩包包含不支持的路径')
    if 'glsd_server_ready/run_standalone.py' not in archive.namelist():
        raise RuntimeError('请上传 glsd_server_ready.zip，不是之前的metst_asymmetric_residual.zip')
    archive.extractall(negative_unpack)
NEG_CODE = negative_unpack / 'glsd_server_ready'
check = subprocess.run([sys.executable, '-c', 'import numpy,pandas,scipy,sklearn'],
                       capture_output=True, text=True)
if check.returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(NEG_CODE / 'requirements.txt')], check=True)
print('COLAB_NEGATIVE_SETUP = PASS')
print('Python =', sys.executable, sys.version.split()[0])
print('响应目录 =', NEG_DUMP)


# %% CELL 2：新审计 + 负残差强度搜索，保存Drive并下载结果
from datetime import datetime, timezone

# 首次运行保持None；中断续跑时填入上次打印的OUTPUT完整路径。
NEG_RESUME = None
negative_base = Path('/content/drive/MyDrive/GLSD_NEGATIVE_STRENGTH')
negative_base.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
NEG_OUTPUT = Path(NEG_RESUME) if NEG_RESUME else negative_base / ('sammlv_' + stamp)
NEG_LOG = Path(str(NEG_OUTPUT) + '.log')
command = [sys.executable, '-u', str(NEG_CODE / 'run_standalone.py'),
           '--dump', str(NEG_DUMP), '--output', str(NEG_OUTPUT)]
if NEG_RESUME:
    command.append('--resume')
negative_env = os.environ.copy()
negative_env['PYTHONUNBUFFERED'] = '1'
# 子进程使用包内作者运行时，避免此前notebook导入或补丁残留。
negative_env.pop('PYTHONPATH', None)
print('OUTPUT =', NEG_OUTPUT, flush=True)
print('LOG =', NEG_LOG, flush=True)
with NEG_LOG.open('a', encoding='utf-8') as log:
    process_negative = subprocess.Popen(command, cwd=NEG_CODE, env=negative_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process_negative.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        negative_exit = process_negative.wait()
    except KeyboardInterrupt:
        process_negative.terminate()
        try:
            process_negative.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process_negative.kill()
            process_negative.wait()
        print('已停止子进程。续跑时将NEG_RESUME设为：', str(NEG_OUTPUT))
        raise

NEG_ZIP = Path(str(NEG_OUTPUT) + '.zip')
with zipfile.ZipFile(NEG_ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(NEG_OUTPUT.rglob('*')):
        if path.is_file():
            archive.write(path, str(Path(NEG_OUTPUT.name) / path.relative_to(NEG_OUTPUT)))
    archive.write(NEG_LOG, NEG_OUTPUT.name + '/colab.log')
print('结果ZIP =', NEG_ZIP)
if negative_exit == 0:
    completed = json.loads((NEG_OUTPUT / 'completion.json').read_text())
    if not completed.get('completed'):
        raise RuntimeError('进程结束，但没有完成标记')
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(NEG_OUTPUT / 'summary_full.csv'))
    print('NEGATIVE_STRENGTH = PASS（执行完成，不代表方法有效）')
files.download(str(NEG_ZIP))
if negative_exit != 0:
    raise RuntimeError(f'运行失败，退出码={negative_exit}。错误日志已保存并打包：{NEG_ZIP}')
