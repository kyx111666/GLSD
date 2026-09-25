# %% CELL 1
from pathlib import Path
import sys, subprocess, zipfile, json

# 在服务器 JupyterLab 的 Python notebook 运行。ZIP 放在 notebook 旁边。
if sys.version_info < (3, 10):
    raise RuntimeError('需要 Python 3.10 或更新版本，建议 3.10–3.13')
locations = [Path.cwd(), Path.home() / 'workspace']
PACKAGE = next((p / 'glsd_server_ready.zip' for p in locations
                if (p / 'glsd_server_ready.zip').is_file()), None)
if PACKAGE is None:
    raise FileNotFoundError('请先上传 glsd_server_ready.zip 到当前 notebook 所在目录')
CODE = PACKAGE.parent / 'glsd_server_ready'
if CODE.exists():
    print('复用已有代码目录：', CODE)
else:
    with zipfile.ZipFile(PACKAGE) as z:
        for name in z.namelist():
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
        z.extractall(PACKAGE.parent)
probe = subprocess.run([sys.executable, '-c', 'import numpy,pandas,scipy,sklearn'],
                       capture_output=True, text=True)
if probe.returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(CODE / 'requirements.txt')], check=True)
print('准备完成：作者运行时代码已包含，不需要 ME-TST 文件夹或旧审计目录。')
print('CODE =', CODE)


# %% CELL 2
# 自动寻找之前解压的29份响应；后台运行新审计和完整实验。
subprocess.run([sys.executable, str(CODE / 'server_job.py'), 'start'], check=True)


# %% CELL 3
# 可重复运行这格查看进度；不要重复创建新实验。
subprocess.run([sys.executable, str(CODE / 'server_job.py'), 'status'], check=True)


# %% CELL 4
# 进程结束后运行：显示结果并提供ZIP下载链接；失败时也会打包日志。
from IPython.display import FileLink, display
subprocess.run([sys.executable, str(CODE / 'server_job.py'), 'result'], check=True)
state = json.loads((CODE / 'server_job.json').read_text())
result = Path(state['output']).with_suffix('.zip')
try:
    display(FileLink(str(result.relative_to(Path.cwd()))))
except ValueError:
    print('请在左侧文件栏找到并下载：', result)
