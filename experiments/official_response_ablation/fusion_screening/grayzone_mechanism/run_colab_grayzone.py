"""Launcher executed by the delivered complete cell in the configured Colab."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
from google.colab import drive, files

drive.mount('/content/drive')
package_dir = Path(GRAY_PACKAGE_DIR)
setting = globals().get('GRAY_SETTING', 'sammlv')
mode = globals().get('GRAY_MODE', 'full')
resume = globals().get('GRAY_RESUME')
assert setting in ('sammlv', 'casme3') and mode in ('probe', 'full')
checkout = Path(os.environ.get('ME_TST_ROOT', '/content/ME-TST'))
assert checkout.is_dir(), '请沿用原 ME-TST Colab 环境；本 cell 不安装依赖、不重新训练或生成响应'
manifest = json.loads((package_dir/'package_manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((package_dir/name).read_bytes()).hexdigest() == expected, '运行包不一致：'+name
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys([str(package_dir),str(checkout)]+
    [p for p in sys.path if p]+[env.get('PYTHONPATH','')]))
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(package_dir),
    '-p','test_grayzone.py','-v'],cwd=str(checkout),env=env,check=True)
base = Path('/content/drive/MyDrive/GLSD_GRAYZONE_SUPPORT')
base.mkdir(parents=True,exist_ok=True)
tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
GRAY_OUTPUT = Path(resume) if resume else base/(mode+'_'+setting+'_'+tag)
log_path = base/('log_'+setting+'_'+tag+'.txt')
cmd = [sys.executable,'-u',str(package_dir/'colab_grayzone_entry.py'),
    '--setting',setting,'--mode',mode,'--output',str(GRAY_OUTPUT)]
if resume:cmd.append('--resume')
print('独立灰区实验输出：',GRAY_OUTPUT,flush=True)
print('日志：',log_path,flush=True)
with log_path.open('x',encoding='utf8') as log:
    process = subprocess.Popen(cmd,cwd=str(checkout),env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True)
    try:
        for line in process.stdout:
            print(line,end='');log.write(line);log.flush()
        status = process.wait()
    except BaseException:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
if status:
    raise RuntimeError('本轮停止；保留日志和输出。修复前不要跳过失败配置：'+str(log_path))
completion = json.loads((GRAY_OUTPUT/'completion.json').read_text())
assert completion['completed'] and completion['mode']==mode
if mode=='full':
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(GRAY_OUTPUT/('metst_'+setting)/'summary_full.csv'))
    display(pd.read_csv(GRAY_OUTPUT/('metst_'+setting)/'paired_comparisons.csv'))
archive = shutil.make_archive('/content/'+GRAY_OUTPUT.name+'_grayzone','zip',
    root_dir=str(GRAY_OUTPUT.parent),base_dir=GRAY_OUTPUT.name)
shutil.copy2(archive,base/Path(archive).name)
print('结果 ZIP 已保存到 Drive：',base/Path(archive).name,flush=True)
files.download(archive)
