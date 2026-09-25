"""Execute from continuation cells in the existing Colab kernel."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys
from google.colab import drive, files

drive.mount('/content/drive')
control_dir = Path('/content/glsd_p8_threshold_control')
mode = globals().get('P8_CONTROL_MODE', 'probe')
setting = globals().get('P8_CONTROL_SETTING', 'sammlv')
resume = globals().get('P8_CONTROL_RESUME')
reuse = Path(globals().get('P8_CONTROL_REUSE', '/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE/full_20260922T053538_684804Z'))
assert mode in ('probe','full') and setting in ('sammlv','casme3')
assert Path('/content/ME-TST').is_dir(), '请先恢复上次 ME-TST 环境，不需要重新生成响应'
assert (reuse/'completion.json').is_file(), '找不到上轮完整结果，请将 P8_CONTROL_REUSE 改为实际 full 目录'
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys([str(control_dir),'/content/ME-TST']+[p for p in sys.path if p]+[env.get('PYTHONPATH','')]))
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(control_dir),'-p','test_p8_threshold_control.py','-v'],
               cwd='/content/ME-TST',env=env,check=True)
base = Path('/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL')
base.mkdir(parents=True,exist_ok=True)
tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
p8_control_output = Path(resume) if resume else base/(mode+'_'+setting+'_'+tag)
log_path = base/('log_'+tag+'.txt')
cmd=[sys.executable,'-u',str(control_dir/'colab_p8_threshold_entry.py'),'--mode',mode,
     '--setting',setting,'--reuse',str(reuse),'--output',str(p8_control_output)]
if resume: cmd.append('--resume')
print('旧结果（只读）:',reuse,flush=True)
print('本轮输出:',p8_control_output,flush=True)
print('日志:',log_path,flush=True)
with log_path.open('x',encoding='utf8') as log:
    process=subprocess.Popen(cmd,cwd='/content/ME-TST',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        for line in process.stdout:
            print(line,end=''); log.write(line); log.flush()
        status=process.wait()
    except BaseException:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        raise
if status: raise RuntimeError('实验停止。请提供末尾报错和日志：'+str(log_path))
assert json.loads((p8_control_output/'completion.json').read_text())['completed']
if mode=='full':
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(p8_control_output/('metst_'+setting)/'summary_full.csv'))
    display(pd.read_csv(p8_control_output/('metst_'+setting)/'paired_comparisons.csv'))
archive=shutil.make_archive('/content/'+p8_control_output.name,'zip',root_dir=str(p8_control_output.parent),base_dir=p8_control_output.name)
files.download(archive)
