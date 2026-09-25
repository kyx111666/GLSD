"""Append to the current configured Colab notebook; static tables avoid UI errors."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from google.colab import drive, files

drive.mount('/content/drive')
eta_dir=Path('/content/glsd_eta_gm')
mode=globals().get('ETA_MODE','probe');setting=globals().get('ETA_SETTING','sammlv')
resume=globals().get('ETA_RESUME')
assert mode in ('probe','full') and setting in ('sammlv','casme3')
defaults={
    'sammlv':'/content/drive/MyDrive/GLSD_JOINT_GM/full_sammlv_20260922T091538_299118Z',
    'casme3':'/content/drive/MyDrive/GLSD_JOINT_GM/full_casme3_20260922T095610_977764Z',
}
root=Path(globals().get('ETA_REUSE_PATHS',defaults)[setting])
assert Path('/content/ME-TST').is_dir(), '先恢复原 ME-TST 环境；不需要重跑训练或响应生成'
assert (root/'completion.json').is_file(), '找不到上轮联合搜索完整目录：'+str(root)
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(eta_dir),'/content/ME-TST']+[p for p in sys.path if p]+[env.get('PYTHONPATH','')]))
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(eta_dir),'-p','test_eta_gm.py','-v'],cwd='/content/ME-TST',env=env,check=True)
base=Path('/content/drive/MyDrive/GLSD_ETA_GM');base.mkdir(parents=True,exist_ok=True)
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
eta_output=Path(resume) if resume else base/(mode+'_'+setting+'_'+tag)
log_path=base/('log_'+tag+'.txt')
cmd=[sys.executable,'-u',str(eta_dir/'colab_eta_gm_entry.py'),'--mode',mode,'--setting',setting,'--reuse',str(root),'--output',str(eta_output)]
if resume:cmd.append('--resume')
print('旧联合结果（只读）:',root,flush=True)
print('本轮输出:',eta_output,flush=True);print('日志:',log_path,flush=True)
with log_path.open('x',encoding='utf8') as log:
    process=subprocess.Popen(cmd,cwd='/content/ME-TST',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        for line in process.stdout:
            print(line,end='');log.write(line);log.flush()
        status=process.wait()
    except BaseException:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
if status:raise RuntimeError('实验停止；请保留末尾报错和日志：'+str(log_path))
assert json.loads((eta_output/'completion.json').read_text())['completed']
import pandas as pd
if mode=='full':
    for name in ('summary_full.csv','paired_comparisons.csv'):
        table=pd.read_csv(eta_output/('metst_'+setting)/name)
        print('\n'+name+'\n'+table.to_string(index=False),flush=True)
else:
    print((eta_output/('metst_'+setting)/'alignment_sensitivity.json').read_text(),flush=True)
print('计算和保存已完成。下面只打包下载；如下载异常，不要重跑实验。',flush=True)
archive=shutil.make_archive('/content/'+eta_output.name+'_eta','zip',root_dir=str(eta_output.parent),base_dir=eta_output.name)
print('RESULT_ZIP =',archive,flush=True)
files.download(archive)
