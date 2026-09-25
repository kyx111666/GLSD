"""Execute inside the existing Colab notebook; print static CSV, persist logs."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from google.colab import drive, files

drive.mount('/content/drive')
rlu_dir=Path('/content/glsd_reliable_local_union')
mode=globals().get('RLU_MODE','probe');setting=globals().get('RLU_SETTING','sammlv')
assert mode in ('check','probe','full') and setting in ('sammlv','casme3')
defaults={
    'sammlv':'/content/drive/MyDrive/GLSD_ETA_GM/full_sammlv_20260922T110018_905231Z',
    'casme3':'/content/drive/MyDrive/GLSD_ETA_GM/full_casme3_20260922T111208_876534Z',
}
old=Path(globals().get('RLU_REUSE_PATHS',defaults)[setting])
assert Path('/content/ME-TST').is_dir(), '先恢复原 ME-TST 环境，无需训练或重新生成响应'
assert (old/'completion.json').is_file(), '找不到上轮完整 ETA 结果目录：'+str(old)
assert (rlu_dir/'package_manifest.json').is_file(), '先运行运行包安装 cell'
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(rlu_dir),'/content/ME-TST']+[p for p in sys.path if p]+[env.get('PYTHONPATH','')]))
base=Path('/content/drive/MyDrive/GLSD_RELIABLE_LOCAL_UNION');base.mkdir(parents=True,exist_ok=True)
resume=globals().get('RLU_RESUME')
existing=sorted(p for p in base.glob(mode+'_'+setting+'_*') if p.is_dir())
if not resume and existing:
    assert len(existing)==1, '已有多个同模式运行，请把 RLU_RESUME 设为要接续的目录，避免重复计算'
    resume=str(existing[0])
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
rlu_output=Path(resume) if resume else base/(mode+'_'+setting+'_'+tag)
log_path=base/('log_'+mode+'_'+setting+'_'+tag+'.txt')
cmd=[sys.executable,'-u',str(rlu_dir/'colab_reliable_entry.py'),'--mode',mode,'--setting',setting,
     '--reuse',str(old),'--output',str(rlu_output)]
if resume:cmd.append('--resume')
print('旧 ETA 结果（只读）:',old,flush=True)
print('本轮输出:',rlu_output,flush=True);print('日志:',log_path,flush=True)
with log_path.open('x',encoding='utf8') as log:
    def logged_call(command):
        process=subprocess.Popen(command,cwd='/content/ME-TST',env=env,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end='');log.write(line);log.flush()
            status=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        if status:raise RuntimeError('运行停止，请查看日志：'+str(log_path))
    logged_call([sys.executable,'-m','unittest','discover','-s',str(rlu_dir),'-p','test_reliable_local_union.py','-v'])
    logged_call(cmd)
completion=json.loads((rlu_output/'completion.json').read_text())
assert completion['completed'] and completion['mode']==mode
if mode=='full':
    for name in ('summary_full.csv','paired_comparisons.csv','search_budget.csv','decision.json'):
        print('\n'+name+'\n'+(rlu_output/('metst_'+setting)/name).read_text(),flush=True)
else:
    for name in ('compatibility.json','feature_audit.json','probe.json'):
        path=rlu_output/('metst_'+setting)/name
        if path.exists():print('\n'+name+'\n'+path.read_text(),flush=True)
print('计算已保存。以下仅打包下载；下载出错时不要重新计算。',flush=True)
archive=shutil.make_archive('/content/'+rlu_output.name+'_reliable_local_union','zip',
                            root_dir=str(rlu_output.parent),base_dir=rlu_output.name)
print('RESULT_ZIP =',archive,flush=True)
files.download(archive)
