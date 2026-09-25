"""Two-dataset evidence diagnostic in the existing configured Colab runtime."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
from google.colab import drive,files

drive.mount('/content/drive')
package=Path(MECHANISM_PACKAGE_DIR)
manifest=json.loads((package/'package_manifest.json').read_text())
for n,h in manifest.items():assert hashlib.sha256((package/n).read_bytes()).hexdigest()==h,n
checkout=Path(os.environ.get('ME_TST_ROOT','/content/ME-TST'))
assert checkout.is_dir(),'请在原先完成grayzone实验的ME-TST Colab中执行'
settings=globals().get('MECHANISM_SETTINGS',['sammlv','casme3'])
assert settings and len(settings)==len(set(settings)) and set(settings)<= {'sammlv','casme3'}
old_paths=globals()['MECHANISM_PREVIOUS']
for setting in settings:
    assert (Path(old_paths[setting])/'completion.json').is_file(),'找不到本轮已完成的grayzone输出：'+old_paths[setting]
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(package),str(checkout)]+[p for p in sys.path if p]+[env.get('PYTHONPATH','')]))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(package),'-p','test_mechanism.py','-v'],
    cwd=str(checkout),env=env,check=True)
base=Path('/content/drive/MyDrive/GLSD_MECHANISM_DIAGNOSTIC');base.mkdir(exist_ok=True,parents=True)
resume=globals().get('MECHANISM_RESUME_ROOT')
root=Path(resume) if resume else base/('diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
if resume:assert root.is_dir(),'恢复目录不存在'
else:root.mkdir()
print('本轮诊断根目录（恢复时填此路径）：',root,flush=True)
for setting in settings:
    out=root/setting
    cmd=[sys.executable,'-u',str(package/'colab_mechanism_entry.py'),'--setting',setting,
         '--reuse',old_paths[setting],'--output',str(out)]
    if out.exists():cmd.append('--resume')
    log_path=root/(setting+'_'+datetime.now(timezone.utc).strftime('%H%M%S_%f')+'.log')
    with log_path.open('x',encoding='utf8') as log:
        process=subprocess.Popen(cmd,cwd=str(checkout),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:print(line,end='');log.write(line);log.flush()
            status=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if status:raise RuntimeError('诊断停止，保留日志和根目录后按说明恢复：'+str(log_path))
    assert json.loads((out/'completion.json').read_text())['completed']
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(out/'conditional_summary.csv'))
archive=shutil.make_archive('/content/'+root.name+'_mechanism','zip',root_dir=str(root.parent),base_dir=root.name)
shutil.copy2(archive,base/Path(archive).name)
print('完成的是机制诊断，不是新的full F1；结果ZIP：',base/Path(archive).name,flush=True)
files.download(archive)
