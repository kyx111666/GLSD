"""Launch joint search from the existing configured Colab notebook."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from google.colab import drive, files

drive.mount('/content/drive')
joint_dir = Path('/content/glsd_joint_gm')
mode = globals().get('JOINT_MODE', 'probe')
setting = globals().get('JOINT_SETTING', 'sammlv')
resume = globals().get('JOINT_RESUME')
assert mode in ('probe', 'full') and setting in ('sammlv', 'casme3')
old = Path(globals().get('JOINT_OLD', '/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE/full_20260922T053538_684804Z'))
default_threshold = {
    'sammlv': '/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/full_sammlv_20260922T070420_025128Z',
    'casme3': '/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/full_casme3_20260922T071214_000063Z',
}
threshold = Path(globals().get('JOINT_THRESHOLD_PATHS', default_threshold)[setting])
assert Path('/content/ME-TST').is_dir(), '请恢复原 ME-TST 环境；不用重新训练或生成响应'
for path in (old, threshold):
    assert (path/'completion.json').is_file(), '找不到已完成旧结果：' + str(path)
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys([str(joint_dir), '/content/ME-TST'] +
                                                [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]))
# Always validate packaged new paths before launching; no dependency installation.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(joint_dir),
                '-p', 'test_joint_gm.py', '-v'], cwd='/content/ME-TST', env=env, check=True)
base = Path('/content/drive/MyDrive/GLSD_JOINT_GM')
base.mkdir(parents=True, exist_ok=True)
tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
joint_output = Path(resume) if resume else base/(mode + '_' + setting + '_' + tag)
log_path = base/('log_' + tag + '.txt')
cmd = [sys.executable, '-u', str(joint_dir/'colab_joint_gm_entry.py'), '--mode', mode,
       '--setting', setting, '--reuse', str(old), '--threshold-reuse', str(threshold),
       '--output', str(joint_output)]
if resume:
    cmd.append('--resume')
print('旧结果只读:', old, threshold, sep='\n', flush=True)
print('本轮输出:', joint_output, flush=True)
print('日志:', log_path, flush=True)
with log_path.open('x', encoding='utf8') as log:
    process = subprocess.Popen(cmd, cwd='/content/ME-TST', env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            print(line, end=''); log.write(line); log.flush()
        status = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        raise
if status:
    raise RuntimeError('停止于新实验。请保留日志和输出目录：' + str(log_path))
assert json.loads((joint_output/'completion.json').read_text())['completed']
if mode == 'full':
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(joint_output/('metst_' + setting)/'summary_full.csv'))
    display(pd.read_csv(joint_output/('metst_' + setting)/'paired_comparisons.csv'))
archive = shutil.make_archive('/content/' + joint_output.name + '_joint', 'zip',
                              root_dir=str(joint_output.parent), base_dir=joint_output.name)
files.download(archive)
