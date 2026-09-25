"""Run in the existing Colab notebook: P8_OTO_MODE=probe (default) or full."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys
from google.colab import drive, files

drive.mount('/content/drive')
p8_oto_dir = Path('/content/glsd_p8_one_to_one')
mode = globals().get('P8_OTO_MODE', 'probe')
assert mode in ('probe', 'full')
resume = globals().get('P8_OTO_RESUME')
assert Path('/content/ME-TST').is_dir(), '请先恢复原 ME-TST 环境'
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(
    [str(p8_oto_dir), '/content/ME-TST'] + [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]))
for test in ('test_p8_phase2.py', 'test_p8_matching.py'):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(p8_oto_dir),
                    '-p', test, '-v'], cwd='/content/ME-TST', env=env, check=True)
tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
base = Path('/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE')
base.mkdir(parents=True, exist_ok=True)
p8_oto_output = Path(resume) if resume else base / (mode + '_' + tag)
log_path = base / ('log_' + tag + '.txt')
cmd = [sys.executable, '-u', str(p8_oto_dir / 'colab_p8_one_to_one_entry.py'),
       '--mode', mode, '--setting', 'sammlv' if mode == 'probe' else 'both',
       '--output', str(p8_oto_output)]
if resume:
    cmd.append('--resume')
print('输出:', p8_oto_output, flush=True)
print('日志:', log_path, flush=True)
with log_path.open('x', encoding='utf-8') as log:
    process = subprocess.Popen(cmd, cwd='/content/ME-TST', env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        status = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
if status:
    raise RuntimeError('诊断/实验停止，请发最后的报错及日志：%s' % log_path)
assert json.loads((p8_oto_output / 'completion.json').read_text())['completed']
if mode == 'probe':
    print('诊断通过。继续时设 P8_OTO_MODE="full"、P8_OTO_RESUME=None，再执行本入口。')
else:
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(p8_oto_output / 'summary_full.csv'))
bundle = shutil.make_archive('/content/' + p8_oto_output.name + '_p8_results', 'zip',
                             root_dir=str(p8_oto_output.parent), base_dir=p8_oto_output.name)
files.download(bundle)
