"""Execute this file in an already configured Colab notebook cell.

Set P8_RESUME to a previous stage-2 output path to resume it; otherwise None.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from google.colab import drive, files
from IPython.display import display

drive.mount('/content/drive')
p8_task_dir = Path('/content/glsd_p8_phase2')
assert (p8_task_dir / 'run_p8_phase2.py').is_file(), '请先上传并解压第二阶段 zip'
assert Path('/content/ME-TST').is_dir(), '原 ME-TST 环境不在本 runtime，请先恢复原环境'
p8_env = dict(os.environ)
p8_env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(
    [str(p8_task_dir), '/content/ME-TST'] + [p for p in sys.path if p] + [p8_env.get('PYTHONPATH', '')]
))

# Same interpreter/compatibility adapter as the successful phase-1 run.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(p8_task_dir),
                '-p', 'test_p8_phase2.py', '-v'], env=p8_env, check=True)
p8_command = [sys.executable, '-u', str(p8_task_dir / 'colab_p8_entry.py'), '--setting', 'both']
subprocess.run(p8_command + ['--check-inputs'], cwd='/content/ME-TST', env=p8_env, check=True)

p8_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
p8_resume = globals().get('P8_RESUME')
p8_base = Path('/content/drive/MyDrive/GLSD_P8_PHASE2')
p8_base.mkdir(parents=True, exist_ok=True)
p8_output = Path(p8_resume) if p8_resume else p8_base / ('p8_' + p8_tag)
p8_log = p8_base / ('run_' + p8_tag + '.log')
p8_command += ['--output', str(p8_output)]
if p8_resume:
    p8_command.append('--resume')
print('第二阶段输出:', p8_output, flush=True)
print('日志:', p8_log, flush=True)
print('若中断，将 P8_RESUME 设为上面的输出路径，再运行此单元格。', flush=True)
with p8_log.open('x', encoding='utf-8') as log:
    process = subprocess.Popen(p8_command, cwd='/content/ME-TST', env=p8_env,
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
    raise RuntimeError('实验停止；请保留日志和输出，先检查错误：%s' % p8_log)
completion = json.loads((p8_output / 'completion.json').read_text())
assert completion['completed'], '缺少完成标志'
import pandas as pd
display(pd.read_csv(p8_output / 'summary_full.csv'))
display(pd.read_csv(p8_output / 'paired_comparisons.csv'))
print('P8_PHASE2_EXECUTION = PASS（程序完成；优劣请看结果表）')
p8_result_zip = shutil.make_archive('/content/' + p8_output.name + '_results', 'zip',
                                    root_dir=str(p8_output.parent), base_dir=p8_output.name)
print('结果已保存至 Drive；下载汇总包:', p8_result_zip)
files.download(p8_result_zip)
