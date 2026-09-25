"""Run SAMMLV screening in the existing Colab kernel runtime."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from google.colab import drive, files

drive.mount('/content/drive')
package_dir = Path(RESIDUAL_DIR)
mode = globals().get('RESIDUAL_MODE', 'screen')
resume = globals().get('RESIDUAL_RESUME')
audit_dir = Path(globals().get('RESIDUAL_AUDIT',
    '/content/drive/MyDrive/GLSD_EVALUATOR_AUDIT/sammlv_20260922T091343_286158Z'))
assert mode in ('probe', 'screen')
base = Path('/content/drive/MyDrive/GLSD_REFERENCE_RESIDUAL')
base.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
residual_output = Path(resume) if resume else base/(mode+'_sammlv_'+stamp)
if residual_output.resolve().parent != base.resolve():
    raise RuntimeError('恢复路径必须是 GLSD_REFERENCE_RESIDUAL 的直接子目录')
log_path = base/('log_'+stamp+'.txt')
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join([str(package_dir), '/content/ME-TST'])
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONWARNINGS'] = 'ignore:The behavior of DataFrame concatenation with empty or all-NA entries is deprecated:FutureWarning'
failure = None
print('本轮输出：', residual_output, flush=True)
print('核验依据：', audit_dir, flush=True)
print('固定 a0=2/rho=2；五方法×19阈值；内层full选参；三项锁参消融。', flush=True)


def run_logged(command, log):
    process = subprocess.Popen(command, cwd='/content/ME-TST', env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        status = process.wait()
        if status:
            raise RuntimeError('实验进程退出，exit='+str(status))
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


with log_path.open('x', encoding='utf8') as log:
    try:
        if not Path('/content/ME-TST/training_utils.py').is_file():
            raise RuntimeError('原 ME-TST 运行环境不存在，请保留之前运行成功的运行时')
        if not (audit_dir/'completion.json').is_file():
            raise RuntimeError('找不到已通过的评价核验目录：'+str(audit_dir))
        log.write('Runtime: '+sys.executable+' '+sys.version+'\n')
        run_logged([sys.executable, '-m', 'unittest', 'discover', '-s', str(package_dir),
                    '-p', 'test_reference_residual.py', '-v'], log)
        command = [sys.executable, '-u', str(package_dir/'colab_reference_residual_entry.py'),
            '--mode', mode, '--setting', 'sammlv', '--output', str(residual_output), '--audit', str(audit_dir)]
        if resume:
            command.append('--resume')
        run_logged(command, log)
    except Exception:
        failure = traceback.format_exc()
        log.write(failure)
        print(failure, flush=True)

residual_output.mkdir(parents=True, exist_ok=True)
shutil.copy2(log_path, residual_output/('run_log_'+stamp+'.txt'))
if not (residual_output/'package_manifest.json').exists():
    shutil.copy2(package_dir/'package_manifest.json', residual_output/'package_manifest.json')
if failure:
    (residual_output/('failure_'+stamp+'.txt')).write_text(failure, encoding='utf8')
archive = shutil.make_archive('/content/'+residual_output.name, 'zip',
    root_dir=str(residual_output.parent), base_dir=residual_output.name)
print('结果 ZIP：', archive, flush=True)
files.download(archive)
if failure:
    raise RuntimeError('实验停止，诊断ZIP已保存并下载；请把它发回。未跳过故障配置。')
assert json.loads((residual_output/'completion.json').read_text())['completed']
if mode == 'screen':
    import pandas as pd
    from IPython.display import display
    display(pd.read_csv(residual_output/'summary_full.csv'))
    display(pd.read_csv(residual_output/'paired_comparisons.csv'))
print('请把下载的结果 ZIP 发回，核对机制和事件差异后再决定是否搜索结构参数。')
