"""Validate the existing server setup and launch one SAMMLV run in background."""
from pathlib import Path
from datetime import datetime, timezone
import os
import subprocess
import sys

root=Path('/root/workspace/glsd_agreement')
code=Path(__file__).resolve().parent
entry=code/'server_entry.py'
for path in [root, Path('/content/ME-TST/training_utils.py'), Path('/content/drive/MyDrive/GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE')]:
    if not path.exists(): raise RuntimeError(f'Missing existing server mapping: {path}')
pidfile=root/'oto_structure_pid.txt'
if pidfile.exists():
    old=int(pidfile.read_text().strip())
    cmdline=Path(f'/proc/{old}/cmdline')
    if cmdline.exists() and str(entry).encode() in cmdline.read_bytes():
        raise RuntimeError(f'A structure experiment is already running: PID {old}')
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(code),'-p','test_structure_bonus.py','-v'],check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(code),'-p','test_one_to_one.py','-v'],check=True)
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
result=root/'results'/('oto_structure_sammlv_'+tag)
log=result.with_suffix('.log')
log.parent.mkdir(parents=True,exist_ok=True)
with log.open('x') as stream:
    proc=subprocess.Popen([sys.executable,'-u',str(entry),'--setting','sammlv','--output',str(result)],
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
pidfile.write_text(str(proc.pid))
(root/'latest_oto_log.txt').write_text(str(log))
print('PID:',proc.pid)
print('RESULT:',result)
print('LOG:',log)
print('BACKGROUND_LAUNCH = PASS (experiment completion is reported in log)')
