# Paste this entire file into ONE new cell in the prepared ME-TST+ notebook.
from google.colab import files
from pathlib import Path
from datetime import datetime, timezone
import io
import os
import subprocess
import sys
import tempfile
import zipfile

# First fix and run SAMMLV. For CAS(ME)3 later set this to "casme3".
SETTING = "sammlv"
THRESHOLD_GRID = "refined"  # All five methods: tau=0.05,...,0.95, step 0.05.
PROBE_ONLY = False  # True runs the regression and Native without the full search.

assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
uploaded_oto = files.upload()  # Select metst_one_to_one_refined_tuning.zip
assert len(uploaded_oto) == 1, "请一次只上传一个新 ZIP 包"
uploaded_name, package_bytes = next(iter(uploaded_oto.items()))
expected = {
    "run_one_to_one_full_tuning.py", "one_to_one_evaluator.py",
    "run_full_fusion_tuning.py", "run_fusion_screening.py",
    "official_response_component_ablation.py", "colab_one_to_one_entry.py",
    "README_ONE_TO_ONE.md",
}
oto_dir = Path(tempfile.mkdtemp(prefix="glsd_one_to_one_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
    assert len(archive.namelist()) == len(expected) and set(archive.namelist()) == expected, "ZIP 包内容不匹配"
    archive.extractall(oto_dir)

run_tag_oto = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
oto_root = Path("/content/drive/MyDrive/GLSD_ONE_TO_ONE_FULL_TUNING")
oto_root.mkdir(parents=True, exist_ok=True)
oto_output = oto_root / ("oto_refined_" + run_tag_oto)
oto_log = oto_root / ("oto_refined_" + run_tag_oto + ".log")
command_oto = [sys.executable, "-u", str(oto_dir / "colab_one_to_one_entry.py"),
               "--setting", SETTING, "--threshold-grid", THRESHOLD_GRID, "--output", str(oto_output)]
if PROBE_ONLY:
    command_oto.append("--probe-only")
env_oto = dict(os.environ)
env_oto["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([
    str(oto_dir), os.getcwd(), *[p for p in sys.path if p], env_oto.get("PYTHONPATH", "")
]))
print("上传文件：", uploaded_name)
print("运行目录：", oto_dir)
print("新结果目录：", oto_output)
print("日志：", oto_log)
with oto_log.open("x", encoding="utf-8") as log:
    process_oto = subprocess.Popen(command_oto, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env_oto)
    try:
        for line in process_oto.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code_oto = process_oto.wait()
    except BaseException:
        process_oto.terminate()
        try:
            process_oto.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process_oto.kill()
            process_oto.wait()
        raise
assert code_oto == 0, f"运行停止，请查看日志：{oto_log}"
assert (oto_output / "completion.json").is_file(), "未生成完成标志"
print("运行完成：", oto_output)

