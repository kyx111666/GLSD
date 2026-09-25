# 在已经配置好的 ME-TST+ Colab notebook 中，新建一个代码 cell，粘贴本文件全文。
from google.colab import files
from pathlib import Path
from datetime import datetime, timezone
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile


assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
uploaded_structure = files.upload()  # 选择 metst_fusion_event_diagnostic.zip
assert len(uploaded_structure) == 1, "请一次只上传一个新 ZIP 包"
structure_name, structure_bytes = next(iter(uploaded_structure.items()))
structure_expected = {
    "run_fusion_event_diagnostic.py", "one_to_one_evaluator.py",
    "run_full_fusion_tuning.py", "run_fusion_screening.py",
    "official_response_component_ablation.py", "run_one_to_one_full_tuning.py",
    "structure_expansion.py", "structure_selected_replay_reference.json",
    "README_EVENT_DIAGNOSTIC_CN.md",
}
structure_dir = Path(tempfile.mkdtemp(prefix="glsd_event_diagnostic_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(structure_bytes)) as archive:
    assert len(archive.namelist()) == len(structure_expected) and set(archive.namelist()) == structure_expected, "请上传新的事件诊断包，不能使用旧搜索包"
    assert archive.testzip() is None, "ZIP 文件损坏"
    archive.extractall(structure_dir)

structure_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
structure_root = Path("/content/drive/MyDrive/GLSD_ONE_TO_ONE_FULL_TUNING")
structure_root.mkdir(parents=True, exist_ok=True)
structure_output = structure_root / ("oto_events_" + structure_tag)
structure_log = structure_root / ("oto_events_" + structure_tag + ".log")
structure_command = [sys.executable, "-u", str(structure_dir / "run_fusion_event_diagnostic.py"),
    "--output", str(structure_output)]
structure_env = dict(os.environ)
structure_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([
    str(structure_dir), os.getcwd(), *[p for p in sys.path if p], structure_env.get("PYTHONPATH", "")
]))
print("上传文件：", structure_name)
print("新结果目录：", structure_output)
print("日志：", structure_log)
with structure_log.open("x", encoding="utf-8") as log:
    structure_process = subprocess.Popen(structure_command, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, env=structure_env)
    try:
        for line in structure_process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        structure_code = structure_process.wait()
    except BaseException:
        structure_process.terminate()
        try:
            structure_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            structure_process.kill()
            structure_process.wait()
        raise
assert structure_code == 0, f"运行停止，请查看日志：{structure_log}"
assert json.loads((structure_output / "completion.json").read_text())["completed"], "实验未完成"
structure_result_zip = shutil.make_archive(str(structure_output), "zip",
    root_dir=structure_output.parent, base_dir=structure_output.name)
print("诊断回放完成，已核对原预测和计数。请发回结果 ZIP 分析互补性。")
print("请下载并发回这个结果 ZIP：", structure_result_zip)
