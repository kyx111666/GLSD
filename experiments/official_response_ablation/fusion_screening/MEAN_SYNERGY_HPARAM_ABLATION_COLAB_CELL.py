# 在已完成 ME-TST+ 环境初始化并挂载 Drive 的 Colab 中粘贴本 cell。
# 先用 PROBE_ONLY=True 做一次环境检查；通过后改为 False，依次运行所有 epsilon。
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

SETTING = "sammlv"  # 第二次运行改成 "casme3"
PROBE_ONLY = True
EPSILONS = (0.0, 0.001, 0.002, 0.003, 0.005)

assert SETTING in {"sammlv", "casme3"}
assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
repo_root = Path("/content/ME-TST")
assert (repo_root / "Utils").is_dir(), "找不到 /content/ME-TST/Utils，请先运行原环境初始化"

uploaded = files.upload()  # 选择 metst_mean_synergy_one_to_one.zip
assert len(uploaded) == 1, "请一次只上传一个 ZIP 包"
package_name, package_bytes = next(iter(uploaded.items()))
assert package_name == "metst_mean_synergy_one_to_one.zip", "请选择新的 Mean synergy 一对一运行包"

expected = {
    "run_full_fusion_tuning.py", "run_one_to_one_full_tuning.py",
    "one_to_one_evaluator.py", "run_fusion_screening.py",
    "official_response_component_ablation.py", "colab_one_to_one_entry.py",
}
package_dir = Path(tempfile.mkdtemp(prefix="glsd_mean_hparam_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
    assert set(archive.namelist()) == expected, "运行包内容不匹配"
    assert archive.testzip() is None, "运行包损坏"
    archive.extractall(package_dir)

output_root = Path("/content/drive/MyDrive/GLSD_MEAN_SYNERGY_HPARAM_ABLATION")
output_root.mkdir(parents=True, exist_ok=True)
epsilons = (0.002,) if PROBE_ONLY else EPSILONS
run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

for epsilon in epsilons:
    epsilon_tag = f"{epsilon:.3f}".replace(".", "p")
    output_dir = output_root / f"{SETTING}_{run_tag}_epsilon_{epsilon_tag}"
    log_path = output_root / f"{SETTING}_{run_tag}_epsilon_{epsilon_tag}.log"
    command = [sys.executable, "-u", str(package_dir / "colab_one_to_one_entry.py"),
               "--setting", SETTING, "--threshold-grid", "legacy",
               "--structure-grid", "legacy", "--beta-grid", "all",
               "--mean-selection", "near_optimal_mean_synergy_v1",
               "--mean-epsilon", str(epsilon), "--output", str(output_dir)]
    if PROBE_ONLY:
        command.append("--probe-only")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(repo_root), str(package_dir), environment.get("PYTHONPATH", "")])
    print(f"epsilon={epsilon:.3f}; 输出：{output_dir}")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env=environment, cwd=str(repo_root))
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = process.wait()
    assert code == 0, f"epsilon={epsilon:.3f} 运行失败，请查看日志：{log_path}"
    completion = json.loads((output_dir / "completion.json").read_text())
    assert completion.get("completed") is True, f"epsilon={epsilon:.3f} 输出未完成"
    result_zip = shutil.make_archive(str(output_dir), "zip",
                                     root_dir=output_dir.parent, base_dir=output_dir.name)
    print("结果 ZIP：", result_zip)

print("MEAN_SYNERGY_HPARAM_ABLATION = PASS；PASS 只表示运行完成，请审计各 epsilon 的结果。")
