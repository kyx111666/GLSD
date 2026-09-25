# 在已经挂载 Drive、并完成原 ME-TST+ 官方响应环境初始化的 Colab 中运行本 cell。
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

SETTING = "sammlv"  # 第二次运行前改成 "casme3"

assert SETTING in {"sammlv", "casme3"}
assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"

# A fresh Colab runtime does not have the small helper package yet. Upload the
# package once when needed; it contains only runner dependencies and no data.
existing_dir = Path("/content/glsd_fusion_screening")
if not (existing_dir / "run_fusion_screening.py").is_file():
    uploaded_base = files.upload()  # 选择 metst_fusion_screening.zip
    assert len(uploaded_base) == 1, "请一次只上传一个基础环境 ZIP"
    base_name, base_bytes = next(iter(uploaded_base.items()))
    assert base_name == "metst_fusion_screening.zip", "请选择 metst_fusion_screening.zip"
    expected_base = {
        "run_fusion_screening.py", "README.md", "fusion_screening_colab.ipynb",
        "official_response_component_ablation.py",
    }
    existing_dir.mkdir(parents=True, exist_ok=True)
    with __import__("zipfile").ZipFile(io.BytesIO(base_bytes)) as archive:
        assert set(archive.namelist()) == expected_base, "基础环境 ZIP 内容不匹配"
        assert archive.testzip() is None, "基础环境 ZIP 已损坏"
        archive.extractall(existing_dir)
    print("基础环境已解压：", existing_dir)

uploaded = files.upload()  # 选择本次修改后的 run_full_fusion_tuning.py
assert len(uploaded) == 1, "请一次只上传一个 runner 文件"
uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
assert uploaded_name.endswith("run_full_fusion_tuning.py"), "请选择修改后的 run_full_fusion_tuning.py"

work_dir = Path(tempfile.mkdtemp(prefix="glsd_mean_retake_", dir="/content"))
runner_path = work_dir / "run_full_fusion_tuning.py"
runner_path.write_bytes(uploaded_bytes)

# The existing environment supplies run_fusion_screening.py, the official
# response component, and the sealed ME-TST checkout. Keep this runner isolated
# from any previous output directory.
assert (existing_dir / "run_fusion_screening.py").is_file(), "缺少原有 glsd_fusion_screening 环境"
isolated_dir = work_dir / "runtime"
isolated_dir.mkdir()
for name in ("run_fusion_screening.py", "official_response_component_ablation.py"):
    source = existing_dir / name
    assert source.is_file(), f"缺少依赖：{source}"
    shutil.copy2(source, isolated_dir / name)
shutil.copy2(runner_path, isolated_dir / runner_path.name)

# The archived ME-TST evaluator calls pandas.DataFrame.append, which was
# removed in current Colab pandas. Restore that old API only inside this
# subprocess; the evaluator's matching logic remains unchanged.
(isolated_dir / "sitecustomize.py").write_text(
    "import pandas as _pd\n"
    "if not hasattr(_pd.DataFrame, 'append') and hasattr(_pd.DataFrame, '_append'):\n"
    "    _pd.DataFrame.append = _pd.DataFrame._append\n",
    encoding="utf-8",
)

run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
output_root = Path("/content/drive/MyDrive/GLSD_MEAN_RETAKE")
output_root.mkdir(parents=True, exist_ok=True)
output_dir = output_root / f"{SETTING}_{run_tag}"
log_path = output_root / f"{SETTING}_{run_tag}.log"
command = [sys.executable, "-u", str(isolated_dir / "run_full_fusion_tuning.py"),
           "--setting", SETTING, "--output", str(output_dir)]
environment = dict(os.environ)
# The official response adapter imports `Utils` from the ME-TST checkout. The
# notebook may have cloned that checkout without leaving it as the current
# directory, so make the import root explicit for the subprocess.
repo_candidates = [Path("/content/ME-TST"), Path.cwd()]
repo_root = next((p for p in repo_candidates if (p / "Utils").is_dir()), None)
if repo_root is None:
    raise RuntimeError(
        "找不到 ME-TST 仓库根目录（需要其中包含 Utils/）。请先运行原 notebook 的 ME-TST 环境初始化单元。"
    )
environment["PYTHONPATH"] = os.pathsep.join(
    [str(repo_root), str(isolated_dir), str(existing_dir), environment.get("PYTHONPATH", "")])

print("上传文件：", uploaded_name)
print("新结果目录：", output_dir)
print("日志：", log_path)
with log_path.open("x", encoding="utf-8") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=environment, cwd=str(repo_root))
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    code = process.wait()
assert code == 0, f"运行失败，请查看日志：{log_path}"
completion = json.loads((output_dir / "completion.json").read_text())
assert completion.get("completed") is True, "输出未完成"
result_zip = shutil.make_archive(str(output_dir), "zip",
                                 root_dir=output_dir.parent, base_dir=output_dir.name)
print("MEAN_RETAKE_EXECUTION = PASS；PASS 只表示运行完成，请审计 F1 和消融结果。")
print("结果 ZIP：", result_zip)
