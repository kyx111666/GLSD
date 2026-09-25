# Native--GLSD 输出级融合：在新的、已经完成原 ME-TST+ 环境初始化的 Colab runtime 中
# 新建一个代码 cell，粘贴本文件全文运行。GPU 不需要。

from google.colab import files, drive
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


SETTING = "metst_sammlv"  # 第二次运行改为 "metst_casme3"
MODE = "Agreement"       # 需要并集对照时改为 "Union"
assert SETTING in {"metst_sammlv", "metst_casme3"}
assert MODE in {"Agreement", "Union"}

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")

assert Path("/content/ME-TST/Utils").is_dir(), (
    "找不到 /content/ME-TST/Utils。请先运行原 ME-TST+ notebook 的环境初始化单元，"
    "不要在旧 G/L 融合 cell 中续跑。"
)

print("请上传：native_glsd_output_fusion.zip")
uploaded = files.upload()
assert len(uploaded) == 1, "请一次只上传一个 ZIP"
zip_name, zip_bytes = next(iter(uploaded.items()))
assert zip_name == "native_glsd_output_fusion.zip", "请选择 native_glsd_output_fusion.zip"

work = Path(tempfile.mkdtemp(prefix="native_glsd_fusion_", dir="/content"))
package = work / "package"
package.mkdir()
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
    names = set(archive.namelist())
    required = {
        "native_glsd_output_fusion/run_native_glsd_output_fusion.py",
        "native_glsd_output_fusion/official_response_component_ablation.py",
    }
    assert required.issubset(names), f"ZIP 内容缺少文件：{required - names}"
    archive.extractall(package)

package = package / "native_glsd_output_fusion"
runner = package / "run_native_glsd_output_fusion.py"
helper = package / "official_response_component_ablation.py"

# 旧版 ME-TST evaluator 使用 DataFrame.append；只在子进程中恢复兼容接口。
(package / "sitecustomize.py").write_text(
    "import pandas as _pd\n"
    "if not hasattr(_pd.DataFrame, 'append') and hasattr(_pd.DataFrame, '_append'):\n"
    "    _pd.DataFrame.append = _pd.DataFrame._append\n",
    encoding="utf-8",
)

output_root = Path("/content/drive/MyDrive/GLSD_NATIVE_GLSD_OUTPUT_FUSION_V1")
output_root.mkdir(parents=True, exist_ok=True)
run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
output = output_root / f"{SETTING}_{run_tag}"
log_path = output_root / f"{SETTING}_{run_tag}.log"

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([
    str(package),
    "/content/ME-TST",
    env.get("PYTHONPATH", ""),
])

command = [
    sys.executable,
    "-u",
    str(runner),
    "--setting",
    SETTING,
    "--output",
    str(output),
    "--mode",
    MODE,
]

print("SETTING:", SETTING)
print("MODE:", MODE)
print("OUTPUT:", output)
print("LOG:", log_path)
print("GPU required: False")

with log_path.open("x", encoding="utf-8") as log:
    process = subprocess.Popen(
        command,
        cwd="/content/ME-TST",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return_code = process.wait()

assert return_code == 0, f"运行失败，请检查日志：{log_path}"
completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
assert completion.get("completed") is True, "缺少 completed=true"

print("\n运行完成。请读取：")
print(output / "summary.csv")
print(output / "ablation_per_subject_counts.csv")
print(output / "outer_folds.csv")
