# Table 3 专用：固定 Table 1 的 outer_selected_configs，只扫描 alpha。
# 先保持 PROBE_ONLY=True；它会对所有被试检查 alpha=0.5 是否逐被试复现 Table 1。
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

SETTING = "sammlv"  # 第二次运行改成 "casme3"；也可用 "both"
PROBE_ONLY = True
ALPHAS = (0.0, 0.3, 0.5, 0.7, 1.0)

assert SETTING in {"sammlv", "casme3", "both"}
assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
assert Path("/content/ME-TST/Utils").is_dir(), "请先运行原 ME-TST 环境初始化"

uploaded = files.upload()  # 选择 metst_locked_table3_alpha.zip
assert len(uploaded) == 1 and "metst_locked_table3_alpha.zip" in uploaded, \
    "请选择 metst_locked_table3_alpha.zip"
package_bytes = uploaded["metst_locked_table3_alpha.zip"]

package_dir = Path(tempfile.mkdtemp(prefix="glsd_locked_table3_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
    expected = {
        "run_locked_alpha_ablation.py", "run_alpha_hparam_ablation.py",
        "run_full_fusion_tuning.py", "run_fusion_screening.py",
        "one_to_one_evaluator.py", "official_response_component_ablation.py",
        "test_alpha_hparam_ablation.py", "test_locked_alpha_ablation.py",
    }
    assert set(archive.namelist()) == expected, f"ZIP 内容不匹配：{sorted(archive.namelist())}"
    assert archive.testzip() is None, "ZIP 文件损坏"
    archive.extractall(package_dir)

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(
    [str(Path("/content/ME-TST")), str(package_dir), env.get("PYTHONPATH", "")]
)

test = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", str(package_dir),
     "-p", "test*.py", "-v"],
    env=env, cwd="/content/ME-TST", check=True,
)
print("LOCKED_TABLE3_UNIT_TESTS = PASS")

# 旧版 ME-TST 使用 DataFrame.append；Colab 当前 pandas 已移除该接口。
# 必须在子进程创建 ME-TST context 前安装兼容别名。
bootstrap = r"""
import runpy, sys
try:
    import pandas as pd
    if not hasattr(pd.DataFrame, "append"):
        def dataframe_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
            return pd.concat([self, other], ignore_index=ignore_index,
                             verify_integrity=verify_integrity, sort=sort)
        pd.DataFrame.append = dataframe_append
except ImportError:
    pass
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
output_root = Path("/content/drive/MyDrive/GLSD_LOCKED_TABLE3_ALPHA")
output_root.mkdir(parents=True, exist_ok=True)
output_dir = output_root / f"{SETTING}_{run_tag}"
log_path = output_root / f"{SETTING}_{run_tag}.log"
alpha_arg = ",".join(f"{x:.6g}" for x in ALPHAS)
command = [
    sys.executable, "-u", "-c", bootstrap, str(package_dir / "run_locked_alpha_ablation.py"),
    "--setting", SETTING, "--alphas", alpha_arg, "--output", str(output_dir),
]
if PROBE_ONLY:
    command.append("--probe-only")

print("输出目录：", output_dir)
print("日志：", log_path)
with log_path.open("x", encoding="utf-8") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env, cwd="/content/ME-TST")
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    returncode = process.wait()
assert returncode == 0, f"运行失败，请检查日志：{log_path}"
completion = json.loads((output_dir / "completion.json").read_text(encoding="utf-8"))
assert completion.get("completed") is True, "缺少 completed=true"
result_zip = shutil.make_archive(str(output_dir), "zip",
                                 root_dir=output_dir.parent, base_dir=output_dir.name)
print("结果 ZIP：", result_zip)
print("LOCKED_TABLE3_ALPHA = PASS；请检查 alpha_0p5_alignment.json 和 alpha_summary.csv。")
