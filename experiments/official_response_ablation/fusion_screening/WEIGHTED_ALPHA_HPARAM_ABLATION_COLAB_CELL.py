# 在已经完成 ME-TST+ 环境初始化并挂载 Drive 的原 Colab 中，粘贴整个 cell。
# 先用 PROBE_ONLY=True 检查路径、官方回放和一对一评价；通过后改为 False。
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

SETTING = "sammlv"  # 第二次运行 CAS(ME)3 时改为 "casme3"；也可用 "both"
PROBE_ONLY = True
ALPHAS = (0.0, 0.3, 0.5, 0.7, 1.0)

assert SETTING in {"sammlv", "casme3", "both"}
assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
assert Path("/content/ME-TST/Utils").is_dir(), "请先运行原 ME-TST 环境初始化"

uploaded = files.upload()  # 选择 metst_weighted_alpha_hparam_ablation.zip
assert len(uploaded) == 1, "请一次只上传一个 ZIP 包"
package_name, package_bytes = next(iter(uploaded.items()))
assert package_name == "metst_weighted_alpha_hparam_ablation.zip", "请选择新的 alpha 消融运行包"

expected = {
    "run_alpha_hparam_ablation.py",
    "run_full_fusion_tuning.py",
    "run_fusion_screening.py",
    "one_to_one_evaluator.py",
    "official_response_component_ablation.py",
    "test_alpha_hparam_ablation.py",
}
package_dir = Path(tempfile.mkdtemp(prefix="glsd_weighted_alpha_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
    assert set(archive.namelist()) == expected, f"ZIP 内容不匹配：{sorted(archive.namelist())}"
    assert archive.testzip() is None, "ZIP 文件损坏"
    archive.extractall(package_dir)

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(
    [str(Path("/content/ME-TST")), str(package_dir), env.get("PYTHONPATH", "")]
)

test = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", str(package_dir),
     "-p", "test_alpha_hparam_ablation.py", "-v"],
    env=env, cwd="/content/ME-TST", check=True,
)
print("ALPHA_HPARAM_UNIT_TESTS = PASS")

run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
output_root = Path("/content/drive/MyDrive/GLSD_WEIGHTED_ALPHA_HPARAM_ABLATION")
output_root.mkdir(parents=True, exist_ok=True)
output_dir = output_root / f"{SETTING}_{run_tag}"
log_path = output_root / f"{SETTING}_{run_tag}.log"
alpha_arg = ",".join(f"{x:.6g}" for x in ALPHAS)
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
command = [sys.executable, "-u", "-c", bootstrap, str(package_dir / "run_alpha_hparam_ablation.py"),
           "--setting", SETTING, "--alphas", alpha_arg, "--output", str(output_dir)]
if PROBE_ONLY:
    command.append("--probe-only")

print("输出目录：", output_dir)
print("日志：", log_path)
with log_path.open("x", encoding="utf-8") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env, cwd="/content/ME-TST")
    try:
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        returncode = process.wait()
    except BaseException:
        process.terminate()
        raise
assert returncode == 0, f"运行失败，请检查日志：{log_path}"
completion = json.loads((output_dir / "completion.json").read_text(encoding="utf-8"))
assert completion.get("completed") is True, "缺少 completed=true"
result_zip = shutil.make_archive(str(output_dir), "zip",
                                 root_dir=output_dir.parent, base_dir=output_dir.name)
print("结果 ZIP：", result_zip)
print("ALPHA_HPARAM_ABLATION = PASS；PASS 只表示执行完成，请审计 alpha_summary.csv。")
