# Table 3: BoostingVRME with Table 1 locked GLSD-90 outer configurations.
# Run once with PROBE_ONLY=True, then repeat with PROBE_ONLY=False.
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

SETTING = "boosting_sammlv"  # then "boosting_casme3"; "both" is also supported
PROBE_ONLY = True
ALPHAS = (0.0, 0.3, 0.5, 0.7, 1.0)

assert SETTING in {"boosting_sammlv", "boosting_casme3", "both"}
assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
assert Path("/content/BoostingVRME").is_dir(), "请先完成 BoostingVRME 环境初始化"

uploaded = files.upload()  # 选择 boosting_locked_table3_alpha.zip
assert len(uploaded) == 1, "请只上传一个 Boosting Table 3 alpha ZIP"
uploaded_name, package_bytes = next(iter(uploaded.items()))
print("上传包：", uploaded_name)

package_dir = Path(tempfile.mkdtemp(prefix="glsd_boosting_locked_table3_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
    expected = {
        "run_locked_alpha_ablation_boosting.py",
        "official_response_component_ablation.py",
        "boosting_official_glds_full.py",
        "boosting_official_glds_full_casme3.py",
    }
    assert set(archive.namelist()) == expected, f"ZIP 内容不匹配：{sorted(archive.namelist())}"
    assert archive.testzip() is None, "ZIP 文件损坏"
    archive.extractall(package_dir)

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(
    [str(Path("/content/BoostingVRME")), str(package_dir), env.get("PYTHONPATH", "")]
)

run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
output_root = Path("/content/drive/MyDrive/GLSD_LOCKED_TABLE3_ALPHA_BOOSTING")
output_root.mkdir(parents=True, exist_ok=True)
output_dir = output_root / f"{SETTING}_{run_tag}"
log_path = output_root / f"{SETTING}_{run_tag}.log"
alpha_arg = ",".join(f"{x:.6g}" for x in ALPHAS)
command = [
    sys.executable, "-u", "-c",
    "import pandas as pd; "
    "pd.DataFrame.append = (lambda self, other, ignore_index=False, verify_integrity=False, sort=False: "
    "pd.concat([self, pd.DataFrame([other]) if isinstance(other, dict) else other.to_frame().T if isinstance(other, pd.Series) else other], "
    "ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)) if not hasattr(pd.DataFrame, 'append') else None; "
    "import runpy, sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')",
    str(package_dir / "run_locked_alpha_ablation_boosting.py"),
    "--setting", SETTING, "--alphas", alpha_arg, "--output", str(output_dir),
]
if PROBE_ONLY:
    command.append("--probe-only")

print("输出目录：", output_dir)
print("日志：", log_path)
with log_path.open("x", encoding="utf-8") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env, cwd="/content/BoostingVRME")
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    returncode = process.wait()
assert returncode == 0, f"运行失败，请检查日志：{log_path}"

for setting in ("boosting_sammlv", "boosting_casme3") if SETTING == "both" else (SETTING,):
    detail = output_dir / setting / "alpha_0p5_alignment.json"
    assert json.loads(detail.read_text(encoding="utf-8"))["mismatches"] == [], detail

result_zip = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
print("结果 ZIP：", result_zip)
print("BOOSTING_LOCKED_TABLE3_ALPHA = PASS；请检查 alpha_summary.csv。")
