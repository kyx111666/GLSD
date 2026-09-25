# %% 1. 接续现有 Colab：上传本路线代码包，复用已有环境和 Drive
from pathlib import Path
import hashlib, json, subprocess, sys, zipfile
from importlib import metadata
from google.colab import files, drive

if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')

# 已上传时填现有路径即可；留空才弹出上传窗口。
PACKAGE_ZIP = ''
if not PACKAGE_ZIP:
    uploaded = files.upload()
    packages = [name for name in uploaded if name.endswith('.zip') and 'shift_control' in name]
    if len(packages) != 1:
        raise RuntimeError('请只选择本路线 shift_control_fusion_ready.zip')
    PACKAGE_ZIP = packages[0]
archive = Path(PACKAGE_ZIP)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
extract = Path('/content')/('shift_control_release_'+digest[:12])
extract.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        dest = (extract/info.filename).resolve()
        if extract.resolve() not in dest.parents:
            raise RuntimeError('Unsafe archive path')
    z.extractall(extract)
CODE = extract/'shift_control_fusion'
manifest = json.loads((CODE/'PACKAGE_MANIFEST.json').read_text())
for rel, expected in manifest['files'].items():
    if hashlib.sha256((CODE/rel).read_bytes()).hexdigest() != expected:
        raise RuntimeError('代码包校验失败: '+rel)

# 只补缺失或不兼容依赖；正式运行在独立 Python 进程中，无需重跑训练环境。
from packaging.requirements import Requirement
missing = []
for line in (CODE/'requirements.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'): continue
    req = Requirement(line)
    try: version = metadata.version(req.name)
    except metadata.PackageNotFoundError: missing.append(line); continue
    if version not in req.specifier: missing.append(line)
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
subprocess.run([sys.executable, str(CODE/'test_shift.py')], check=True)
print('CODE =', CODE, '\nPACKAGE_SHA256 =', digest)

# %% 2. 配置数据集和独立输出，先输入审计与真实 probe
# 首轮先 SAM。机制复核后，切换为 casme3 执行同一冻结规则；不重训 backbone。
DATASET = 'sammlv'  # sammlv / casme3
DATA_ROOT = Path('/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP')
DUMP = DATA_ROOT/({'sammlv':'SAMMLV_method1_strategy1', 'casme3':'CASME_3_method1_strategy1'}[DATASET])
RUN_ROOT = Path('/content/drive/MyDrive/GLSD_SHIFT_CONTROL_FIXED_V1')/DATASET
# 若实际响应位置不同，只改 DUMP。必须是直接含 subject_*.pkl 的目录，不是旧缓存。
RUN_ROOT.mkdir(parents=True, exist_ok=True)

def launch(mode, output, extra=()):
    command = [sys.executable, '-u', str(CODE/'run_shift.py'), '--dataset', DATASET,
               '--dump', str(DUMP), '--output', str(output), '--mode', mode, *extra]
    if (output/'run_manifest.json').exists(): command.append('--resume')
    subprocess.run(command, check=True)

launch('audit', RUN_ROOT/'input_audit')
launch('probe', RUN_ROOT/'probe', ('--probe-grid',))
print(json.loads((RUN_ROOT/'probe/probe.json').read_text())['scope'])

# %% 3. 首轮固定结构全量运行；中断后重跑此格安全接续
# 七评分×301阈值；每折仅在其余被试 pooled full F1 选阈值。
# 首轮结构是单点，不扩搜结构，不自动追加权重或分位数。
OUTPUT = RUN_ROOT/'screen'
launch('screen', OUTPUT)
print((OUTPUT/'completion.json').read_text())
for name in ('fixed_structure_baselines.csv', 'ablations_retuned.csv', 'ablations_locked_diagnostic.csv'):
    print('\n'+name+'\n'+(OUTPUT/name).read_text())

# %% 4. 独立审计和结果回传
subprocess.run([sys.executable, str(CODE/'verify_results.py'), str(OUTPUT)], check=True)
# screen.zip 已包含计数、候选、事件账本、选择、输入身份和检查点。
RESULT_ZIP = Path(str(OUTPUT)+'.zip')
print('请回传:', RESULT_ZIP)
files.download(str(RESULT_ZIP))
