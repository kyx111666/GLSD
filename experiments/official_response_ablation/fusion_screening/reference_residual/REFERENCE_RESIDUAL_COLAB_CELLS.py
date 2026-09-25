# %% CELL 1：上传残差筛查包
from google.colab import files
from pathlib import Path
import hashlib, io, json, tempfile, zipfile

uploaded = files.upload()  # 只选择 metst_reference_residual.zip
assert len(uploaded) == 1, '请只上传一个残差筛查 ZIP'
data = next(iter(uploaded.values()))
assert hashlib.sha256(data).hexdigest() == 'a28da80c055b7afeb2d155b4f0293cc4df4440dd2a2fc8b20798c62f68093e6f', 'ZIP版本不符，请重新下载'
RESIDUAL_DIR = Path(tempfile.mkdtemp(prefix='glsd_reference_residual_', dir='/content'))
with zipfile.ZipFile(io.BytesIO(data)) as z:
    manifest = json.loads(z.read('package_manifest.json'))
    assert set(z.namelist()) == set(manifest) | {'package_manifest.json'}
    for name, digest in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest
    z.extractall(RESIDUAL_DIR)
print('RESIDUAL_PACKAGE = PASS', RESIDUAL_DIR)


# %% CELL 2：SAMMLV固定结构筛查
RESIDUAL_MODE = 'screen'
RESIDUAL_RESUME = None
RESIDUAL_AUDIT = '/content/drive/MyDrive/GLSD_EVALUATOR_AUDIT/sammlv_20260922T091343_286158Z'
entry = RESIDUAL_DIR / 'run_colab_reference_residual.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
