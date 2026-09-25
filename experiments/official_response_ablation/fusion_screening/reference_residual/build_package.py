"""Build the audited fixed-structure residual screening package and Colab cells."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve()
ROOT = HERE.parent
METST = ROOT.parent/'metst_fusion_screening'
FILES = {
    'run_reference_residual.py': ROOT/'run_reference_residual.py',
    'residual_features.py': ROOT/'residual_features.py',
    'typed_matching.py': ROOT/'typed_matching.py',
    'test_reference_residual.py': ROOT/'test_reference_residual.py',
    'sealed_decoder_fixture.py': ROOT/'sealed_decoder_fixture.py',
    'remote_official_fixture.py': ROOT/'remote_official_fixture.py',
    'colab_reference_residual_entry.py': ROOT/'colab_reference_residual_entry.py',
    'run_colab_reference_residual.py': ROOT/'run_colab_reference_residual.py',
    'run_server_reference_residual.py': ROOT/'run_server_reference_residual.py',
    'SERVER_SETUP.md': ROOT/'SERVER_SETUP.md',
    'run_p8_phase2.py': METST/'run_p8_phase2.py',
    'run_p8_one_to_one.py': METST/'run_p8_one_to_one.py',
    'run_p8_threshold_control.py': METST/'run_p8_threshold_control.py',
    'run_generalized_mean_screening.py': METST/'run_generalized_mean_screening.py',
    'official_response_component_ablation.py': METST/'official_response_component_ablation.py',
    'colab_p8_entry.py': METST/'colab_p8_entry.py',
    'one_to_one_evaluator.py': METST.parent/'one_to_one_evaluator.py',
}
manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in FILES.items()}
archive = ROOT/'metst_reference_residual.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, path in FILES.items(): z.write(path, name)
    z.writestr('package_manifest.json', json.dumps(manifest, indent=2)+'\n')
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

upload = '''from google.colab import files
from pathlib import Path
import hashlib, io, json, tempfile, zipfile

uploaded = files.upload()  # 只选择 metst_reference_residual.zip
assert len(uploaded) == 1, '请只上传一个残差筛查 ZIP'
data = next(iter(uploaded.values()))
assert hashlib.sha256(data).hexdigest() == %r, 'ZIP版本不符，请重新下载'
RESIDUAL_DIR = Path(tempfile.mkdtemp(prefix='glsd_reference_residual_', dir='/content'))
with zipfile.ZipFile(io.BytesIO(data)) as z:
    manifest = json.loads(z.read('package_manifest.json'))
    assert set(z.namelist()) == set(manifest) | {'package_manifest.json'}
    for name, digest in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest
    z.extractall(RESIDUAL_DIR)
print('RESIDUAL_PACKAGE = PASS', RESIDUAL_DIR)
''' % archive_sha
run = '''RESIDUAL_MODE = 'screen'
RESIDUAL_RESUME = None
RESIDUAL_AUDIT = '/content/drive/MyDrive/GLSD_EVALUATOR_AUDIT/sammlv_20260922T091343_286158Z'
entry = RESIDUAL_DIR / 'run_colab_reference_residual.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
'''
(ROOT/'REFERENCE_RESIDUAL_COLAB_CELLS.py').write_text(
    '# %% CELL 1：上传残差筛查包\n'+upload+'\n\n# %% CELL 2：SAMMLV固定结构筛查\n'+run,
    encoding='utf8')
cells = [dict(cell_type='markdown', metadata={}, id='intro', source=[
    '# GLSD 参考尺度残差：SAMMLV固定结构筛查\n',
    '\n在已通过评价核验的 P8 笔记本末尾依次运行下面两格。\n',
    '固定 a0=2、rho=2；五种评分各自内层选阈值；不训练、不生成响应。\n'])]
for i, (title, code) in enumerate([('CELL 1：上传残差筛查包', upload), ('CELL 2：SAMMLV固定结构筛查', run)]):
    cells += [dict(cell_type='markdown', metadata={}, id='title'+str(i), source=['## '+title+'\n']),
              dict(cell_type='code', metadata={}, id='code'+str(i), source=code.splitlines(keepends=True), execution_count=None, outputs=[])]
nb = dict(nbformat=4, nbformat_minor=5,
          metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')),
          cells=cells)
(ROOT/'Colab_参考尺度残差_SAMMLV固定结构筛查.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('ZIP SHA256:', archive_sha)
print(archive)
