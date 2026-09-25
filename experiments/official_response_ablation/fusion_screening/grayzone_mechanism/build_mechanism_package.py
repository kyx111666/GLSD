"""Reproducible diagnostic ZIP and complete Colab cell bound by SHA256."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent
old=root.parent/'grayzone_support'
with zipfile.ZipFile(old/'metst_grayzone_support_v1.zip') as z:
    inherited=[n for n in json.loads(z.read('package_manifest.json')) if n.endswith('.py')]
    for n in inherited:assert (root/n).read_bytes()==z.read(n),n+' changed'
new=['mechanism_features.py','analyze_export.py','run_mechanism.py','colab_mechanism_entry.py',
     'run_colab_mechanism.py','test_mechanism.py','sealed_core_test_fixture.py','README_MECHANISM_CN.md']
sources=inherited+new
hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sources}
(root/'package_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
zip_path=root/'metst_mechanism_diagnostic_v1.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n in sorted(sources+['package_manifest.json']):
        info=zipfile.ZipInfo(n,(2026,9,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
        z.writestr(info,(root/n).read_bytes())
digest=hashlib.sha256(zip_path.read_bytes()).hexdigest()
cell='''# 接在刚才完成 Gray 实验的原 ME-TST Colab 末尾；只上传诊断包。
# 默认两数据集顺序执行；不要重新运行旧搜索单元格。
MECHANISM_SETTINGS = ['sammlv', 'casme3']
MECHANISM_RESUME_ROOT = None  # 中断恢复时填本轮日志中的 diagnostic_<UTC> 根目录
MECHANISM_PREVIOUS = {
    'sammlv': '/content/drive/MyDrive/GLSD_GRAYZONE_SUPPORT/full_sammlv_20260922T125802_413296Z',
    'casme3': '/content/drive/MyDrive/GLSD_GRAYZONE_SUPPORT/full_casme3_20260922T134807_975740Z',
}

from google.colab import files
from pathlib import Path
import hashlib, io, json, tempfile, zipfile
uploaded = files.upload()  # 只选 metst_mechanism_diagnostic_v1.zip
assert len(uploaded) == 1, '请只上传本轮机制诊断包'
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest() == %r, 'ZIP 与 cell 不对应'
expected = %r
MECHANISM_PACKAGE_DIR = Path(tempfile.mkdtemp(prefix='glsd_mechanism_', dir='/content'))
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    assert len(z.namelist()) == len(expected)+1 and set(z.namelist()) == set(expected)|{'package_manifest.json'}
    assert json.loads(z.read('package_manifest.json')) == expected
    for name, digest in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    z.extractall(MECHANISM_PACKAGE_DIR)
entry = MECHANISM_PACKAGE_DIR / 'run_colab_mechanism.py'
exec(compile(entry.read_text(encoding='utf8'), str(entry), 'exec'))
'''%(digest,hashes)
(root/'MECHANISM_COMPLETE_CELL.py').write_text(cell,encoding='utf8')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[
 dict(cell_type='markdown',metadata={},id='intro',source=['# G/L条件区分与逐尺度诊断\n','接在已配置并完成Gray实验的原Colab中运行。一个完整cell、同一ZIP、默认两套数据，不扩灰区阈值网格。\n']),
 dict(cell_type='code',metadata={},id='diagnostic',source=cell.splitlines(keepends=True),outputs=[],execution_count=None)])
(root/'Colab_G_L机制诊断接续.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(zip=str(zip_path),sha256=digest,size=zip_path.stat().st_size),indent=2))
