"""Build corresponding ZIP and two self-contained Colab cells reproducibly."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent
parent=root.parent
inherited=['official_response_component_ablation.py','run_generalized_mean_screening.py',
    'run_p8_phase2.py','run_p8_one_to_one.py','run_p8_threshold_control.py','colab_p8_entry.py',
    'test_p8_matching.py','test_p8_phase2.py','test_p8_threshold_control.py']
audit={}
for n in inherited+['one_to_one_evaluator.py']:
    source=parent/n if n=='one_to_one_evaluator.py' else parent/'metst_fusion_screening'/n
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==hashlib.sha256((root/n).read_bytes()).hexdigest(),n+' inherited bytes changed'
    audit[n]=dict(source=str(source.relative_to(parent)),sha256=digest,exact_copy=True)
(root/'SOURCE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
names=inherited+['one_to_one_evaluator.py','run_grayzone.py','test_grayzone.py',
    'colab_grayzone_entry.py','run_colab_grayzone.py','README_GRAYZONE_CN.md','SOURCE_AUDIT.json']
hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
(root/'package_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
zip_path=root/'metst_grayzone_support_v1.zip'
# Fixed timestamp/order permits byte-for-byte deterministic rebuilds.
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n in sorted(names+['package_manifest.json']):
        info=zipfile.ZipInfo(n,date_time=(2026,9,22,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o644 << 16
        z.writestr(info,(root/n).read_bytes())
zip_sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()

def complete_cell(setting):
    return '''# 在已配置的原 ME-TST Colab 中运行；只上传 metst_grayzone_support_v1.zip。
# P8 占用同一内核时请排队，或使用另一个已配置的会话并行运行。
GRAY_SETTING = %r
GRAY_MODE = 'full'  # 自动检查后执行全量实验；只查环境可设 'probe'
GRAY_RESUME = None  # 中断后填写本轮日志中实际输出目录；保持相同 ZIP/mode/setting

from google.colab import files
from pathlib import Path
import hashlib, io, json, tempfile, zipfile

uploaded = files.upload()
assert len(uploaded) == 1, '请只上传对应的灰区支持 ZIP'
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest() == %r, 'ZIP 与本 cell 不对应，请使用一同交付的包'
GRAY_PACKAGE_DIR = Path(tempfile.mkdtemp(prefix='glsd_grayzone_', dir='/content'))
expected = %r
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    assert len(z.namelist()) == len(expected)+1 and set(z.namelist()) == set(expected)|{'package_manifest.json'}
    assert json.loads(z.read('package_manifest.json')) == expected
    for name, digest in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name+' 文件哈希不一致'
    z.extractall(GRAY_PACKAGE_DIR)
entry = GRAY_PACKAGE_DIR / 'run_colab_grayzone.py'
exec(compile(entry.read_text(encoding='utf8'), str(entry), 'exec'))
''' % (setting,zip_sha,hashes)

cells=[]
for i,setting in enumerate(('sammlv','casme3')):
    code=complete_cell(setting)
    (root/('GRAYZONE_'+setting.upper()+'_COMPLETE_CELL.py')).write_text(code,encoding='utf8')
    cells.append(dict(cell_type='markdown',metadata={},id='label-'+str(i),source=[
        '## '+setting+'：完整独立 cell\n',
        '在现有已配置 Colab 执行；上传同一个灰区 ZIP。默认 full 自动完成 preflight→搜索→事件导出→下载。\n']))
    cells.append(dict(cell_type='code',metadata={},id='code-'+str(i),source=code.splitlines(keepends=True),outputs=[],execution_count=None))
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)
(root/'Colab_灰区支持接续.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(zip=str(zip_path),sha256=zip_sha,size=zip_path.stat().st_size),indent=2))
