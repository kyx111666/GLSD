"""Build a content-verified standalone continuation package and four Colab cells."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
inherited=json.loads((ROOT/'inherited_manifest.json').read_text())
for n,h in inherited.items():
    if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('changed inherited source: '+n)
names=list(inherited)+['reliable_features.py','run_reliable_local_union.py','test_reliable_local_union.py',
    'validate_local.py','colab_reliable_entry.py','run_colab_reliable.py','build_package.py',
    'inherited_manifest.json','official_runtime_reference.json','README_CN.md','VALIDATION.md']
data={n:(ROOT/n).read_bytes() for n in names}
for n,b in data.items():
    if n.endswith('.py'):ast.parse(b,filename=n)
hashes={n:hashlib.sha256(b).hexdigest() for n,b in data.items()}
archive=ROOT/'metst_reliable_local_union.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in data.items():
        info=zipfile.ZipInfo(n,date_time=(2026,9,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    info=zipfile.ZipInfo('package_manifest.json',date_time=(2026,9,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
    z.writestr(info,json.dumps(hashes,indent=2)+'\n')
install='''from pathlib import Path
import hashlib, io, json, zipfile
from google.colab import files

# 新运行包：可先放到 /content；已有官方响应和旧结果无需重新上传。
package_path = Path('/content/metst_reliable_local_union.zip')
if package_path.is_file():
    package_bytes = package_path.read_bytes()
else:
    uploaded = files.upload()  # 仅选择本轮新包 metst_reliable_local_union.zip
    assert len(uploaded) == 1, '只选择本轮新运行包'
    package_bytes = next(iter(uploaded.values()))
rlu_dir = Path('/content/glsd_reliable_local_union')
expected = %r
with zipfile.ZipFile(io.BytesIO(package_bytes)) as z:
    assert set(z.namelist()) == set(expected) | {'package_manifest.json'}
    assert json.loads(z.read('package_manifest.json')) == expected, '运行包版本不符'
    for name, digest in expected.items():
        assert Path(name).name == name
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name + ' 哈希错误'
    rlu_dir.mkdir(exist_ok=True)
    unexpected = {p.name for p in rlu_dir.glob('*.py')} - set(expected)
    assert not unexpected, '代码目录存在旧文件，请使用本轮独立目录：' + str(unexpected)
    for name in z.namelist():
        target = rlu_dir / name
        if target.exists():
            assert target.read_bytes() == z.read(name), '已有文件版本不同：' + name
        else:
            target.write_bytes(z.read(name))
RLU_REUSE_PATHS = {
    'sammlv': '/content/drive/MyDrive/GLSD_ETA_GM/full_sammlv_20260922T110018_905231Z',
    'casme3': '/content/drive/MyDrive/GLSD_ETA_GM/full_casme3_20260922T111208_876534Z',
}
print('运行包已验证。下一格先做 SAM 输入、特征和实际解码 probe。')
'''%hashes

def launch(mode,setting):
    return """RLU_MODE = %r
RLU_SETTING = %r
RLU_RESUME = None  # 已有唯一同模式目录会自动接续；多个目录时填写确切路径
entry = rlu_dir / 'run_colab_reliable.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
"""%(mode,setting)
steps=[('CELL 1 安装本轮独立运行包',install),('CELL 2 SAMMLV 输入检查及 probe',launch('probe','sammlv')),
       ('CELL 3 SAMMLV 完整开发实验',launch('full','sammlv')),('CELL 4 CAS(ME)3 同协议实验（内部先 probe）',launch('full','casme3'))]
(ROOT/'RELIABLE_LOCAL_UNION_COLAB_CELLS.py').write_text('\n\n'.join('# %% '+t+'\n'+s for t,s in steps)+'\n',encoding='utf8')
def cell(kind,s,i):
    c=dict(cell_type=kind,metadata={},id='reliable-'+str(i),source=s.splitlines(keepends=True))
    if kind=='code':c.update(outputs=[],execution_count=None)
    return c
cells=[cell('markdown','# 可靠局部证据并联融合：Colab 接续\n\n把四格依次复制到已配置好的当前 Colab 末尾。无需重跑训练、响应或旧搜索。先 SAM probe，再 SAM full，最后 CAS（自动先 probe）。本地验证范围见 VALIDATION.md；这里尚无远端执行结果。\n',0)]
for i,(title,s) in enumerate(steps,1):
    ast.parse(s);cells.extend([cell('markdown','## '+title+'\n',2*i-1),cell('code',s,2*i)])
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)
(ROOT/'Colab_可靠局部证据并联融合_接续.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(ROOT/'package_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(archive)
