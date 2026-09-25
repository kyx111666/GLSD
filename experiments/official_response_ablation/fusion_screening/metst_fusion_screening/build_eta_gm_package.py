"""Standalone eta continuation with exact inherited joint-run sources."""
import hashlib
import json
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parent
with zipfile.ZipFile(root/'metst_joint_gm.zip') as z:
    inherited=json.loads(z.read('package_manifest.json'))
    data={name:z.read(name) for name in inherited if name.endswith('.py')}
    for name,body in data.items():
        if hashlib.sha256(body).hexdigest()!=inherited[name]:raise RuntimeError('inherited package hash mismatch')
for n in ('run_eta_gm.py','test_eta_gm.py','colab_eta_gm_entry.py','run_colab_eta_gm.py','README_ETA_GM.md'):
    data[n]=(root/n).read_bytes()
hashes={n:hashlib.sha256(body).hexdigest() for n,body in data.items()}
with zipfile.ZipFile(root/'metst_eta_gm.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name,body in data.items():z.writestr(name,body)
    z.writestr('package_manifest.json',json.dumps(hashes,indent=2))
upload='''from google.colab import files
from pathlib import Path
import hashlib, io, json, zipfile

uploaded = files.upload()  # 只选 metst_eta_gm.zip
assert len(uploaded) == 1, '请只上传本轮eta运行包'
eta_dir = Path('/content/glsd_eta_gm')
eta_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as z:
    expected = %r
    assert set(z.namelist()) == set(expected) | {'package_manifest.json'}, '不是本轮运行包'
    assert json.loads(z.read('package_manifest.json')) == expected
    for name, digest in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name + ' 哈希错误'
    z.extractall(eta_dir)
ETA_REUSE_PATHS = {
    'sammlv': '/content/drive/MyDrive/GLSD_JOINT_GM/full_sammlv_20260922T091538_299118Z',
    'casme3': '/content/drive/MyDrive/GLSD_JOINT_GM/full_casme3_20260922T095610_977764Z',
}
print('eta运行包已就绪，下一格先做特征敏感性和事件probe。')
''' % hashes

def launch(mode,setting):
    return """ETA_MODE = %r
ETA_SETTING = %r
ETA_RESUME = None
entry = eta_dir / 'run_colab_eta_gm.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
"""%(mode,setting)
steps=[('CELL 1 上传eta新包',upload),('CELL 2 SAMMLV特征检查与probe',launch('probe','sammlv')),
       ('CELL 3 SAMMLV full',launch('full','sammlv')),('CELL 4 CAS(ME)3 full（内部先probe）',launch('full','casme3'))]
(root/'ETA_GM_COLAB_CELLS.py').write_text('\n\n'.join('# %% '+title+'\n'+code for title,code in steps),encoding='utf8')
def cell(kind,s,i):
    c=dict(cell_type=kind,metadata={},id='eta-'+str(i),source=s.splitlines(keepends=True))
    if kind=='code':c.update(outputs=[],execution_count=None)
    return c
cells=[cell('markdown','# GLSD 对齐容差 eta 接续实验\n\n将四个代码单元格依次复制到当前配置好的Colab末尾。旧结果只读，不训练、不重跑旧搜索。Cell2先检查全候选特征变化与事件接口；PASS后再继续full。结果仅用于开发验证，不保证最终性能。\n',0)]
for i,(title,code) in enumerate(steps,1):cells.extend([cell('markdown','## '+title+'\n',2*i-1),cell('code',code,2*i)])
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)
(root/'Colab_ETA_对齐容差接续.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(root/'metst_eta_gm.zip')
print(root/'Colab_ETA_对齐容差接续.ipynb')
