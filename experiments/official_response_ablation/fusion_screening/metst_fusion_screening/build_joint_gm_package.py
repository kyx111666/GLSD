"""Bundle the joint experiment plus exact inherited sources, without modifying them."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent
inherited=['official_response_component_ablation.py','run_p8_threshold_control.py','run_p8_phase2.py',
 'colab_p8_threshold_entry.py','run_generalized_mean_screening.py','test_p8_matching.py','colab_p8_entry.py',
 'colab_p8_one_to_one_entry.py','test_p8_threshold_control.py','test_p8_phase2.py','run_p8_one_to_one.py',
 'run_colab_p8_threshold.py']
names=inherited+['run_joint_gm.py','test_joint_gm.py','colab_joint_gm_entry.py','run_colab_joint_gm.py','README_JOINT_GM.md']
sources={n:root/n for n in names}
sources['one_to_one_evaluator.py']=root.parent/'one_to_one_evaluator.py'
hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sources.items()}
with zipfile.ZipFile(root/'metst_joint_gm.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n,p in sources.items():z.write(p,n)
 z.writestr('package_manifest.json',json.dumps(hashes,indent=2))
upload='''from google.colab import files
from pathlib import Path
import hashlib, io, json, zipfile

uploaded = files.upload()  # 只选 metst_joint_gm.zip
assert len(uploaded) == 1, '请只上传本轮联合搜索包'
joint_dir = Path('/content/glsd_joint_gm')
joint_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as z:
    expected = %r
    assert set(z.namelist()) == set(expected) | {'package_manifest.json'}, '不是本轮运行包'
    assert json.loads(z.read('package_manifest.json')) == expected
    for name, digest in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name + ' 哈希错误'
    z.extractall(joint_dir)
JOINT_OLD = '/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE/full_20260922T053538_684804Z'
JOINT_THRESHOLD_PATHS = {
    'sammlv': '/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/full_sammlv_20260922T070420_025128Z',
    'casme3': '/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/full_casme3_20260922T071214_000063Z',
}
print('联合搜索包已就绪，下一格运行 probe。')
''' % hashes

def launch(mode,setting):
 return """JOINT_MODE = %r
JOINT_SETTING = %r
JOINT_RESUME = None
entry = joint_dir / 'run_colab_joint_gm.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
"""%(mode,setting)
steps=[('CELL 1 上传联合搜索包',upload),('CELL 2 SAMMLV probe',launch('probe','sammlv')),
 ('CELL 3 SAMMLV full',launch('full','sammlv')),('CELL 4 CAS(ME)3 full，内部先probe',launch('full','casme3'))]
(root/'JOINT_GM_COLAB_CELLS.py').write_text('\n\n'.join('# %% '+title+'\n'+code for title,code in steps),encoding='utf8')
def cell(kind,text,i):
 c=dict(cell_type=kind,metadata={},id='joint-'+str(i),source=text.splitlines(keepends=True))
 if kind=='code':c.update(outputs=[],execution_count=None)
 return c
cells=[cell('markdown','# 广义均值联合选参接续\n\n把四个代码单元格复制到现有已配置的Colab末尾。先上传新ZIP，再probe、SAMMLV full、CAS full。不要重新运行旧搜索或训练。新增513组融合、144组G阈值；旧计数和等价事件验证后复用。\n',0)]
for i,(title,code) in enumerate(steps,1):
 cells.extend([cell('markdown','## '+title+'\n',2*i-1),cell('code',code,2*i)])
cells.append(cell('markdown','## 恢复与结果\n\n中断后按README设置JOINT_RESUME为本轮实际输出目录，保持mode/setting/代码一致。每个full完成显示JOINT_GM_FULL = PASS并下载结果ZIP。真实Colab实验尚未运行；不能将本地测试当作实验结果。\n',9))
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)
(root/'Colab_GM_联合选参接续.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(root/'metst_joint_gm.zip')
print(root/'Colab_GM_联合选参接续.ipynb')
if __name__=='__main__': pass
