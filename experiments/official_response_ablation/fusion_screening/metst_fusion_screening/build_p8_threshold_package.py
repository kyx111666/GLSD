"""Build isolated continuation bundle; preserve previous experiment files."""
import hashlib
import json
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parent
names=['run_p8_threshold_control.py','colab_p8_threshold_entry.py','run_colab_p8_threshold.py',
       'test_p8_threshold_control.py','README_P8_THRESHOLD_CONTROL.md',
       'run_p8_phase2.py','run_p8_one_to_one.py','run_generalized_mean_screening.py',
       'official_response_component_ablation.py','colab_p8_entry.py','colab_p8_one_to_one_entry.py',
       'test_p8_phase2.py','test_p8_matching.py']
sources={n:root/n for n in names}
sources['one_to_one_evaluator.py']=root.parent/'one_to_one_evaluator.py'
hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sources.items()}
with zipfile.ZipFile(root/'metst_p8_threshold_control.zip','w',zipfile.ZIP_DEFLATED) as z:
    for n,p in sources.items(): z.write(p,n)
    z.writestr('package_manifest.json',json.dumps(hashes,indent=2))
upload='''from google.colab import files
from pathlib import Path
import hashlib, io, json, zipfile

uploaded = files.upload()  # 只选 metst_p8_threshold_control.zip
assert len(uploaded) == 1, '请只上传本轮新运行包'
control_dir = Path('/content/glsd_p8_threshold_control')
control_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as z:
    expected = %r
    assert set(z.namelist()) == set(expected) | {'package_manifest.json'}, '不是本轮运行包'
    assert json.loads(z.read('package_manifest.json')) == expected
    for name, digest in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name + ' 哈希错误'
    z.extractall(control_dir)
P8_CONTROL_REUSE = '/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE/full_20260922T053538_684804Z'
print('新运行包就绪；旧结果只读：', P8_CONTROL_REUSE)
''' % hashes

def launch(mode,setting):
    return """P8_CONTROL_MODE = %r
P8_CONTROL_SETTING = %r
P8_CONTROL_RESUME = None
entry = control_dir / 'run_colab_p8_threshold.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
""" % (mode,setting)
steps=[('CELL 1：上传新运行包',upload),('CELL 2：SAMMLV probe；通过后再运行下一格',launch('probe','sammlv')),
       ('CELL 3：SAMMLV full；完成后再运行下一格',launch('full','sammlv')),
       ('CELL 4：CAS(ME)3 full（自带该数据集 probe）',launch('full','casme3'))]
(root/'P8_THRESHOLD_REPAIR_CELL.py').write_text('# 上传修复包并重跑 probe；保留当前 Colab 环境。\n'+upload+'\n'+launch('probe','sammlv'),encoding='utf8')
(root/'P8_THRESHOLD_CONTROL_COLAB_CELLS.py').write_text('\n\n'.join('# %% '+title+'\n'+code for title,code in steps),encoding='utf8')
def cell(kind,source,i):
    c=dict(cell_type=kind,metadata={},id='p8control-'+str(i),source=source.splitlines(keepends=True))
    if kind=='code': c.update(outputs=[],execution_count=None)
    return c
cells=[cell('markdown','# P8 阈值对照与事件级消融接续\n\n把下面四个代码单元格依次复制到上次配置好的 Colab 笔记本末尾。旧结果只读；输出到 GLSD_P8_THRESHOLD_CONTROL。先 probe，再 SAMMLV，最后 CAS(ME)3。不要重新执行旧 full 搜索。\n',0)]
for i,(title,code) in enumerate(steps,1):
    cells.extend([cell('markdown','## '+title+'\n',2*i-1),cell('code',code,2*i)])
cells.append(cell('markdown','## 中断恢复\n\n使用 README 中的恢复代码，仅将 P8_CONTROL_RESUME 设为本轮失败运行的实际输出目录。保持 mode、setting 和代码一致。正常进入下一阶段时恢复为 None。\n\n本包尚未在真实 Colab 响应上运行；本地回归测试不能替代上述 probe。\n',9))
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)
(root/'Colab_P8_阈值对照与事件消融接续.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(root/'metst_p8_threshold_control.zip')
print(root/'Colab_P8_阈值对照与事件消融接续.ipynb')
