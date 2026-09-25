"""Build a self-contained, hash-manifested ZIP and executable Colab notebook."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    cells=[dict(cell_type='markdown',metadata={},id='intro',source=[
        '# 时间错位对照融合：首轮固定结构实验\n',
        '复用 ME-TST+ 官方冻结响应；先输入审计与真实 probe，再跑全量。',
        '七评分，A={1,1.5,2,2.5,3,4}、a0=2、rho=3，tau=-1:0.01:2。',
        '软件 PASS 和 probe 不是性能结果。SAM 后复核机制，再以同一规则接续 CAS。'])]
    for i,block in enumerate((ROOT/'COLAB_CELLS.py').read_text().split('# %% ')[1:]):
        title,code=block.split('\n',1)
        compile(code,title,'exec')
        cells.extend([dict(cell_type='markdown',metadata={},id=f'title{i}',source=[title]),
            dict(cell_type='code',metadata={},id=f'code{i}',execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    (ROOT/'Colab_时间错位对照融合_首轮接续.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    files={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='PACKAGE_MANIFEST.json':continue
        if p.suffix not in ('.py','.json','.md','.txt','.ipynb'):continue
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
        files[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    manifest=dict(protocol='shift_control_fixed_a2_r3_six_scales_tau_minus100_200_v1',
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    data=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
    (ROOT/'PACKAGE_MANIFEST.json').write_bytes(data)
    files['PACKAGE_MANIFEST.json']=data
    archive=ROOT/'shift_control_fusion_ready.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            info=zipfile.ZipInfo('shift_control_fusion/'+name,(2026,9,23,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    print(archive)
    print('files=',len(files),'sha256=',hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__=='__main__':main()
