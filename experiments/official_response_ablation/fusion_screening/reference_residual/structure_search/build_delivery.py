"""Build a separate self-contained structure ZIP and two-cell Colab notebook."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'server_standalone'
source = (ROOT / 'COLAB_STRUCTURE_CELLS.py').read_text()
cells = []
for block in source.split('# %% ')[1:]:
    title, code = block.split('\n', 1)
    compile(code, title, 'exec')
    cells.extend([
        dict(cell_type='markdown', metadata={}, id=f'title{len(cells)}', source=[title]),
        dict(cell_type='code', metadata={}, id=f'code{len(cells)}', execution_count=None,
             outputs=[], source=code.splitlines(keepends=True))])
cells.insert(0, dict(cell_type='markdown', metadata={}, id='intro', source=[
    '# 参考尺度残差：共同结构搜索\n\n',
    '可在当前Colab笔记本末尾追加这两格，或直接打开本笔记本。无需重跑旧的安装、训练和负权重实验。\n\n',
    'Cell 1上传本轮 `glsd_structure_ready.zip`；Cell 2运行并下载结果。',
    '数据仍使用Drive上原来的29份SAMMLV冻结响应。\n\n',
    'a0∈{1,1.5,2}、rho∈{1,2,3}、tau=0.05到0.95步长0.05。',
    'G/L/Mean/PositiveResidual/RefResidual各自内层选参，不再搜索lambda。',
    'CPU即可，不调用训练或骨干网络。新结果保存在GLSD_RESIDUAL_STRUCTURE。\n']))
nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
    display_name='Python 3', language='python', name='python3')), cells=cells)
(ROOT / 'Colab_参考残差_共同结构搜索.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2)+'\n')
files = {str(p.relative_to(RUNTIME)): p for p in RUNTIME.rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.txt')
         and p.name not in ('build_delivery.py', 'server_job.json', 'server_job.py')}
files['run_structure.py'] = ROOT / 'run_structure.py'
files['README.md'] = ROOT / 'README.md'
with zipfile.ZipFile(ROOT / 'glsd_structure_ready.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name, path in sorted(files.items()):
        if path.suffix == '.py':
            compile(path.read_text(), name, 'exec')
        info = zipfile.ZipInfo('glsd_structure_ready/'+name, (2026, 9, 22, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, path.read_bytes())
print(ROOT / 'glsd_structure_ready.zip')
print(ROOT / 'Colab_参考残差_共同结构搜索.ipynb')
