"""Build the expanded-scale Colab package."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'server_standalone'
STRUCTURE = ROOT.parent / 'structure_search'
cells = (ROOT / 'COLAB_EXPANDED_CELLS.py').read_text()
for block in cells.split('# %% ')[1:]:
    title, code = block.split('\n', 1)
    compile(code, title, 'exec')
nbcells = [dict(cell_type='markdown', metadata={}, id='intro', source=[
    '# 参考尺度残差：扩展边界结构搜索\n\n',
    '所有方法共用扩展物理尺度集合，实验不调用训练。两格依次运行。\n\n',
    '默认复现细阈值结果：a0={1,1.5,2,2.5,3,4}、rho={0.5,1,2,3,4,6}和tau=0.01:0.01:0.99。',
    '每个方法都在内层full F1选参；候选池随a0变化，同一a0下方法共享候选。\n'])]
for i, block in enumerate(cells.split('# %% ')[1:], 1):
    title, code = block.split('\n', 1)
    nbcells += [dict(cell_type='markdown', metadata={}, id=f'title{i}', source=[title]),
                dict(cell_type='code', metadata={}, id=f'code{i}', execution_count=None,
                     outputs=[], source=code.splitlines(keepends=True))]
nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
    display_name='Python 3', language='python', name='python3')), cells=nbcells)
(ROOT / 'Colab_扩展边界_共同结构搜索.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2)+'\n')
files = {str(p.relative_to(RUNTIME)): p for p in RUNTIME.rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.txt')
         and p.name not in ('build_delivery.py', 'server_job.py', 'server_job.json')}
files.update({'run_structure.py': STRUCTURE/'run_structure.py',
              'run_expanded.py': ROOT/'run_expanded.py',
              'run_expanded_fine.py': ROOT/'run_expanded_fine.py',
              'run_original_fine.py': ROOT/'run_original_fine.py',
              'supplement_no_global.py': ROOT/'supplement_no_global.py',
              'README.md': ROOT/'README.md'})
with zipfile.ZipFile(ROOT/'glsd_expanded_structure_ready.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name, path in sorted(files.items()):
        compile(path.read_text(), name, 'exec') if path.suffix == '.py' else None
        info = zipfile.ZipInfo('glsd_expanded_structure_ready/'+name, (2026,9,22,0,0,0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, path.read_bytes())
print(ROOT/'glsd_expanded_structure_ready.zip')
print(ROOT/'Colab_扩展边界_共同结构搜索.ipynb')
