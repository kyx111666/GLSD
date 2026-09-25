"""Build a flat upload archive and the two-cell continuation notebook."""
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
names = ['run_p8_phase2.py', 'run_generalized_mean_screening.py',
         'official_response_component_ablation.py', 'colab_p8_entry.py',
         'run_colab_p8.py', 'test_p8_phase2.py', 'README_P8_PHASE2.md']
archive_path = root / 'metst_p8_phase2.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        archive.write(root / name, name)

upload = '''from google.colab import files
from pathlib import Path
import io, zipfile

uploaded = files.upload()  # 选择 metst_p8_phase2.zip
name = 'metst_p8_phase2.zip'
assert name in uploaded, '请上传新的 metst_p8_phase2.zip'
p8_task_dir = Path('/content/glsd_p8_phase2')
p8_task_dir.mkdir(exist_ok=True)
allowed = %r
with zipfile.ZipFile(io.BytesIO(uploaded[name])) as archive:
    assert set(archive.namelist()) == allowed, '脚本包内容不一致'
    archive.extractall(p8_task_dir)
print('P8 第二阶段脚本已准备:', p8_task_dir)
''' % set(names)
run = '''# 新实验保持 None；中断续跑时改为上次第二阶段的完整输出路径。
P8_RESUME = None
entry = p8_task_dir / 'run_colab_p8.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
'''

def cell(kind, source, name):
    row = dict(cell_type=kind, metadata=dict(id=name), id=name, source=source.splitlines(keepends=True))
    if kind == 'code':
        row.update(execution_count=None, outputs=[])
    return row

notebook = dict(nbformat=4, nbformat_minor=5,
    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                  language_info=dict(name='python'), colab=dict(name='Colab_P8_第二阶段接续.ipynb')),
    cells=[cell('markdown', '# P8 第二阶段：追加到原 Colab notebook\n\n'
               '把下面两个代码单元格复制到刚才已配置好环境的 notebook 最后，依次运行。'
               '不要点击原 notebook 的全部运行。此文件是接续单元格，不含环境安装。\n\n'
               '固定 p=8，调 a0/rho/tau；比较 G、L、Mean、GM_p8。'
               '主选择指标为 inner full F1。结果写入 GLSD_P8_PHASE2 新目录。\n', 'p8-help'),
           cell('code', upload, 'p8-upload'), cell('code', run, 'p8-run')])
notebook_path = root / 'Colab_P8_第二阶段接续.ipynb'
notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(archive_path)
print(notebook_path)
