"""Build a standalone P8 evaluator-fix package and Colab continuation cells."""
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
names = ['run_p8_phase2.py', 'run_generalized_mean_screening.py', 'official_response_component_ablation.py',
         'colab_p8_entry.py', 'run_p8_one_to_one.py', 'colab_p8_one_to_one_entry.py',
         'run_colab_p8_one_to_one.py', 'test_p8_phase2.py', 'test_p8_matching.py', 'README_P8_ONE_TO_ONE.md']
sources = {name: root / name for name in names}
sources['one_to_one_evaluator.py'] = root.parent / 'one_to_one_evaluator.py'
bundle = root / 'metst_p8_one_to_one.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, path in sources.items():
        z.write(path, name)

upload = '''from google.colab import files
from pathlib import Path
import io, zipfile

uploaded = files.upload()  # 选择 metst_p8_one_to_one.zip
zips = [n for n in uploaded if n.endswith('.zip')]
assert len(zips) == 1, '每次只上传一个新的诊断包'
p8_oto_dir = Path('/content/glsd_p8_one_to_one')
p8_oto_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(uploaded[zips[0]])) as archive:
    assert sorted(archive.namelist()) == %r, '不是本次 P8 一对一评价诊断包'
    archive.extractall(p8_oto_dir)
''' % sorted(sources)
probe = '''P8_OTO_MODE = 'probe'
P8_OTO_RESUME = None
entry = p8_oto_dir / 'run_colab_p8_one_to_one.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
'''
full = '''# 先确认上面输出 P8_ONE_TO_ONE_PROBE = PASS。
# 初次 full 必须新建输出目录；不要填旧评价协议或 probe 的输出目录。
P8_OTO_MODE = 'full'
P8_OTO_RESUME = None
entry = p8_oto_dir / 'run_colab_p8_one_to_one.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
'''
cell_path = root / 'P8_ONE_TO_ONE_COLAB_CELL.py'
cell_path.write_text('# 复制全文到原 Colab notebook 末尾的新单元格；仅诊断，不搜索。\n' + upload + '\n' + probe, encoding='utf-8')
def cell(kind, source, name):
    result = dict(cell_type=kind, metadata=dict(id=name), id=name, source=source.splitlines(keepends=True))
    if kind == 'code':
        result.update(outputs=[], execution_count=None)
    return result
nb = dict(nbformat=4, nbformat_minor=5,
          metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')),
          cells=[cell('markdown', '# P8 一对一评价诊断与继续实验\n\n在原已配好环境的 Colab notebook 末尾追加以下单元格。先上传、probe；通过后才运行 full。新目录 GLSD_P8_ONE_TO_ONE，不能沿用旧评价器计数。\n', 'oto-help'),
                 cell('code', upload, 'oto-upload'), cell('code', probe, 'oto-probe'), cell('code', full, 'oto-full')])
nb_path = root / 'Colab_P8_评价修复接续.ipynb'
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(bundle)
print(cell_path)
print(nb_path)
