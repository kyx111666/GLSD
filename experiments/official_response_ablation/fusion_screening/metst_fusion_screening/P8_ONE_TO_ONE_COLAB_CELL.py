# 复制全文到原 Colab notebook 末尾的新单元格；仅诊断，不搜索。
from google.colab import files
from pathlib import Path
import io, zipfile

uploaded = files.upload()  # 选择 metst_p8_one_to_one.zip
zips = [n for n in uploaded if n.endswith('.zip')]
assert len(zips) == 1, '每次只上传一个新的诊断包'
p8_oto_dir = Path('/content/glsd_p8_one_to_one')
p8_oto_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(uploaded[zips[0]])) as archive:
    assert sorted(archive.namelist()) == ['README_P8_ONE_TO_ONE.md', 'colab_p8_entry.py', 'colab_p8_one_to_one_entry.py', 'official_response_component_ablation.py', 'one_to_one_evaluator.py', 'run_colab_p8_one_to_one.py', 'run_generalized_mean_screening.py', 'run_p8_one_to_one.py', 'run_p8_phase2.py', 'test_p8_matching.py', 'test_p8_phase2.py'], '不是本次 P8 一对一评价诊断包'
    archive.extractall(p8_oto_dir)

P8_OTO_MODE = 'probe'
P8_OTO_RESUME = None
entry = p8_oto_dir / 'run_colab_p8_one_to_one.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
