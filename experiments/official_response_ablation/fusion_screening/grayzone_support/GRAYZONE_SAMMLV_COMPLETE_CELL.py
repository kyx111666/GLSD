# 在已配置的原 ME-TST Colab 中运行；只上传 metst_grayzone_support_v1.zip。
# P8 占用同一内核时请排队，或使用另一个已配置的会话并行运行。
GRAY_SETTING = 'sammlv'
GRAY_MODE = 'full'  # 自动检查后执行全量实验；只查环境可设 'probe'
GRAY_RESUME = None  # 中断后填写本轮日志中实际输出目录；保持相同 ZIP/mode/setting

from google.colab import files
from pathlib import Path
import hashlib, io, json, tempfile, zipfile

uploaded = files.upload()
assert len(uploaded) == 1, '请只上传对应的灰区支持 ZIP'
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest() == 'ed45843a42a7caeddbbff35d8c57322bc49866f17eb219f17dc4659d50f3c1b7', 'ZIP 与本 cell 不对应，请使用一同交付的包'
GRAY_PACKAGE_DIR = Path(tempfile.mkdtemp(prefix='glsd_grayzone_', dir='/content'))
expected = {'official_response_component_ablation.py': 'e439bc088dcf833bfe1cc2fdc2cc9ecc27d5588e30f7c12cb1215383427193f4', 'run_generalized_mean_screening.py': 'b4d4e97f4f9ce18c57998eac372294ac27743effb6090134fc6190befa42cb86', 'run_p8_phase2.py': 'bfd7d6bed67f00b20ef801180763a864b0134e0ec4e1035ffa943c2aab2be442', 'run_p8_one_to_one.py': '509312a1d1afb3d9dc04ebc134609fb915abb4e301eb46a6f651223e522c1795', 'run_p8_threshold_control.py': 'fb1c446f01200c390fb7b5c16961e60bf28af811e41ba39aba148bdd5f698b34', 'colab_p8_entry.py': '30d184886a80c25f9360fd4f79245371a87530ba9a7bc04c7c7c6ea8994f322c', 'test_p8_matching.py': '1ec9e0cebda3d8945b768288ac5d21fc42dc4670aae3d6cbe526e2c259ced203', 'test_p8_phase2.py': '5b1e7806500667c5dbe4a34b0d7a6d05701f2f15a09efbd1cd46f1bf612f90da', 'test_p8_threshold_control.py': 'b6d0b945436683cf6f4335669ceb3e8fc9e32152bc4c686d06a4818e7547d447', 'one_to_one_evaluator.py': '0b61a8eac4be63b9785cba4dc1ac5364575f9becd3447258079e2a11f9831008', 'run_grayzone.py': '9040d7a20b475ecf2527494c6ce9758fea435bf5babadd213a40ec3820e39610', 'test_grayzone.py': '3c5855b013651b0f927d5ce682891c812d5f9f19ccea37d2c214aa6052c5bc6b', 'colab_grayzone_entry.py': '15132b2ad9965d77ed6318acac5899e4a86c32c46ff449c2d7631dccda6f30e5', 'run_colab_grayzone.py': 'df65b0cca7fdbb57e6d71f2c09ced345fa541dcf0462d4353eb4d6d949e27945', 'README_GRAYZONE_CN.md': '292ed92ce669d7be4ecfd2d529dd722b293714c0f486ff1457a53bd88db59561', 'SOURCE_AUDIT.json': '9bc49bb3d897c6472552057046f86cca6808c6cefd5ea5e6d222be028f8480aa'}
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    assert len(z.namelist()) == len(expected)+1 and set(z.namelist()) == set(expected)|{'package_manifest.json'}
    assert json.loads(z.read('package_manifest.json')) == expected
    for name, digest in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name+' 文件哈希不一致'
    z.extractall(GRAY_PACKAGE_DIR)
entry = GRAY_PACKAGE_DIR / 'run_colab_grayzone.py'
exec(compile(entry.read_text(encoding='utf8'), str(entry), 'exec'))
