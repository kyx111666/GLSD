# 在已配置的原 ME-TST Colab 中运行；只上传 metst_grayzone_rho3_v1.zip。
# 本轮使用独立 Drive 输出目录 GLSD_GRAYZONE_RHO3，不读取当前 P8 任务输出。
GRAY_SETTING = 'sammlv'
GRAY_MODE = 'full'  # 先做环境和协议检查可改为 'probe'
GRAY_RESUME = None  # 中断后填入同一轮实际输出目录，并保持同一 ZIP

from google.colab import files
from pathlib import Path
import hashlib, io, json, tempfile, zipfile

uploaded = files.upload()
assert len(uploaded) == 1, '请只上传 metst_grayzone_rho3_v1.zip'
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest() == 'edff0db9422b1bfe0b4dd3210cdcc0f662b55928f40c0cf4da9bdb595601ee64', 'ZIP 与本 cell 不对应'
GRAY_PACKAGE_DIR = Path(tempfile.mkdtemp(prefix='glsd_grayzone_rho3_', dir='/content'))
expected = {'official_response_component_ablation.py': 'e439bc088dcf833bfe1cc2fdc2cc9ecc27d5588e30f7c12cb1215383427193f4', 'run_generalized_mean_screening.py': 'b4d4e97f4f9ce18c57998eac372294ac27743effb6090134fc6190befa42cb86', 'run_p8_phase2.py': 'bfd7d6bed67f00b20ef801180763a864b0134e0ec4e1035ffa943c2aab2be442', 'run_p8_one_to_one.py': '509312a1d1afb3d9dc04ebc134609fb915abb4e301eb46a6f651223e522c1795', 'run_p8_threshold_control.py': 'fb1c446f01200c390fb7b5c16961e60bf28af811e41ba39aba148bdd5f698b34', 'colab_p8_entry.py': '30d184886a80c25f9360fd4f79245371a87530ba9a7bc04c7c7c6ea8994f322c', 'test_p8_matching.py': '1ec9e0cebda3d8945b768288ac5d21fc42dc4670aae3d6cbe526e2c259ced203', 'test_p8_phase2.py': '5b1e7806500667c5dbe4a34b0d7a6d05701f2f15a09efbd1cd46f1bf612f90da', 'test_p8_threshold_control.py': 'b6d0b945436683cf6f4335669ceb3e8fc9e32152bc4c686d06a4818e7547d447', 'one_to_one_evaluator.py': '0b61a8eac4be63b9785cba4dc1ac5364575f9becd3447258079e2a11f9831008', 'run_grayzone.py': '601b7881bc25e95275a4fe93289912fa4109a1bf6f67f1c2d75c514cfa8f5cfb', 'run_grayzone_rho3.py': '2a25666d9904b99532d5134a3ef5f875ae0502adc9ce81eddf136c59d327625c', 'test_grayzone_rho3.py': 'a1cb7258dbb43c60005bc9bef0e10a1acf8b37c6465dab8e27aaaa7a8a7257d3', 'colab_grayzone_rho3_entry.py': 'd8b84975510a73e0671239ba516a2d26552a3fe480c583aff012be00fe18e2c4', 'run_colab_grayzone_rho3.py': 'ac62f59c46c89bdab43b79b9cd2f467798f445b3c28a952c9e4a63ade187e653', 'README_GRAYZONE_RHO3_CN.md': 'f3b1a2e9e7ec4ac04c9d4543f5e5a0148e7fb858213e53e3297da094aa8984aa', 'SOURCE_AUDIT.json': '9bc49bb3d897c6472552057046f86cca6808c6cefd5ea5e6d222be028f8480aa'}
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    names = set(z.namelist())
    assert names == set(expected) | {'package_manifest.json'}, 'ZIP 文件清单不匹配'
    assert json.loads(z.read('package_manifest.json')) == expected, 'package_manifest 不匹配'
    for name, file_hash in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == file_hash, name + ' 文件哈希不一致'
    z.extractall(GRAY_PACKAGE_DIR)
entry = GRAY_PACKAGE_DIR / 'run_colab_grayzone_rho3.py'
exec(compile(entry.read_text(encoding='utf8'), str(entry), 'exec'))
