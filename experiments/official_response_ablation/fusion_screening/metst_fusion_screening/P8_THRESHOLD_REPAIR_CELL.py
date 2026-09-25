# 上传修复包并重跑 probe；保留当前 Colab 环境。
from google.colab import files
from pathlib import Path
import hashlib, io, json, zipfile

uploaded = files.upload()  # 只选 metst_p8_threshold_control.zip
assert len(uploaded) == 1, '请只上传本轮新运行包'
control_dir = Path('/content/glsd_p8_threshold_control')
control_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as z:
    expected = {'run_p8_threshold_control.py': 'fb1c446f01200c390fb7b5c16961e60bf28af811e41ba39aba148bdd5f698b34', 'colab_p8_threshold_entry.py': 'a0ea99b628053ae6f4b5827c5ddc58cb3c585fe23307318d79cdaaf8d3acf42b', 'run_colab_p8_threshold.py': 'f8a6767882301715914e0d4de639e5e0e7494a8276d6939c488782f7dff446a4', 'test_p8_threshold_control.py': 'b6d0b945436683cf6f4335669ceb3e8fc9e32152bc4c686d06a4818e7547d447', 'README_P8_THRESHOLD_CONTROL.md': '8575e13bd1328887705145634747f2e237359186123ce82ed6c27db0f43f364c', 'run_p8_phase2.py': 'bfd7d6bed67f00b20ef801180763a864b0134e0ec4e1035ffa943c2aab2be442', 'run_p8_one_to_one.py': '509312a1d1afb3d9dc04ebc134609fb915abb4e301eb46a6f651223e522c1795', 'run_generalized_mean_screening.py': 'b4d4e97f4f9ce18c57998eac372294ac27743effb6090134fc6190befa42cb86', 'official_response_component_ablation.py': 'c9519c241fcc57f4916853d5375d4978da5a48a8b19f76fe3f91e5dab2801bc7', 'colab_p8_entry.py': '915c46e8dd79bcc9cc8cc87cf3d451e5ca754695292575fd9c48765cbe9fe792', 'colab_p8_one_to_one_entry.py': '8e48f9def2859acb8011aef324ddab99b36f34f2874cb4d33907731de1c1afe1', 'test_p8_phase2.py': '5b1e7806500667c5dbe4a34b0d7a6d05701f2f15a09efbd1cd46f1bf612f90da', 'test_p8_matching.py': '1ec9e0cebda3d8945b768288ac5d21fc42dc4670aae3d6cbe526e2c259ced203', 'one_to_one_evaluator.py': '0b61a8eac4be63b9785cba4dc1ac5364575f9becd3447258079e2a11f9831008'}
    assert set(z.namelist()) == set(expected) | {'package_manifest.json'}, '不是本轮运行包'
    assert json.loads(z.read('package_manifest.json')) == expected
    for name, digest in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name + ' 哈希错误'
    z.extractall(control_dir)
P8_CONTROL_REUSE = '/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE/full_20260922T053538_684804Z'
print('新运行包就绪；旧结果只读：', P8_CONTROL_REUSE)

P8_CONTROL_MODE = 'probe'
P8_CONTROL_SETTING = 'sammlv'
P8_CONTROL_RESUME = None
entry = control_dir / 'run_colab_p8_threshold.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
