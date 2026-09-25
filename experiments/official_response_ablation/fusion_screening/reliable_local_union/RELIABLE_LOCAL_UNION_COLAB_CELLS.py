# %% CELL 1 安装本轮独立运行包
from pathlib import Path
import hashlib, io, json, zipfile
from google.colab import files

# 新运行包：可先放到 /content；已有官方响应和旧结果无需重新上传。
package_path = Path('/content/metst_reliable_local_union.zip')
if package_path.is_file():
    package_bytes = package_path.read_bytes()
else:
    uploaded = files.upload()  # 仅选择本轮新包 metst_reliable_local_union.zip
    assert len(uploaded) == 1, '只选择本轮新运行包'
    package_bytes = next(iter(uploaded.values()))
rlu_dir = Path('/content/glsd_reliable_local_union')
expected = {'official_response_component_ablation.py': 'c9519c241fcc57f4916853d5375d4978da5a48a8b19f76fe3f91e5dab2801bc7', 'run_p8_threshold_control.py': 'fb1c446f01200c390fb7b5c16961e60bf28af811e41ba39aba148bdd5f698b34', 'run_p8_phase2.py': 'bfd7d6bed67f00b20ef801180763a864b0134e0ec4e1035ffa943c2aab2be442', 'colab_p8_threshold_entry.py': 'a0ea99b628053ae6f4b5827c5ddc58cb3c585fe23307318d79cdaaf8d3acf42b', 'run_generalized_mean_screening.py': 'b4d4e97f4f9ce18c57998eac372294ac27743effb6090134fc6190befa42cb86', 'test_p8_matching.py': '1ec9e0cebda3d8945b768288ac5d21fc42dc4670aae3d6cbe526e2c259ced203', 'colab_p8_entry.py': '915c46e8dd79bcc9cc8cc87cf3d451e5ca754695292575fd9c48765cbe9fe792', 'colab_p8_one_to_one_entry.py': '8e48f9def2859acb8011aef324ddab99b36f34f2874cb4d33907731de1c1afe1', 'test_p8_threshold_control.py': 'b6d0b945436683cf6f4335669ceb3e8fc9e32152bc4c686d06a4818e7547d447', 'test_p8_phase2.py': '5b1e7806500667c5dbe4a34b0d7a6d05701f2f15a09efbd1cd46f1bf612f90da', 'run_p8_one_to_one.py': '509312a1d1afb3d9dc04ebc134609fb915abb4e301eb46a6f651223e522c1795', 'run_colab_p8_threshold.py': 'f8a6767882301715914e0d4de639e5e0e7494a8276d6939c488782f7dff446a4', 'run_joint_gm.py': '7d7481223a81362cfb37db20ea7cec6466b4e1da4ddffce39f97d67048952cc8', 'test_joint_gm.py': '0fe351515a80d369c0f099bfedd8dd8750250e5974e381f2d6d0cc7beed51a60', 'colab_joint_gm_entry.py': '660e69fe9606d3af0d163f19ca543f7bcb3c9a9b7dc041f6b349b91ab7ade0f1', 'run_colab_joint_gm.py': 'ee8728e5911cd1510965fbe37641d7b96edf8783f9fac71b620e83c0c1fc008e', 'one_to_one_evaluator.py': '0b61a8eac4be63b9785cba4dc1ac5364575f9becd3447258079e2a11f9831008', 'run_eta_gm.py': '2d5fad4695f98ef905e2e1617f1cee0a29187980b531b1931ecfc4797a552198', 'test_eta_gm.py': '4f3f86a42a98571a5ad6fadcec808d2d3459d1dba839065531f96b1ce270ba17', 'colab_eta_gm_entry.py': '1331402ee56af421a4e8feb952b390c7e405e01f0f0803e39507182be7d50fb8', 'run_colab_eta_gm.py': '51055a13ef1982616e03d0e362c15c28fad9709c9992155be2a725521ac2990c', 'reliable_features.py': '2a44bee41f041c68f416b9e9722b6e70e6404174ce97194549f92843a978b87c', 'run_reliable_local_union.py': '9a8dab4b4e15e9cbfe2b86a1b2bb2856dddf56549b777e99463408578f020000', 'test_reliable_local_union.py': '7914dad8fef6fe9595b0e05b3a3737601afb497c169d7fddc3cdbb138dc23635', 'validate_local.py': '0ac99029268e2e0a99181c685fa5a48095e54a3b9e5aae1ba379a33c62c84a7a', 'colab_reliable_entry.py': 'fae4e89fcea55d1264ea871ccfe687ea3e03122e41190b4003bcdb954113c8b2', 'run_colab_reliable.py': '19c289abf1c370eb59fff5d4ae70c3ce7986cb88631674d6d94df0310097e88a', 'build_package.py': '67aca5ce5c5abb5a50186de6a709daaf8556a68b6ce3e039afd95df0e6894800', 'inherited_manifest.json': 'e746af9f96e359afd6071ae4b7b58d53994fdfd861d35f75301e754d4fd3472f', 'official_runtime_reference.json': 'd3823dffd39fdbf585d426fe3ef2a44568fb73aef17e40fae7fcdcfa2c2936df', 'README_CN.md': '56b5ddfa6f97adf5a5488933ecd0f71f01fb867ed592efe04157b289eaffd4b6', 'VALIDATION.md': 'ae144e15e1bab2d478f356a2bc4ebe22e08fe7c232dd9f212eb4a34157e90c43'}
with zipfile.ZipFile(io.BytesIO(package_bytes)) as z:
    assert set(z.namelist()) == set(expected) | {'package_manifest.json'}
    assert json.loads(z.read('package_manifest.json')) == expected, '运行包版本不符'
    for name, digest in expected.items():
        assert Path(name).name == name
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name + ' 哈希错误'
    rlu_dir.mkdir(exist_ok=True)
    unexpected = {p.name for p in rlu_dir.glob('*.py')} - set(expected)
    assert not unexpected, '代码目录存在旧文件，请使用本轮独立目录：' + str(unexpected)
    for name in z.namelist():
        target = rlu_dir / name
        if target.exists():
            assert target.read_bytes() == z.read(name), '已有文件版本不同：' + name
        else:
            target.write_bytes(z.read(name))
RLU_REUSE_PATHS = {
    'sammlv': '/content/drive/MyDrive/GLSD_ETA_GM/full_sammlv_20260922T110018_905231Z',
    'casme3': '/content/drive/MyDrive/GLSD_ETA_GM/full_casme3_20260922T111208_876534Z',
}
print('运行包已验证。下一格先做 SAM 输入、特征和实际解码 probe。')


# %% CELL 2 SAMMLV 输入检查及 probe
RLU_MODE = 'probe'
RLU_SETTING = 'sammlv'
RLU_RESUME = None  # 已有唯一同模式目录会自动接续；多个目录时填写确切路径
entry = rlu_dir / 'run_colab_reliable.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))


# %% CELL 3 SAMMLV 完整开发实验
RLU_MODE = 'full'
RLU_SETTING = 'sammlv'
RLU_RESUME = None  # 已有唯一同模式目录会自动接续；多个目录时填写确切路径
entry = rlu_dir / 'run_colab_reliable.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))


# %% CELL 4 CAS(ME)3 同协议实验（内部先 probe）
RLU_MODE = 'full'
RLU_SETTING = 'casme3'
RLU_RESUME = None  # 已有唯一同模式目录会自动接续；多个目录时填写确切路径
entry = rlu_dir / 'run_colab_reliable.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))

