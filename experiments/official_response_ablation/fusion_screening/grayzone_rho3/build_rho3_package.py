"""Build the independent rho3 ZIP, complete cells, notebook, and manifest."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
INHERITED = [
    'official_response_component_ablation.py',
    'run_generalized_mean_screening.py',
    'run_p8_phase2.py',
    'run_p8_one_to_one.py',
    'run_p8_threshold_control.py',
    'colab_p8_entry.py',
    'test_p8_matching.py',
    'test_p8_phase2.py',
    'test_p8_threshold_control.py',
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def source_audit():
    audit = {}
    for name in INHERITED + ['one_to_one_evaluator.py']:
        source = PARENT / name if name == 'one_to_one_evaluator.py' else PARENT / 'metst_fusion_screening' / name
        local = ROOT / name
        if digest(source) != digest(local):
            raise RuntimeError(f'inherited bytes changed: {name}')
        audit[name] = {
            'source': str(source.relative_to(PARENT)),
            'sha256': digest(source),
            'exact_copy': True,
        }
    write_json(ROOT / 'SOURCE_AUDIT.json', audit)
    return audit


def package_names():
    return INHERITED + [
        'one_to_one_evaluator.py',
        'run_grayzone.py',
        'run_grayzone_rho3.py',
        'test_grayzone_rho3.py',
        'colab_grayzone_rho3_entry.py',
        'run_colab_grayzone_rho3.py',
        'README_GRAYZONE_RHO3_CN.md',
        'SOURCE_AUDIT.json',
    ]


def make_cell(setting: str, zip_sha: str, hashes: dict[str, str]) -> str:
    expected = repr(hashes)
    return f'''# 在已配置的原 ME-TST Colab 中运行；只上传 metst_grayzone_rho3_v1.zip。
# 本轮使用独立 Drive 输出目录 GLSD_GRAYZONE_RHO3，不读取当前 P8 任务输出。
GRAY_SETTING = {setting!r}
GRAY_MODE = 'full'  # 先做环境和协议检查可改为 'probe'
GRAY_RESUME = None  # 中断后填入同一轮实际输出目录，并保持同一 ZIP

from google.colab import files
from pathlib import Path
import hashlib, io, json, tempfile, zipfile

uploaded = files.upload()
assert len(uploaded) == 1, '请只上传 metst_grayzone_rho3_v1.zip'
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest() == {zip_sha!r}, 'ZIP 与本 cell 不对应'
GRAY_PACKAGE_DIR = Path(tempfile.mkdtemp(prefix='glsd_grayzone_rho3_', dir='/content'))
expected = {expected}
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    names = set(z.namelist())
    assert names == set(expected) | {{'package_manifest.json'}}, 'ZIP 文件清单不匹配'
    assert json.loads(z.read('package_manifest.json')) == expected, 'package_manifest 不匹配'
    for name, file_hash in expected.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == file_hash, name + ' 文件哈希不一致'
    z.extractall(GRAY_PACKAGE_DIR)
entry = GRAY_PACKAGE_DIR / 'run_colab_grayzone_rho3.py'
exec(compile(entry.read_text(encoding='utf8'), str(entry), 'exec'))
'''


def main():
    source_audit()
    names = package_names()
    hashes = {name: digest(ROOT / name) for name in names}
    write_json(ROOT / 'package_manifest.json', hashes)
    zip_path = ROOT / 'metst_grayzone_rho3_v1.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names + ['package_manifest.json']):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (ROOT / name).read_bytes())
    zip_sha = digest(zip_path)

    cells = []
    for index, setting in enumerate(('sammlv', 'casme3')):
        code = make_cell(setting, zip_sha, hashes)
        (ROOT / f'GRAYZONE_RHO3_{setting.upper()}_COMPLETE_CELL.py').write_text(code, encoding='utf8')
        cells.extend([
            {
                'cell_type': 'markdown', 'metadata': {}, 'id': f'label-{index}',
                'source': [f'## {setting}：rho3 完整独立 cell\n',
                           '沿用现有已配置 Colab；上传同一个 rho3 ZIP。默认 full 会执行 preflight、全量搜索、事件账本和下载。\n'],
            },
            {
                'cell_type': 'code', 'metadata': {}, 'id': f'code-{index}',
                'source': code.splitlines(keepends=True), 'outputs': [], 'execution_count': None,
            },
        ])
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
        'cells': cells,
    }
    (ROOT / 'Colab_灰区rho3接续.ipynb').write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

    # Static package checks belong to the delivery record; runtime checks are run separately.
    for name in names:
        ast.parse((ROOT / name).read_text(encoding='utf8')) if name.endswith('.py') else None
    validation = {
        'package': zip_path.name,
        'package_sha256': zip_sha,
        'package_files': names,
        'package_file_hashes': hashes,
        'grid_budget': {'G': 231, 'L': 924, 'Mean': 480, 'P8_control': 480, 'Gray': 5088, 'total': 7203},
        'protocol': 'grayzone_relative_support_rho3_full_loso_v1',
        'evaluation': 'event_one_to_one_v1_prediction_order',
        'status': 'package_built; 30 local package tests passed; official full experiment not run',
    }
    write_json(ROOT / 'VALIDATION.json', validation)
    print(json.dumps({'zip': str(zip_path), 'sha256': zip_sha, 'size': zip_path.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
