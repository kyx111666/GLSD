"""Build the server package for the negative-residual-strength screen."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
METST = ROOT.parent/'metst_fusion_screening'
FILES = {
    'run_reference_residual.py': ROOT/'run_reference_residual.py',
    'run_asymmetric_residual.py': ROOT/'run_asymmetric_residual.py',
    'run_server_asymmetric_residual.py': ROOT/'run_server_asymmetric_residual.py',
    'residual_features.py': ROOT/'residual_features.py',
    'typed_matching.py': ROOT/'typed_matching.py',
    'test_reference_residual.py': ROOT/'test_reference_residual.py',
    'sealed_decoder_fixture.py': ROOT/'sealed_decoder_fixture.py',
    'remote_official_fixture.py': ROOT/'remote_official_fixture.py',
    'run_p8_phase2.py': METST/'run_p8_phase2.py',
    'run_p8_one_to_one.py': METST/'run_p8_one_to_one.py',
    'run_p8_threshold_control.py': METST/'run_p8_threshold_control.py',
    'run_generalized_mean_screening.py': METST/'run_generalized_mean_screening.py',
    'official_response_component_ablation.py': METST/'official_response_component_ablation.py',
    'colab_p8_entry.py': METST/'colab_p8_entry.py',
    'one_to_one_evaluator.py': METST.parent/'one_to_one_evaluator.py',
}
for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)
manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in FILES.items()}
archive = ROOT/'metst_asymmetric_residual.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, path in FILES.items():
        z.write(path, name)
    z.writestr('package_manifest.json', json.dumps(manifest, indent=2)+'\n')
print('ZIP SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())
print(archive)
