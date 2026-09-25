#!/usr/bin/env python3
"""Reproducible Phase-0 readiness audit for frozen temporal features.

This script does not run a relation probe.  It refuses to claim a scientific
GO/NO-GO until aligned hidden features can be exported from frozen checkpoints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
CACHES = {
    "me_tst_plus__SAMMLV": ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl",
    "me_tst_plus__CASME_3": ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl",
    "boostingvrme__SAMMLV": ROOT / "caches/boostingvrme/sammlv_curves.pkl",
    "boostingvrme__CASME_3": ROOT / "caches/boostingvrme/casme3_curves.pkl",
}
OFFICIAL_COMMIT = "ae85bcdf5163d1fb3d8acec6208d25ab5621afd4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_fields(mapping: dict) -> dict:
    output = {}
    for key, value in mapping.items():
        example = value
        if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            example = value[0]
        if isinstance(example, np.ndarray):
            output[key] = {"shape": list(example.shape), "dtype": str(example.dtype)}
    return output


def inspect_cache(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "records" in payload:
        example = payload["records"][0]
        videos = len(payload["records"])
        subjects = len({str(row["subject"]) for row in payload["records"]})
        gt = sum(len(row["samples"]) for row in payload["records"])
        frames = sum(len(row["score"]) for row in payload["records"])
        schema = "me_tst_strategy_output_records"
    else:
        example = payload["subject_curves"][0]
        videos = sum(len(row["score"]) for row in payload["subject_curves"])
        subjects = len(payload["subject_curves"])
        gt = sum(len(samples) for row in payload["subject_curves"] for samples in row["samples"])
        frames = sum(len(score) for row in payload["subject_curves"] for score in row["score"])
        schema = "boostingvrme_subject_curves"
    top_keys = sorted(payload.keys())
    example_keys = sorted(example.keys())
    hidden_names = {"hidden", "feature", "features", "embedding", "representation", "x_spot"}
    hidden_present = bool(hidden_names.intersection(top_keys) or hidden_names.intersection(example_keys))
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "schema": schema,
        "top_level_keys": top_keys,
        "example_keys": example_keys,
        "example_array_fields": array_fields(example),
        "subjects": subjects,
        "videos": videos,
        "gt_events": gt,
        "score_time_points": frames,
        "hidden_temporal_feature_present": hidden_present,
    }


def module_status(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_runtime() -> dict:
    if not module_status("torch"):
        return {"available": False, "cuda_available": False, "mps_available": False}
    import torch
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_audit = {name: inspect_cache(path) for name, path in CACHES.items()}
    local_requirements = {
        "network_sf.py": list(ROOT.rglob("network_sf.py")),
        "network.py": list(ROOT.rglob("network.py")),
        "SAMMLV_dataset.pkl": list(ROOT.rglob("SAMMLV_dataset.pkl")),
        "CASME_3_dataset_parts": list(ROOT.rglob("CASME_3_dataset_*.pkl")),
        "subject_checkpoints": [
            path for path in ROOT.rglob("subject_*.pkl")
            if "weights" in path.parts
        ],
    }
    local_requirements = {
        key: [str(path.resolve()) for path in paths]
        for key, paths in local_requirements.items()
    }
    blockers = [
        "No aligned hidden [T,D] tensor is stored in any of the four frozen curve/output caches.",
        "The current project contains no official ME-TST+ preprocessing cache or subject checkpoint.",
        "The current project does not contain the official network_sf.py model source.",
        "The active environment lacks mamba_ssm and has neither CUDA nor MPS acceleration.",
        "The BoostingVRME model source/checkpoints and a checkpoint identifier are absent, so its feature tap cannot be authenticated.",
    ]
    report = {
        "status": "blocked_phase0",
        "scientific_verdict": "not_evaluated",
        "reason": "required aligned frozen temporal features cannot be produced from the current project",
        "official_me_tst_source": {
            "repository": "https://github.com/zizheng-guo/ME-TST",
            "commit": OFFICIAL_COMMIT,
            "primary_tap": "METST_SF.spot_pathway output immediately before fc_spot",
            "window_tensor_shape": "[B,T,384]",
            "score_shape": "[B,T]",
            "local_alignment": "one hidden vector per pre-sigmoid spotting-head temporal index",
            "required_global_alignment": "stitch hidden windows with the exact same batch-sensitive logic used for cached score",
            "secondary_tap_not_run": "Stem output [B,128,T]",
        },
        "cache_audit": cache_audit,
        "local_asset_inventory": local_requirements,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch_runtime(),
            "einops_available": module_status("einops"),
            "timm_available": module_status("timm"),
            "mamba_ssm_available": module_status("mamba_ssm"),
        },
        "blockers": blockers,
        "relation_metrics_computed": False,
        "forbidden_substitutions": [
            "recognition logits [T,5] are not treated as a hidden temporal representation",
            "scalar score is not expanded or reconstructed into pseudo-features",
            "no random or uninitialized checkpoint is used",
        ],
        "required_to_resume": [
            "exact official/paper-aligned input feature caches for SAMMLV and CASME3",
            "all subject-specific pretrained checkpoints with hashes",
            "Linux/CUDA environment compatible with the recorded mamba_ssm implementation",
            "model source and dependency versions pinned to an identified commit",
            "one-video dump whose stitched score matches the existing frozen score before full export",
        ],
    }
    output = OUT / "feature_relation_results.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "output": str(output),
        "status": report["status"],
        "scientific_verdict": report["scientific_verdict"],
        "relation_metrics_computed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
