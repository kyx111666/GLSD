#!/usr/bin/env python3
"""Colab/server runner for the locked smoke40 Qwen3-VL experiment.

The module deliberately has no ML imports at import time.  Input integrity and
label-leakage gates therefore run before inference dependencies are installed
or the model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_NAME_LOCKED = "Qwen/Qwen3-VL-8B-Instruct"
SEED_LOCKED = 100
ROLES_LOCKED = ["B1", "B2", "B3", "C1", "C2", "C3", "A1", "A2", "A3"]
GENERATION_CONFIG = {"do_sample": False, "max_new_tokens": 128, "batch_size": 1, "seed": SEED_LOCKED}
EXPECTED_OUTPUT_KEYS = (
    "local_facial_change",
    "brief_transient_change",
    "return_toward_baseline",
    "global_motion_artifact",
    "verdict",
)


def stage(number: int, title: str) -> None:
    print(f"\n[{number}] {title}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def locate_deployment(deployment_root: str | Path) -> dict[str, Path]:
    root = Path(deployment_root).expanduser().resolve()
    paths = {
        "root": root,
        "manifest": root / "data/manifests/smoke40_inference_manifest.json",
        "labels": root / "data/manifests/smoke40_labels_PRIVATE.json",
        "lockfile": root / "data/audit/SMOKE40_LOCKFILE.json",
        "prompt": root / "config/verifier_prompt_v1.txt",
        "protocol": root / "config/smoke40_protocol_v1.yaml",
        "candidates": root / "data/smoke40/candidates",
        "contact_sheets": root / "data/smoke40/contact_sheets",
        "evaluator": root / "src/evaluate_smoke40.py",
    }
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.exists()]
    if missing:
        raise FileNotFoundError("DEPLOYMENT_PACKAGE_INCOMPLETE:\n" + "\n".join(missing))
    print(f"Deployment package: {root}")
    return paths


def detect_gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as error:
        print("CUDA_GPU_NOT_AVAILABLE")
        raise RuntimeError("CUDA_GPU_NOT_AVAILABLE") from error
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"Could not parse nvidia-smi output: {line}")
    cuda_version = None
    try:
        banner = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout
        match = re.search(r"CUDA Version:\s*([0-9.]+)", banner)
        cuda_version = match.group(1) if match else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    gpu = {
        "model": fields[0],
        "total_vram_gib": round(float(fields[1]) / 1024, 3),
        "available_vram_gib": round(float(fields[2]) / 1024, 3),
        "compute_capability": fields[3],
        "cuda_version_driver": cuda_version,
    }
    print(f"GPU: {gpu['model']}")
    print(f"Total VRAM: {gpu['total_vram_gib']:.1f} GiB")
    print(f"Available VRAM: {gpu['available_vram_gib']:.1f} GiB")
    print(f"CUDA version: {gpu['cuda_version_driver'] or 'unavailable'}")
    print(f"Compute capability: {gpu['compute_capability']}")
    return gpu


def verify_lockfile(paths: dict[str, Path]) -> dict[str, Any]:
    lock = load_json(paths["lockfile"])
    if lock.get("status") != "SMOKE40_LOCKFILE_CREATED" or lock.get("candidate_count") != 40:
        print("SMOKE40_LOCKFILE_VERIFICATION_FAILED")
        raise RuntimeError("SMOKE40_LOCKFILE_VERIFICATION_FAILED")
    result = {
        "data": lock,
        "sha256": sha256_file(paths["lockfile"]),
    }
    print(f"SMOKE40_LOCKFILE_OK: {result['sha256']}")
    return result


def _fail_lock(message: str) -> None:
    print("SMOKE40_LOCKFILE_VERIFICATION_FAILED")
    raise RuntimeError(f"SMOKE40_LOCKFILE_VERIFICATION_FAILED: {message}")


def verify_input_hashes(paths: dict[str, Path], lock_state: dict[str, Any]) -> dict[str, Any]:
    lock = lock_state["data"]
    tracked = {
        "smoke40_inference_manifest.json": paths["manifest"],
        "verifier_prompt_v1.txt": paths["prompt"],
        "smoke40_protocol_v1.yaml": paths["protocol"],
    }
    hashes: dict[str, str] = {}
    for name, path in tracked.items():
        actual = sha256_file(path)
        expected = lock.get("locked_files", {}).get(name)
        if actual != expected:
            _fail_lock(f"{name} hash mismatch")
        hashes[name] = actual

    manifest = load_json(paths["manifest"])
    rows = manifest.get("candidates", [])
    if len(rows) != 40 or len({row.get("candidate_id") for row in rows}) != 40:
        _fail_lock("manifest must contain 40 unique candidates")
    locked_images = lock.get("smoke40_sorted_file_hash_manifest", {})
    checked_images: set[str] = set()
    for row in rows:
        frames = row.get("frame_files", [])
        if [frame.get("role") for frame in frames] != ROLES_LOCKED:
            _fail_lock(f"frame role order changed for {row.get('candidate_id')}")
        for frame in frames:
            path = (paths["manifest"].parent / frame["file"]).resolve()
            try:
                relative = path.relative_to(paths["root"] / "data/smoke40").as_posix()
            except ValueError as error:
                _fail_lock(f"frame escaped deployment smoke40 root: {path}")
                raise AssertionError from error
            expected = locked_images.get(relative)
            if expected is None or not path.is_file() or sha256_file(path) != expected:
                _fail_lock(f"candidate image mismatch: {relative}")
            checked_images.add(relative)
    if len(checked_images) != 40 * 9:
        _fail_lock(f"expected 360 unique candidate images, found {len(checked_images)}")
    hashes["candidate_image_count"] = len(checked_images)
    print("SMOKE40_INPUT_GATE_PASS")
    return {"manifest": manifest, "hashes": hashes}


_FORBIDDEN_KEYS = {
    "tp", "fp", "gt", "g", "l", "ground_truth", "match_label", "true_positive",
    "false_positive", "positive", "negative", "s_glsd", "glsd_score", "threshold",
}
_FORBIDDEN_TEXT = re.compile(
    r"(?<![A-Za-z0-9])(?:TP|FP|GT|ground[ _-]?truth|match[ _-]?label|"
    r"true[ _-]?positive|false[ _-]?positive|positive|negative|S_GLSD|GLSD[ _-]?score|"
    r"threshold|G|L)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _check_metadata(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in _FORBIDDEN_KEYS:
                raise RuntimeError(f"LABEL_LEAKAGE_GATE_FAILED: forbidden metadata key {location}.{key}")
            _check_metadata(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_metadata(child, f"{location}[{index}]")


def verify_label_leakage(paths: dict[str, Path], manifest: dict[str, Any]) -> None:
    try:
        _check_metadata(manifest, "inference_manifest")
        prompt = paths["prompt"].read_text(encoding="utf-8")
        match = _FORBIDDEN_TEXT.search(prompt)
        if match:
            raise RuntimeError(f"forbidden prompt token {match.group(0)!r}")
        for directory in (paths["candidates"], paths["contact_sheets"]):
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(directory).as_posix()
                match = _FORBIDDEN_TEXT.search(relative)
                if match:
                    raise RuntimeError(
                        f"forbidden token {match.group(0)!r} in model-facing filename {relative}"
                    )
    except Exception as error:
        print("LABEL_LEAKAGE_GATE_FAILED")
        if str(error).startswith("LABEL_LEAKAGE_GATE_FAILED"):
            raise
        raise RuntimeError(f"LABEL_LEAKAGE_GATE_FAILED: {error}") from error
    print("LABEL_LEAKAGE_GATE_PASS")


def precision_policy(total_vram_gib: float) -> tuple[str, str, int]:
    if total_vram_gib >= 35:
        return "bf16", "VRAM >= 35 GiB", 1024 * 1024
    if total_vram_gib >= 20:
        return "int4", "20 GiB <= VRAM < 35 GiB", 768 * 768
    return "int4_low_memory", "VRAM < 20 GiB", 512 * 512


def select_precision(gpu: dict[str, Any], results_root: str | Path) -> dict[str, Any]:
    mode, reason, max_pixels = precision_policy(gpu["total_vram_gib"])
    config = {
        "model_name": MODEL_NAME_LOCKED,
        "precision_mode": mode,
        "precision_reason": reason,
        "batch_size": 1,
        "seed": SEED_LOCKED,
        "visual_preprocessing": {
            "frame_count": 9,
            "frame_roles": ROLES_LOCKED,
            "image_patch_size": 16,
            "min_pixels": 16 * 16 * 256,
            "max_pixels": max_pixels,
            "uniform_for_all_candidates": True,
        },
        "generation_config": GENERATION_CONFIG,
        "gpu": gpu,
    }
    print(f"Selected precision mode: {mode}")
    print(f"Reason: {reason}")
    print(f"GPU: {gpu['model']}")
    print(f"VRAM: {gpu['total_vram_gib']:.1f} GiB")
    write_json(Path(results_root) / "runtime_config.json", config)
    return config


def install_dependencies(quantized: bool) -> None:
    packages = ["transformers>=4.57.0", "accelerate", "qwen-vl-utils", "Pillow"]
    try:
        import torch  # noqa: F401
    except ImportError:
        packages.insert(0, "torch")
    if quantized:
        packages.append("bitsandbytes")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages], check=True)
    print("INFERENCE_DEPENDENCIES_INSTALLED")


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_versions() -> dict[str, Any]:
    import torch

    versions = {
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "qwen-vl-utils": package_version("qwen-vl-utils"),
        "Pillow": package_version("Pillow"),
        "bitsandbytes": package_version("bitsandbytes"),
        "torch_cuda": torch.version.cuda,
    }
    for name, value in versions.items():
        print(f"{name} version: {value or 'not installed'}")
    return versions


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() or error.__class__.__name__ == "OutOfMemoryError"


def _clear_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def load_model(runtime: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    def load(mode: str) -> tuple[Any, Any]:
        kwargs: dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
        if mode == "bf16":
            kwargs["dtype"] = torch.bfloat16
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        processor = AutoProcessor.from_pretrained(MODEL_NAME_LOCKED)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME_LOCKED, **kwargs)
        return model, processor

    mode = runtime["precision_mode"]
    try:
        model, processor = load(mode)
    except Exception as error:
        if mode != "bf16" or not _is_oom(error):
            if _is_oom(error):
                print("COLAB_HARDWARE_INSUFFICIENT")
            raise
        print("MODEL_LOAD_OOM")
        print("Applying declared fallback: bf16 -> int4")
        _clear_cuda()
        if package_version("bitsandbytes") is None:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "bitsandbytes"],
                check=True,
            )
        try:
            model, processor = load("int4")
        except Exception as fallback_error:
            if _is_oom(fallback_error):
                print("COLAB_HARDWARE_INSUFFICIENT")
            raise
        runtime = dict(runtime)
        runtime["precision_mode"] = "int4"
        runtime["precision_reason"] = "bf16 model-load OOM; declared fallback to int4"

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise RuntimeError("FROZEN_MODEL_GATE_FAILED")
    print("trainable parameters = 0")
    print("MODEL_LOAD_PASS")
    return model, processor, runtime


def validate_parsed_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(EXPECTED_OUTPUT_KEYS):
        raise ValueError("output must contain exactly the five locked keys")
    if any(not isinstance(value[key], bool) for key in EXPECTED_OUTPUT_KEYS[:-1]):
        raise ValueError("the four diagnostic fields must be booleans")
    if value["verdict"] not in {"keep", "reject"}:
        raise ValueError("verdict must be keep or reject")
    return {key: value[key] for key in EXPECTED_OUTPUT_KEYS}


def parse_qwen_output(raw: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    cleaned = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return validate_parsed_output(json.loads(cleaned)), False, None
    except (json.JSONDecodeError, ValueError) as first_error:
        decoder = json.JSONDecoder()
        for start in (match.start() for match in re.finditer(r"\{", cleaned)):
            try:
                value, _ = decoder.raw_decode(cleaned[start:])
                return validate_parsed_output(value), True, None
            except (json.JSONDecodeError, ValueError):
                continue
        left, right = cleaned.find("{"), cleaned.rfind("}")
        if left < 0 or right < left:
            return None, True, str(first_error)
        repaired = re.sub(r",\s*([}\]])", r"\1", cleaned[left : right + 1])
        try:
            return validate_parsed_output(json.loads(repaired)), True, None
        except (json.JSONDecodeError, ValueError) as repair_error:
            return None, True, f"{first_error}; repair failed: {repair_error}"


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def infer_candidate(
    row: dict[str, Any],
    prompt: str,
    manifest_path: Path,
    model: Any,
    processor: Any,
    runtime: dict[str, Any],
    identity: dict[str, str],
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    visual = runtime["visual_preprocessing"]
    content = []
    for frame in row["frame_files"]:
        image_path = (manifest_path.parent / frame["file"]).resolve()
        content.append(
            {
                "type": "image",
                "image": image_path.as_uri(),
                "min_pixels": visual["min_pixels"],
                "max_pixels": visual["max_pixels"],
            }
        )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    ).to(_model_device(model))
    torch.manual_seed(SEED_LOCKED)
    torch.cuda.manual_seed_all(SEED_LOCKED)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    parsed, repaired, error = parse_qwen_output(raw)
    result = {
        "candidate_id": row["candidate_id"],
        "model_name": MODEL_NAME_LOCKED,
        "precision_mode": runtime["precision_mode"],
        "prompt_sha256": identity["prompt_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "smoke40_lock_sha256": identity["lockfile_sha256"],
        "preprocessing_sha256": sha256_json(runtime["visual_preprocessing"]),
        "generation_config_sha256": sha256_json(GENERATION_CONFIG),
        "raw_output": raw,
        "parsed_output": parsed or {},
        "verdict": parsed["verdict"] if parsed else None,
        "parse_ok": error is None,
        "format_repair_attempted": repaired,
        "parse_error": error,
        "mock_output": False,
        "completed_at_utc": utc_now(),
    }
    del inputs, generated, trimmed
    _clear_cuda()
    return result


def result_is_resumable(
    result: dict[str, Any], candidate_id: str, runtime: dict[str, Any], identity: dict[str, str]
) -> bool:
    expected = {
        "candidate_id": candidate_id,
        "model_name": MODEL_NAME_LOCKED,
        "precision_mode": runtime["precision_mode"],
        "prompt_sha256": identity["prompt_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "smoke40_lock_sha256": identity["lockfile_sha256"],
        "preprocessing_sha256": sha256_json(runtime["visual_preprocessing"]),
        "generation_config_sha256": sha256_json(GENERATION_CONFIG),
        "mock_output": False,
    }
    return all(result.get(key) == value for key, value in expected.items()) and isinstance(
        result.get("parse_ok"), bool
    )


def enforce_resume_precision(results_root: Path, precision_mode: str) -> None:
    per_candidate = results_root / "formal/per_candidate"
    found_modes = set()
    for path in per_candidate.glob("*.json"):
        try:
            found_modes.add(load_json(path).get("precision_mode"))
        except (OSError, json.JSONDecodeError):
            continue
    found_modes.discard(None)
    if found_modes and found_modes != {precision_mode}:
        raise RuntimeError(
            "FORMAL_PRECISION_MISMATCH_RESTART_REQUIRED: existing formal results use "
            f"{sorted(found_modes)}, current run requires {precision_mode}. Use an empty RESULTS_ROOT "
            "or deliberately move the prior run; mixed precision is forbidden."
        )


def run_preflight(
    rows: list[dict[str, Any]], prompt: str, paths: dict[str, Path], results_root: Path,
    model: Any, processor: Any, runtime: dict[str, Any], identity: dict[str, str]
) -> None:
    selected = random.Random(SEED_LOCKED).sample(rows, 2)
    output_dir = results_root / "preflight"
    for row in selected:
        result = infer_candidate(row, prompt, paths["manifest"], model, processor, runtime, identity)
        write_json(output_dir / f"{row['candidate_id']}.json", result)
        if not result["parse_ok"]:
            raise RuntimeError(f"TECHNICAL_PREFLIGHT_FAILED: JSON parse failed for {row['candidate_id']}")
    print("TECHNICAL_PREFLIGHT_PASS")


def run_formal(
    rows: list[dict[str, Any]], prompt: str, paths: dict[str, Path], results_root: Path,
    model: Any, processor: Any, runtime: dict[str, Any], identity: dict[str, str]
) -> None:
    enforce_resume_precision(results_root, runtime["precision_mode"])
    output_dir = results_root / "formal/per_candidate"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        output_path = output_dir / f"{row['candidate_id']}.json"
        valid = False
        if output_path.is_file():
            try:
                valid = result_is_resumable(load_json(output_path), row["candidate_id"], runtime, identity)
            except (OSError, json.JSONDecodeError):
                valid = False
        if valid:
            print(f"[{index:02d}/40] SKIP valid checkpoint: {row['candidate_id']}")
        else:
            result = infer_candidate(row, prompt, paths["manifest"], model, processor, runtime, identity)
            write_json(output_path, result)
            print(f"[{index:02d}/40] SAVED: {row['candidate_id']}")
        completed = sum(
            result_is_resumable(load_json(path), path.stem, runtime, identity)
            for path in output_dir.glob("*.json")
        )
        print(f"Completed: {completed} / 40")
        print(f"Remaining: {40 - completed} / 40")


def verify_completion(
    rows: list[dict[str, Any]], results_root: Path, runtime: dict[str, Any], identity: dict[str, str]
) -> list[dict[str, Any]]:
    expected_ids = [row["candidate_id"] for row in rows]
    files = list((results_root / "formal/per_candidate").glob("*.json"))
    if len(files) != 40:
        raise RuntimeError(f"FORMAL_COMPLETION_GATE_FAILED: found {len(files)} result files")
    by_id = {}
    for path in files:
        result = load_json(path)
        candidate_id = result.get("candidate_id")
        if candidate_id in by_id or not result_is_resumable(result, candidate_id, runtime, identity):
            raise RuntimeError(f"FORMAL_COMPLETION_GATE_FAILED: invalid result {path.name}")
        by_id[candidate_id] = result
    if set(by_id) != set(expected_ids) or any(item.get("mock_output") for item in by_id.values()):
        raise RuntimeError("FORMAL_COMPLETION_GATE_FAILED: identities or mock-output gate failed")
    ordered = [by_id[candidate_id] for candidate_id in expected_ids]
    jsonl = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered)
    write_text(results_root / "formal/smoke40_raw.jsonl", jsonl)
    summary = {
        "MOCK_OUTPUT": False,
        "model": MODEL_NAME_LOCKED,
        "precision": runtime["precision_mode"],
        "candidate_count": 40,
        "results": ordered,
    }
    write_json(results_root / "formal/inference_summary.json", summary)
    print("FORMAL_SMOKE40_INFERENCE_COMPLETE")
    return ordered


def run_replay(
    rows: list[dict[str, Any]], formal: list[dict[str, Any]], prompt: str,
    paths: dict[str, Path], results_root: Path, model: Any, processor: Any,
    runtime: dict[str, Any], identity: dict[str, str]
) -> dict[str, Any]:
    selected = random.Random(SEED_LOCKED).sample(rows, 5)
    formal_by_id = {item["candidate_id"]: item for item in formal}
    output_dir = results_root / "deterministic_replay/per_candidate"
    comparisons = []
    for row in selected:
        replay = infer_candidate(row, prompt, paths["manifest"], model, processor, runtime, identity)
        write_json(output_dir / f"{row['candidate_id']}.json", replay)
        original = formal_by_id[row["candidate_id"]]
        comparisons.append(
            {
                "candidate_id": row["candidate_id"],
                "verdict_equal": original["verdict"] == replay["verdict"],
                "parsed_json_equal": original["parsed_output"] == replay["parsed_output"],
                "raw_text_equal": original["raw_output"] == replay["raw_output"],
            }
        )
    verdict_pass = all(item["verdict_equal"] for item in comparisons)
    exact_json_pass = all(item["parsed_json_equal"] for item in comparisons)
    replay_summary = {
        "seed": SEED_LOCKED,
        "candidate_count": 5,
        "comparisons": comparisons,
        "verdict_status": "PASS" if verdict_pass else "FAIL",
        "parsed_json_status": "PASS" if exact_json_pass else "DIFF",
    }
    write_json(results_root / "deterministic_replay/replay_summary.json", replay_summary)
    if verdict_pass:
        print("DETERMINISTIC_REPLAY_PASS")
        if not exact_json_pass:
            print("DETERMINISTIC_REPLAY_PARSED_JSON_DIFFERENCE_RECORDED")
    else:
        print("DETERMINISTIC_REPLAY_FAILED")
        raise RuntimeError("DETERMINISTIC_REPLAY_FAILED")
    return replay_summary


def run_evaluation(paths: dict[str, Path], results_root: Path) -> dict[str, Any]:
    summary = results_root / "formal/inference_summary.json"
    formal = load_json(summary)
    if formal.get("candidate_count") != 40 or formal.get("MOCK_OUTPUT") is not False:
        raise RuntimeError("PRIVATE_LABEL_ACCESS_BLOCKED: formal completion gate is not satisfied")
    output = results_root / "smoke40_evaluation.json"
    temporary_report = results_root / "SMOKE40_REPORT.evaluator.tmp.md"
    subprocess.run(
        [
            sys.executable,
            str(paths["evaluator"]),
            "--raw", str(summary),
            "--labels", str(paths["labels"]),
            "--output", str(output),
            "--report", str(temporary_report),
        ],
        check=True,
        cwd=paths["root"],
    )
    evaluation = load_json(output)
    evaluation["json_parse_success_rate"] = sum(item["parse_ok"] for item in formal["results"]) / 40
    write_json(output, evaluation)
    temporary_report.unlink(missing_ok=True)
    print(f"JSON parse success rate: {evaluation['json_parse_success_rate']:.3f}")
    return evaluation


def save_runtime_metadata(
    results_root: Path, runtime: dict[str, Any], environment: dict[str, Any],
    identity: dict[str, str]
) -> Path:
    metadata = {
        "date_time_utc": utc_now(),
        "gpu_model": runtime["gpu"]["model"],
        "vram_gib": runtime["gpu"]["total_vram_gib"],
        "cuda_version": environment.get("torch_cuda") or runtime["gpu"].get("cuda_version_driver"),
        "compute_capability": runtime["gpu"].get("compute_capability"),
        **{key: environment.get(key) for key in ("torch", "transformers", "accelerate", "qwen-vl-utils", "bitsandbytes")},
        "model_checkpoint": MODEL_NAME_LOCKED,
        "precision_mode": runtime["precision_mode"],
        "prompt_sha256": identity["prompt_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "lockfile_sha256": identity["lockfile_sha256"],
        "visual_preprocessing": runtime["visual_preprocessing"],
        "generation_config": GENERATION_CONFIG,
        "trainable_parameters": 0,
    }
    path = results_root / "runtime_metadata.json"
    write_json(path, metadata)
    return path


def generate_report(
    results_root: Path, runtime: dict[str, Any], evaluation: dict[str, Any], replay: dict[str, Any]
) -> Path:
    report = (
        "# Smoke40 Report\n\n"
        f"Model: {MODEL_NAME_LOCKED}\n\n"
        f"Precision: {runtime['precision_mode']}\n\n"
        f"GPU: {runtime['gpu']['model']}\n\n"
        "Candidate count: 40\n\n"
        "TP count: 20\n\n"
        "FP count: 20\n\n"
        f"TP kept: {evaluation['TP_kept']}\n\n"
        f"TP rejected: {evaluation['TP_rejected']}\n\n"
        f"FP kept: {evaluation['FP_kept']}\n\n"
        f"FP rejected: {evaluation['FP_rejected']}\n\n"
        f"TP retention: {evaluation['TP_retention']:.3f}\n\n"
        f"FP removal: {evaluation['FP_removal']:.3f}\n\n"
        f"JSON parse success: {evaluation['json_parse_success_rate']:.3f}\n\n"
        f"Deterministic replay: {replay['verdict_status']} "
        f"(parsed JSON: {replay['parsed_json_status']})\n\n"
        f"Exploratory status: {evaluation['exploration_gate']}\n"
    )
    path = results_root / "SMOKE40_REPORT.md"
    write_text(path, report)
    return path


def generate_result_lock(
    results_root: Path, runtime: dict[str, Any], identity: dict[str, str],
    evaluation_path: Path, metadata_path: Path, report_path: Path
) -> Path:
    result_files = sorted(
        path for directory in (results_root / "formal", results_root / "deterministic_replay")
        for path in directory.rglob("*") if path.is_file()
    )
    lock = {
        "status": "SMOKE40_RESULT_LOCK_CREATED",
        "model": MODEL_NAME_LOCKED,
        "precision": runtime["precision_mode"],
        "prompt_sha256": identity["prompt_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "input_lock_sha256": identity["lockfile_sha256"],
        "result_hashes": {
            str(path.relative_to(results_root)): sha256_file(path) for path in result_files
        },
        "evaluation_result_sha256": sha256_file(evaluation_path),
        "runtime_metadata_sha256": sha256_file(metadata_path),
        "report_sha256": sha256_file(report_path),
    }
    path = results_root / "SMOKE40_RESULT_LOCK.json"
    write_json(path, lock)
    return path


def build_identity(paths: dict[str, Path], lock_state: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt_sha256": sha256_file(paths["prompt"]),
        "protocol_sha256": sha256_file(paths["protocol"]),
        "lockfile_sha256": lock_state["sha256"],
    }


def mock_dry_run(deployment_root: str | Path, results_root: str | Path) -> None:
    paths = locate_deployment(deployment_root)
    lock_state = verify_lockfile(paths)
    verified = verify_input_hashes(paths, lock_state)
    verify_label_leakage(paths, verified["manifest"])
    sample = json.dumps(
        {
            "local_facial_change": True,
            "brief_transient_change": True,
            "return_toward_baseline": True,
            "global_motion_artifact": False,
            "verdict": "keep",
        }
    )
    parsed, _, error = parse_qwen_output(f"```json\n{sample}\n```")
    if error or parsed is None:
        raise RuntimeError("MOCK_DRY_RUN_FAILED: parser")
    output = Path(results_root) / "mock_validation.json"
    write_json(
        output,
        {
            "MOCK_OUTPUT": True,
            "evaluation_status": "NOT_FOR_EVALUATION",
            "candidate_count": len(verified["manifest"]["candidates"]),
            "input_lock_sha256": lock_state["sha256"],
            "parser_result": parsed,
        },
    )
    print("MOCK_DRY_RUN_PASS")
    print("MLLM_INFERENCE_NOT_RUN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--mock-dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock_dry_run:
        mock_dry_run(args.deployment_root, args.results_root)
        return
    raise RuntimeError(
        "Use colab_run_smoke40_qwen3vl.ipynb for the staged formal run. "
        "The module functions are also reusable on an independent GPU server."
    )


if __name__ == "__main__":
    main()
