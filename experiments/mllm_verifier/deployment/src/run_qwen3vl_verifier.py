#!/usr/bin/env python3
"""Qwen3-VL verifier entry point; GPU imports occur only for real inference."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from common import ROLES, assert_no_leakage, load_json, sha256_file, write_json
from parse_qwen_output import parse_qwen_output


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PRECISIONS = ("bf16", "fp16", "int8", "int4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/smoke40_inference_manifest.json")
    parser.add_argument("--prompt", type=Path, default=ROOT / "config/verifier_prompt_v1.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke40_raw.json")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--precision", choices=PRECISIONS)
    parser.add_argument("--labels", type=Path, help="Forbidden: evaluation labels may not enter inference")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-output", action="store_true")
    return parser.parse_args()


def read_config(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if ":" not in stripped or stripped.startswith("-"):
            continue
        key, value = stripped.split(":", 1)
        if value.strip():
            values[key.strip()] = value.strip().strip("\"'")
    return values


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict], str]:
    if args.labels is not None:
        raise RuntimeError("LABEL_LEAKAGE_GATE_FAILED: labels input is forbidden")
    manifest = load_json(args.manifest)
    assert_no_leakage(manifest, "model_input_metadata")
    rows = manifest.get("candidates", [])
    if len(rows) != 40:
        raise RuntimeError(f"Expected 40 inference candidates, found {len(rows)}")
    for row in rows:
        if set(row) != {"candidate_id", "subject_id", "video_id", "frame_files"}:
            raise RuntimeError(f"Unexpected inference fields for {row.get('candidate_id')}")
        if [frame.get("role") for frame in row["frame_files"]] != ROLES:
            raise RuntimeError(f"Invalid frame role order for {row['candidate_id']}")
        for frame in row["frame_files"]:
            path = (args.manifest.parent / frame["file"]).resolve()
            with Image.open(path) as image:
                image.verify()
            frame["resolved_path"] = str(path)
    prompt = args.prompt.read_text(encoding="utf-8")
    assert_no_leakage(prompt, "prompt")
    return rows, prompt


def mock_result(row: dict) -> dict:
    raw = json.dumps(
        {
            "local_facial_change": True,
            "brief_transient_change": True,
            "return_toward_baseline": True,
            "global_motion_artifact": False,
            "verdict": "keep",
        }
    )
    parsed, repaired, error = parse_qwen_output(raw)
    return {
        "candidate_id": row["candidate_id"],
        "model_output_raw": raw,
        "structured_output": parsed,
        "verdict": parsed["verdict"] if parsed else None,
        "parse_ok": error is None,
        "format_repair_attempted": repaired,
        "parse_error": error,
    }


def real_inference(rows: list[dict], prompt: str, args: argparse.Namespace) -> list[dict]:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("GPU inference dependencies are missing; no fallback is permitted") from error
    if args.precision == "auto":
        args.precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
        print(f"AUTO_PRECISION_SELECTED={args.precision}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "int8": None, "int4": None}[args.precision]
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    else:
        try:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=args.precision == "int8", load_in_4bit=args.precision == "int4"
            )
        except Exception as error:
            raise RuntimeError(f"Precision mode {args.precision} is unsupported in this environment") from error
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 0:
        raise RuntimeError("Frozen-model gate failed")
    random.seed(100)
    torch.manual_seed(100)
    outputs = []
    for row in rows:
        content = [{"type": "image", "image": Path(frame["resolved_path"]).as_uri()} for frame in row["frame_files"]]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[rendered], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = [result[len(source):] for source, result in zip(inputs.input_ids, generated)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        parsed, repaired, error = parse_qwen_output(raw)
        outputs.append({"candidate_id": row["candidate_id"], "model_output_raw": raw, "structured_output": parsed, "verdict": parsed["verdict"] if parsed else None, "parse_ok": error is None, "format_repair_attempted": repaired, "parse_error": error})
    return outputs


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    args.model = args.model or config.get("model_name", MODEL)
    args.precision = args.precision or config.get("precision_mode", "bf16")
    if args.model != MODEL:
        raise RuntimeError("MODEL_VARIANT_CHANGED")
    if args.precision not in {*PRECISIONS, "auto"}:
        raise RuntimeError(f"Unsupported precision mode: {args.precision}")
    rows, prompt = validate_inputs(args)
    if args.mock_output and not args.dry_run:
        raise RuntimeError("--mock-output requires --dry-run")
    if args.dry_run:
        if not args.mock_output:
            raise RuntimeError("CPU dry run requires --mock-output")
        results = [mock_result(row) for row in rows]
        payload = {
            "MOCK_OUTPUT": True,
            "evaluation_status": "NOT_FOR_EVALUATION",
            "model_loaded": False,
            "candidate_count": len(results),
            "prompt_sha256": sha256_file(args.prompt),
            "results": results,
        }
        write_json(args.output, payload)
        print("MOCK_DRY_RUN_PASS")
        return
    results = real_inference(rows, prompt, args)
    write_json(args.output, {"MOCK_OUTPUT": False, "model": args.model, "precision": args.precision, "results": results})
    print("QWEN3VL_INFERENCE_COMPLETE")


if __name__ == "__main__":
    main()
