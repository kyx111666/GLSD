#!/usr/bin/env python3
"""Frozen Qwen3-VL candidate verifier with leakage and determinism gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

from parse_output import parse_model_output
from protocol import SEED, assert_no_label_leakage, load_json


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SYSTEM_PROMPT = """You are given an ordered sequence of facial frames surrounding a temporal candidate detected by a separate temporal decoder.

The frames are ordered as:

Before → Candidate → After.

Your task is NOT to classify emotion.

Determine whether the candidate region contains a brief, localized, non-global facial change that is visually consistent with a transient facial event.

Evaluate only visible evidence.

Consider:

1. whether a localized facial region changes;
2. whether the change is brief and temporally concentrated;
3. whether the face tends to return toward the preceding appearance afterward;
4. whether the apparent change is instead dominated by global head motion, camera motion, blur, occlusion, lighting change, or other artifacts.

Return ONLY valid JSON."""
USER_PROMPT = """Inspect the nine ordered facial frames in this exact order:
B1, B2, B3, C1, C2, C3, A1, A2, A3.
B denotes Before, C denotes Candidate, and A denotes After.

Return exactly this JSON schema, with verdict equal to either \"keep\" or \"reject\":
{
  \"local_facial_change\": true,
  \"brief_transient_change\": true,
  \"return_toward_baseline\": true,
  \"global_motion_artifact\": false,
  \"verdict\": \"keep\"
}"""


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + USER_PROMPT).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=root / "data/inference_manifest.json"
    )
    parser.add_argument("--output", type=Path, default=root / "results/smoke40_raw.jsonl")
    parser.add_argument("--metadata", type=Path, default=root / "logs/inference_metadata.json")
    parser.add_argument(
        "--determinism-output",
        type=Path,
        default=root / "results/determinism_gate.json",
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--determinism-candidates", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_messages(frame_paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "text", "text": USER_PROMPT}]
    content.extend(
        {"type": "image", "image": Path(path).resolve().as_uri()} for path in frame_paths
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    assert_no_label_leakage(manifest)
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Inference manifest has no candidates")
    seen = set()
    for row in rows:
        if set(row) - {
            "candidate_id",
            "ordered_frame_paths",
            "ordered_frame_roles",
            "contact_sheet_path",
            "extraction_warnings",
        }:
            raise RuntimeError(f"Unexpected inference fields for {row.get('candidate_id')}")
        candidate_id = row["candidate_id"]
        if candidate_id in seen:
            raise RuntimeError(f"Duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        if row["ordered_frame_roles"] != [
            "B1",
            "B2",
            "B3",
            "C1",
            "C2",
            "C3",
            "A1",
            "A2",
            "A3",
        ]:
            raise RuntimeError(f"Invalid frame order for {candidate_id}")
        paths = [Path(path) for path in row["ordered_frame_paths"]]
        if len(paths) != 9 or any(not path.is_file() for path in paths):
            raise RuntimeError(f"Missing ordered frame(s) for {candidate_id}")
    return rows


def seed_everything(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module, "use_deterministic_algorithms"):
        torch_module.use_deterministic_algorithms(True, warn_only=True)


def load_frozen_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Missing inference dependencies. Install transformers, accelerate, "
            "qwen-vl-utils, and a compatible PyTorch build."
        ) from error

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError as error:
            raise RuntimeError("Installed Transformers does not support Qwen3-VL") from error

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[args.dtype],
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.quantization != "none":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("Quantized loading requires bitsandbytes support") from error
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quantization == "8bit",
            load_in_4bit=args.quantization == "4bit",
        )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = ModelClass.from_pretrained(args.model, **kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"Frozen-model gate failed: trainable parameters={trainable}")
    return model, processor, process_vision_info, (torch, transformers)


def move_inputs(inputs: Any, model: Any) -> Any:
    try:
        device = next(model.parameters()).device
        return inputs.to(device)
    except (StopIteration, AttributeError):
        return inputs


def infer_one(
    row: dict[str, Any],
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch_module: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = build_messages(row["ordered_frame_paths"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs(inputs, model)
    with torch_module.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    parsed, parse_ok, repaired, parse_error = parse_model_output(raw)
    return {
        "candidate_id": row["candidate_id"],
        "model_output_raw": raw,
        "parsed_output": parsed,
        "verdict": parsed["verdict"] if parsed else None,
        "parse_ok": parse_ok,
        "format_repair_attempted": repaired,
        "parse_error": parse_error,
    }


def main() -> None:
    args = parse_args()
    base_metadata = {
        "model_name": args.model,
        "checkpoint": args.model,
        "prompt_sha256": prompt_hash(),
        "seed": args.seed,
        "generation_config": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        },
        "dtype": args.dtype,
        "quantization": args.quantization,
        "python_version": sys.version,
        "platform": platform.platform(),
        "candidate_count": None,
        "label_leakage_gate": "NOT_RUN",
        "trainable_parameters_required": 0,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run and not args.manifest.is_file():
        base_metadata.update(
            {
                "status": "DRY_RUN_INTERFACE_ONLY",
                "model_loaded": False,
                "note": (
                    "Fixed prompt, output parser, model interface, and generation "
                    "configuration are defined. Manifest/frame/leakage runtime checks "
                    "are pending a valid balanced smoke40 manifest."
                ),
            }
        )
        args.metadata.write_text(json.dumps(base_metadata, indent=2) + "\n", encoding="utf-8")
        print("DRY_RUN_INTERFACE_ONLY")
        return
    manifest = load_json(args.manifest)
    rows = validate_manifest(manifest)
    base_metadata["candidate_count"] = len(rows)
    base_metadata["label_leakage_gate"] = "PASS"
    if args.dry_run:
        base_metadata.update(
            {
                "status": "DRY_RUN_PASS",
                "model_loaded": False,
                "note": "Manifest, frames, fixed prompt, and leakage gate validated.",
            }
        )
        args.metadata.write_text(json.dumps(base_metadata, indent=2) + "\n", encoding="utf-8")
        print("DRY_RUN_PASS")
        return

    model, processor, process_vision_info, modules = load_frozen_model(args)
    torch_module, transformers_module = modules
    seed_everything(torch_module, args.seed)
    base_metadata.update(
        {
            "status": "INFERENCE_STARTED",
            "torch_version": torch_module.__version__,
            "transformers_version": transformers_module.__version__,
            "cuda_version": torch_module.version.cuda,
            "cuda_available": torch_module.cuda.is_available(),
            "trainable_parameters": 0,
        }
    )
    args.metadata.write_text(json.dumps(base_metadata, indent=2) + "\n", encoding="utf-8")
    outputs = []
    for row in rows:
        try:
            outputs.append(
                infer_one(
                    row,
                    model,
                    processor,
                    process_vision_info,
                    torch_module,
                    args.max_new_tokens,
                )
            )
        except Exception as error:  # preserve candidate-level failure evidence
            outputs.append(
                {
                    "candidate_id": row["candidate_id"],
                    "model_output_raw": "",
                    "parsed_output": None,
                    "verdict": None,
                    "parse_ok": False,
                    "format_repair_attempted": False,
                    "parse_error": None,
                    "inference_error": repr(error),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in outputs),
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    replay_rows = rng.sample(rows, min(args.determinism_candidates, len(rows)))
    replay = []
    for row in replay_rows:
        try:
            first = infer_one(
                row, model, processor, process_vision_info, torch_module, args.max_new_tokens
            )
            second = infer_one(
                row, model, processor, process_vision_info, torch_module, args.max_new_tokens
            )
            error = None
        except Exception as replay_error:
            first = {"verdict": None}
            second = {"verdict": None}
            error = repr(replay_error)
        replay.append(
            {
                "candidate_id": row["candidate_id"],
                "first_verdict": first["verdict"],
                "second_verdict": second["verdict"],
                "consistent": first["verdict"] is not None
                and first["verdict"] == second["verdict"],
                "error": error,
            }
        )
    determinism = {
        "seed": args.seed,
        "requested_candidates": args.determinism_candidates,
        "tested_candidates": len(replay),
        "status": "PASS" if replay and all(item["consistent"] for item in replay) else "FAIL",
        "results": replay,
    }
    args.determinism_output.write_text(
        json.dumps(determinism, indent=2) + "\n", encoding="utf-8"
    )
    base_metadata["status"] = "INFERENCE_COMPLETE"
    base_metadata["determinism_gate"] = determinism["status"]
    args.metadata.write_text(json.dumps(base_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; determinism={determinism['status']}")


if __name__ == "__main__":
    main()
