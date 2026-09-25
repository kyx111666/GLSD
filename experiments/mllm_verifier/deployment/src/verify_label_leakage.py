#!/usr/bin/env python3
"""Fail closed if model-facing smoke40 inputs expose evaluation labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import assert_no_leakage, forbidden_term, load_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data/manifests/smoke40_inference_manifest.json")
    parser.add_argument("--candidates", type=Path, default=root / "data/smoke40/candidates")
    parser.add_argument("--contact-sheets", type=Path, default=root / "data/smoke40/contact_sheets")
    parser.add_argument("--prompt", type=Path, default=root / "config/verifier_prompt_v1.txt")
    args = parser.parse_args()

    assert_no_leakage(load_json(args.manifest), "inference_manifest")
    assert_no_leakage(args.prompt.read_text(encoding="utf-8"), "prompt")
    for directory in (args.candidates, args.contact_sheets):
        if not directory.is_dir():
            raise RuntimeError(f"LABEL_LEAKAGE_GATE_FAILED: missing directory {directory}")
        for path in directory.rglob("*"):
            relative = str(path.relative_to(directory))
            term = forbidden_term(relative)
            if term:
                raise RuntimeError(
                    f"LABEL_LEAKAGE_GATE_FAILED: {term!r} in model-facing path {relative}"
                )
    print("LABEL_LEAKAGE_GATE_PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("LABEL_LEAKAGE_GATE_FAILED")
        raise

