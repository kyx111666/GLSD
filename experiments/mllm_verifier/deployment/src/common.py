"""Shared, GPU-independent smoke40 helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROLES = ["B1", "B2", "B3", "C1", "C2", "C3", "A1", "A2", "A3"]
FORBIDDEN_TERMS = (
    "tp",
    "fp",
    "ground_truth",
    "gt_label",
    "match_label",
    "positive",
    "negative",
    "true_positive",
    "false_positive",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbidden_term(text: str) -> str | None:
    normalized = text.lower()
    for term in FORBIDDEN_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized):
            return term
    return None


def assert_no_leakage(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            term = forbidden_term(str(key))
            if term:
                raise RuntimeError(
                    f"LABEL_LEAKAGE_GATE_FAILED: {term!r} in key {location}.{key}"
                )
            assert_no_leakage(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_leakage(child, f"{location}[{index}]")
    elif isinstance(value, str):
        term = forbidden_term(value)
        if term:
            raise RuntimeError(
                f"LABEL_LEAKAGE_GATE_FAILED: {term!r} in value at {location}"
            )

