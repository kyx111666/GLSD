"""Strict parser for the locked five-field verifier response."""

from __future__ import annotations

import json
import re
from typing import Any


EXPECTED_KEYS = (
    "local_facial_change",
    "brief_transient_change",
    "return_toward_baseline",
    "global_motion_artifact",
    "verdict",
)


def validate_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(EXPECTED_KEYS):
        raise ValueError("output must contain exactly the five locked keys")
    if any(not isinstance(value[key], bool) for key in EXPECTED_KEYS[:-1]):
        raise ValueError("the four diagnostic fields must be booleans")
    if value["verdict"] not in {"keep", "reject"}:
        raise ValueError("verdict must be keep or reject")
    return {key: value[key] for key in EXPECTED_KEYS}


def parse_qwen_output(raw: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    try:
        return validate_output(json.loads(raw)), False, None
    except (json.JSONDecodeError, ValueError) as first_error:
        left, right = raw.find("{"), raw.rfind("}")
        if left < 0 or right < left:
            return None, True, str(first_error)
        repaired = re.sub(r",\s*([}\]])", r"\1", raw[left : right + 1])
        try:
            return validate_output(json.loads(repaired)), True, None
        except (json.JSONDecodeError, ValueError) as second_error:
            return None, True, f"{first_error}; repair failed: {second_error}"

