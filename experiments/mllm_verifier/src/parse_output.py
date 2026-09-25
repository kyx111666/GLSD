"""Strict JSON parser with at most one deterministic, syntax-only repair."""

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


def validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(EXPECTED_KEYS):
        raise ValueError("output must contain exactly the five schema keys")
    for key in EXPECTED_KEYS[:-1]:
        if not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be boolean")
    if payload["verdict"] not in {"keep", "reject"}:
        raise ValueError("verdict must be keep or reject")
    return {key: payload[key] for key in EXPECTED_KEYS}


def deterministic_repair(raw: str) -> str:
    """Remove wrappers/trailing commas only; never alter semantic values."""
    left, right = raw.find("{"), raw.rfind("}")
    if left < 0 or right < left:
        raise ValueError("no JSON object found")
    candidate = raw[left : right + 1]
    return re.sub(r",\s*([}\]])", r"\1", candidate)


def parse_model_output(raw: str) -> tuple[dict[str, Any] | None, bool, bool, str | None]:
    try:
        return validate(json.loads(raw)), True, False, None
    except (json.JSONDecodeError, ValueError) as first_error:
        try:
            repaired = deterministic_repair(raw)
            return validate(json.loads(repaired)), True, True, None
        except (json.JSONDecodeError, ValueError) as second_error:
            return None, False, True, f"{first_error}; repair failed: {second_error}"

