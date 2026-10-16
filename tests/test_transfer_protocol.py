#!/usr/bin/env python3
"""Dependency-light smoke tests; these do not read official responses."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from transfer_common import canonical_subject, paired_bootstrap, selected_config_rows
from run_transfer_target import assert_source_k_never_passed, target_config, theta_of


def test_canonical_subjects() -> None:
    assert canonical_subject("sammlv", "6") == "006"
    assert canonical_subject("sammlv", "006") == "006"
    assert canonical_subject("casme3", "015") == "15"


def test_selected_config_rejects_k() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["outer_subject", "reference_scale", "local_radius", "threshold", "k_p"])
            writer.writeheader()
            writer.writerow({"outer_subject": "006", "reference_scale": 1, "local_radius": 2, "threshold": 0.3, "k_p": 17})
        try:
            selected_config_rows(path, "sammlv")
        except RuntimeError as exc:
            assert "forbidden transfer fields" in str(exc)
        else:
            raise AssertionError("source k was not rejected")


def test_target_config_is_target_owned() -> None:
    grid = [
        {"reference_scale": 1.0, "local_radius": 2.0, "threshold": 0.3, "target_only": 7},
        {"reference_scale": 1.5, "local_radius": 2.0, "threshold": 0.3, "target_only": 8},
    ]
    selected = target_config(grid, (1.5, 2.0, 0.3))
    assert selected["target_only"] == 8
    assert theta_of(selected) == (1.5, 2.0, 0.3)


def test_explicit_source_k_assertion() -> None:
    assert_source_k_never_passed({"a0": 1.0, "rho": 2.0, "tau": 0.3})
    try:
        assert_source_k_never_passed({"a0": 1.0, "rho": 2.0, "tau": 0.3, "source_k": 17})
    except AssertionError:
        pass
    else:
        raise AssertionError("explicit source-k assertion did not fire")


def test_bootstrap_is_paired_and_deterministic() -> None:
    native = np.asarray([[1, 2, 3], [2, 1, 2], [0, 1, 1]])
    tuned = np.asarray([[1, 1, 3], [2, 1, 2], [0, 1, 1]])
    transferred = np.asarray([[2, 1, 2], [2, 1, 2], [0, 1, 1]])
    first = paired_bootstrap(native, tuned, transferred, iterations=100, seed=100)
    second = paired_bootstrap(native, tuned, transferred, iterations=100, seed=100)
    assert first == second
    assert first["paired_resampling"] is True
    assert first["unit"] == "subject"


if __name__ == "__main__":
    test_canonical_subjects()
    test_selected_config_rejects_k()
    test_target_config_is_target_owned()
    test_explicit_source_k_assertion()
    test_bootstrap_is_paired_and_deterministic()
    print("TRANSFER_PROTOCOL_SMOKE_TESTS = PASS")
