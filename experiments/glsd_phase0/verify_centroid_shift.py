#!/usr/bin/env python3
"""Offline diagnostic for temporal probability-centroid re-anchoring.

Examples
--------
# A directory containing 009_3.npy, 012_3.npy, ... (uses embedded cases):
python verify_centroid_shift.py --scores /path/to/score_arrays

# The frozen ME-TST+ project cache (reconstructs its tuned smoothing curve):
python verify_centroid_shift.py --scores sammlv_strategy1_outputs.pkl

# A CSV/JSON case list and one NPZ whose keys are video names:
python verify_centroid_shift.py --cases cases.csv --scores frozen_scores.npz

The case list must contain video_name (or video), pred_peak, pred_onset,
pred_offset, gt_onset, and gt_offset.  The oracle CSV produced by this project
is accepted directly via its nearest_gt_onset/nearest_gt_offset column names.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


GO_MESSAGE = "[DECISION: GO] 质心重锚定假设成立，继续推进解码器实现"
NO_GO_MESSAGE = "[DECISION: NO-GO] 质心无法有效纠正漂移，停止该路线"

# The 11 Oracle Recoverable Cases from the locked ME-TST+ SAMMLV diagnostic.
# Ten have pred_peak outside GT; the final case is retained as a control and is
# excluded from the GO/NO-GO denominator by default.
EMBEDDED_CASES = [
    {"video_name": "009_3", "pred_peak": 422, "pred_onset": 416, "pred_offset": 428, "gt_onset": 404, "gt_offset": 416, "c_s": 2.0},
    {"video_name": "012_3", "pred_peak": 668, "pred_onset": 662, "pred_offset": 674, "gt_onset": 660, "gt_offset": 667, "c_s": 2.0},
    {"video_name": "017_3", "pred_peak": 217, "pred_onset": 211, "pred_offset": 223, "gt_onset": 198, "gt_offset": 213, "c_s": 2.0},
    {"video_name": "019_4", "pred_peak": 65, "pred_onset": 59, "pred_offset": 71, "gt_onset": 70, "gt_offset": 78, "c_s": 2.0},
    {"video_name": "020_4", "pred_peak": 485, "pred_onset": 479, "pred_offset": 491, "gt_onset": 491, "gt_offset": 499, "c_s": 2.0},
    {"video_name": "026_2", "pred_peak": 572, "pred_onset": 566, "pred_offset": 578, "gt_onset": 573, "gt_offset": 585, "c_s": 2.0},
    {"video_name": "026_3", "pred_peak": 252, "pred_onset": 246, "pred_offset": 258, "gt_onset": 243, "gt_offset": 248, "c_s": 2.0},
    {"video_name": "031_3", "pred_peak": 56, "pred_onset": 50, "pred_offset": 62, "gt_onset": 47, "gt_offset": 55, "c_s": 2.0},
    {"video_name": "032_3", "pred_peak": 69, "pred_onset": 63, "pred_offset": 75, "gt_onset": 60, "gt_offset": 66, "c_s": 2.0},
    {"video_name": "032_6", "pred_peak": 42, "pred_onset": 36, "pred_offset": 48, "gt_onset": 43, "gt_offset": 52, "c_s": 2.0},
    {"video_name": "037_4", "pred_peak": 223, "pred_onset": 218, "pred_offset": 228, "gt_onset": 212, "gt_offset": 225, "c_s": 1.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether local probability centroids correct peak drift."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        help="Score directory, .npz, .json, ME-TST+ .pkl, or one-video .npy.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="Optional CSV/JSON case list. Defaults to the embedded 11 cases.",
    )
    parser.add_argument("--tau-valley", type=float, default=0.5)
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument(
        "--frame-base",
        type=int,
        choices=(0, 1),
        default=0,
        help="Index base used by case frame numbers (NumPy arrays remain 0-based).",
    )
    parser.add_argument(
        "--go-threshold",
        type=int,
        default=5,
        help="Minimum off-target cases moved into GT for GO.",
    )
    parser.add_argument(
        "--include-peak-in-gt",
        action="store_true",
        help="Also include control cases whose original peak is already in GT.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in deterministic tests; --scores is not required.",
    )
    return parser.parse_args()


def normalized_case(raw: dict[str, Any], row_number: int) -> dict[str, Any]:
    aliases = {
        "video_name": ("video_name", "video"),
        "pred_peak": ("pred_peak",),
        "pred_onset": ("pred_onset",),
        "pred_offset": ("pred_offset",),
        "gt_onset": ("gt_onset", "nearest_gt_onset"),
        "gt_offset": ("gt_offset", "nearest_gt_offset"),
    }
    case: dict[str, Any] = {}
    for target, choices in aliases.items():
        value = next((raw[name] for name in choices if name in raw and raw[name] != ""), None)
        if value is None:
            raise ValueError(f"case row {row_number}: missing {target!r}")
        case[target] = str(value) if target == "video_name" else int(value)
    if "score_path" in raw and raw["score_path"]:
        case["score_path"] = str(raw["score_path"])
    if "score_key" in raw and raw["score_key"]:
        case["score_key"] = str(raw["score_key"])
    if "c_s" in raw and raw["c_s"] not in (None, ""):
        case["c_s"] = float(raw["c_s"])
    elif raw.get("tuned_native_config"):
        config = raw["tuned_native_config"]
        config = json.loads(config) if isinstance(config, str) else config
        case["c_s"] = float(config["c_s"])
    return case


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(case) for case in EMBEDDED_CASES]
    if not path.is_file():
        raise FileNotFoundError(f"case list not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            raw_cases: Any = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            raw_cases = json.load(handle)
        if isinstance(raw_cases, dict):
            raw_cases = raw_cases.get("cases", raw_cases.get("samples"))
    else:
        raise ValueError("--cases must be a .csv or .json file")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case list must be a non-empty JSON list/CSV table")
    return [normalized_case(raw, index) for index, raw in enumerate(raw_cases, 1)]


def validate_scores(scores: Any, label: str) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64).squeeze()
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label}: expected a non-empty 1D score array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: scores contain NaN or infinity")
    if np.any(array < 0):
        raise ValueError(f"{label}: scores must be non-negative for fractional gamma")
    return array


class ScoreSource:
    def __init__(self, path: Path, unique_video_count: int):
        if not path.exists():
            raise FileNotFoundError(f"score source not found: {path}")
        self.path = path
        self.unique_video_count = unique_video_count
        self.archive: Any = None
        self.me_tst_scores: dict[str, np.ndarray] | None = None
        self.k_p: int | None = None
        if path.is_file() and path.suffix.lower() == ".npz":
            self.archive = np.load(path, allow_pickle=False)
        elif path.is_file() and path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as handle:
                self.archive = json.load(handle)
        elif path.is_file() and path.suffix.lower() in (".pkl", ".pickle"):
            # Pickle can execute code: only use a cache produced by a trusted run.
            with path.open("rb") as handle:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    payload = pickle.load(handle)
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                raise ValueError(f"{path}: not a recognized ME-TST+ score cache")
            self.k_p = int(payload["k_p"])
            self.me_tst_scores = {
                str(record["video"]): validate_scores(record["score"], str(record["video"]))
                for record in payload["records"]
            }

    def close(self) -> None:
        if isinstance(self.archive, np.lib.npyio.NpzFile):
            self.archive.close()

    def get(self, case: dict[str, Any]) -> np.ndarray:
        video = case["video_name"]
        key = case.get("score_key", video)
        if self.me_tst_scores is not None:
            if video not in self.me_tst_scores:
                raise KeyError(f"{self.path}: no ME-TST+ record for video {video!r}")
            if "c_s" not in case:
                raise ValueError(f"{video}: c_s is required to reconstruct the tuned curve")
            width = max(1, int(round(float(case["c_s"]) * int(self.k_p))))
            kernel = np.ones(width, dtype=np.float64) / width
            return np.convolve(self.me_tst_scores[video], kernel, mode="same")
        if isinstance(self.archive, np.lib.npyio.NpzFile):
            if key not in self.archive.files:
                raise KeyError(f"{self.path}: no NPZ key {key!r}; available={self.archive.files}")
            return validate_scores(self.archive[key], f"{video}/{key}")
        if isinstance(self.archive, dict):
            if key not in self.archive:
                raise KeyError(f"{self.path}: no JSON score key {key!r}")
            return validate_scores(self.archive[key], f"{video}/{key}")
        if self.path.is_file():
            if self.path.suffix.lower() != ".npy":
                raise ValueError("--scores file must be .npy, .npz, .json, or .pkl")
            if self.unique_video_count != 1:
                raise ValueError("a single .npy can only be used when all cases share one video")
            return validate_scores(np.load(self.path, allow_pickle=False), video)

        candidates = []
        if case.get("score_path"):
            candidates.append(self.path / case["score_path"])
        candidates.extend(
            [
                self.path / f"{video}.npy",
                self.path / video / "scores.npy",
                self.path / f"{video.replace('/', '_')}.npy",
            ]
        )
        score_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if score_path is None:
            tried = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"no score array for {video!r}; tried: {tried}")
        return validate_scores(np.load(score_path, allow_pickle=False), str(score_path))


def is_local_minimum(scores: np.ndarray, index: int) -> bool:
    if index <= 0 or index >= len(scores) - 1:
        return False
    return (
        scores[index] <= scores[index - 1]
        and scores[index] <= scores[index + 1]
        and (scores[index] < scores[index - 1] or scores[index] < scores[index + 1])
    )


def support_island(scores: np.ndarray, peak: int, tau_valley: float) -> tuple[int, int]:
    """Return inclusive bounds; the threshold/minimum stopping frame is included."""
    threshold = float(scores[peak]) * tau_valley
    start = peak
    for index in range(peak - 1, -1, -1):
        start = index
        if scores[index] <= threshold or is_local_minimum(scores, index):
            break
    end = peak
    for index in range(peak + 1, len(scores)):
        end = index
        if scores[index] <= threshold or is_local_minimum(scores, index):
            break
    return start, end


def probability_centroid(
    scores: np.ndarray, start: int, end: int, gamma: float
) -> int:
    positions = np.arange(start, end + 1, dtype=np.float64)
    weights = np.power(scores[start : end + 1], gamma)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError(f"support [{start}, {end}] has zero total weight")
    centroid = float(np.dot(positions, weights) / weight_sum)
    # Explicit half-up rounding is stable and natural for non-negative frame IDs.
    return int(math.floor(centroid + 0.5))


def analyze(
    cases: list[dict[str, Any]],
    source: Any,
    tau_valley: float,
    gammas: list[float],
    frame_base: int,
    include_peak_in_gt: bool,
) -> list[dict[str, Any]]:
    results = []
    score_cache: dict[str, np.ndarray] = {}
    for case in cases:
        video = case["video_name"]
        if video not in score_cache:
            score_cache[video] = source.get(case)
        scores = score_cache[video]
        peak = case["pred_peak"] - frame_base
        gt_onset = case["gt_onset"] - frame_base
        gt_offset = case["gt_offset"] - frame_base
        if not 0 <= peak < len(scores):
            raise IndexError(
                f"{video}: pred_peak={case['pred_peak']} maps to {peak}, "
                f"outside score array [0, {len(scores) - 1}]"
            )
        if not 0 <= gt_onset <= gt_offset < len(scores):
            raise IndexError(
                f"{video}: GT [{case['gt_onset']}, {case['gt_offset']}] maps "
                f"outside score array [0, {len(scores) - 1}]"
            )
        peak_in_gt = gt_onset <= peak <= gt_offset
        if peak_in_gt and not include_peak_in_gt:
            continue
        start, end = support_island(scores, peak, tau_valley)
        gt_center = (gt_onset + gt_offset) / 2.0
        peak_distance = abs(peak - gt_center)
        for gamma in gammas:
            p_star = probability_centroid(scores, start, end, gamma)
            centroid_distance = abs(p_star - gt_center)
            results.append(
                {
                    "video": video,
                    "gamma": gamma,
                    "peak": peak + frame_base,
                    "gt": f"[{gt_onset + frame_base}, {gt_offset + frame_base}]",
                    "support": f"[{start + frame_base}, {end + frame_base}]",
                    "p_star": p_star + frame_base,
                    "moved_in": gt_onset <= p_star <= gt_offset,
                    "peak_dist": peak_distance,
                    "centroid_dist": centroid_distance,
                    "reduction": peak_distance - centroid_distance,
                }
            )
    if not results:
        raise ValueError("no cases remain after filtering")
    return results


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines)


def print_report(results: list[dict[str, Any]], go_threshold: int) -> bool:
    detail_rows = []
    for row in results:
        detail_rows.append(
            [
                row["video"],
                f"{row['gamma']:g}",
                row["peak"],
                row["gt"],
                row["support"],
                row["p_star"],
                "YES" if row["moved_in"] else "NO",
                f"{row['peak_dist']:.2f}",
                f"{row['centroid_dist']:.2f}",
                f"{row['reduction']:+.2f}",
            ]
        )
    print("\n## Per-case centroid diagnostic\n")
    print(
        markdown_table(
            ["video", "gamma", "peak", "GT", "support", "p*", "moved in", "peak |d|", "p* |d|", "reduction"],
            detail_rows,
        )
    )

    summaries = []
    gamma_values = sorted({row["gamma"] for row in results})
    for gamma in gamma_values:
        subset = [row for row in results if row["gamma"] == gamma]
        moved = sum(row["moved_in"] for row in subset)
        shortened = sum(row["reduction"] > 0 for row in subset)
        positive_reductions = [row["reduction"] for row in subset if row["reduction"] > 0]
        summaries.append(
            {
                "gamma": gamma,
                "count": len(subset),
                "moved": moved,
                "shortened": shortened,
                "shortened_ratio": shortened / len(subset),
                "mean_peak_distance": float(np.mean([row["peak_dist"] for row in subset])),
                "mean_centroid_distance": float(np.mean([row["centroid_dist"] for row in subset])),
                "mean_reduction": float(np.mean(positive_reductions)) if positive_reductions else 0.0,
            }
        )
    print("\n## Aggregate diagnostic\n")
    print(
        markdown_table(
            ["gamma", "cases", "moved into GT", "distance shortened", "shortened ratio", "mean peak |d|", "mean p* |d|", "mean shrink (improved)"],
            [
                [
                    f"{row['gamma']:g}",
                    row["count"],
                    f"{row['moved']}/{row['count']}",
                    f"{row['shortened']}/{row['count']}",
                    f"{100 * row['shortened_ratio']:.1f}%",
                    f"{row['mean_peak_distance']:.2f}",
                    f"{row['mean_centroid_distance']:.2f}",
                    f"{row['mean_reduction']:+.2f}",
                ]
                for row in summaries
            ],
        )
    )
    best = max(summaries, key=lambda row: (row["moved"], -row["gamma"]))
    go = best["moved"] >= go_threshold
    print(
        f"\nBest gamma={best['gamma']:g}: {best['moved']}/{best['count']} "
        f"cases moved into GT (GO threshold: {go_threshold})."
    )
    print(GO_MESSAGE if go else NO_GO_MESSAGE)
    return go


def run_self_test() -> None:
    symmetric = np.array([0.0, 0.4, 0.8, 1.0, 0.8, 0.4, 0.0])
    assert support_island(symmetric, 3, 0.5) == (1, 5)
    assert probability_centroid(symmetric, 1, 5, 1.0) == 3

    asymmetric = np.array([0.0, 0.4, 0.8, 1.0, 0.9, 0.8, 0.7, 0.4, 0.0])
    start, end = support_island(asymmetric, 3, 0.5)
    p_star = probability_centroid(asymmetric, start, end, 1.0)
    assert (start, end) == (1, 7)
    assert p_star > 3

    valley = np.array([0.9, 1.0, 0.8, 0.7, 0.75, 0.9])
    assert support_island(valley, 1, 0.5)[1] == 3

    class MemorySource:
        def get(self, case: dict[str, Any]) -> np.ndarray:
            return asymmetric

    cases = [
        {"video_name": "shifted", "pred_peak": 3, "pred_onset": 2, "pred_offset": 5, "gt_onset": 4, "gt_offset": 6},
        {"video_name": "control", "pred_peak": 3, "pred_onset": 2, "pred_offset": 5, "gt_onset": 2, "gt_offset": 4},
    ]
    results = analyze(cases, MemorySource(), 0.5, [1.0, 2.0, 3.0], 0, False)
    assert len(results) == 3
    assert {row["video"] for row in results} == {"shifted"}
    print("SELF-TEST PASSED: threshold, asymmetry, centroid, and local-valley checks")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.scores is None:
        raise ValueError("--scores is required unless --self-test is used")
    if not 0 < args.tau_valley <= 1:
        raise ValueError("--tau-valley must be in (0, 1]")
    if not args.gammas or any(gamma <= 0 for gamma in args.gammas):
        raise ValueError("all --gammas values must be positive")
    if args.go_threshold < 1:
        raise ValueError("--go-threshold must be >= 1")

    cases = load_cases(args.cases)
    unique_video_count = len({case["video_name"] for case in cases})
    source = ScoreSource(args.scores, unique_video_count)
    try:
        results = analyze(
            cases,
            source,
            args.tau_valley,
            list(dict.fromkeys(args.gammas)),
            args.frame_base,
            args.include_peak_in_gt,
        )
    finally:
        source.close()
    print(
        f"Cases loaded: {len(cases)}; analyzed: "
        f"{len(results) // len(set(args.gammas))}; tau_valley={args.tau_valley:g}; "
        f"frame_base={args.frame_base}"
    )
    print_report(results, args.go_threshold)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, IndexError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
