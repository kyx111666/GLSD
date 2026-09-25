"""Final one-shot expanded-setting search for the locked GLSD-v1 method.

Only the candidate values of (a0, rho, tau) are expanded.  The implementation,
evaluator, fold construction, interval geometry, and tie-break are imported from
the exact historical GLSD-90 reproduction and are not modified here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences


PROJECT = Path(__file__).resolve().parents[3]
REPRO = PROJECT / "historical_gl_exact_fresh_reproduction"
ENGINE_ROOT = REPRO / "fresh_run" / "boostingvrme"
ARCHIVE = REPRO / "source_archive" / "a"
OUTPUT = PROJECT / "RethinkFuse_reproduction" / "results" / "glsd_v1_expanded_final"
sys.path.insert(0, str(ENGINE_ROOT))

import equiscale_fair_validation as fair  # noqa: E402
import unified_persistence as engine  # noqa: E402

# The historical ME-TST source intentionally contains a Windows path literal.
# Its archived POSIX replay resolves that literal beneath ENGINE_ROOT.
if not fair.ME_CACHE.is_absolute():
    fair.ME_CACHE = ENGINE_ROOT / fair.ME_CACHE


OLD_A0 = (1.0, 1.5, 2.0)
OLD_RHO = (1.0, 2.0, 3.0)
OLD_TAU = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75)
EXPANDED_A0 = (0.75, 1.0, 1.5, 2.0, 2.25)
EXPANDED_RHO = (0.5, 1.0, 2.0, 3.0, 3.5)
EXPANDED_TAU = (
    0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.7, 0.75,
)
SETTINGS = (
    ("metst", "sammlv"),
    ("metst", "casme3"),
    ("boostingvrme", "sammlv"),
    ("boostingvrme", "casme3"),
)
DISPLAY = {
    "metst": "ME-TST+",
    "boostingvrme": "BoostingVRME",
    "sammlv": "SAMMLV",
    "casme3": "CAS(ME)3",
}
EXPECTED = {
    ("metst", "sammlv"): {
        "native": (53, 184, 106), "old": (48, 126, 111),
        "native_f1": 0.2676767676767677, "old_f1": 0.2882882882882883,
    },
    ("metst", "casme3"): {
        "native": (81, 912, 777), "old": (96, 1085, 762),
        "native_f1": 0.08752025931928688, "old_f1": 0.09416380578715057,
    },
    ("boostingvrme", "sammlv"): {
        "native": (49, 145, 110), "old": (42, 95, 117),
        "native_f1": 0.2776203966005666, "old_f1": 0.28378378378378377,
    },
    ("boostingvrme", "casme3"): {
        "native": (93, 818, 765), "old": (120, 1148, 738),
        "native_f1": 0.10514414923685698, "old_f1": 0.11288805268109126,
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_digest(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def historical_signature(value) -> str:
    """Match the historical runner's default json.dumps signature encoding."""
    payload = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def mode_for(backbone: str) -> str:
    return "native" if backbone == "boostingvrme" else "fixed"


def historical_root(backbone: str, dataset: str) -> Path:
    return (
        ARCHIVE / backbone / "results" / "pure_persistence_matched_v1"
        / dataset / mode_for(backbone)
    )


def configs_for(a0_values, rho_values, tau_values):
    return [
        engine.Config("unified", a0, 0.0, tau, rho)
        for a0 in a0_values
        for rho in rho_values
        for tau in tau_values
    ]


class ExpandedCurveFeatures:
    """Exact GLSD-v1 features with an expanded reference-scale domain.

    The three support scales remain locked.  Additional a0 values create only
    reference candidates and G values; they do not add votes to the median L.
    """

    def __init__(self, curve, k):
        self.curve = np.asarray(curve, dtype=float)
        if self.curve.ndim != 1 or len(self.curve) < 3:
            raise ValueError("Expected a one-dimensional curve with >=3 samples")
        if not np.isfinite(self.curve).all() or k < 1:
            raise ValueError("Invalid curve or k")
        self.k = int(k)
        self.references = {}
        self.local_cache = {}
        self.evidence_cache = {}
        physical = {}
        for scale in tuple(engine.SCALES) + tuple(EXPANDED_A0):
            width = min(len(self.curve), max(1, round(scale * self.k)))
            if width not in physical:
                smooth = np.convolve(
                    self.curve, np.ones(width, dtype=float) / width, mode="same"
                )
                peaks = (
                    find_peaks(smooth, distance=self.k)[0]
                    if np.ptp(self.curve) > 0 else np.empty(0, dtype=int)
                )
                spread = max(float(np.ptp(smooth)), 1e-12)
                mean = float(smooth.mean())
                height = np.maximum(
                    0.0,
                    (smooth[peaks] - mean)
                    / max(float(smooth.max() - mean), 1e-12),
                )
                global_values = peak_prominences(smooth, peaks)[0] / spread
                physical[width] = (peaks, height, global_values, smooth, spread)
            self.references[scale] = physical[width]
        self.support_scales = {}
        for scale in engine.SCALES:
            width = min(len(self.curve), max(1, round(scale * self.k)))
            self.support_scales[width] = physical[width]

    def evidence(self, reference, radius=1.0):
        key = (reference, radius)
        if key in self.evidence_cache:
            return self.evidence_cache[key]
        peaks, height, global_values, _, _ = self.references[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))
        for width, (other_peaks, _, _, smooth, spread) in self.support_scales.items():
            if (width, window) not in self.local_cache:
                values = []
                for point in other_peaks:
                    left = float(np.min(smooth[max(0, point - window):point + 1]))
                    right = float(
                        np.min(smooth[point:min(len(smooth), point + window + 1)])
                    )
                    values.append(max(0.0, float(smooth[point]) - max(left, right)) / spread)
                self.local_cache[width, window] = np.asarray(values)
        local = []
        for point in peaks:
            aligned = []
            for width, (other_peaks, _, _, _, _) in self.support_scales.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    aligned.append(0.0)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (
                        abs(int(other_peaks[index]) - point), -values[index]
                    ),
                )
                aligned.append(float(values[chosen]))
            local.append(float(np.median(aligned)))
        output = peaks, np.column_stack((height, global_values, local))
        self.evidence_cache[key] = output
        return output


def exact_old_replay(backbone: str, dataset: str) -> dict:
    records, subjects, _, cache_path = fair.load_data(backbone, dataset)
    outer_k, _ = fair.fold_priors(records, subjects, backbone)
    source_protocol = read_json(
        ARCHIVE / backbone / "results" / "unified_persistence_final"
        / dataset / "PROTOCOL.json"
    )
    signed = {key: value for key, value in source_protocol.items() if key != "signature"}
    if historical_signature(signed) != source_protocol["signature"]:
        raise AssertionError("Historical protocol signature mismatch")
    if digest(cache_path) != source_protocol["input_sha256"]:
        raise AssertionError("Frozen cache checksum mismatch")
    for name, expected_hash in source_protocol["source_sha256"].items():
        if digest(ENGINE_ROOT / name) != expected_hash:
            raise AssertionError(f"Locked engine source mismatch: {name}")
    protocol_configs = [engine.Config(**row) for row in source_protocol["protocol"]["configurations"]]
    if protocol_configs != engine.configuration_grid():
        raise AssertionError("GLSD-90 grid does not match locked engine")

    locked_root = historical_root(backbone, dataset)
    selections = {
        row["subject"]: protocol_configs[int(row["config_id"])]
        for row in read_csv(locked_root / "outer_loso_selections.csv")
        if row["family"] == "pure"
    }
    locked_predictions = read_json(locked_root / "selected_predictions.json")
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    counts = np.zeros((len(subjects), 3), dtype=np.int64)
    if len(records) != len(locked_predictions):
        raise AssertionError("Historical prediction video count mismatch")
    for record, locked in zip(records, locked_predictions):
        if (record["subject"], record["video"], record["gt"]) != (
            locked["subject"], locked["video"], locked["gt"]
        ):
            raise AssertionError("Historical prediction order/provenance mismatch")
        sid = subject_index[record["subject"]]
        config = selections[record["subject"]]
        features = engine.CurveFeatures(record["curve"], int(outer_k[sid]))
        expanded_features = ExpandedCurveFeatures(record["curve"], int(outer_k[sid]))
        for old_a0 in OLD_A0:
            for old_rho in OLD_RHO:
                original_evidence = features.evidence(old_a0, old_rho)
                expanded_evidence = expanded_features.evidence(old_a0, old_rho)
                np.testing.assert_array_equal(original_evidence[0], expanded_evidence[0])
                np.testing.assert_array_equal(original_evidence[1], expanded_evidence[1])
        prepared = engine.prepare(
            record, features, config.reference, backbone, mode_for(backbone), config.radius
        )
        values, details = prepared.evaluate([config], details=True)
        expanded_prepared = engine.prepare(
            record, expanded_features, config.reference, backbone,
            mode_for(backbone), config.radius,
        )
        expanded_values, expanded_details = expanded_prepared.evaluate([config], details=True)
        np.testing.assert_array_equal(values, expanded_values)
        if details != expanded_details:
            raise AssertionError("Expanded implementation changed an old configuration")
        if details[0] != locked["predictions"]["pure"]:
            raise AssertionError(
                f"Old prediction replay mismatch: {backbone}/{dataset}/{record['video']}"
            )
        counts[sid] += values[0]
    total = tuple(int(value) for value in counts.sum(axis=0))
    report = read_json(locked_root / "report.json")
    locked_metric = report["metrics"]["pure"]
    expected = EXPECTED[backbone, dataset]
    if total != expected["old"] or total != tuple(
        int(locked_metric[key]) for key in ("TP", "FP", "FN")
    ):
        raise AssertionError("Old GLSD-90 TP/FP/FN replay mismatch")
    replay_f1 = fair.metrics(total)["F1"]
    if abs(replay_f1 - expected["old_f1"]) > 1e-12:
        raise AssertionError("Old GLSD-90 F1 replay mismatch")
    native_metric = report["metrics"]["original_native"]
    if tuple(int(native_metric[key]) for key in ("TP", "FP", "FN")) != expected["native"]:
        raise AssertionError("Locked Native counts mismatch")
    return {
        "backbone": backbone,
        "dataset": dataset,
        "cache_path": str(cache_path),
        "cache_sha256": digest(cache_path),
        "protocol_signature": source_protocol["signature"],
        "subjects": len(subjects),
        "videos": len(records),
        "TP": total[0], "FP": total[1], "FN": total[2], "F1": replay_f1,
        "selected_config_exact": True,
        "all_predictions_exact": True,
        "status": "PASS",
    }


def effective_mapping() -> tuple[list[dict], list[dict]]:
    contexts = []
    for backbone, dataset in SETTINGS:
        records, _, _, _ = fair.load_data(backbone, dataset)
        outer_k, inner_k = fair.fold_priors(
            records, sorted({r["subject"] for r in records}, key=lambda x: (int(x), x)), backbone
        )
        for k in sorted(set(outer_k.tolist()) | set(inner_k[inner_k > 0].tolist())):
            for length in sorted({len(r["curve"]) for r in records}):
                contexts.append((backbone, dataset, int(k), int(length)))
    groups = defaultdict(list)
    configs = configs_for(EXPANDED_A0, EXPANDED_RHO, EXPANDED_TAU)
    for config in configs:
        signature = (
            config.threshold,
            tuple(
                (
                    backbone, dataset, k,
                    min(length, max(1, round(config.reference * k))),
                    max(1, round(config.radius * k)),
                )
                for backbone, dataset, k, length in contexts
            ),
        )
        groups[signature].append(config)
    unique = [values[0] for values in groups.values()]
    duplicates = [
        {
            "effective_config": asdict(values[0]),
            "nominal_configs": [asdict(value) for value in values],
        }
        for values in groups.values() if len(values) > 1
    ]
    return [asdict(config) for config in unique], duplicates


def freeze_manifest(gate_rows: list[dict]) -> dict:
    unique, duplicates = effective_mapping()
    old_configs = configs_for(OLD_A0, OLD_RHO, OLD_TAU)
    expanded_configs = configs_for(EXPANDED_A0, EXPANDED_RHO, EXPANDED_TAU)
    cache_hashes = {
        f"{row['backbone']}/{row['dataset']}": {
            "path": row["cache_path"], "sha256": row["cache_sha256"]
        }
        for row in gate_rows
    }
    core = {
        "old_grid": {"a0": OLD_A0, "rho": OLD_RHO, "tau": OLD_TAU},
        "expanded_grid": {
            "a0": EXPANDED_A0, "rho": EXPANDED_RHO, "tau": EXPANDED_TAU
        },
        "old_config_count": len(old_configs),
        "nominal_config_count": len(expanded_configs),
        "effective_unique_config_count": len(unique),
        "effective_unique_configs": unique,
        "duplicate_mapping": duplicates,
        "support_scales_locked": list(engine.SCALES),
        "new_a0_semantics": (
            "Additional a0 values extend only the reference smoothing/candidate domain; "
            "the locked three-scale median-L support set is unchanged."
        ),
        "runner_sha256": digest(Path(__file__)),
        "locked_engine_sha256": digest(ENGINE_ROOT / "unified_persistence.py"),
        "git_commit": None,
        "cache_hashes": cache_hashes,
        "evaluator_identity": (
            "locked equiscale_fair_validation.Prepared: inclusive IoU>=0.5, "
            "chronological greedy matching and locked backbone interval/NMS"
        ),
        "protocol_identity": (
            "outer LOSO + inner LOSO; pooled inner TP/FP/FN Raw F1; "
            "tie-break F1, precision, fewer FP, grid order"
        ),
        "grid_frozen_before_outer_evaluation": True,
    }
    path = OUTPUT / "expanded_glsd_search_grid.json"
    if path.exists():
        existing = read_json(path)
        existing_core = {
            key: value for key, value in existing.items()
            if key not in ("creation_time", "manifest_sha256")
        }
        if existing_core != json.loads(json.dumps(core)):
            raise RuntimeError("Existing frozen expanded-grid manifest differs")
        if json_digest(existing_core | {"creation_time": existing["creation_time"]}) != existing["manifest_sha256"]:
            raise RuntimeError("Existing expanded-grid manifest signature mismatch")
        return existing
    manifest = {
        **core,
        "creation_time": datetime.now(timezone.utc).isoformat(),
    }
    manifest["manifest_sha256"] = json_digest(manifest)
    write_json(path, manifest)
    return manifest


def evaluate_counts(records, subjects, backbone, k, needed, configs):
    stats = np.zeros((len(configs), len(subjects), 3), dtype=np.int32)
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    batches = defaultdict(list)
    for index, config in enumerate(configs):
        batches[(config.reference, config.radius)].append(index)
    completed = 0
    eligible = sum(needed[subject_ids[record["subject"]]] for record in records)
    for record in records:
        sid = subject_ids[record["subject"]]
        if not needed[sid]:
            continue
        features = ExpandedCurveFeatures(record["curve"], k)
        for (reference, radius), indexes in batches.items():
            prepared = engine.prepare(
                record, features, reference, backbone, mode_for(backbone), radius
            )
            stats[indexes, sid] += prepared.evaluate([configs[i] for i in indexes])
        completed += 1
        if completed % 100 == 0 or completed == eligible:
            print(
                f"[{backbone}/{dataset_for_log(records)}] k={k}: "
                f"{completed}/{eligible} videos",
                flush=True,
            )
    return stats


def dataset_for_log(records) -> str:
    return "sammlv" if len(records) == 79 else "casme3"


def load_or_compute_stats(backbone, dataset, records, subjects, outer_k, inner_k, configs, signature):
    stats = {}
    cache_dir = OUTPUT / "expanded_count_cache" / backbone / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    for k in sorted(set(outer_k.tolist()) | set(inner_k[inner_k > 0].tolist())):
        needed = fair.required_subjects(k, outer_k, inner_k)
        path = cache_dir / f"stats_k{k}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["signature"]) != signature:
                    raise AssertionError("Expanded count-cache signature mismatch")
                if not np.array_equal(cached["needed"], needed):
                    raise AssertionError("Expanded count-cache subject mask mismatch")
                stats[k] = cached["stats"]
            print(f"[{backbone}/{dataset}] reused k={k} expanded counts", flush=True)
        else:
            print(f"[{backbone}/{dataset}] computing k={k}, configs={len(configs)}", flush=True)
            stats[k] = evaluate_counts(
                records, subjects, backbone, int(k), needed, configs
            )
            np.savez_compressed(
                path, stats=stats[k], needed=needed, signature=np.array(signature)
            )
    return stats


def run_setting(backbone: str, dataset: str, manifest: dict) -> dict:
    records, subjects, legacy_k, _ = fair.load_data(backbone, dataset)
    outer_k, inner_k = fair.fold_priors(records, subjects, backbone)
    configs = [engine.Config(**row) for row in manifest["effective_unique_configs"]]
    stats = load_or_compute_stats(
        backbone, dataset, records, subjects, outer_k, inner_k, configs,
        manifest["manifest_sha256"],
    )
    choices = []
    selection_rows = []
    subject_counts = np.zeros((len(subjects), 3), dtype=np.int64)
    all_indexes = np.arange(len(configs), dtype=int)
    for held, subject in enumerate(subjects):
        inner = engine.inner_counts(stats, held, inner_k)
        selected = engine.choose(inner, all_indexes)
        choices.append(selected)
        test = stats[int(outer_k[held])][selected, held]
        subject_counts[held] = test
        config = configs[selected]
        inner_metric = fair.metrics(inner[selected])
        selection_rows.append({
            "backbone": DISPLAY[backbone], "dataset": DISPLAY[dataset],
            "outer_subject": subject,
            "selected_a0": config.reference,
            "selected_rho": config.radius,
            "selected_tau": config.threshold,
            "inner_tp": inner_metric["TP"], "inner_fp": inner_metric["FP"],
            "inner_fn": inner_metric["FN"], "inner_f1": inner_metric["F1"],
            "outer_tp": int(test[0]), "outer_fp": int(test[1]),
            "outer_fn": int(test[2]),
        })

    sid_lookup = {subject: index for index, subject in enumerate(subjects)}
    replay = np.zeros_like(subject_counts)
    prediction_rows = []
    for record in records:
        sid = sid_lookup[record["subject"]]
        config = configs[choices[sid]]
        features = ExpandedCurveFeatures(record["curve"], int(outer_k[sid]))
        prepared = engine.prepare(
            record, features, config.reference, backbone, mode_for(backbone), config.radius
        )
        values, details = prepared.evaluate([config], details=True)
        replay[sid] += values[0]
        prediction_rows.append({
            "subject": record["subject"], "video": record["video"],
            "gt": record["gt"], "selected_theta": {
                "a0": config.reference, "rho": config.radius, "tau": config.threshold
            },
            "predictions": details[0],
        })
    if not np.array_equal(replay, subject_counts):
        raise AssertionError(f"Expanded prediction replay failed: {backbone}/{dataset}")
    prediction_path = OUTPUT / "expanded_glsd_predictions" / f"{backbone}_{dataset}.json"
    write_json(prediction_path, prediction_rows)

    old_root = historical_root(backbone, dataset)
    old_report = read_json(old_root / "report.json")
    native_metric = old_report["metrics"]["original_native"]
    old_metric = old_report["metrics"]["pure"]
    native_subjects = subject_family_counts(old_root / "subject_counts.csv", "original_native", subjects)
    tuned_path = (
        PROJECT / "RethinkFuse_reproduction" / "results" / "fair_tuned_native_v2"
        / backbone / dataset / "subject_counts.csv"
    )
    tuned_subjects = subject_family_counts(tuned_path, "single_tuned", subjects) if tuned_path.exists() else None
    bootstrap = [
        {
            "backbone": DISPLAY[backbone], "dataset": DISPLAY[dataset],
            "comparison": "Expanded GLSD - Native",
            **paired_bootstrap(subject_counts, native_subjects),
        }
    ]
    if tuned_subjects is not None:
        bootstrap.append({
            "backbone": DISPLAY[backbone], "dataset": DISPLAY[dataset],
            "comparison": "Expanded GLSD - Fair Tuned Native",
            **paired_bootstrap(subject_counts, tuned_subjects),
        })
    expanded_metric = fair.metrics(subject_counts.sum(axis=0))
    return {
        "selection_rows": selection_rows,
        "subject_counts": subject_counts,
        "subjects": subjects,
        "native_metric": native_metric,
        "old_metric": old_metric,
        "expanded_metric": expanded_metric,
        "bootstrap": bootstrap,
        "prediction_path": str(prediction_path),
        "legacy_k": legacy_k,
        "outer_k": outer_k,
    }


def subject_family_counts(path: Path, family: str, subjects: list[str]) -> np.ndarray:
    rows = [row for row in read_csv(path) if row["family"] == family]
    by_subject = {
        row["subject"]: np.array(
            [int(float(row[key])) for key in ("TP", "FP", "FN")], dtype=np.int64
        )
        for row in rows
    }
    if set(by_subject) != set(subjects):
        raise AssertionError(f"Subject-count mismatch in {path}")
    return np.stack([by_subject[subject] for subject in subjects])


def paired_bootstrap(first, second, repeats=10_000, seed=100):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(first), size=(repeats, len(first)))

    def f1(values):
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0], denominator,
            out=np.zeros(repeats, dtype=float), where=denominator > 0,
        )

    deltas = f1(first) - f1(second)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "point_delta_f1": fair.metrics(first.sum(axis=0))["F1"]
        - fair.metrics(second.sum(axis=0))["F1"],
        "bootstrap_mean_delta_f1": float(np.mean(deltas)),
        "ci95_low": float(low), "ci95_high": float(high),
        "p_delta_gt_0": float(np.mean(deltas > 0)),
        "repetitions": repeats, "seed": seed,
    }


def frequency_rows(selection_rows: list[dict]) -> list[dict]:
    output = []
    groups = defaultdict(list)
    for row in selection_rows:
        groups[(row["backbone"], row["dataset"])].append(row)
    for (backbone, dataset), rows in groups.items():
        for parameter, key in (
            ("a0", "selected_a0"), ("rho", "selected_rho"), ("tau", "selected_tau")
        ):
            for value, count in Counter(row[key] for row in rows).most_common():
                output.append({
                    "backbone": backbone, "dataset": dataset,
                    "parameter": parameter, "value": value, "count": count,
                })
        tuples = Counter(
            (row["selected_a0"], row["selected_rho"], row["selected_tau"])
            for row in rows
        )
        for value, count in tuples.most_common():
            output.append({
                "backbone": backbone, "dataset": dataset,
                "parameter": "theta", "value": json.dumps(value), "count": count,
            })
    return output


def make_report(manifest, main_rows, selections, bootstraps, gate_rows) -> str:
    improved = any(float(row["delta_vs_old_glsd90"]) > 0 for row in main_rows)
    decision = "ADOPT_EXPANDED" if improved else "KEEP_GLSD90"
    lines = [
        "# GLSD-v1 Expanded-Setting Search 最终报告", "",
        "## A. Protocol Gate", "",
        "- 方法公式未修改：是。runner 直接复用锁定 evaluator/decoder；G、L、median 与 S=(G+L)/2 均保持不变。",
        "- frozen caches 未修改：是；搜索前后均按 SHA-256 校验。",
        "- evaluator、outer/inner LOSO、seed 与 tie-break 未修改：是。",
        "- old GLSD-90 exact replay：四组逐视频 predictions、逐折选择及 TP/FP/FN 全部通过。",
        "- expanded grid：在任何 outer evaluation 前写入并签名冻结。",
        "- outer-test：仅用于所选 theta 的最终评价，未参与选择。", "",
        "**PROTOCOL_VALIDATED**", "",
        "## B. Search Space", "",
        f"- old a0: `{list(OLD_A0)}`", f"- expanded a0: `{list(EXPANDED_A0)}`",
        f"- old rho: `{list(OLD_RHO)}`", f"- expanded rho: `{list(EXPANDED_RHO)}`",
        f"- old tau: `{list(OLD_TAU)}`", f"- expanded tau: `{list(EXPANDED_TAU)}`",
        f"- old configurations: {manifest['old_config_count']}",
        f"- expanded nominal configurations: {manifest['nominal_config_count']}",
        f"- expanded effective unique configurations: {manifest['effective_unique_config_count']}",
        f"- duplicate effective mappings: {len(manifest['duplicate_mapping'])}", "",
        "`a0` 是候选参考曲线的平滑尺度乘子；`rho` 是局部双侧窗口的 k 倍数，实际窗口为 `max(1, round(rho*k))`；`tau` 是最终 `(G+L)/2` 阈值。固定的三尺度 L 证据集合没有扩展。", "",
        "## C. Main Results", "",
        "| Backbone | Dataset | Native | GLSD-90 | Expanded GLSD | Δ vs Native | Δ vs GLSD-90 |",
        "| - | - | -: | -: | -: | -: | -: |",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['backbone']} | {row['dataset']} | {float(row['native_f1']):.6f} "
            f"({row['native_tp']}/{row['native_fp']}/{row['native_fn']}) | "
            f"{float(row['old_glsd90_f1']):.6f} "
            f"({row['old_glsd90_tp']}/{row['old_glsd90_fp']}/{row['old_glsd90_fn']}) | "
            f"{float(row['expanded_f1']):.6f} "
            f"({row['expanded_tp']}/{row['expanded_fp']}/{row['expanded_fn']}) | "
            f"{float(row['delta_vs_native']):+.6f} | "
            f"{float(row['delta_vs_old_glsd90']):+.6f} |"
        )
    lines += ["", "TP/FP/FN 均按表中括号顺序给出。", "", "### Paired bootstrap", "",
              "按被试配对，10,000 次，seed=100。", "",
              "| Backbone | Dataset | Comparison | ΔF1 | 95% CI | P(Δ>0) |",
              "| - | - | - | -: | - | -: |"]
    for row in bootstraps:
        lines.append(
            f"| {row['backbone']} | {row['dataset']} | {row['comparison']} | "
            f"{row['point_delta_f1']:+.6f} | [{row['ci95_low']:+.6f}, "
            f"{row['ci95_high']:+.6f}] | {row['p_delta_gt_0']:.4f} |"
        )
    lines += ["", "## D. Fold-level Selection", ""]
    for backbone in ("ME-TST+", "BoostingVRME"):
        for dataset in ("SAMMLV", "CAS(ME)3"):
            rows = [r for r in selections if r["backbone"] == backbone and r["dataset"] == dataset]
            a = Counter(r["selected_a0"] for r in rows)
            rho = Counter(r["selected_rho"] for r in rows)
            tau = Counter(r["selected_tau"] for r in rows)
            boundary = sum(
                r["selected_a0"] in (min(EXPANDED_A0), max(EXPANDED_A0))
                or r["selected_rho"] in (min(EXPANDED_RHO), max(EXPANDED_RHO))
                or r["selected_tau"] in (min(EXPANDED_TAU), max(EXPANDED_TAU))
                for r in rows
            )
            lines.append(
                f"- {backbone}/{dataset}: a0={dict(a)}; rho={dict(rho)}; "
                f"tau={dict(tau)}；任一参数落边界 {boundary}/{len(rows)} folds。"
            )
    lines += [
        "", "边界选择仅作诊断，本任务未据此补点或启动第二轮。", "",
        "## E. Result Decision", "", f"**{decision}**", "",
        (
            "至少一个主要 setting 的 expanded Raw Spotting F1 高于 GLSD-90，且协议闸门通过。"
            if improved else
            "expanded search 未在任何主要 setting 上提高 Raw Spotting F1，因此保留 GLSD-90。"
        ),
        "", "本任务在第一轮预定义 expanded grid 完成后停止；未运行 GLSD-v2，未修改论文、LaTeX、方法图或锁定结果。",
        "", "## Reproduction Gate Records", "",
    ]
    for row in gate_rows:
        lines.append(
            f"- {DISPLAY[row['backbone']]}/{DISPLAY[row['dataset']]}: "
            f"TP/FP/FN={row['TP']}/{row['FP']}/{row['FN']}, F1={row['F1']:.12f}, PASS。"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gate_rows = []
    try:
        for backbone, dataset in SETTINGS:
            print(f"[gate] exact replay {backbone}/{dataset}", flush=True)
            gate_rows.append(exact_old_replay(backbone, dataset))
    except Exception as error:
        failure = {
            "status": "BLOCKED / REPRODUCTION_GATE_FAILED",
            "error": f"{type(error).__name__}: {error}",
            "completed": gate_rows,
        }
        write_json(OUTPUT / "REPRODUCTION_GATE_FAILED.json", failure)
        raise
    write_json(OUTPUT / "old_glsd90_exact_replay_gate.json", {
        "status": "PASS", "settings": gate_rows
    })
    print("[gate] all four old GLSD-90 settings passed", flush=True)

    manifest = freeze_manifest(gate_rows)
    print(
        f"[manifest] frozen: nominal={manifest['nominal_config_count']}, "
        f"effective={manifest['effective_unique_config_count']}", flush=True,
    )
    results = {}
    for backbone, dataset in SETTINGS:
        print(f"[expanded] start {backbone}/{dataset}", flush=True)
        results[backbone, dataset] = run_setting(backbone, dataset, manifest)
        print(
            f"[expanded] complete {backbone}/{dataset}: "
            f"F1={results[backbone, dataset]['expanded_metric']['F1']:.12f}",
            flush=True,
        )

    selections = [
        row for result in results.values() for row in result["selection_rows"]
    ]
    selection_fields = (
        "backbone", "dataset", "outer_subject", "selected_a0", "selected_rho",
        "selected_tau", "inner_tp", "inner_fp", "inner_fn", "inner_f1",
        "outer_tp", "outer_fp", "outer_fn",
    )
    write_csv(
        OUTPUT / "expanded_glsd_selected_theta_per_fold.csv",
        selections, selection_fields,
    )
    write_csv(
        OUTPUT / "expanded_glsd_selected_config_frequency.csv",
        frequency_rows(selections),
        ("backbone", "dataset", "parameter", "value", "count"),
    )

    main_rows = []
    bootstraps = []
    for (backbone, dataset), result in results.items():
        native = result["native_metric"]
        old = result["old_metric"]
        expanded = result["expanded_metric"]
        main_rows.append({
            "backbone": DISPLAY[backbone], "dataset": DISPLAY[dataset],
            "native_tp": native["TP"], "native_fp": native["FP"],
            "native_fn": native["FN"], "native_f1": native["F1"],
            "old_glsd90_tp": old["TP"], "old_glsd90_fp": old["FP"],
            "old_glsd90_fn": old["FN"], "old_glsd90_f1": old["F1"],
            "expanded_tp": expanded["TP"], "expanded_fp": expanded["FP"],
            "expanded_fn": expanded["FN"], "expanded_f1": expanded["F1"],
            "delta_vs_native": expanded["F1"] - native["F1"],
            "delta_vs_old_glsd90": expanded["F1"] - old["F1"],
        })
        bootstraps.extend(result["bootstrap"])
    write_csv(OUTPUT / "expanded_glsd_main_results.csv", main_rows)
    write_csv(OUTPUT / "expanded_glsd_bootstrap.csv", bootstraps)
    write_json(OUTPUT / "expanded_glsd_bootstrap.json", {
        "paired_unit": "subject", "repetitions": 10_000, "seed": 100,
        "comparisons": bootstraps,
    })

    current_hashes = {
        f"{row['backbone']}/{row['dataset']}": digest(Path(row["cache_path"]))
        for row in gate_rows
    }
    expected_hashes = {
        key: value["sha256"] for key, value in manifest["cache_hashes"].items()
    }
    if current_hashes != expected_hashes:
        raise AssertionError("Frozen cache changed during expanded search")
    report = make_report(manifest, main_rows, selections, bootstraps, gate_rows)
    (OUTPUT / "EXPANDED_GLSD_REPORT_CN.md").write_text(report, encoding="utf-8")
    write_json(OUTPUT / "run_complete.json", {
        "status": "PROTOCOL_VALIDATED",
        "decision": "ADOPT_EXPANDED" if any(
            row["delta_vs_old_glsd90"] > 0 for row in main_rows
        ) else "KEEP_GLSD90",
        "manifest_sha256": manifest["manifest_sha256"],
        "cache_hashes_unchanged": True,
        "runner_sha256": digest(Path(__file__)),
    })
    print("[complete] all requested artifacts written", flush=True)


if __name__ == "__main__":
    main()
