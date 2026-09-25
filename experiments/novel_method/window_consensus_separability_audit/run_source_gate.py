#!/usr/bin/env python3
"""Source gate for the pre-stitch window-consensus audit.

The scientific audit is intentionally not run when raw overlapping-window
outputs are absent. Blocked placeholders are explicit status artifacts, not
empty scientific results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/window_consensus_separability_audit"
OUTPUTS = OUT / "outputs"
SAM_CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
CAS_CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"
STITCH_SOURCE = ROOT / "senior_original/me_tst_video/me-tst-video/compare_paper_aligned_strategies.py"
PATH_AUDIT = ROOT / "my_method/PRE_STITCH_PATH_AUDIT_CN.md"
AGREEMENT_AUDIT = ROOT / "my_method/PRE_STITCH_CONTEXT_AGREEMENT_AUDIT_CN.md"
DECODER_AUDIT = ROOT / "my_method/PRE_STITCH_DECODER_EQUIVALENCE_AUDIT_CN.md"
AGREEMENT_RESULT = ROOT / "my_method/pre_stitch_context_feasibility/outputs/context_agreement_results.json"
DECODER_RESULT = ROOT / "my_method/pre_stitch_context_feasibility/outputs/decoder_equivalence_results.json"
STATUS = "BLOCKED-WINDOW-CONSENSUS-SOURCE"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_inventory(path):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    records = payload["records"]
    first = records[0]
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), "dataset": payload["dataset"],
            "subjects": int(payload["num_subjects"]), "videos": int(payload["num_videos"]),
            "GT": int(payload["num_gt"]), "k_p": int(payload["k_p"]), "record_count": len(records),
            "stored_record_fields": list(first.keys()), "score_shape_example": [len(first["score"])],
            "logits_shape_example": list(first["logits"].shape),
            "raw_window_score_present": any(key in first for key in ("raw_score", "raw_window_score", "window_score", "window_scores")),
            "window_id_present": "window_id" in first, "local_t_present": "local_t" in first,
            "global_t_mapping_present": "global_t" in first}


def write_blocked_csv(path, artifact):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "scientific_audit_executed", "artifact", "reason"])
        writer.writeheader()
        writer.writerow({"status": STATUS, "scientific_audit_executed": False, "artifact": artifact,
                         "reason": "raw pre-stitch overlapping-window outputs unavailable"})


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    required = (SAM_CACHE, CAS_CACHE, STITCH_SOURCE, PATH_AUDIT, AGREEMENT_AUDIT, DECODER_AUDIT,
                AGREEMENT_RESULT, DECODER_RESULT)
    for path in required:
        if not path.exists():
            raise RuntimeError(f"required source-audit evidence missing: {path}")
    compact = {"SAMMLV": compact_inventory(SAM_CACHE), "CASME3": compact_inventory(CAS_CACHE)}
    local_raw_candidates = sorted({str(path.resolve()) for pattern in (
        "*window*.npz", "*prestitch*.npz", "*pre_stitch*.npz", "*raw*window*", "*context*audit*.npz")
        for path in ROOT.rglob(pattern) if path.is_file()})
    # Known fresh outputs are stitched hidden/score files, not raw windows; none is local now.
    usable_raw = [path for path in local_raw_candidates if "raw" in Path(path).name.lower() and
                  "window" in Path(path).name.lower()]
    agreement = json.loads(AGREEMENT_RESULT.read_text(encoding="utf-8"))
    decoder = json.loads(DECODER_RESULT.read_text(encoding="utf-8"))
    if usable_raw:
        raise RuntimeError("unexpected possible raw-window source found; inspect before changing blocked status")
    summary = {
        "status": STATUS, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_audit_executed": False, "final_GO_WEAK_NO_GO_grade": None,
        "stop_reason": "Only stitched compact outputs are locally available. Raw per-window 30-frame predictions and their window/local/global indices are absent, so exact reconstruction and candidate-level window trajectories cannot be computed.",
        "compact_cache_inventory": compact,
        "pre_stitch_path_definition": {
            "source_path": str(STITCH_SOURCE.resolve()), "source_sha256": sha256(STITCH_SOURCE),
            "window_length": 30, "stride": 15, "subject_batch_size": 32,
            "window_spotting_shape": "[B,30] (single window [30])",
            "window_logits_shape": "[B,30,C]",
            "local_to_global_mapping": "global_t = video_window_id * 15 + local_t",
            "stitching_rule": "batch-sensitive overwrite: batch-local i==0 writes full window; otherwise writes local 15:30",
            "subject_video_correspondence": "subject windows flattened in original video order; batch-local i does not reset at video boundaries"},
        "workspace_raw_window_search": {"root": str(ROOT.resolve()), "candidate_matches": local_raw_candidates,
                                        "usable_raw_window_sources": usable_raw, "usable_count": 0},
        "prior_alignment_evidence": {
            "strict_numeric_alignment_status": agreement["status"],
            "one_video_max_abs_error": agreement["alignment"]["max_abs_error"],
            "one_video_mean_abs_error": agreement["alignment"]["mean_abs_error"],
            "one_video_Pearson": agreement["alignment"]["Pearson_correlation"],
            "one_video_np_allclose": agreement["alignment"]["np_allclose"],
            "later_one_video_decoder_equivalence": decoder["status"],
            "decoder_equivalence_scope": decoder["scope"],
            "decoder_equivalence_limitation": "The fresh NPZ contains stitched hidden/score/valid_mask only, not raw overlapping-window predictions; it covers one SAMMLV video and cannot support either full SAMMLV or CASME3 window-consensus features."},
        "requested_equivalence_gate": {"max_absolute_error": None, "mean_absolute_error": None,
                                       "Pearson_correlation": None, "event_level_Native_reproduction": None,
                                       "status": "NOT_RUN_SOURCE_MISSING"},
        "candidate_bank": {"status": "NOT_RUN_SOURCE_MISSING", "Final_Strong_Native_candidates_collected": 0},
        "OLD_PRESTITCH_FEATURES": {"available_implementation": str((ROOT / "my_method/pre_stitch_context_feasibility/run_context_agreement_scientific_audit.py").resolve()),
                                   "definition": "stitched peak Height, scalar cross-window Agreement/disagreement, and fixed Height+Agreement",
                                   "scientific_result": "NOT_RUN because its raw context cache/alignment gate was unavailable"},
        "NEW_WINDOW_TRAJECTORY_FEATURES": {"definition": "V1 support ratio; V2 peak-location MAD; V3 peak-confidence CV; V4 center-edge consistency; V5 displacement/support-run/transitions",
                                           "computed": False, "reason": "each requires genuine per-window local observations"},
        "outputs_semantics": "CSV files contain one explicit blocked-status row and no fabricated candidates, features, OOF scores, or metrics.",
        "forbidden_actions": {"backbone_forward": False, "decoder_created": False, "threshold_tuned": False,
                              "events_pruned": False, "events_rescued": False, "hidden_features_added": False,
                              "recognition_rerun": False}}
    dump_path = OUTPUTS / "window_consensus_summary.json"
    dump_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for filename in ("window_candidate_bank.csv", "window_consensus_features.csv", "window_feature_metrics.csv",
                     "window_oof_predictions.csv"):
        write_blocked_csv(OUTPUTS / filename, filename)
    (OUTPUTS / "window_bootstrap.json").write_text(json.dumps({"status": STATUS, "scientific_audit_executed": False,
        "bootstrap_executed": False, "reason": "raw pre-stitch source missing"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# ME-TST+ Pre-Stitch Window-Consensus Separability Audit

最终状态：`{STATUS}`。

## Source gate

当前 workspace 没有可用的 raw pre-stitch overlapping-window outputs。SAMMLV 与 CASME3 compact cache 均只保存 stitched `score[T]`、`logits[T,5]` 等记录，不含 `window_id/local_t/global_t/raw_window_score`，因此无法计算同一 global peak 的多窗口支持、位置轨迹或 center/edge consistency。

已确认的原路径定义：window length=30、stride=15、subject batch size=32；自然映射为 `global_t=video_window_id*15+local_t`。原 stitching 是 batch-sensitive overwrite，而不是窗口平均。

## Equivalence evidence

- 历史 one-video strict numeric gate：max abs error={agreement['alignment']['max_abs_error']:.12f}，mean abs error={agreement['alignment']['mean_abs_error']:.12f}，Pearson={agreement['alignment']['Pearson_correlation']:.12f}，`np.allclose=False`。
- 后续 006/006_1 的 Native event-decoder equivalence 为 `{decoder['status']}`，但其 fresh NPZ 只有 stitched hidden/score/valid_mask，没有 raw window views；该证据不能生成 full SAMMLV 或 CASME3 trajectory features。
- 因 source 缺失，本协议要求的 full reconstruction max/mean error、Pearson 与 event-level Native reproduction 均为 `NOT_RUN_SOURCE_MISSING`。

## Old vs new feature distinction

- `OLD_PRESTITCH_FEATURES`：旧实现计划使用 Height、单一 Agreement/disagreement 及固定 Height+Agreement；科学阶段同样未运行。
- `NEW_WINDOW_TRAJECTORY_FEATURES`：V1 support、V2 location MAD、V3 confidence CV、V4 center-edge、V5 displacement/support-run/transitions。它们都依赖真实 raw window predictions，不能由 stitched cache 反推。

## Stop-rule outcome

没有构建 candidate bank，没有计算 V1–V5，没有运行 LR/LOSO/bootstrap，也没有产生 GO/WEAK-GO/NO-GO 科学等级。所有 CSV 仅包含明确的 blocked-status 行，避免伪造空结果。

`{STATUS}`

已停止；未 forward backbone，未修改数据源，未创建 decoder。

## Outputs

- `{(OUT / 'WINDOW_CONSENSUS_SEPARABILITY_AUDIT_CN.md').resolve()}`
- `{(OUTPUTS / 'window_candidate_bank.csv').resolve()}`
- `{(OUTPUTS / 'window_consensus_features.csv').resolve()}`
- `{(OUTPUTS / 'window_feature_metrics.csv').resolve()}`
- `{(OUTPUTS / 'window_oof_predictions.csv').resolve()}`
- `{(OUTPUTS / 'window_bootstrap.json').resolve()}`
- `{(OUTPUTS / 'window_consensus_summary.json').resolve()}`
- `{(HERE / 'run_source_gate.py').resolve()}`
"""
    (OUT / "WINDOW_CONSENSUS_SEPARABILITY_AUDIT_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": STATUS, "usable_raw_window_sources": 0,
                      "scientific_audit_executed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
