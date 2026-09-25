"""Rebuild BoostingVRME spotting curves and evaluate fixed persistence policies.

This script never writes to the BoostingVRME checkout.  It reads its published
cache and weights, then writes curves and reports next to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from natsort import natsorted
from scipy.signal import find_peaks, peak_prominences


PROJECT_ROOT = Path(r"D:\workspace\BoostingVRME-main\BoostingVRME-main")
PUBLISHED_ROOT = Path(r"D:\workspace\BoostingVRME-main\google_drive_download\BoostVRME")
RAW_LABEL_ROOT = Path(r"D:\workspace\ME-TST-main\dataset")
OUTPUT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = OUTPUT_ROOT / "curve_cache"
RESULT_ROOT = OUTPUT_ROOT / "results"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame_skip: int
    window: int
    native_peak_ratio: float
    default_batch_size: int


SPECS = {
    "sammlv": DatasetSpec("SAMMLV", frame_skip=7, window=30, native_peak_ratio=0.55, default_batch_size=128),
    "casme3": DatasetSpec("CASME_3", frame_skip=1, window=50, native_peak_ratio=0.55, default_batch_size=128),
}


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    return np.convolve(values, np.ones(width, dtype=float) / width, mode="same")


def canonical_subject(value: object) -> str:
    return str(value).split(".")[-1].strip()


def map_emotion(dataset: str, emotion: object) -> str | None:
    name = str(emotion).lower().strip()
    if dataset == "SAMMLV":
        if name in {"anger", "contempt", "disgust", "fear", "sadness"}:
            return "negative"
        if name == "happiness":
            return "positive"
        if name == "surprise":
            return "surprise"
        if name == "other":
            return "others"
    else:
        if name in {"anger", "disgust", "fear", "sad"}:
            return "negative"
        if name == "happy":
            return "positive"
        if name == "surprise":
            return "surprise"
        if name == "others":
            return "others"
    return None


def load_label_table(spec: DatasetSpec, label_root: Path) -> pd.DataFrame:
    if spec.name == "SAMMLV":
        dataset_dir = label_root / "SAMMLV"
        long_cols = ["Subject", "Filename", "Inducement Code", "Onset", "Apex", "Offset", "Duration", "Type", "Action Units", "Notes"]
        micro_cols = ["Subject", "Filename", "Inducement Code", "Onset", "Apex", "Offset", "Duration", "Type", "Action Units", "Emotion", "Classes", "Notes"]
        long_table = pd.ExcelFile(dataset_dir / "SAMM_LongVideos_V3_Release.xlsx").parse(
            header=None, names=long_cols, skiprows=list(range(10))
        )
        micro_table = pd.ExcelFile(dataset_dir / "SAMM_Micro_FACS_Codes_v2.xlsx").parse(
            header=None, names=micro_cols, skiprows=[0]
        )
        micro_table["Filename"] = micro_table["Filename"].astype("object")
        table = long_table.merge(micro_table[["Filename", "Type", "Emotion"]], on=["Filename", "Type"])
        table["videoCode"] = [str(name).split("_")[0] + "_" + str(name).split("_")[1] for name in table["Filename"]]
        table["subjectCode"] = [str(name).split("_")[0] for name in table["Filename"]]
        table["Type"] = table["Type"].replace({"Micro - 1/2": "micro-expression"})
        return table.rename(columns={"Type": "type", "Onset": "onset", "Apex": "apex", "Offset": "offset", "Emotion": "emotion"})

    table = pd.ExcelFile(label_root / "CASME_3" / "casme3_label.xlsx").parse(0)
    table = table.rename(
        columns={"Subject": "subject", "Filename": "video", "Onset": "onset", "Apex": "apex", "Offset": "offset"}
    )
    table["videoCode"] = table["video"].astype(str)
    table["subjectCode"] = table["subject"].astype(str)
    table["type"] = "micro-expression"
    return table


def source_video_order(spec: DatasetSpec, table: pd.DataFrame, label_root: Path) -> List[Tuple[str, str]]:
    if spec.name == "SAMMLV":
        videos_dir = label_root / "SAMMLV" / "SAMM_longvideos"
        filesystem_videos = [item.name for item in videos_dir.iterdir()] if videos_dir.exists() else []
        if filesystem_videos:
            ordered = natsorted(filesystem_videos)
            return [(name.split("_")[0], name) for name in ordered]
    else:
        data_dir = label_root / "CASME_3" / "data"
        pairs: List[Tuple[str, str]] = []
        if data_dir.exists():
            for subject_dir in natsorted([item for item in data_dir.iterdir() if item.is_dir()], key=lambda item: item.name):
                subject = canonical_subject(subject_dir.name)
                for video_dir in natsorted([item for item in subject_dir.iterdir() if item.is_dir()], key=lambda item: item.name):
                    pairs.append((subject, video_dir.name))
            if pairs:
                return pairs

    pairs = {(canonical_subject(row.subjectCode), str(row.videoCode)) for row in table.itertuples()}
    return natsorted(pairs, key=lambda pair: (int(pair[0]) if pair[0].isdigit() else pair[0], pair[1]))


def build_metadata(spec: DatasetSpec, label_root: Path) -> List[dict]:
    table = load_label_table(spec, label_root)
    accepted = {"repression", "anger", "contempt", "disgust", "fear", "sadness", "sad", "negative", "happy", "happiness", "positive", "surprise", "others", "other"}
    records: List[dict] = []

    for subject, video in source_video_order(spec, table, label_root):
        matched = table[
            (table["subjectCode"].astype(str).map(canonical_subject) == subject)
            & (table["videoCode"].astype(str) == str(video))
            & (table["type"] == "micro-expression")
        ]
        ranges: List[List[int]] = []
        emotions: List[str] = []
        for row in matched.itertuples():
            emotion_raw = str(row.emotion).lower().strip()
            if emotion_raw not in accepted:
                continue
            emotion = map_emotion(spec.name, emotion_raw)
            if emotion is not None:
                emotions.append(emotion)
            onset, apex, offset = int(row.onset), int(row.apex), int(row.offset)
            if offset == 0:
                ranges.append([onset - 1, apex - 1, apex - 1])
            elif onset != 0:
                ranges.append([onset - 1, apex - 1, offset - 1])
        if ranges and emotions:
            records.append(
                {
                    "subject": subject,
                    "video": str(video),
                    "gt": [[int(value / spec.frame_skip) for value in event] for event in ranges],
                    "emotions": emotions,
                }
            )
    return records


def load_windows(spec: DatasetSpec) -> List[np.ndarray]:
    cache_dir = PUBLISHED_ROOT / "cache"
    if spec.name == "SAMMLV":
        paths = [cache_dir / "SAMMLV_dataset.pkl"]
    else:
        paths = [cache_dir / f"CASME_3_dataset_{index}.pkl" for index in range(1, 6)]
    output: List[np.ndarray] = []
    for path in paths:
        with path.open("rb") as handle:
            output.extend(pickle.load(handle))
    return output


class ReferenceMamba(nn.Module):
    """The public Mamba v1 inference equations using ordinary PyTorch ops."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        inner = d_model * expand
        dt_rank = (d_model + 15) // 16
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1, groups=inner, bias=True)
        self.x_proj = nn.Linear(inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(inner, d_state))
        self.D = nn.Parameter(torch.empty(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        x, gate = self.in_proj(hidden).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2))
        projection = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        delta_raw, b_term, c_term = torch.split(projection, [dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        delta = F.softplus(F.linear(delta_raw, self.dt_proj.weight) + self.dt_proj.bias)
        a_term = -torch.exp(self.A_log.float())
        state = x.new_zeros((batch, x.shape[-1], self.A_log.shape[1]))
        outputs: List[torch.Tensor] = []
        for index in range(length):
            delta_i = delta[:, index].unsqueeze(-1)
            state = torch.exp(delta_i * a_term) * state + delta_i * b_term[:, index].unsqueeze(1) * x[:, index].unsqueeze(-1)
            outputs.append((state * c_term[:, index].unsqueeze(1)).sum(dim=-1) + self.D * x[:, index])
        y = torch.stack(outputs, dim=1) * F.silu(gate)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = ReferenceMamba(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(x))


class BlockMamba(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MambaLayer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm1(self.attn(x))


class LateralConnection(nn.Module):
    def __init__(self, fast_channels: int = 128, slow_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(fast_channels, slow_channels, 3, 2, 1), nn.BatchNorm1d(slow_channels), nn.ReLU())

    def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        return self.conv(fast.transpose(1, 2)).transpose(1, 2) + slow


class BoostingVRMEReference(nn.Module):
    def __init__(self, out_channels: int = 5) -> None:
        super().__init__()
        dim = 128
        self.Stem = nn.Sequential(
            nn.Conv1d(36, dim, 3, 1, 1), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 3, 1, 1), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
        )
        self.Stem_slow = nn.Sequential(nn.Conv1d(dim, dim * 2, 3, 2, 1), nn.BatchNorm1d(dim * 2), nn.ReLU(inplace=True))
        self.block1 = nn.ModuleList([BlockMamba(dim * 2) for _ in range(4)])
        self.block2 = nn.ModuleList([BlockMamba(dim * 2) for _ in range(4)])
        self.block3 = nn.ModuleList([BlockMamba(dim * 2) for _ in range(4)])
        self.block1_fast = nn.ModuleList([BlockMamba(dim) for _ in range(4)])
        self.block2_fast = nn.ModuleList([BlockMamba(dim) for _ in range(4)])
        self.block3_fast = nn.ModuleList([BlockMamba(dim) for _ in range(4)])
        self.fuse_1 = LateralConnection()
        self.fuse_2 = LateralConnection()
        self.upsample = nn.Upsample(scale_factor=2)
        self.fc_spot = nn.Sequential(nn.Linear(dim * 3, 1))
        self.fc_recog = nn.Sequential(nn.Linear(dim * 3, out_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.squeeze(1)
        x = self.Stem(x)
        fast = x.transpose(1, 2)
        slow = self.Stem_slow(x).transpose(1, 2)
        for block in self.block1:
            slow = block(slow)
        for block in self.block1_fast:
            fast = block(fast)
        slow = self.fuse_1(slow, fast)
        for block in self.block2:
            slow = block(slow)
        for block in self.block2_fast:
            fast = block(fast)
        slow = self.fuse_2(slow, fast)
        for block in self.block3:
            slow = block(slow)
        for block in self.block3_fast:
            fast = block(fast)
        slow = self.upsample(slow.transpose(1, 2)).transpose(1, 2)
        fusion = torch.cat((slow, fast), dim=2)
        return self.sigmoid(self.fc_spot(fusion)).squeeze(-1), self.fc_recog(fusion)


def subject_weight(spec: DatasetSpec, subject: str) -> Path:
    folder = "SAMMLV_4emo" if spec.name == "SAMMLV" else "CASME_3_4emo"
    filename = f"subject_{int(subject):03d}.pkl" if spec.name == "SAMMLV" else f"subject_{int(subject)}.pkl"
    candidates = [PUBLISHED_ROOT / "weights" / folder / filename, PROJECT_ROOT / "weights" / folder / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Official {spec.name} weight was not found: {candidates}")


def load_model(spec: DatasetSpec, subject: str, device: torch.device) -> BoostingVRMEReference:
    weight_path = subject_weight(spec, subject)
    state = torch.load(weight_path, map_location="cpu")
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model = BoostingVRMEReference().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def reassemble_window_scores(window_scores: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    curve = np.zeros((len(window_scores) + 1) * half, dtype=float)
    for index, score in enumerate(window_scores):
        if index == 0:
            curve[:window] = score
        else:
            curve[(index + 1) * half : (index + 2) * half] = score[half:]
    return curve


def build_curves(spec: DatasetSpec, records: List[dict], batch_size: int, device: torch.device, force: bool) -> dict:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"{spec.name.lower()}_curves.pkl"
    if cache_path.exists() and not force:
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    windows = load_windows(spec)
    if len(records) != len(windows):
        raise RuntimeError(f"{spec.name}: labels have {len(records)} videos but the official cache has {len(windows)}. Refusing to misalign GT.")
    if any(tuple(np.asarray(video).shape[1:]) != (36, spec.window) for video in windows):
        raise RuntimeError(f"{spec.name}: cache window shape differs from the official network input.")

    video_indices = list(range(len(records)))
    by_subject: Dict[str, List[int]] = OrderedDict()
    for index, record in enumerate(records):
        by_subject.setdefault(record["subject"], []).append(index)

    curves: List[np.ndarray | None] = [None] * len(records)
    for position, (subject, indices) in enumerate(by_subject.items(), start=1):
        print(f"[{spec.name}] subject {position}/{len(by_subject)}: {subject}, videos={len(indices)}", flush=True)
        model = load_model(spec, subject, device)
        video_lengths = [len(windows[index]) for index in indices]
        subject_windows = np.concatenate([np.asarray(windows[index], dtype=np.float32) for index in indices], axis=0)
        with torch.inference_mode():
            outputs: List[np.ndarray] = []
            for batch_start in range(0, len(subject_windows), batch_size):
                batch = torch.from_numpy(subject_windows[batch_start : batch_start + batch_size]).unsqueeze(1).to(device)
                scores, _ = model(batch)
                outputs.append(scores.cpu().numpy())
        subject_scores = np.concatenate(outputs, axis=0)
        offset = 0
        for video_index, length in zip(indices, video_lengths):
            curves[video_index] = reassemble_window_scores(subject_scores[offset : offset + length], spec.window)
            offset += length
        del model

    package = {
        "dataset": spec.name,
        "window": spec.window,
        "records": records,
        "curves": curves,
        "weight_root": str(PUBLISHED_ROOT / "weights"),
        "cache_root": str(PUBLISHED_ROOT / "cache"),
    }
    with cache_path.open("wb") as handle:
        pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return package


def k_prime(records: Sequence[dict]) -> int:
    durations = sorted(event[2] - event[0] for record in records for event in record["gt"])
    return int((durations[len(durations) // 2] + 1) / 2)


def source_interval(curve: np.ndarray, peak: int, k_p: int) -> Tuple[int, int]:
    length = len(curve)
    left = max(0, peak - k_p)
    right = min(length - 1, peak + k_p)
    mean = float(curve.mean())
    start = next((index - 1 for index in range(left + 1, peak) if curve[index] > curve[index - 1] and curve[index] > mean / 1.5), left)
    end = next((index + 1 for index in range(right - 1, peak, -1) if curve[index] > curve[index + 1] and curve[index] > mean / 1.5), right)
    for index in range(start - 1, max(-1, peak - k_p * 5) - 1, -1):
        if index > 0 and curve[index] < mean and curve[index - 1] < mean:
            start = index + 1
            break
    for index in range(end + 1, min(length - 1, peak + k_p * 5) + 1):
        if index < length - 1 and curve[index] < mean and curve[index + 1] < mean:
            end = index - 1
            break
    return max(0, start), min(length - 1, end)


def interval_iou(first: Sequence[int], second: Sequence[int]) -> float:
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    intersection = max(0, right - left + 1)
    union = (first[1] - first[0] + 1) + (second[1] - second[0] + 1) - intersection
    return intersection / union if union else 0.0


def source_nms_iou(first: Sequence[int], second: Sequence[int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 0.0


def native_predictions(curve: np.ndarray, k_p: int, ratio: float) -> List[Tuple[int, int, int]]:
    smooth = moving_average(curve, k_p * 2)
    threshold = float(smooth.mean() + ratio * (smooth.max() - smooth.mean()))
    peaks, _ = find_peaks(smooth, height=threshold, distance=k_p)
    candidates = [(*source_interval(curve, int(peak), k_p), int(peak)) for peak in peaks]
    kept: List[Tuple[int, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if all(source_nms_iou(candidate[:2], old[:2]) < 0.2 and abs(candidate[2] - old[2]) > k_p for old in kept):
            kept.append(candidate)
    return kept


def persistence_predictions(curve: np.ndarray, k_p: int, ratio: float, prominence_ratio: float, scales: Sequence[float]) -> List[Tuple[int, int, int]]:
    candidates: List[Tuple[float, int]] = []
    value_range = max(float(curve.max() - curve.min()), 1e-9)
    for scale in scales:
        smooth = moving_average(curve, max(3, round(k_p * 2 * scale)))
        threshold = float(smooth.mean() + ratio * (smooth.max() - smooth.mean()))
        peaks, _ = find_peaks(smooth, height=threshold, distance=max(1, round(k_p * scale)))
        if len(peaks):
            persistences = peak_prominences(smooth, peaks)[0]
            candidates.extend((float(persistence), int(peak)) for peak, persistence in zip(peaks, persistences) if persistence >= prominence_ratio * value_range)
    candidates.sort(reverse=True)
    selected: List[Tuple[int, int, int]] = []
    for _, peak in candidates:
        candidate = (*source_interval(curve, peak, k_p), peak)
        if all(abs(peak - previous[2]) > k_p and source_nms_iou(candidate[:2], previous[:2]) < 0.2 for previous in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item[0])


POLICIES = {
    "Native": lambda curve, k: native_predictions(curve, k, 0.55),
    "Global_Persistence": lambda curve, k: persistence_predictions(curve, k, 0.65, 0.12, (1.0,)),
    "Global_Moderate": lambda curve, k: persistence_predictions(curve, k, 0.55, 0.08, (1.0,)),
    "Local_Multiscale": lambda curve, k: persistence_predictions(curve, k, 0.35, 0.04, (0.75, 1.25, 2.0)),
    "Local_Conservative": lambda curve, k: persistence_predictions(curve, k, 0.45, 0.06, (0.75, 1.25, 2.0)),
}


def evaluate(predictions: Sequence[Sequence[Tuple[int, int, int]]], records: Sequence[dict], include_subjects: set[str] | None = None) -> dict:
    tp = fp = fn = 0
    for predicted, record in zip(predictions, records):
        if include_subjects is not None and record["subject"] not in include_subjects:
            continue
        matched: set[int] = set()
        for predicted_interval in predicted:
            overlaps = [interval_iou(predicted_interval[:2], (event[0], event[2])) for event in record["gt"]]
            best = int(np.argmax(overlaps)) if overlaps else -1
            if best >= 0 and overlaps[best] >= 0.5 and best not in matched:
                tp += 1
                matched.add(best)
            else:
                fp += 1
        fn += len(record["gt"]) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def selector_predictions(policy_predictions: Dict[str, List[List[Tuple[int, int, int]]]], records: Sequence[dict]) -> Tuple[List[List[Tuple[int, int, int]]], Counter]:
    subjects = sorted({record["subject"] for record in records}, key=lambda value: int(value) if value.isdigit() else value)
    selected = [[] for _ in records]
    choices: Counter = Counter()
    for held_subject in subjects:
        training_subjects = set(subjects) - {held_subject}
        best_policy = max(POLICIES, key=lambda name: (evaluate(policy_predictions[name], records, training_subjects)["F1"], name))
        choices[best_policy] += 1
        for index, record in enumerate(records):
            if record["subject"] == held_subject:
                selected[index] = policy_predictions[best_policy][index]
    return selected, choices


def render_report(spec: DatasetSpec, package: dict) -> str:
    records = package["records"]
    curves = package["curves"]
    k_p = k_prime(records)
    policy_predictions = {name: [function(curve, k_p) for curve in curves] for name, function in POLICIES.items()}
    metrics = {name: evaluate(predictions, records) for name, predictions in policy_predictions.items()}
    selected, choices = selector_predictions(policy_predictions, records)
    metrics["Nested_Selector"] = evaluate(selected, records)
    rows = []
    for name, values in metrics.items():
        rows.append(f"| {name} | {values['TP']} | {values['FP']} | {values['FN']} | {values['precision']:.4f} | {values['recall']:.4f} | {values['F1']:.4f} |")
    details = {
        "dataset": spec.name,
        "videos": len(records),
        "subjects": len({record["subject"] for record in records}),
        "gt_events": sum(len(record["gt"]) for record in records),
        "k_prime": k_p,
        "metrics": metrics,
        "selector_choices": dict(choices),
        "fixed_policies": {
            "Native": "官方 smooth(k'=2) + p=0.55 + 原始区间延展/NMS",
            "Global_Persistence": "单尺度、p=0.65、显著性>=0.12*全局幅度",
            "Global_Moderate": "单尺度、p=0.55、显著性>=0.08*全局幅度",
            "Local_Multiscale": "0.75/1.25/2.0 三尺度、p=0.35、显著性>=0.04*全局幅度",
            "Local_Conservative": "0.75/1.25/2.0 三尺度、p=0.45、显著性>=0.06*全局幅度",
        },
    }
    (RESULT_ROOT / f"{spec.name.lower()}_metrics.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = f"""# BoostingVRME Persistence 策略测试：{spec.name}

## 数据与复现范围

- 曲线来自 BoostingVRME 官方发布的预处理缓存和逐被试权重；原项目未被改写。
- GT 仅使用完整原始数据目录内的 Excel 标注，并按官方 `load_label.py` 的规则转换和按 `frame_skip` 缩放。
- 视频数：{details['videos']}；被试数：{details['subjects']}；GT 事件数：{details['gt_events']}；官方 `k_p`：{k_p}。
- 指标为原项目同一 IoU=0.5 的逐视频贪心匹配定义；TP/FP/FN 后汇总计算 F1。

## 结果

| 策略 | TP | FP | FN | Precision | Recall | F1 |
| - | -: | -: | -: | -: | -: | -: |
{chr(10).join(rows)}

## 策略选择

`Nested_Selector` 对每个留出被试，只在其余被试上比较五个固定候选，然后将最优候选应用到留出被试。它不是把当前被试的 GT 用于选参。

各候选被选为留出策略的次数：`{dict(choices)}`。

## 固定候选

- `Native`：官方峰检测、区间延展和 NMS，不作新增筛选。
- `Global_Persistence`/`Global_Moderate`：单尺度峰的全局显著性筛选，适合高分背景下压低 FP。
- `Local_Multiscale`/`Local_Conservative`：在三个平滑尺度上保留显著峰，适合低分且表达峰分散的曲线。

所有候选共享官方的区间延展逻辑；差异仅在峰的候选和显著性筛选。
"""
    (RESULT_ROOT / f"{spec.name.lower()}_SUMMARY.md").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sammlv", "casme3", "both"], default="both")
    parser.add_argument("--stage", choices=["metadata", "curves", "evaluate", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected PyTorch environment.")
    device = torch.device(args.device)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_specs = list(SPECS.values()) if args.dataset == "both" else [SPECS[args.dataset]]
    for spec in selected_specs:
        records = build_metadata(spec, RAW_LABEL_ROOT)
        print(f"[{spec.name}] metadata: videos={len(records)}, subjects={len({row['subject'] for row in records})}", flush=True)
        if args.stage == "metadata":
            continue
        package = build_curves(spec, records, args.batch_size, device, args.force)
        if args.stage in {"evaluate", "all"}:
            report = render_report(spec, package)
            print(report, flush=True)


if __name__ == "__main__":
    main()
