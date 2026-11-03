from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn


DEFAULT_TARGET_COLUMNS = ("property_value_si",)


def load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return {key.replace("-", "_"): value for key, value in config.items()}


def normalize_config_defaults(config: dict[str, Any], path_arg_names: set[str]) -> dict[str, Any]:
    normalized = dict(config)
    for name in path_arg_names:
        if name in normalized and normalized[name] is not None:
            normalized[name] = Path(normalized[name])
    if isinstance(normalized.get("target_columns"), list):
        normalized["target_columns"] = ",".join(str(column) for column in normalized["target_columns"])
    return normalized


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading Excel input requires pandas. Use CSV or install pandas.") from exc
        return pd.read_excel(path).to_dict(orient="records")

    raise ValueError(f"Unsupported data format: {path}")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def property_names_from_columns(target_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column.strip().lower() for column in target_columns)


def split_indices(length: int, val_size: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(length * val_size))
    if length > 1:
        val_count = min(max(val_count, 1), length - 1)
    else:
        val_count = 0
    return indices[val_count:], indices[:val_count]


def subset_dataset(dataset, indices: Sequence[int]) -> list[Any]:
    return [dataset[index] for index in indices]


def compute_target_stats(targets: Tensor) -> tuple[Tensor, Tensor]:
    mean = targets.mean(dim=0)
    std = targets.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(torch.sqrt(torch.mean(error.pow(2))).item()),
        "mae": float(torch.mean(torch.abs(error)).item()),
    }


def evaluate_regressor(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    target_mean: Tensor,
    target_std: Tensor,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch, target in loader:
            batch = batch.to(device)
            target = target.to(device)
            scaled_target = (target - target_mean) / target_std
            scaled_prediction = model(batch)
            loss = criterion(scaled_prediction, scaled_target)

            total_loss += float(loss.item()) * target.size(0)
            total_items += target.size(0)
            predictions.append(scaled_prediction * target_std + target_mean)
            targets.append(target)

    if total_items == 0:
        return float("nan"), {"rmse": float("nan"), "mae": float("nan")}

    prediction_tensor = torch.cat(predictions, dim=0).cpu()
    target_tensor = torch.cat(targets, dim=0).cpu()
    return total_loss / total_items, regression_metrics(prediction_tensor, target_tensor)
