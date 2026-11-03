from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def _find_qsar_dir(start: str | Path) -> Path:
    for candidate in Path(start).resolve().parents:
        if candidate.name == "qsar":
            return candidate
    raise RuntimeError(f"Could not locate a parent directory named 'qsar' from {start}")


QSAR_DIR = _find_qsar_dir(__file__)
PROJECT_ROOT = QSAR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar.tasks.dynamic_viscosity.models.conditional_mpnn import build_conditional_viscosity_mpnn
from qsar.tasks.dynamic_viscosity.train.train_temperature_mpnn import (
    DynamicViscosityDataset,
    ViscosityObservation,
    grouped_split_indices,
    read_rows,
    subset,
)
from qsar.tasks.flash_point.models.molecule_graph import collate_molecule_graphs
from qsar.tasks.flash_point.train.common import load_json_config, normalize_config_defaults


DEFAULT_DATA_PATH = QSAR_DIR / "tasks" / "dynamic_viscosity" / "data" / "dynamic_viscosity_observations_with_smiles.csv"
DEFAULT_OUTPUT_PATH = QSAR_DIR / "tasks" / "dynamic_viscosity" / "artifacts" / "conditional_communicative_mpnn.pt"
PATH_ARG_NAMES = {"data_path", "output_path"}


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = normalize_config_defaults(load_json_config(config_args.config), PATH_ARG_NAMES)
    for name in PATH_ARG_NAMES:
        value = config_defaults.get(name)
        if value is not None and not value.is_absolute():
            config_defaults[name] = QSAR_DIR / value

    parser = argparse.ArgumentParser(
        description="Train a temperature/pressure-conditioned MPNN for dynamic viscosity.",
        parents=[config_parser],
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--compound-column", default="compound_id")
    parser.add_argument("--temperature-column", default="temperature_K")
    parser.add_argument("--pressure-column", default="pressure_kPa")
    parser.add_argument("--default-pressure-kPa", type=float, default=101.325)
    parser.add_argument("--target-column", default="log_dynamic_viscosity")
    parser.add_argument("--model", choices=("directed", "communicative"), default="communicative")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--molecule-dim", type=int, default=None)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--reference-temperature-K", type=float, default=298.15)
    parser.add_argument("--reference-pressure-kPa", type=float, default=101.325)
    parser.add_argument("--temperature-scale-K", type=float, default=100.0)
    parser.add_argument("--pressure-scale-kPa", type=float, default=100000.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.set_defaults(**config_defaults)
    return parser.parse_args(remaining_args)


def collate_batch(items: Sequence[ViscosityObservation]) -> tuple[Any, Tensor, Tensor, Tensor]:
    return (
        collate_molecule_graphs([item.graph for item in items]),
        torch.tensor([item.temperature_K for item in items], dtype=torch.float32),
        torch.tensor([item.pressure_kPa for item in items], dtype=torch.float32),
        torch.tensor([item.log_viscosity for item in items], dtype=torch.float32),
    )


def run_epoch(model, loader, criterion, target_mean, target_std, device, optimizer=None) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    sum_squared_error = 0.0
    sum_absolute_error = 0.0

    with torch.set_grad_enabled(is_train):
        for batch, temperature, pressure, target in loader:
            non_blocking = device.type == "cuda"
            batch = batch.to(device, non_blocking=non_blocking)
            temperature = temperature.to(device, non_blocking=non_blocking)
            pressure = pressure.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            scaled_target = (target - target_mean) / target_std

            if is_train:
                optimizer.zero_grad(set_to_none=True)
            scaled_prediction = model(batch, temperature, pressure)
            loss = criterion(scaled_prediction, scaled_target)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item()) * target.size(0)
            total_items += target.size(0)
            prediction = scaled_prediction.detach() * target_std + target_mean
            error = prediction - target
            sum_squared_error += float(error.pow(2).sum().item())
            sum_absolute_error += float(error.abs().sum().item())

    if total_items == 0:
        return float("nan"), {"rmse_log_eta": float("nan"), "mae_log_eta": float("nan")}
    return total_loss / total_items, {
        "rmse_log_eta": math.sqrt(sum_squared_error / total_items),
        "mae_log_eta": sum_absolute_error / total_items,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = read_rows(args.data_path)
    dataset = DynamicViscosityDataset(
        rows,
        smiles_column=args.smiles_column,
        compound_column=args.compound_column,
        temperature_column=args.temperature_column,
        pressure_column=args.pressure_column,
        default_pressure_kPa=args.default_pressure_kPa,
        target_column=args.target_column,
        add_self_loops=(args.model != "directed"),
    )
    if not dataset:
        raise ValueError("No valid dynamic-viscosity observations found.")

    print("Using device:", device)
    train_indices, val_indices = grouped_split_indices(dataset, args.val_size, args.seed)
    from ester_oil.audit import save_split
    save_split(args, train_indices, val_indices, dataset)
    train_data, val_data = subset(dataset, train_indices), subset(dataset, val_indices)
    train_targets = torch.tensor([item.log_viscosity for item in train_data], dtype=torch.float32, device=device)
    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False).clamp_min(1e-6)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_batch,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    model_config = {
        "backbone_name": args.model,
        "node_feature_dim": 130,
        "edge_feature_dim": 10,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "readout_dim": args.readout_dim,
        "molecule_dim": args.molecule_dim,
        "condition_dim": args.condition_dim,
        "dropout": args.dropout,
        "reference_temperature_K": args.reference_temperature_K,
        "reference_pressure_kPa": args.reference_pressure_kPa,
        "temperature_scale_K": args.temperature_scale_K,
        "pressure_scale_kPa": args.pressure_scale_kPa,
    }
    model = build_conditional_viscosity_mpnn(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_loss, best_epoch, history = float("inf"), 0, []
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, target_mean, target_std, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, target_mean, target_std, device)
        score_loss = val_loss if val_data else train_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **train_metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if score_loss < best_loss:
            best_loss, best_epoch = score_loss, epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "conditional_viscosity_mpnn",
                    "model_config": model_config,
                    "target_mean": target_mean.detach().cpu(),
                    "target_std": target_std.detach().cpu(),
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "data_path": str(args.data_path),
                    "n_rows": len(rows),
                    "n_observations": len(dataset),
                    "n_train": len(train_data),
                    "n_val": len(val_data),
                    "n_compounds": len({item.compound_id for item in dataset.items}),
                    "n_skipped": dataset.skipped_rows,
                },
                args.output_path,
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_rmse_log_eta={val_metrics['rmse_log_eta']:.6f}"
            )

    with args.output_path.with_suffix(".metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "n_observations": len(dataset),
                "n_train": len(train_data),
                "n_val": len(val_data),
                "n_compounds": len({item.compound_id for item in dataset.items}),
                "target_mean": float(target_mean.detach().cpu().item()),
                "target_std": float(target_std.detach().cpu().item()),
                "history": history,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
