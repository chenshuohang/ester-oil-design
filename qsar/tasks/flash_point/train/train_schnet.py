from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


def _find_qsar_dir(start: str | Path) -> Path:
    for candidate in Path(start).resolve().parents:
        if candidate.name == "qsar":
            return candidate
    raise RuntimeError(f"Could not locate a parent directory named 'qsar' from {start}")


QSAR_DIR = _find_qsar_dir(__file__)
PROJECT_ROOT = QSAR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar.tasks.flash_point.models.schnet import (
    SchNetFeaturizer,
    SchNetMoleculeGraph,
    build_schnet_flash_point_gnn,
    collate_schnet_graphs,
)
from qsar.tasks.flash_point.train.common import (
    load_json_config,
    normalize_config_defaults,
    parse_float,
    read_rows,
    split_indices,
)


DEFAULT_DATA_PATH = QSAR_DIR / "datas" / "insulating_oil_property_database" / "processed" / "oil_like_multitask.csv"
DEFAULT_OUTPUT_PATH = QSAR_DIR / "tasks" / "flash_point" / "artifacts" / "baselines" / "schnet_flash_point.pt"
PATH_ARG_NAMES = {"data_path", "output_path"}


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = normalize_config_defaults(load_json_config(config_args.config), PATH_ARG_NAMES)

    parser = argparse.ArgumentParser(description="Train a SchNet 3D regressor for flash-point prediction.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--target-column", default="flash_point_K")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-atomic-number", type=int, default=100)
    parser.add_argument("--no-add-hydrogens", action="store_true")
    parser.add_argument("--conformer-seed", type=int, default=2026)
    parser.add_argument("--optimize-steps", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.set_defaults(**config_defaults)
    return parser.parse_args(remaining_args)


@dataclass(frozen=True)
class SchNetFlashPointObservation:
    graph: SchNetMoleculeGraph
    target: float


class SchNetFlashPointDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        smiles_column: str,
        target_column: str,
        featurizer: SchNetFeaturizer,
    ) -> None:
        self.items: list[SchNetFlashPointObservation] = []
        self.skipped_rows = 0
        graph_cache: dict[str, SchNetMoleculeGraph] = {}

        for row in rows:
            smiles = str(row.get(smiles_column, "")).strip()
            target = parse_float(row.get(target_column))
            if not smiles or target is None:
                self.skipped_rows += 1
                continue
            try:
                graph = graph_cache.get(smiles)
                if graph is None:
                    graph = featurizer(smiles)
                    graph_cache[smiles] = graph
            except ValueError:
                self.skipped_rows += 1
                continue
            self.items.append(SchNetFlashPointObservation(graph, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> SchNetFlashPointObservation:
        return self.items[index]


def collate_batch(items: Sequence[SchNetFlashPointObservation]) -> tuple[Any, Tensor]:
    return (
        collate_schnet_graphs([item.graph for item in items]),
        torch.tensor([[item.target] for item in items], dtype=torch.float32),
    )


def subset_dataset(dataset: Dataset, indices: Sequence[int]) -> list[Any]:
    return [dataset[index] for index in indices]


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(torch.sqrt(torch.mean(error.pow(2))).item()),
        "mae": float(torch.mean(torch.abs(error)).item()),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    target_mean: Tensor,
    target_std: Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    predictions = []
    targets = []

    with torch.set_grad_enabled(is_train):
        for batch, target in loader:
            non_blocking = device.type == "cuda"
            batch = batch.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            scaled_target = (target - target_mean) / target_std

            if is_train:
                optimizer.zero_grad(set_to_none=True)
            scaled_prediction = model(batch)
            loss = criterion(scaled_prediction, scaled_target)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item()) * target.size(0)
            total_items += target.size(0)
            predictions.append((scaled_prediction.detach() * target_std + target_mean).cpu())
            targets.append(target.detach().cpu())

    if total_items == 0:
        return float("nan"), {"rmse": float("nan"), "mae": float("nan")}
    return total_loss / total_items, regression_metrics(torch.cat(predictions, dim=0), torch.cat(targets, dim=0))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = read_rows(args.data_path)
    featurizer_config = {
        "cutoff": args.cutoff,
        "add_hydrogens": not args.no_add_hydrogens,
        "random_seed": args.conformer_seed,
        "optimize_steps": args.optimize_steps,
    }
    dataset = SchNetFlashPointDataset(
        rows,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        featurizer=SchNetFeaturizer(**featurizer_config),
    )
    if not dataset:
        raise ValueError("No valid flash-point observations found.")
    print("Using device:", device)
    print(f"SchNet featurizer skipped rows: {dataset.skipped_rows}")

    train_indices, val_indices = split_indices(len(dataset), args.val_size, args.seed)
    from ester_oil.audit import save_split
    save_split(args, train_indices, val_indices, dataset)
    train_data, val_data = subset_dataset(dataset, train_indices), subset_dataset(dataset, val_indices)
    train_targets = torch.tensor([[item.target] for item in train_data], dtype=torch.float32, device=device)
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0, unbiased=False).clamp_min(1e-6)

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
        "max_atomic_number": args.max_atomic_number,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_gaussians": args.num_gaussians,
        "cutoff": args.cutoff,
        "readout_dim": args.readout_dim,
        "dropout": args.dropout,
        "output_dim": 1,
    }
    model = build_schnet_flash_point_gnn(**model_config).to(device)
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
                    "model_name": "schnet_flash_point_gnn",
                    "model_config": model_config,
                    "featurizer_config": featurizer_config,
                    "property_names": ("flash_point_k",),
                    "target_columns": (args.target_column,),
                    "smiles_column": args.smiles_column,
                    "target_mean": target_mean.detach().cpu(),
                    "target_std": target_std.detach().cpu(),
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "data_path": str(args.data_path),
                    "n_rows": len(rows),
                    "n_train": len(train_data),
                    "n_val": len(val_data),
                    "n_skipped": dataset.skipped_rows,
                },
                args.output_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
            )

    metrics_path = args.output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": "schnet_flash_point_gnn",
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "property_names": ("flash_point_k",),
                "target_columns": (args.target_column,),
                "n_rows": len(rows),
                "n_train": len(train_data),
                "n_val": len(val_data),
                "n_skipped": dataset.skipped_rows,
                "target_mean": float(target_mean.detach().cpu().reshape(()).item()),
                "target_std": float(target_std.detach().cpu().reshape(()).item()),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved best checkpoint: {args.output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
