from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

def _is_qsar_dir(path: Path) -> bool:
    return path.name == "qsar" and (path / "core").is_dir() and (path / "tasks").is_dir()


def _candidate_dirs(start: str | Path) -> list[Path]:
    start_path = Path(start).resolve()
    anchors = [start_path.parent if start_path.is_file() else start_path, Path.cwd().resolve()]
    env_qsar_dir = os.environ.get("QSAR_DIR")
    if env_qsar_dir:
        anchors.insert(0, Path(env_qsar_dir).expanduser().resolve())

    candidates: list[Path] = []
    for anchor in anchors:
        candidates.extend([anchor, *anchor.parents])
        candidates.extend([anchor / "qsar", *(parent / "qsar" for parent in anchor.parents)])
    return candidates


def _find_qsar_dir(start: str | Path) -> Path:
    seen: set[Path] = set()
    for candidate in _candidate_dirs(start):
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_qsar_dir(candidate):
            return candidate
    raise RuntimeError(
        f"Could not locate the qsar project directory from {start}. "
        "Set QSAR_DIR to the absolute path of the qsar directory."
    )


QSAR_DIR = _find_qsar_dir(__file__)
PROJECT_ROOT = QSAR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar.tasks.flash_point.models.mpnn_baselines import build_mpnn_baseline
from qsar.tasks.flash_point.models.molecule_graph import (
    MoleculeGraphFeaturizer,
    MoleculeGraphTensors,
    collate_molecule_graphs,
)
from qsar.tasks.flash_point.train.common import (
    DEFAULT_TARGET_COLUMNS,
    compute_target_stats,
    evaluate_regressor,
    load_json_config,
    normalize_config_defaults,
    parse_float,
    property_names_from_columns,
    read_rows,
    split_indices,
    subset_dataset,
)


DEFAULT_DATA_PATH = QSAR_DIR / "datas" / "insulating_oil_property_database" / "processed" / "oil_like_multitask.csv"
DEFAULT_OUTPUT_DIR = QSAR_DIR / "tasks" / "flash_point" / "artifacts" / "baselines"
PATH_ARG_NAMES = {"data_path", "output_dir", "output_path"}


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None, help="JSON training config.")
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = normalize_config_defaults(load_json_config(config_args.config), PATH_ARG_NAMES)

    parser = argparse.ArgumentParser(
        description="Train Directed MPNN or Communicative MPNN QSAR baselines.",
        parents=[config_parser],
    )
    parser.add_argument("--model", choices=("directed", "communicative", "communitative"), default="directed")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument(
        "--target-columns",
        type=str,
        default=",".join(DEFAULT_TARGET_COLUMNS),
        help="Comma-separated target columns in model output order.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.set_defaults(**config_defaults)
    return parser.parse_args(remaining_args)


class BaselineMoleculePropertyDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        smiles_column: str,
        target_columns: Sequence[str],
        *,
        add_self_loops: bool,
    ) -> None:
        self.graphs: list[MoleculeGraphTensors] = []
        self.targets: list[Tensor] = []
        self.skipped_rows = 0
        featurizer = MoleculeGraphFeaturizer(add_self_loops=add_self_loops)

        for row in rows:
            smiles = str(row.get(smiles_column, "")).strip()
            targets = [parse_float(row.get(column)) for column in target_columns]
            if not smiles or any(value is None for value in targets):
                self.skipped_rows += 1
                continue

            try:
                graph = featurizer(smiles)
            except ValueError:
                self.skipped_rows += 1
                continue
            self.graphs.append(graph)
            self.targets.append(torch.tensor(targets, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> tuple[MoleculeGraphTensors, Tensor]:
        return self.graphs[index], self.targets[index]


def collate_batch(items: Sequence[tuple[MoleculeGraphTensors, Tensor]]):
    graphs, targets = zip(*items)
    return collate_molecule_graphs(graphs), torch.stack(targets, dim=0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    target_mean: Tensor,
    target_std: Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch, target in loader:
        batch = batch.to(device)
        target = target.to(device)
        scaled_target = (target - target_mean) / target_std

        optimizer.zero_grad(set_to_none=True)
        scaled_prediction = model(batch)
        loss = criterion(scaled_prediction, scaled_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * target.size(0)
        total_items += target.size(0)

    return total_loss / max(total_items, 1)


def main() -> None:
    args = parse_args()
    target_columns = tuple(column.strip() for column in args.target_columns.split(",") if column.strip())
    if not target_columns:
        raise ValueError("At least one target column is required.")
    property_names = property_names_from_columns(target_columns)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    normalized_model_name = "communicative" if args.model == "communitative" else args.model

    rows = read_rows(args.data_path)
    dataset = BaselineMoleculePropertyDataset(
        rows=rows,
        smiles_column=args.smiles_column,
        target_columns=target_columns,
        add_self_loops=(normalized_model_name != "directed"),
    )
    if len(dataset) == 0:
        raise ValueError("No valid training molecules found.")

    train_indices, val_indices = split_indices(len(dataset), args.val_size, args.seed)
    from ester_oil.audit import save_split
    save_split(args, train_indices, val_indices, dataset)
    train_data = subset_dataset(dataset, train_indices)
    val_data = subset_dataset(dataset, val_indices)

    train_targets = torch.stack([target for _, target in train_data], dim=0).to(device)
    target_mean, target_std = compute_target_stats(train_targets)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model_config = {
        "node_feature_dim": 130,
        "edge_feature_dim": 10,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "readout_dim": args.readout_dim,
        "output_dim": len(target_columns),
        "dropout": args.dropout,
    }
    model = build_mpnn_baseline(normalized_model_name, **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    checkpoint_path = args.output_path or args.output_dir / f"{normalized_model_name}_mpnn.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_path.with_suffix(".metrics.json")

    best_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, target_mean, target_std, device)
        val_loss, val_metrics = evaluate_regressor(model, val_loader, criterion, target_mean, target_std, device)
        score_loss = val_loss if val_data else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})

        if score_loss < best_loss:
            best_loss = score_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": normalized_model_name,
                "model_config": model_config,
                "property_names": property_names,
                "target_columns": target_columns,
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
            }
            torch.save(checkpoint, checkpoint_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"model={normalized_model_name} epoch={epoch:04d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} val_rmse={val_metrics['rmse']:.6f} "
                f"val_mae={val_metrics['mae']:.6f}"
            )

    metrics = {
        "model_name": normalized_model_name,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "property_names": property_names,
        "target_columns": target_columns,
        "n_rows": len(rows),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_skipped": dataset.skipped_rows,
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved best checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
