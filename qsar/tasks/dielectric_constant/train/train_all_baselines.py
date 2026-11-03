from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _find_qsar_dir(start: str | Path) -> Path:
    for candidate in Path(start).resolve().parents:
        if candidate.name == "qsar":
            return candidate
    raise RuntimeError(f"Could not locate a parent directory named 'qsar' from {start}")


QSAR_DIR = _find_qsar_dir(__file__)
TASK_DIR = QSAR_DIR / "tasks" / "dielectric_constant"
CONFIG_DIR = TASK_DIR / "configs"
ARTIFACT_DIR = TASK_DIR / "artifacts" / "baselines"
SUMMARY_PATH = TASK_DIR / "artifacts" / "baseline_comparison.json"

BASELINES = (
    (
        "directed_mpnn",
        "qsar.tasks.dielectric_constant.train.train_mpnn_baselines",
        CONFIG_DIR / "directed_mpnn.json",
        ARTIFACT_DIR / "directed_mpnn_dielectric_constant.metrics.json",
    ),
    (
        "communicative_mpnn",
        "qsar.tasks.dielectric_constant.train.train_mpnn_baselines",
        CONFIG_DIR / "communicative_mpnn.json",
        ARTIFACT_DIR / "communicative_mpnn_dielectric_constant.metrics.json",
    ),
    (
        "schnet",
        "qsar.tasks.dielectric_constant.train.train_schnet",
        CONFIG_DIR / "schnet.json",
        ARTIFACT_DIR / "schnet_dielectric_constant.metrics.json",
    ),
    (
        "ecfp",
        "qsar.tasks.dielectric_constant.train.train_ecfp",
        CONFIG_DIR / "ecfp.json",
        ARTIFACT_DIR / "ecfp_dielectric_constant.metrics.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all dielectric-constant baseline models.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[name for name, *_ in BASELINES],
        help="Optional subset of baselines to train.",
    )
    return parser.parse_args()


def metric_at_best_epoch(metrics: dict) -> dict:
    best_epoch = metrics.get("best_epoch")
    history = metrics.get("history") or []
    best_record = next((record for record in history if record.get("epoch") == best_epoch), {})
    return {
        "best_epoch": best_epoch,
        "best_loss": metrics.get("best_loss"),
        "val_rmse": best_record.get("val_rmse", best_record.get("rmse")),
        "val_mae": best_record.get("val_mae", best_record.get("mae")),
        "n_train": metrics.get("n_train"),
        "n_val": metrics.get("n_val"),
        "n_skipped": metrics.get("n_skipped"),
    }


def main() -> None:
    args = parse_args()
    selected = set(args.only or [name for name, *_ in BASELINES])
    summary = {}

    for name, module, config_path, metrics_path in BASELINES:
        if name not in selected:
            continue
        print(f"Training {name} with {config_path}")
        subprocess.run(
            [sys.executable, "-m", module, "--config", str(config_path)],
            cwd=QSAR_DIR.parent,
            check=True,
        )
        if metrics_path.exists():
            summary[name] = metric_at_best_epoch(json.loads(metrics_path.read_text(encoding="utf-8")))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved comparison summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
