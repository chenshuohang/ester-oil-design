from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


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

DEFAULT_OBSERVATIONS = (
    QSAR_DIR / "datas" / "insulating_oil_property_database" / "processed" / "dielectric_constant_experimental.csv"
)
DEFAULT_COMPOUND_MASTER = (
    QSAR_DIR / "datas" / "insulating_oil_property_database" / "processed" / "compound_master.csv"
)
DEFAULT_OUTPUT = (
    QSAR_DIR / "tasks" / "dielectric_constant" / "data" / "dielectric_constant_observations_with_smiles.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an observation-level dielectric-constant dataset with canonical SMILES."
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--compound-master", type=Path, default=DEFAULT_COMPOUND_MASTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_dataset(observations_path: Path, compound_master_path: Path) -> pd.DataFrame:
    observations = pd.read_csv(observations_path)
    compound_master = pd.read_csv(compound_master_path, usecols=["compound_id", "canonical_smiles"])
    dataset = observations.merge(
        compound_master,
        on="compound_id",
        how="left",
        validate="many_to_one",
    )

    if dataset["canonical_smiles"].isna().any():
        missing = int(dataset["canonical_smiles"].isna().sum())
        raise ValueError(f"Missing canonical_smiles for {missing} observation rows.")
    if dataset["property_value_si"].isna().any():
        missing = int(dataset["property_value_si"].isna().sum())
        raise ValueError(f"Missing property_value_si for {missing} observation rows.")

    front_columns = [
        "observation_id",
        "compound_id",
        "canonical_smiles",
        "property_value_si",
        "unit_si",
    ]
    remaining_columns = [column for column in dataset.columns if column not in front_columns]
    return dataset[front_columns + remaining_columns]


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.observations, args.compound_master)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Saved dataset: {args.output}")
    print(f"Rows: {len(dataset)}")


if __name__ == "__main__":
    main()
