from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _find_qsar_dir(start: str | Path) -> Path:
    for candidate in Path(start).resolve().parents:
        if candidate.name == "qsar":
            return candidate
    raise RuntimeError(f"Could not locate a parent directory named 'qsar' from {start}")


QSAR_DIR = _find_qsar_dir(__file__)

DEFAULT_OBSERVATIONS = (
    QSAR_DIR / "datas" / "insulating_oil_property_database" / "processed" / "dynamic_viscosity_experimental.csv"
)
DEFAULT_COMPOUND_MASTER = (
    QSAR_DIR / "datas" / "insulating_oil_property_database" / "processed" / "compound_master.csv"
)
DEFAULT_OUTPUT = QSAR_DIR / "tasks" / "dynamic_viscosity" / "data" / "dynamic_viscosity_observations_with_smiles.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an observation-level dynamic-viscosity dataset with canonical SMILES."
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--compound-master", type=Path, default=DEFAULT_COMPOUND_MASTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_dataset(observations_path: Path, compound_master_path: Path) -> pd.DataFrame:
    observations = pd.read_csv(observations_path)
    compound_master = pd.read_csv(compound_master_path, usecols=["compound_id", "canonical_smiles"])
    dataset = observations.merge(compound_master, on="compound_id", how="left", validate="many_to_one")

    dataset["property_value_si"] = pd.to_numeric(dataset["property_value_si"], errors="coerce")
    dataset["temperature_K"] = pd.to_numeric(dataset["temperature_K"], errors="coerce")
    dataset = dataset.dropna(subset=["canonical_smiles", "property_value_si", "temperature_K"]).copy()
    dataset = dataset[(dataset["property_value_si"] > 0.0) & (dataset["temperature_K"] > 0.0)].copy()
    dataset["log_dynamic_viscosity"] = np.log(dataset["property_value_si"])

    front_columns = [
        "observation_id",
        "compound_id",
        "canonical_smiles",
        "temperature_K",
        "property_value_si",
        "log_dynamic_viscosity",
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
    print(f"Compounds: {dataset['compound_id'].nunique()}")


if __name__ == "__main__":
    main()
