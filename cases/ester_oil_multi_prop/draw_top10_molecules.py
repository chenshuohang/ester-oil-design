from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PKL_PATH = RESULTS_DIR / "diffusion_evo_opt.pkl"
CONFIG_PATH = RESULTS_DIR / "config.json"
OUT_DIR = RESULTS_DIR / "molecule_grids"
TOP_K = 10
DEDUPLICATE_SMILES = True


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with PKL_PATH.open("rb") as f:
        evo = pickle.load(f)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    objective_specs = config["qsar_objectives"]
    properties = [spec["name"] for spec in objective_specs]
    directions = {spec["name"]: spec.get("direction", "minimize") for spec in objective_specs}
    targets = {spec["name"]: spec.get("target") for spec in objective_specs}

    def objective_order_values(series: pd.Series, objective: str) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        direction = directions[objective]
        if direction == "maximize":
            return -values
        if direction == "target":
            if targets[objective] is None:
                raise ValueError(f"{objective} uses target direction but no target is configured")
            return (values - float(targets[objective])).abs()
        return values

    def generation_frame(generation: int) -> pd.DataFrame:
        frame = evo["pred_df"][generation].copy().reset_index(drop=True)
        frame["SMILES"] = evo["smiles"][generation]
        frame.insert(0, "generation", generation)
        for prop in properties:
            frame[prop] = pd.to_numeric(frame[prop], errors="coerce")
        if DEDUPLICATE_SMILES:
            frame = frame.drop_duplicates("SMILES", keep="first").reset_index(drop=True)
        return frame

    index_rows = []
    for generation in range(len(evo["pred_df"])):
        frame = generation_frame(generation)
        for objective in properties:
            ranked = frame.assign(_sort_value=objective_order_values(frame[objective], objective))
            ranked = ranked.replace([np.inf, -np.inf], np.nan).dropna(subset=["_sort_value"])
            ranked = ranked.sort_values("_sort_value", ascending=True).head(TOP_K).reset_index(drop=True)
            if ranked.empty:
                continue

            mols = []
            legends = []
            drawn_rows = []
            for rank, row in ranked.iterrows():
                smiles = str(row["SMILES"])
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mols.append(mol)
                legends.append(
                    f"#{rank + 1} {objective}\n"
                    f"FP {row['Flash_Point']:.1f} K | Vis {row['Dynamic_Viscosity']:.3g}\n"
                    f"DC {row['Dielectric_Constant']:.2f}"
                )
                drawn_rows.append(
                    {
                        "generation": generation,
                        "selected_by": objective,
                        "rank": rank + 1,
                        "SMILES": smiles,
                        **{prop: row[prop] for prop in properties},
                    }
                )

            if not mols:
                continue

            objective_dir = OUT_DIR / slugify(objective)
            objective_dir.mkdir(parents=True, exist_ok=True)
            image_path = objective_dir / f"generation_{generation:03d}_top{TOP_K}.png"
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=5,
                subImgSize=(360, 260),
                legends=legends,
                useSVG=False,
            )
            img.save(str(image_path))

            for item in drawn_rows:
                item["image_path"] = str(image_path.relative_to(RESULTS_DIR))
                index_rows.append(item)

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(OUT_DIR / "top10_molecule_grid_index.csv", index=False)
    print(f"Generated {index_df['image_path'].nunique()} grid images")
    print(OUT_DIR)
    print(index_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
