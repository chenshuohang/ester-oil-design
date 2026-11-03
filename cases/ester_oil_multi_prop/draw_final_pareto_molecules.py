from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


RESULTS_DIR = Path(__file__).resolve().parent / "results_no_threshold"
INPUT_CSV = RESULTS_DIR / "pareto_front" / "final_generation_pareto_front.csv"
OUT_DIR = RESULTS_DIR / "pareto_front" / "final_generation_pareto_molecule_images"


def filename_safe_smiles(smiles: str, max_len: int = 48) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", smiles).strip("_")
    return safe[:max_len].strip("_") or "molecule"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if "pareto_score_norm_sum" in df.columns:
        df = df.sort_values("pareto_score_norm_sum", ascending=True)
    df = df.reset_index(drop=True)

    index_rows: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        smiles = str(row["SMILES"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            index_rows.append(
                {
                    "pareto_rank": idx + 1,
                    "SMILES": smiles,
                    "image_path": "",
                    "draw_status": "invalid_smiles",
                }
            )
            continue

        legend = (
            f"Pareto #{idx + 1} | Gen {int(row['generation'])}\n"
            f"FP {row['Flash_Point']:.1f} K | Visc {row['Dynamic_Viscosity']:.4g} | "
            f"DC {row['Dielectric_Constant']:.2f}"
        )
        image_name = f"pareto_{idx + 1:03d}_{filename_safe_smiles(smiles)}.png"
        image_path = OUT_DIR / image_name
        image = Draw.MolToImage(mol, size=(1100, 760), legend=legend)
        image.save(image_path)

        index_rows.append(
            {
                "pareto_rank": idx + 1,
                "generation": row["generation"],
                "SMILES": smiles,
                "Flash_Point": row["Flash_Point"],
                "Dynamic_Viscosity": row["Dynamic_Viscosity"],
                "Dielectric_Constant": row["Dielectric_Constant"],
                "pareto_score_norm_sum": row.get("pareto_score_norm_sum", ""),
                "image_path": str(image_path.relative_to(RESULTS_DIR)),
                "draw_status": "ok",
            }
        )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(OUT_DIR / "final_generation_pareto_molecule_image_index.csv", index=False)
    print(f"Input rows: {len(df)}")
    print(f"Images drawn: {(index_df['draw_status'] == 'ok').sum()}")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
