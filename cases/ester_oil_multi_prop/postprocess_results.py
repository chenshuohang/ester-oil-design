from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OBJECTIVES = [
    {"name": "Flash_Point", "direction": "maximize"},
    {"name": "Dynamic_Viscosity", "direction": "minimize"},
    {"name": "Dielectric_Constant", "direction": "minimize"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess ester oil multi-objective evolution results.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing diffusion_evo_opt.pkl.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    return parser.parse_args()


def load_config(results_dir: Path, explicit_path: Path | None) -> dict[str, Any]:
    candidates = [explicit_path, results_dir / "config.json", results_dir.parent / "config.json"]
    for path in candidates:
        if path is not None and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {"qsar_objectives": DEFAULT_OBJECTIVES}


def objective_sort_values(values: pd.Series, spec: dict[str, Any]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    direction = str(spec.get("direction", "minimize")).lower()
    if direction == "maximize":
        return -numeric
    if direction == "target":
        target = spec.get("target")
        if target is None:
            raise ValueError(f"{spec['name']} uses target direction but no target is configured.")
        return (numeric - float(target)).abs()
    return numeric


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    finite = np.isfinite(costs).all(axis=1)
    efficient = np.zeros(costs.shape[0], dtype=bool)
    finite_indices = np.where(finite)[0]
    finite_costs = costs[finite]
    for local_idx, candidate in enumerate(finite_costs):
        dominated = np.all(finite_costs <= candidate, axis=1) & np.any(finite_costs < candidate, axis=1)
        if not dominated.any():
            efficient[finite_indices[local_idx]] = True
    return efficient


def generation_frame(evo: dict[str, Any], generation: int, properties: list[str]) -> pd.DataFrame:
    frame = evo["pred_df"][generation].copy().reset_index(drop=True)
    if "SMILES" not in frame.columns:
        frame.insert(0, "SMILES", evo["smiles"][generation])
    frame.insert(0, "generation", generation)
    for prop in properties:
        frame[prop] = pd.to_numeric(frame[prop], errors="coerce")
    if "constraints" in evo:
        constraints = np.asarray(evo["constraints"][generation])
        frame["feasible"] = np.all(constraints < 0, axis=1)
        for idx in range(constraints.shape[1]):
            frame[f"constraint_{idx}"] = constraints[:, idx]
    if "constraint_scores" in evo:
        constraint_scores = np.asarray(evo["constraint_scores"][generation])
        if constraint_scores.shape[1] >= 1:
            frame["unique_ok"] = constraint_scores[:, 0].astype(bool)
        if constraint_scores.shape[1] >= 2:
            frame["substructure_ok"] = constraint_scores[:, 1].astype(bool)
        if constraint_scores.shape[1] >= 3:
            frame["tanimoto"] = constraint_scores[:, 2]
    return frame


def add_pareto_columns(frame: pd.DataFrame, objective_specs: list[dict[str, Any]]) -> pd.DataFrame:
    out = frame.copy()
    costs = np.column_stack([objective_sort_values(out[spec["name"]], spec).to_numpy() for spec in objective_specs])
    out["pareto_efficient"] = is_pareto_efficient(costs)
    out["pareto_rank_proxy"] = costs.sum(axis=1)
    return out


def save_pairwise_scatter(frame: pd.DataFrame, properties: list[str], path: Path, title: str) -> None:
    n = len(properties)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 3.5 * n), squeeze=False)
    colors = np.where(frame.get("pareto_efficient", False), "#d62728", "#4c78a8")
    for row, y_col in enumerate(properties):
        for col, x_col in enumerate(properties):
            ax = axes[row][col]
            if row == col:
                ax.hist(frame[x_col].dropna(), bins=24, color="#4c78a8", alpha=0.85)
            else:
                ax.scatter(frame[x_col], frame[y_col], s=22, c=colors, alpha=0.75, linewidths=0)
            if row == n - 1:
                ax.set_xlabel(x_col)
            if col == 0:
                ax.set_ylabel(y_col)
            ax.grid(alpha=0.2)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_3d_scatter(frame: pd.DataFrame, properties: list[str], path: Path, title: str) -> None:
    if len(properties) < 3:
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(frame.get("pareto_efficient", False), "#d62728", "#4c78a8")
    ax.scatter(frame[properties[0]], frame[properties[1]], frame[properties[2]], c=colors, s=28, alpha=0.8)
    ax.set_xlabel(properties[0])
    ax.set_ylabel(properties[1])
    ax.set_zlabel(properties[2])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary_outputs(results_dir: Path, frames: list[pd.DataFrame], objective_specs: list[dict[str, Any]]) -> None:
    properties = [spec["name"] for spec in objective_specs]
    best_rows = []
    top10_rows = []
    for frame in frames:
        generation = int(frame["generation"].iloc[0])
        unique = frame.drop_duplicates("SMILES", keep="first").reset_index(drop=True)
        for spec in objective_specs:
            prop = spec["name"]
            ranked = unique.assign(_sort_value=objective_sort_values(unique[prop], spec))
            ranked = ranked.replace([np.inf, -np.inf], np.nan).dropna(subset=["_sort_value"])
            ranked = ranked.sort_values("_sort_value", ascending=True)
            if ranked.empty:
                continue
            best = ranked.iloc[0].to_dict()
            best_rows.append({"generation": generation, "objective": prop, **{k: best[k] for k in ["SMILES", *properties]}})
            means = ranked.head(10)[properties].mean(numeric_only=True).to_dict()
            top10_rows.append({"generation": generation, "objective": prop, **means})

    best_df = pd.DataFrame(best_rows)
    top10_df = pd.DataFrame(top10_rows)
    best_df.to_csv(results_dir / "best_molecule_by_objective_per_generation.csv", index=False)
    top10_df.to_csv(results_dir / "top10_average_by_objective_per_generation.csv", index=False)

    best_wide = best_df.pivot(index="generation", columns="objective", values=properties)
    best_wide.to_csv(results_dir / "best_molecule_properties_wide.csv")
    top10_wide = top10_df.pivot(index="generation", columns="objective", values=properties)
    top10_wide.to_csv(results_dir / "top10_average_properties_wide.csv")

    for df, filename, title in [
        (best_df, "best_objective_value_per_generation.png", "Best objective value per generation"),
        (top10_df, "top10_average_objective_value_per_generation.png", "Top-10 average objective value per generation"),
    ]:
        fig, axes = plt.subplots(len(properties), 1, figsize=(9, 3 * len(properties)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, prop in zip(axes, properties):
            part = df[df["objective"] == prop]
            ax.plot(part["generation"], part[prop], marker="o", linewidth=1.8)
            ax.set_ylabel(prop)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("generation")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=220)
        plt.close(fig)


def write_final_outputs(results_dir: Path, final_frame: pd.DataFrame, objective_specs: list[dict[str, Any]]) -> None:
    properties = [spec["name"] for spec in objective_specs]
    out_dir = results_dir / "final_generation_properties"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_frame.to_csv(out_dir / "final_generation_predictions.csv", index=False)
    final_frame[properties].describe().T.to_csv(out_dir / "final_generation_property_summary.csv")
    save_pairwise_scatter(
        final_frame,
        properties,
        out_dir / "final_generation_property_pairwise_scatter.png",
        "Final generation properties",
    )
    save_3d_scatter(
        final_frame,
        properties,
        out_dir / "final_generation_property_3d_scatter.png",
        "Final generation properties",
    )


def write_pareto_outputs(results_dir: Path, frames: list[pd.DataFrame], objective_specs: list[dict[str, Any]]) -> None:
    out_dir = results_dir / "pareto_front"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_frame = frames[-1]
    final_pareto = final_frame[final_frame["pareto_efficient"]].sort_values("pareto_rank_proxy")
    final_pareto.to_csv(out_dir / "final_generation_pareto_front.csv", index=False)

    all_unique = pd.concat(frames, ignore_index=True).drop_duplicates("SMILES", keep="last").reset_index(drop=True)
    all_unique = add_pareto_columns(all_unique.drop(columns=["pareto_efficient", "pareto_rank_proxy"], errors="ignore"), objective_specs)
    global_pareto = all_unique[all_unique["pareto_efficient"]].sort_values("pareto_rank_proxy")
    global_pareto.to_csv(out_dir / "global_unique_pareto_front.csv", index=False)

    properties = [spec["name"] for spec in objective_specs]
    save_pairwise_scatter(
        final_pareto.head(200) if len(final_pareto) else final_frame,
        properties,
        out_dir / "final_generation_pareto_front_pairwise_scatter.png",
        "Final generation Pareto front",
    )
    save_pairwise_scatter(
        global_pareto.head(300) if len(global_pareto) else all_unique,
        properties,
        out_dir / "global_unique_pareto_front_pairwise_scatter.png",
        "Global unique Pareto front",
    )


def write_first10_outputs(results_dir: Path, frames: list[pd.DataFrame], objective_specs: list[dict[str, Any]]) -> None:
    out_dir = results_dir / "first10_generations_pareto"
    out_dir.mkdir(parents=True, exist_ok=True)
    first_frames = frames[: min(10, len(frames))]
    all_molecules = pd.concat(first_frames, ignore_index=True)
    all_molecules.to_csv(out_dir / "first10_generations_all_molecules.csv", index=False)
    unique = all_molecules.drop_duplicates("SMILES", keep="last").reset_index(drop=True)
    unique = add_pareto_columns(unique.drop(columns=["pareto_efficient", "pareto_rank_proxy"], errors="ignore"), objective_specs)
    unique.to_csv(out_dir / "first10_generations_unique_molecules.csv", index=False)
    unique[unique["pareto_efficient"]].sort_values("pareto_rank_proxy").to_csv(
        out_dir / "first10_generations_pareto_front.csv", index=False
    )
    save_pairwise_scatter(
        unique,
        [spec["name"] for spec in objective_specs],
        out_dir / "first10_generations_pareto_pairwise_scatter.png",
        "First generations unique molecules",
    )


def write_quality_outputs(results_dir: Path, frames: list[pd.DataFrame]) -> None:
    out_dir = results_dir / "generation_quality_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in frames:
        smiles = frame["SMILES"].astype(str)
        n_total = len(smiles)
        n_unique = smiles.nunique()
        rows.append(
            {
                "generation": int(frame["generation"].iloc[0]),
                "n_total": n_total,
                "n_unique": n_unique,
                "unique_fraction": n_unique / n_total if n_total else np.nan,
                "n_complete_predictions": int(frame.dropna().shape[0]),
                "complete_prediction_fraction": float(frame.dropna().shape[0] / n_total) if n_total else np.nan,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "generation_quality_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(metrics["generation"], metrics["unique_fraction"], marker="o", label="unique fraction")
    ax.plot(metrics["generation"], metrics["complete_prediction_fraction"], marker="o", label="complete predictions")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("generation")
    ax.set_ylabel("fraction")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "generation_quality_metrics_barplot.png", dpi=220)
    plt.close(fig)


def write_generation_trend_outputs(
    results_dir: Path,
    frames: list[pd.DataFrame],
    objective_specs: list[dict[str, Any]],
) -> None:
    out_dir = results_dir / "generation_trends"
    out_dir.mkdir(parents=True, exist_ok=True)
    properties = [spec["name"] for spec in objective_specs]
    all_frames = pd.concat(frames, ignore_index=True)
    all_frames.to_csv(out_dir / "all_generation_predictions.csv", index=False)

    summary_rows = []
    top_rows = []
    for frame in frames:
        generation = int(frame["generation"].iloc[0])
        unique = frame.drop_duplicates("SMILES", keep="first").reset_index(drop=True)
        row: dict[str, Any] = {
            "generation": generation,
            "n_total": len(frame),
            "n_unique": unique["SMILES"].nunique(),
            "pareto_count": int(frame["pareto_efficient"].sum()),
            "pareto_fraction": float(frame["pareto_efficient"].mean()),
        }
        for spec in objective_specs:
            prop = spec["name"]
            values = pd.to_numeric(frame[prop], errors="coerce")
            row[f"{prop}_min"] = values.min()
            row[f"{prop}_q25"] = values.quantile(0.25)
            row[f"{prop}_median"] = values.median()
            row[f"{prop}_mean"] = values.mean()
            row[f"{prop}_q75"] = values.quantile(0.75)
            row[f"{prop}_max"] = values.max()
            row[f"{prop}_std"] = values.std()

            ranked = unique.assign(_sort_value=objective_sort_values(unique[prop], spec))
            ranked = ranked.replace([np.inf, -np.inf], np.nan).dropna(subset=["_sort_value"])
            ranked = ranked.sort_values("_sort_value", ascending=True).head(20).reset_index(drop=True)
            for rank, top_row in ranked.iterrows():
                top_rows.append(
                    {
                        "generation": generation,
                        "objective": prop,
                        "rank": rank + 1,
                        "SMILES": top_row["SMILES"],
                        **{name: top_row[name] for name in properties},
                    }
                )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "generation_property_summary_wide.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out_dir / "top20_by_objective_per_generation.csv", index=False)

    long_rows = []
    for _, row in summary.iterrows():
        for prop in properties:
            long_rows.append(
                {
                    "generation": row["generation"],
                    "property": prop,
                    "min": row[f"{prop}_min"],
                    "q25": row[f"{prop}_q25"],
                    "median": row[f"{prop}_median"],
                    "mean": row[f"{prop}_mean"],
                    "q75": row[f"{prop}_q75"],
                    "max": row[f"{prop}_max"],
                    "std": row[f"{prop}_std"],
                }
            )
    pd.DataFrame(long_rows).to_csv(out_dir / "generation_property_summary_long.csv", index=False)

    fig, axes = plt.subplots(len(properties), 1, figsize=(10, 3.2 * len(properties)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, prop in zip(axes, properties):
        ax.plot(summary["generation"], summary[f"{prop}_median"], marker="o", label="median", color="#1f77b4")
        ax.plot(summary["generation"], summary[f"{prop}_mean"], marker="s", label="mean", color="#ff7f0e")
        ax.fill_between(
            summary["generation"].to_numpy(),
            summary[f"{prop}_q25"].to_numpy(),
            summary[f"{prop}_q75"].to_numpy(),
            color="#1f77b4",
            alpha=0.15,
            label="IQR",
        )
        ax.plot(summary["generation"], summary[f"{prop}_min"], linestyle="--", linewidth=1, color="#2ca02c", label="min")
        ax.plot(summary["generation"], summary[f"{prop}_max"], linestyle="--", linewidth=1, color="#d62728", label="max")
        ax.set_ylabel(prop)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncol=5, fontsize=8)
    axes[-1].set_xlabel("generation")
    fig.suptitle("Property distribution statistics by generation")
    fig.tight_layout()
    fig.savefig(out_dir / "property_distribution_stats_by_generation.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(len(properties), 1, figsize=(10, 3.1 * len(properties)), sharex=True)
    axes = np.atleast_1d(axes)
    generations = sorted(all_frames["generation"].unique())
    for ax, prop in zip(axes, properties):
        data = [all_frames.loc[all_frames["generation"] == gen, prop].dropna().to_numpy() for gen in generations]
        ax.boxplot(data, positions=generations, widths=0.55, showfliers=False)
        ax.set_ylabel(prop)
        ax.grid(alpha=0.25, axis="y")
    axes[-1].set_xlabel("generation")
    fig.suptitle("Per-generation property distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "property_boxplots_by_generation.png", dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(summary["generation"], summary["pareto_count"], marker="o", color="#d62728", label="Pareto count")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("Pareto count")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(summary["generation"], summary["n_unique"], marker="s", color="#4c78a8", label="Unique SMILES")
    ax2.set_ylabel("Unique SMILES")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_and_uniqueness_by_generation.png", dpi=220)
    plt.close(fig)


def write_constrained_comparison_outputs(
    results_dir: Path,
    frames: list[pd.DataFrame],
    objective_specs: list[dict[str, Any]],
) -> None:
    if "feasible" not in frames[0].columns:
        return

    out_dir = results_dir / "constrained_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    properties = [spec["name"] for spec in objective_specs]

    constrained_frames = [frame[frame["feasible"]].copy().reset_index(drop=True) for frame in frames]
    constrained_all = pd.concat(constrained_frames, ignore_index=True)
    constrained_all.to_csv(out_dir / "all_feasible_generation_predictions.csv", index=False)

    best_rows = []
    top10_rows = []
    comparison_rows = []
    feasibility_rows = []

    for frame, feasible_frame in zip(frames, constrained_frames):
        generation = int(frame["generation"].iloc[0])
        feasibility_rows.append(
            {
                "generation": generation,
                "n_total": len(frame),
                "n_feasible": len(feasible_frame),
                "feasible_fraction": len(feasible_frame) / len(frame) if len(frame) else np.nan,
                "n_unique_feasible": feasible_frame["SMILES"].nunique() if len(feasible_frame) else 0,
                "unique_ok_count": int(frame.get("unique_ok", pd.Series(False, index=frame.index)).sum()),
                "substructure_ok_count": int(frame.get("substructure_ok", pd.Series(False, index=frame.index)).sum()),
                "tanimoto_ok_count": int((frame.get("tanimoto", pd.Series(np.nan, index=frame.index)) > 0.5).sum()),
            }
        )

        all_unique = frame.drop_duplicates("SMILES", keep="first").reset_index(drop=True)
        feasible_unique = feasible_frame.drop_duplicates("SMILES", keep="first").reset_index(drop=True)
        for spec in objective_specs:
            objective = spec["name"]

            all_ranked = all_unique.assign(_sort_value=objective_sort_values(all_unique[objective], spec))
            all_ranked = all_ranked.replace([np.inf, -np.inf], np.nan).dropna(subset=["_sort_value"])
            all_ranked = all_ranked.sort_values("_sort_value", ascending=True).reset_index(drop=True)

            feasible_ranked = feasible_unique.assign(_sort_value=objective_sort_values(feasible_unique[objective], spec))
            feasible_ranked = feasible_ranked.replace([np.inf, -np.inf], np.nan).dropna(subset=["_sort_value"])
            feasible_ranked = feasible_ranked.sort_values("_sort_value", ascending=True).reset_index(drop=True)

            if not feasible_ranked.empty:
                best = feasible_ranked.iloc[0]
                best_rows.append(
                    {
                        "generation": generation,
                        "objective": objective,
                        "SMILES": best["SMILES"],
                        **{prop: best[prop] for prop in properties},
                    }
                )
                top10_means = feasible_ranked.head(10)[properties].mean(numeric_only=True).to_dict()
                top10_rows.append({"generation": generation, "objective": objective, **top10_means})
            else:
                top10_means = {prop: np.nan for prop in properties}

            all_best_value = all_ranked.iloc[0][objective] if not all_ranked.empty else np.nan
            constrained_best_value = feasible_ranked.iloc[0][objective] if not feasible_ranked.empty else np.nan
            all_top10_value = all_ranked.head(10)[objective].mean() if not all_ranked.empty else np.nan
            constrained_top10_value = feasible_ranked.head(10)[objective].mean() if not feasible_ranked.empty else np.nan
            comparison_rows.append(
                {
                    "generation": generation,
                    "objective": objective,
                    "n_all_unique": len(all_unique),
                    "n_feasible_unique": len(feasible_unique),
                    "all_best_objective_value": all_best_value,
                    "feasible_best_objective_value": constrained_best_value,
                    "all_top10_mean_objective_value": all_top10_value,
                    "feasible_top10_mean_objective_value": constrained_top10_value,
                    "best_delta_feasible_minus_all": constrained_best_value - all_best_value,
                    "top10_delta_feasible_minus_all": constrained_top10_value - all_top10_value,
                }
            )

    best_df = pd.DataFrame(best_rows)
    top10_df = pd.DataFrame(top10_rows)
    comparison = pd.DataFrame(comparison_rows)
    feasibility = pd.DataFrame(feasibility_rows)

    best_df.to_csv(out_dir / "feasible_best_molecule_by_objective_per_generation.csv", index=False)
    top10_df.to_csv(out_dir / "feasible_top10_average_by_objective_per_generation.csv", index=False)
    comparison.to_csv(out_dir / "feasible_vs_all_objective_comparison.csv", index=False)
    feasibility.to_csv(out_dir / "constraint_feasibility_by_generation.csv", index=False)

    final_feasible = constrained_frames[-1]
    final_feasible.to_csv(out_dir / "final_generation_feasible_predictions.csv", index=False)
    if len(final_feasible):
        final_feasible[properties].describe().T.to_csv(out_dir / "final_generation_feasible_property_summary.csv")

    for column_pair, filename, title in [
        (
            ("all_best_objective_value", "feasible_best_objective_value"),
            "feasible_vs_all_best_objective_value_per_generation.png",
            "Best objective value: all vs feasible",
        ),
        (
            ("all_top10_mean_objective_value", "feasible_top10_mean_objective_value"),
            "feasible_vs_all_top10_objective_value_per_generation.png",
            "Top-10 mean objective value: all vs feasible",
        ),
    ]:
        fig, axes = plt.subplots(len(objective_specs), 1, figsize=(9, 3 * len(objective_specs)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, spec in zip(axes, objective_specs):
            part = comparison[comparison["objective"] == spec["name"]]
            ax.plot(part["generation"], part[column_pair[0]], marker="o", label="all")
            ax.plot(part["generation"], part[column_pair[1]], marker="s", label="feasible only")
            ax.set_ylabel(spec["name"])
            ax.grid(alpha=0.25)
            ax.legend()
        axes[-1].set_xlabel("generation")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(feasibility["generation"], feasibility["feasible_fraction"], marker="o", label="feasible fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("generation")
    ax.set_ylabel("fraction")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "constraint_feasible_fraction_by_generation.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    with (results_dir / "diffusion_evo_opt.pkl").open("rb") as handle:
        evo = pickle.load(handle)
    config = load_config(results_dir, args.config)
    objective_specs = config.get("qsar_objectives") or DEFAULT_OBJECTIVES
    properties = [spec["name"] for spec in objective_specs]

    if not (results_dir / "config.json").exists() and config:
        with (results_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    frames = [add_pareto_columns(generation_frame(evo, idx, properties), objective_specs) for idx in range(len(evo["pred_df"]))]
    write_summary_outputs(results_dir, frames, objective_specs)
    write_final_outputs(results_dir, frames[-1], objective_specs)
    write_pareto_outputs(results_dir, frames, objective_specs)
    write_first10_outputs(results_dir, frames, objective_specs)
    write_quality_outputs(results_dir, frames)
    write_generation_trend_outputs(results_dir, frames, objective_specs)
    write_constrained_comparison_outputs(results_dir, frames, objective_specs)
    print(f"Postprocessed {len(frames)} generations into {results_dir}")


if __name__ == "__main__":
    main()
