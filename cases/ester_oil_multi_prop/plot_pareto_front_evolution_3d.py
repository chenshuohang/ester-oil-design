"""Plot the per-generation Pareto-front evolution in objective space."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR))

from cases.ester_oil_multi_prop.postprocess_results import (  # noqa: E402
    DEFAULT_OBJECTIVES,
    add_pareto_columns,
    generation_frame,
)
def apply_plot_style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})


PROPERTIES = ["Flash_Point", "Dynamic_Viscosity", "Dielectric_Constant"]
GRID_COLOR = "#D9DDE2"
GENERATION_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
GENERATION_MARKERS = ["o", "^", "s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=CASE_DIR / "results",
        help="Results directory containing diffusion_evo_opt.pkl and config.json.",
    )
    parser.add_argument("--generations", type=int, nargs="+", default=[0, 5, 10],
                        help="Zero-based generations to compare (default: 0 5 10).")
    parser.add_argument("--elev", type=float, default=35, help="Camera elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-75, help="Camera azimuth in degrees.")
    return parser.parse_args()


def load_fronts(results_dir: Path, selected: list[int]) -> tuple[list[pd.DataFrame], list[int]]:
    with (results_dir / "diffusion_evo_opt.pkl").open("rb") as handle:
        evo = pickle.load(handle)
    with (results_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    objective_specs = config.get("qsar_objectives") or DEFAULT_OBJECTIVES
    fronts: list[pd.DataFrame] = []
    for generation in selected:
        if generation < 0 or generation >= len(evo["pred_df"]):
            raise ValueError(f"Generation {generation} is unavailable; valid range: 0–{len(evo['pred_df']) - 1}.")
        frame = generation_frame(evo, generation, PROPERTIES)
        if "feasible" in frame.columns:
            frame = frame[frame["feasible"]].copy()
        frame = frame.drop_duplicates("SMILES", keep="first")
        frame = add_pareto_columns(frame, objective_specs)
        front = frame[frame["pareto_efficient"]].copy()
        front["generation"] = generation
        fronts.append(front)
    return fronts, selected


def style_3d_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", direction="in", width=0.5, length=3, pad=2)
    ax.zaxis.set_tick_params(which="both", direction="in", width=0.5, length=3, pad=2)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "-"
        axis.line.set_linewidth(0.5)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.3)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    fronts, generations = load_fronts(results_dir, list(dict.fromkeys(args.generations)))
    if not any(len(front) for front in fronts):
        raise RuntimeError("No non-empty Pareto fronts were found.")

    apply_plot_style()
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(5.8, 4.8))
    ax = fig.add_axes([0.04, 0.08, 0.84, 0.80], projection="3d")
    summary_rows = []
    for index, (generation, front) in enumerate(zip(generations, fronts)):
        color = GENERATION_COLORS[index % len(GENERATION_COLORS)]
        marker = GENERATION_MARKERS[index % len(GENERATION_MARKERS)]
        ax.scatter(
            front[PROPERTIES[0]], front[PROPERTIES[1]] * 1000,
            front[PROPERTIES[2]], s=42 if generation == 0 else 24,
            color=color, marker=marker, alpha=0.9, depthshade=False,
            edgecolors="white", linewidths=0.4,
            label=f"Generation {generation} (n={len(front)})",
        )
        summary_rows.append({"generation": generation, "n_pareto": len(front)})

    ax.set_xlabel("Flash point (K)", labelpad=7)
    ax.set_ylabel("Dynamic viscosity (mPa s)", labelpad=7)
    ax.set_zlabel("")
    ax.text2D(1.10, 0.55, "Dielectric constant", transform=ax.transAxes,
              rotation=90, va="center", ha="center")
    ax.set_proj_type("ortho")
    all_fronts = pd.concat(fronts, ignore_index=True)
    # Pad the actual data bounds so no selected Pareto point is clipped.
    for prop, scale, set_limit in zip(PROPERTIES, [1, 1000, 1],
                                      [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        values = all_fronts[prop] * scale
        lo, hi = values.min(), values.max()
        pad = max((hi - lo) * 0.08, abs(lo) * 0.01, 1e-6)
        set_limit(lo - pad, hi + pad)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect((1.15, 1.0, 0.95))
    style_3d_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.12), frameon=False,
              fontsize=7, labelspacing=0.7)
    fig.suptitle("Pareto fronts at generations " + ", ".join(map(str, generations)),
                 fontsize=9, y=0.98)

    suffix = "_".join(map(str, generations))
    output_dir = results_dir / "redrawn_rcparams_7pt_lw05"
    stem = f"pareto_front_generations_{suffix}_3d_7pt_lw05"
    save_figure(fig, output_dir, stem)
    all_fronts.to_csv(output_dir / f"{stem}.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
