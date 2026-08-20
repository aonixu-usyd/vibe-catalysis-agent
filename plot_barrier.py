#!/usr/bin/env python3
"""Render one NEB/CI-NEB result as a CatMAP-style energy diagram."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms


def _smooth_segment(x0, x1, y0, y1, n=80):
    t = np.linspace(0.0, 1.0, n)
    smooth = 0.5 - 0.5 * np.cos(np.pi * t)
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * smooth


def plot_barrier(result_path, output, initial_label="Initial state", final_label="Final state"):
    data = json.loads(Path(result_path).read_text())
    activation_energy = float(data["forward_barrier_eV"])
    reaction_energy = float(data["reaction_energy_eV"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    color = "#1756D8"
    ax.hlines(0.0, 0.15, 0.85, color=color, lw=2.2)
    x1, y1 = _smooth_segment(0.85, 1.5, 0.0, activation_energy)
    x2, y2 = _smooth_segment(1.5, 2.15, activation_energy, reaction_energy)
    ax.plot(x1, y1, color=color, lw=2.2)
    ax.plot(x2, y2, color=color, lw=2.2)
    ax.hlines(reaction_energy, 2.15, 2.85, color=color, lw=2.2)

    pad = max(0.08, 0.035 * max(abs(activation_energy), abs(reaction_energy), 1.0))
    ax.text(0.5, pad, initial_label, ha="center", va="bottom")
    ax.text(1.5, activation_energy + pad,
            f"TS candidate\n$E_a$ = {activation_energy:.2f} eV",
            ha="center", va="bottom")
    ax.text(2.5, reaction_energy + pad, final_label, ha="center", va="bottom")
    ax.annotate(f"$\\Delta E$ = {reaction_energy:+.2f} eV", xy=(2.5, reaction_energy),
                xytext=(2.5, reaction_energy - 4 * pad), ha="center", va="top")

    ax.set_xlim(0, 3)
    ax.set_ylim(min(0.0, reaction_energy) - max(0.45, 6 * pad),
                max(activation_energy, reaction_energy, 0.0) + max(0.55, 7 * pad))
    ax.set_xticks([])
    ax.set_ylabel("Relative electronic energy (eV)")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300}),
                              ("tiff", {"dpi": 600})):
        fig.savefig(output.with_suffix("." + extension), bbox_inches="tight", **kwargs)
    plt.close(fig)

    with output.with_name(output.name + "_source_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state", "relative_energy_eV"])
        writer.writeheader()
        writer.writerows([
            {"state": initial_label, "relative_energy_eV": 0.0},
            {"state": "TS candidate", "relative_energy_eV": activation_energy},
            {"state": final_label, "relative_energy_eV": reaction_energy},
        ])
    output.with_name(output.name + "_caption.txt").write_text(
        f"CatMAP-style {data.get('method', 'NEB')} electronic-energy diagram. "
        f"Forward barrier: {activation_energy:.3f} eV; reaction energy: "
        f"{reaction_energy:+.3f} eV. The highest climbing image is a transition-state "
        "candidate; validate with saddle refinement and a one-imaginary-mode frequency "
        "calculation.\n"
    )


def plot_top_views(initial_path, transition_state_path, final_path, output,
                   initial_label="Initial state", final_label="Final state"):
    """Render the optimized endpoints and TS candidate along the surface normal."""
    paths = [Path(initial_path), Path(transition_state_path), Path(final_path)]
    atoms = [read(path, -1) for path in paths]
    labels = [initial_label, "TS candidate", final_label]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True)
    for ax, structure, label in zip(axes, atoms, labels):
        plot_atoms(structure, ax=ax, rotation="0x,0y,0z", show_unit_cell=1, radii=0.72)
        ax.set_title(label, fontsize=10, pad=8)
        ax.set_facecolor("#F8FAFC")
    fig.suptitle("NEB structures · top view", fontsize=12, weight="bold")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300}),
                              ("tiff", {"dpi": 600})):
        fig.savefig(output.with_suffix("." + extension), bbox_inches="tight", **kwargs)
    plt.close(fig)
    output.with_name(output.name + "_manifest.json").write_text(json.dumps({
        "view": "top", "rotation": "0x,0y,0z",
        "panels": [{"label": label, "structure": str(path.resolve())}
                   for label, path in zip(labels, paths)],
    }, indent=2))
    output.with_name(output.name + "_caption.txt").write_text(
        "Top views of the relaxed initial state, highest climbing-image transition-state "
        "candidate, and relaxed final state. ASE standard element colours and radii are "
        "used; the periodic unit cell is shown.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-label", default="Initial state")
    parser.add_argument("--final-label", default="Final state")
    parser.add_argument("--initial-structure", type=Path)
    parser.add_argument("--transition-state-structure", type=Path)
    parser.add_argument("--final-structure", type=Path)
    args = parser.parse_args()
    plot_barrier(args.result, args.output, args.initial_label, args.final_label)
    structure_paths = (args.initial_structure, args.transition_state_structure, args.final_structure)
    if any(structure_paths) and not all(structure_paths):
        parser.error("top views require --initial-structure, --transition-state-structure, and --final-structure")
    if all(structure_paths):
        plot_top_views(*structure_paths, args.output.with_name("structure_top_views"),
                       args.initial_label, args.final_label)
    print(args.output)


if __name__ == "__main__":
    main()
