#!/usr/bin/env python3
"""Create polished energy summaries and top views for UMA prediction jobs."""

from __future__ import annotations

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

ASE_GUI_RADII = 0.90

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.linewidth": 1.8,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "legend.fontsize": 11,
    "savefig.dpi": 300,
})


INK = "#172033"
MUTED = "#667085"
BLUE = "#2563EB"
TEAL = "#0F9F8F"
RED = "#DC4C64"
GRID = "#E4E7EC"
PAPER = "#F8FAFC"
CHE_H_COUNTS = {"CO": 0, "CHO": 1, "COH": 1, "CHOH": 2, "CH2OH": 3}
KB_EV_PER_K = 8.617333262e-5


def load_job(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    with (path / "candidates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = []
    for row in rows:
        if row["geometry_status"] != "accepted" or not row["relaxed_adsorption_eV"]:
            continue
        row["energy"] = float(row["relaxed_adsorption_eV"])
        row["total_energy"] = float(row["relaxed_total_eV"])
        accepted.append(row)
    return {"path": path, "summary": summary, "accepted": accepted}


def best_by_site(job: dict) -> list[dict]:
    selected = {}
    for row in job["accepted"]:
        if row["site"] not in selected or row["energy"] < selected[row["site"]]["energy"]:
            selected[row["site"]] = row
    return sorted(selected.values(), key=lambda row: row["energy"])


def structure_for(job: dict, row: dict):
    return read(job["path"] / "structures" / f"{row['candidate']}_final.extxyz")


def draw_top_view(ax, atoms, title: str) -> None:
    # Use ASE's own renderer and standard element colours/radii.  With no
    # rotation the view is along +z, i.e. the slab top view.
    plot_atoms(
        atoms,
        ax=ax,
        rotation="0x,0y,0z",
        show_unit_cell=1,
        radii=ASE_GUI_RADII,
    )
    ax.set_title(title, fontsize=12, color=INK, pad=9, weight="bold")
    ax.set_facecolor(PAPER)


def style_energy_axis(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0)
    ax.axhline(0, color=INK, linewidth=1.4, zorder=1)
    ax.set_ylabel("Adsorption energy, eV", color=INK)


def compute_che_states(labels: list[str], totals: list[float], h2_energy: float,
                       potential_v: float = 0.0, ph: float = 0.0,
                       temperature_k: float = 298.15) -> tuple[int, list[float], list[dict]]:
    """Return CHE energies relative to the least-hydrogenated supplied state."""
    reference_index = min(range(len(labels)), key=lambda i: CHE_H_COUNTS[labels[i]])
    reference_h = CHE_H_COUNTS[labels[reference_index]]
    ph_term = KB_EV_PER_K * temperature_k * np.log(10.0) * ph
    energies, rows = [], []
    for label, total in zip(labels, totals):
        delta_h = CHE_H_COUNTS[label] - reference_h
        electronic = total - totals[reference_index] - 0.5 * delta_h * h2_energy
        corrected = electronic + delta_h * (potential_v + ph_term)
        energies.append(corrected)
        rows.append({"state": f"{label}*", "hydrogen_count_relative": delta_h,
                     "relaxed_total_eV": total, "delta_E_CHE_eV": electronic,
                     "delta_G_CHE_approx_eV": corrected})
    return reference_index, energies, rows


def classify_che_comparison(labels: list[str]) -> str:
    """Choose a chart from reaction topology, never merely from state count."""
    if len(labels) == 2:
        return "single_reaction"
    hydrogen_counts = [CHE_H_COUNTS[label] for label in labels]
    if len(labels) >= 3 and all(
        hydrogen_counts[i + 1] > hydrogen_counts[i]
        for i in range(len(hydrogen_counts) - 1)
    ):
        return "sequential_path"
    return "branch_comparison"


def reaction_rows(labels: list[str], state_energies: list[float],
                  reference_index: int, topology: str) -> list[dict]:
    if topology == "sequential_path":
        edges = [(i - 1, i) for i in range(1, len(labels))]
    else:
        edges = [(reference_index, i) for i in range(len(labels)) if i != reference_index]
    return [
        {"reaction": f"{labels[a]}* -> {labels[b]}*", "reactant": f"{labels[a]}*",
         "product": f"{labels[b]}*", "reaction_energy_eV": state_energies[b] - state_energies[a]}
        for a, b in edges
    ]


def render_single_job(job_dir: Path, output: Path | None = None) -> Path:
    job = load_job(job_dir)
    summary = job["summary"]
    rows = best_by_site(job)
    if not rows:
        raise ValueError(f"No accepted candidates in {job_dir}")
    crystal = summary.get("crystal_structure", "fcc" if summary.get("facet") in {"111", "100", "110"} else "")
    if summary.get("structure_source", {}).get("source") == "uploaded_structure":
        surface = f"{summary['metal']} · uploaded slab"
    else:
        surface = f"{summary['metal']} {crystal}({summary['facet'].replace('m', '-')})".replace("  ", " ")
    output = output or job_dir / "energy_and_topviews.png"
    count = min(len(rows), 4)
    fig = plt.figure(figsize=(11.8, 7.0), facecolor="white")
    grid = fig.add_gridspec(2, max(count, 2), height_ratios=(1.45, 1), hspace=0.42, wspace=0.18)
    fig.subplots_adjust(top=0.84, bottom=0.09)
    energy_ax = fig.add_subplot(grid[0, :])
    fig.suptitle(f"{summary['adsorbate']} adsorption · {surface}", x=0.06, y=0.965,
                 ha="left", fontsize=19, color=INK, weight="bold")
    fig.text(0.06, 0.916, "FAIR-Chem UMA · relaxed ASE structures · lower is more stable",
             color=MUTED, fontsize=11)

    if len(rows) == 1:
        value = rows[0]["energy"]
        energy_ax.axis("off")
        energy_ax.text(0.03, 0.58, f"{value:+.3f}", transform=energy_ax.transAxes,
                       fontsize=46, color=BLUE, weight="bold", va="center")
        energy_ax.text(0.305, 0.58, "eV", transform=energy_ax.transAxes,
                       fontsize=19, color=MUTED, va="center")
        energy_ax.text(0.035, 0.30, f"{rows[0]['site']} · {rows[0]['anchor']}-down · {rows[0]['azimuth_deg']}°",
                       transform=energy_ax.transAxes, fontsize=12, color=INK)
        mode = "single_energy"
    else:
        labels = [row["site"] for row in rows]
        values = [row["energy"] for row in rows]
        colors = [BLUE if i == 0 else TEAL for i in range(len(rows))]
        bars = energy_ax.bar(labels, values, color=colors, width=0.62, zorder=2)
        style_energy_axis(energy_ax)
        for bar, value in zip(bars, values):
            va = "top" if value < 0 else "bottom"
            offset = -5 if value < 0 else 5
            energy_ax.annotate(f"{value:+.3f}", (bar.get_x() + bar.get_width() / 2, value),
                               xytext=(0, offset), textcoords="offset points", ha="center", va=va,
                               fontsize=11, color=INK, weight="bold")
        energy_ax.set_title("Lowest accepted orientation at each adsorption site", loc="left",
                            fontsize=13, color=INK, pad=10)
        mode = "site_comparison"

    for i, row in enumerate(rows[:count]):
        ax = fig.add_subplot(grid[1, i])
        draw_top_view(ax, structure_for(job, row), f"{row['site']}  {row['energy']:+.3f} eV")
    for i in range(count, max(count, 2)):
        fig.add_subplot(grid[1, i]).axis("off")
    fig.text(0.06, 0.025, summary.get("scientific_label", "UMA prediction · not a Catalysis-Hub DFT benchmark"),
             fontsize=10, color=MUTED)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {"visualization_mode": mode, "image": str(output), "plotted_candidates": rows[:count]}
    (job_dir / "visualization.json").write_text(json.dumps(metadata, indent=2))
    return output


def render_profile(job_dirs: list[Path], output: Path, potential_v: float = 0.0,
                   ph: float = 0.0, temperature_k: float = 298.15) -> Path:
    jobs = [load_job(path) for path in job_dirs]
    best = [min(job["accepted"], key=lambda row: row["energy"]) for job in jobs]
    summaries = [job["summary"] for job in jobs]
    energies = [row["energy"] for row in best]
    labels = [summary["adsorbate"] for summary in summaries]
    che_mode = all(label in CHE_H_COUNTS for label in labels) and len(set(labels)) == len(labels)
    che_rows = []
    if che_mode:
        h2_values = [summary.get("che_h2_reference", {}).get("relaxed_eV") for summary in summaries]
        che_mode = all(value is not None for value in h2_values)
    if che_mode:
        h2_energy = float(np.mean(h2_values))
        reference_index, energies, che_rows = compute_che_states(
            labels, [row["total_energy"] for row in best], h2_energy,
            potential_v, ph, temperature_k,
        )
        topology = classify_che_comparison(labels)
        reactions = reaction_rows(labels, energies, reference_index, topology)
    else:
        topology, reactions = "independent_states", []
    fig = plt.figure(figsize=(12.4, 7.0), facecolor="white")
    grid = fig.add_gridspec(2, max(len(jobs), 2), height_ratios=(1.45, 1), hspace=0.42, wspace=0.18)
    fig.subplots_adjust(top=0.84, bottom=0.09)
    ax = fig.add_subplot(grid[0, :])
    che_title = {"single_reaction": "CHE reaction energy",
                 "branch_comparison": "Competing CHE reaction energies",
                 "sequential_path": "CHE hydrogenation-energy profile"}.get(topology)
    fig.suptitle(che_title if che_mode else "Best adsorption-energy comparison", x=0.06, y=0.965, ha="left",
                 fontsize=19, color=INK, weight="bold")
    fig.text(0.06, 0.916, (f"½H₂ reference · U={potential_v:+.2f} V vs SHE · pH {ph:g} · {temperature_k:g} K"
             if che_mode else "Step-style comparison · independent molecular gas references"),
             color=MUTED, fontsize=11)
    if che_mode and topology == "single_reaction":
        value = reactions[0]["reaction_energy_eV"]
        ax.axis("off")
        ax.text(0.04, 0.58, f"{value:+.3f}", transform=ax.transAxes,
                fontsize=46, color=BLUE, weight="bold", va="center")
        ax.text(0.31, 0.58, "eV", transform=ax.transAxes,
                fontsize=19, color=MUTED, va="center")
        ax.text(0.045, 0.31, reactions[0]["reaction"].replace(" -> ", " → "),
                transform=ax.transAxes, fontsize=13, color=INK)
    elif che_mode and topology == "branch_comparison":
        reaction_values = [row["reaction_energy_eV"] for row in reactions]
        reaction_labels = [row["reaction"].replace(" -> ", "→") for row in reactions]
        x = np.arange(len(reactions), dtype=float)
        bars = ax.bar(x, reaction_values, color=[BLUE] + [TEAL] * (len(x) - 1), width=0.58, zorder=3)
        for bar, value in zip(bars, reaction_values):
            ax.annotate(f"{value:+.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 6 if value >= 0 else -7), textcoords="offset points",
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=11,
                        color=INK, weight="bold")
        ax.set_xticks(x, reaction_labels)
    else:
        x = np.arange(len(energies), dtype=float)
        half = 0.32
        for i, (xi, energy) in enumerate(zip(x, energies)):
            ax.hlines(energy, xi - half, xi + half, color=BLUE, linewidth=5, zorder=3)
            ax.text(xi, energy + (0.07 if energy >= 0 else -0.07), f"{energy:+.3f}",
                    ha="center", va="bottom" if energy >= 0 else "top", color=INK,
                    fontsize=11, weight="bold")
            if i < len(energies) - 1:
                ax.plot([xi + half, x[i + 1] - half], [energy, energies[i + 1]],
                        color="#98A2B3", linewidth=1.5, zorder=2)
        ax.set_xticks(x, labels)
    style_energy_axis(ax)
    if che_mode:
        ax.set_ylabel("Reaction energy, eV" if topology != "sequential_path" else "Energy relative to initial state, eV", color=INK)
        ax.set_title(("Bars are independent reactions sharing one reactant" if topology == "branch_comparison"
                      else "One reaction value" if topology == "single_reaction"
                      else "Connected levels are consecutive reaction states") +
                     " · electronic-energy CHE approximation", loc="left",
                     fontsize=12, color=RED, pad=10)
    else:
        ax.set_title("For screening only — this is not a balanced reaction free-energy diagram", loc="left",
                     fontsize=12, color=RED, pad=10)
    for i, (job, row) in enumerate(zip(jobs, best)):
        top = fig.add_subplot(grid[1, i])
        draw_top_view(top, structure_for(job, row), f"{labels[i]} · {row['site']}")
    for i in range(len(jobs), max(len(jobs), 2)):
        fig.add_subplot(grid[1, i]).axis("off")
    fig.text(0.06, 0.025, ("ΔE(CO*→CHO*) = E(CHO*) − E(CO*) − ½E(H₂); potential/pH corrections use the CHE convention."
             if che_mode else "Eads values use a different gas-phase molecular reference for each intermediate."),
             fontsize=10, color=MUTED)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {
        "visualization_mode": (("che_reaction_energy_card" if topology == "single_reaction" else
                                "che_reaction_bar_comparison" if topology == "branch_comparison" else
                                "che_sequential_energy_profile") if che_mode else "adsorption_energy_comparison"),
        "scientific_warning": ("Electronic-energy CHE approximation; ZPE, entropy, solvation and field effects are absent."
                               if che_mode else "Not a balanced reaction or free-energy diagram; each Eads uses its own gas reference."),
        "states": (che_rows if che_mode else
                   [{"adsorbate": label, "energy_eV": energy, "job": str(job["path"])}
                    for label, energy, job in zip(labels, energies, jobs)]),
    }
    if che_mode:
        metadata["che"] = {"reference_state": f"{labels[reference_index]}*", "h2_energy_eV": h2_energy,
                           "potential_V_vs_SHE": potential_v, "pH": ph, "temperature_K": temperature_k,
                           "formula": "DeltaG ~= E(state)-E(reference)-DeltaN_H/2*E(H2)+DeltaN_H*(U+kBT ln(10)*pH)"}
        metadata["reaction_topology"] = topology
        metadata["reactions"] = reactions
        with output.with_name("che_energies.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(che_rows[0]))
            writer.writeheader(); writer.writerows(che_rows)
        with output.with_name("che_reaction_energies.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reactions[0]))
            writer.writeheader(); writer.writerows(reactions)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UMA adsorption energies with top-view structures")
    parser.add_argument("jobs", nargs="+", type=Path, help="One or more completed prediction directories")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--potential-v", type=float, default=0.0, help="Electrode potential vs SHE for CHE correction")
    parser.add_argument("--ph", type=float, default=0.0)
    parser.add_argument("--temperature-k", type=float, default=298.15)
    args = parser.parse_args()
    if len(args.jobs) == 1:
        print(render_single_job(args.jobs[0], args.output))
    else:
        output = args.output or args.jobs[0].parent / "adsorption_energy_profile.png"
        print(render_profile(args.jobs, output, args.potential_v, args.ph, args.temperature_k))


if __name__ == "__main__":
    main()
