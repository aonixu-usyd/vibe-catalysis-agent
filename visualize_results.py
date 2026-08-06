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
from ase.data import covalent_radii
from ase.io import read


INK = "#172033"
MUTED = "#667085"
BLUE = "#2563EB"
TEAL = "#0F9F8F"
RED = "#DC4C64"
GRID = "#E4E7EC"
PAPER = "#F8FAFC"


def load_job(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    with (path / "candidates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = []
    for row in rows:
        if row["geometry_status"] != "accepted" or not row["relaxed_adsorption_eV"]:
            continue
        row["energy"] = float(row["relaxed_adsorption_eV"])
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


def atom_color(symbol: str, is_adsorbate: bool) -> str:
    if symbol == "O":
        return RED
    if symbol == "C":
        return "#202938"
    if symbol == "H":
        return "#FFFFFF"
    return "#AAB4C3" if not is_adsorbate else TEAL


def draw_top_view(ax, atoms, title: str) -> None:
    tags = atoms.get_tags()
    xy = atoms.positions[:, :2]
    order = np.argsort(tags)  # substrate first, adsorbate on top
    for index in order:
        ads = bool(tags[index] == 2)
        radius = covalent_radii[atoms.numbers[index]]
        size = (130 if ads else 72) * max(radius, 0.55) ** 1.25
        ax.scatter(
            xy[index, 0], xy[index, 1], s=size,
            c=atom_color(atoms[index].symbol, ads), edgecolors="#FFFFFF" if not ads else INK,
            linewidths=0.7 if not ads else 1.1, zorder=3 if ads else 2,
        )
    ax.set_aspect("equal")
    pad = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) * 0.08 + 0.35
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color=INK, pad=8, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(PAPER)


def style_energy_axis(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(0, color=INK, linewidth=0.9, zorder=1)
    ax.set_ylabel("Adsorption energy, eV", color=INK, fontsize=10)


def render_single_job(job_dir: Path, output: Path | None = None) -> Path:
    job = load_job(job_dir)
    summary = job["summary"]
    rows = best_by_site(job)
    if not rows:
        raise ValueError(f"No accepted candidates in {job_dir}")
    crystal = summary.get("crystal_structure", "fcc" if summary.get("facet") in {"111", "100", "110"} else "")
    surface = f"{summary['metal']} {crystal}({summary['facet'].replace('m', '-')})".replace("  ", " ")
    output = output or job_dir / "energy_and_topviews.png"
    count = min(len(rows), 4)
    fig = plt.figure(figsize=(10.8, 6.3), facecolor="white")
    grid = fig.add_gridspec(2, max(count, 2), height_ratios=(1.45, 1), hspace=0.42, wspace=0.18)
    fig.subplots_adjust(top=0.84, bottom=0.09)
    energy_ax = fig.add_subplot(grid[0, :])
    fig.suptitle(f"{summary['adsorbate']} adsorption · {surface}", x=0.06, y=0.965,
                 ha="left", fontsize=17, color=INK, weight="bold")
    fig.text(0.06, 0.916, "FAIR-Chem UMA · relaxed ASE structures · lower is more stable",
             color=MUTED, fontsize=9.5)

    if len(rows) == 1:
        value = rows[0]["energy"]
        energy_ax.axis("off")
        energy_ax.text(0.03, 0.58, f"{value:+.3f}", transform=energy_ax.transAxes,
                       fontsize=43, color=BLUE, weight="bold", va="center")
        energy_ax.text(0.305, 0.58, "eV", transform=energy_ax.transAxes,
                       fontsize=17, color=MUTED, va="center")
        energy_ax.text(0.035, 0.30, f"{rows[0]['site']} · {rows[0]['anchor']}-down · {rows[0]['azimuth_deg']}°",
                       transform=energy_ax.transAxes, fontsize=11, color=INK)
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
                               fontsize=9, color=INK, weight="bold")
        energy_ax.set_title("Lowest accepted orientation at each adsorption site", loc="left",
                            fontsize=10.5, color=INK, pad=10)
        mode = "site_comparison"

    for i, row in enumerate(rows[:count]):
        ax = fig.add_subplot(grid[1, i])
        draw_top_view(ax, structure_for(job, row), f"{row['site']}  {row['energy']:+.3f} eV")
    for i in range(count, max(count, 2)):
        fig.add_subplot(grid[1, i]).axis("off")
    fig.text(0.06, 0.025, "UMA prediction on ASE-generated structures · not a Catalysis-Hub DFT benchmark",
             fontsize=8.5, color=MUTED)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {"visualization_mode": mode, "image": str(output), "plotted_candidates": rows[:count]}
    (job_dir / "visualization.json").write_text(json.dumps(metadata, indent=2))
    return output


def render_profile(job_dirs: list[Path], output: Path) -> Path:
    jobs = [load_job(path) for path in job_dirs]
    best = [min(job["accepted"], key=lambda row: row["energy"]) for job in jobs]
    summaries = [job["summary"] for job in jobs]
    energies = [row["energy"] for row in best]
    labels = [summary["adsorbate"] for summary in summaries]
    fig = plt.figure(figsize=(11.5, 6.5), facecolor="white")
    grid = fig.add_gridspec(2, max(len(jobs), 2), height_ratios=(1.45, 1), hspace=0.42, wspace=0.18)
    fig.subplots_adjust(top=0.84, bottom=0.09)
    ax = fig.add_subplot(grid[0, :])
    fig.suptitle("Best adsorption-energy profile", x=0.06, y=0.965, ha="left",
                 fontsize=17, color=INK, weight="bold")
    fig.text(0.06, 0.916, "Step-style comparison · independent molecular gas references",
             color=MUTED, fontsize=9.5)
    x = np.arange(len(energies), dtype=float)
    half = 0.32
    for i, (xi, energy) in enumerate(zip(x, energies)):
        ax.hlines(energy, xi - half, xi + half, color=BLUE, linewidth=5, zorder=3)
        ax.text(xi, energy + (0.07 if energy >= 0 else -0.07), f"{energy:+.3f}",
                ha="center", va="bottom" if energy >= 0 else "top", color=INK,
                fontsize=9.5, weight="bold")
        if i < len(energies) - 1:
            ax.plot([xi + half, x[i + 1] - half], [energy, energies[i + 1]],
                    color="#98A2B3", linewidth=1.5, zorder=2)
    ax.set_xticks(x, labels)
    style_energy_axis(ax)
    ax.set_title("For screening only — this is not a balanced reaction free-energy diagram", loc="left",
                 fontsize=10.5, color=RED, pad=10)
    for i, (job, row) in enumerate(zip(jobs, best)):
        top = fig.add_subplot(grid[1, i])
        draw_top_view(top, structure_for(job, row), f"{labels[i]} · {row['site']}")
    for i in range(len(jobs), max(len(jobs), 2)):
        fig.add_subplot(grid[1, i]).axis("off")
    fig.text(0.06, 0.025, "Eads values use a different gas-phase molecular reference for each intermediate.",
             fontsize=8.5, color=MUTED)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "visualization_mode": "adsorption_energy_profile",
        "scientific_warning": "Not a balanced reaction or free-energy diagram; each Eads uses its own gas reference.",
        "states": [{"adsorbate": label, "energy_eV": energy, "job": str(job["path"])}
                   for label, energy, job in zip(labels, energies, jobs)],
    }, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UMA adsorption energies with top-view structures")
    parser.add_argument("jobs", nargs="+", type=Path, help="One or more completed prediction directories")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.jobs) == 1:
        print(render_single_job(args.jobs[0], args.output))
    else:
        output = args.output or args.jobs[0].parent / "adsorption_energy_profile.png"
        print(render_profile(args.jobs, output))


if __name__ == "__main__":
    main()
