#!/usr/bin/env python3
"""Create the final table, fit metrics, and parity plot for the UMA subset."""

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from catbench.utils.data_utils import load_catbench_json
from ase.constraints import FixAtoms


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_data/Mamun_noble_C1_subset_adsorption.json"
RESULT = ROOT / "result/UMA-s-1p2-OC20/UMA-s-1p2-OC20_result.json"
OUT = ROOT / "results"
CARBON = {"C", "CH", "CH2", "CH3"}


def labels(key):
    metal = re.match(r"([A-Z][a-z]?)12_", key).group(1)
    adsorbate = re.search(r"->\s*([A-Za-z0-9]+)\*", key).group(1)
    return metal, adsorbate


def fixed_indices(atoms):
    fixed = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed.update(int(i) for i in constraint.get_indices())
    return sorted(fixed)


def fit_metrics(rows, pred_col, carbon_only=False):
    chosen = [r for r in rows if not carbon_only or r["adsorbate"] in CARBON]
    x = np.array([r["cathub_dft_eV"] for r in chosen])
    y = np.array([r[pred_col] for r in chosen])
    slope, intercept = np.polyfit(x, y, 1)
    yfit = slope * x + intercept
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    errors = y - x
    return {
        "scope": "C/CH/CH2/CH3" if carbon_only else "all H/O/OH/C/CH/CH2/CH3",
        "mode": "single_point" if pred_col == "uma_single_point_eV" else "fixed_bottom_relax",
        "n": len(x), "slope": slope, "intercept_eV": intercept,
        "r2": 1 - ss_res / ss_tot, "mae_eV": np.mean(np.abs(errors)),
        "rmse_eV": np.sqrt(np.mean(errors**2)), "bias_eV": np.mean(errors),
    }


def main():
    source = load_catbench_json(str(RAW))
    result = json.loads(RESULT.read_text())
    rows = []
    for key, value in result.items():
        if key == "calculation_settings":
            continue
        metal, adsorbate = labels(key)
        raw = source[key]["raw"]
        slab = raw["star"]["atoms"]
        ad_key = next(name for name in raw if name.endswith("star") and name != "star")
        adslab = raw[ad_key]["atoms"]
        slab_fixed, ad_fixed = fixed_indices(slab), fixed_indices(adslab)
        dft = value["reference"]["ads_eng"]
        sp = value["single_calculation"]["ads_eng"]
        relax = value["final"]["ads_eng_median"]
        rows.append({
            "case_id": f"{adsorbate}_{metal}111", "metal": metal,
            "facet": "111", "adsorbate": adsorbate, "reaction": key.split("_", 1)[1].rsplit("_", 1)[0] if re.search(r"_\d+$", key) else key.split("_", 1)[1],
            "cathub_dft_eV": dft, "uma_single_point_eV": sp,
            "single_point_error_eV": sp - dft, "uma_fixed_bottom_relax_eV": relax,
            "relax_error_eV": relax - dft, "constraint_source": source[key].get("constraint_source", "unknown"),
            "slab_fixed_atoms": len(slab_fixed), "adslab_fixed_atoms": len(ad_fixed),
            "fixed_indices_match": slab_fixed == ad_fixed[:len(slab_fixed)],
            "slab_steps": value["final"]["steps_total_slab"], "adslab_steps": value["final"]["steps_total_adslab"],
            "slab_max_displacement_A": value["final"]["slab_max_disp"],
            "adslab_max_displacement_A": value["final"]["adslab_max_disp"],
        })
    rows.sort(key=lambda r: (("Cu", "Ag", "Au", "Pt", "Pd").index(r["metal"]), ("H", "O", "OH", "C", "CH", "CH2", "CH3").index(r["adsorbate"])))

    table_path = OUT / "cathub_uma_noble_C1_benchmark.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    metrics = []
    for pred in ("uma_single_point_eV", "uma_fixed_bottom_relax_eV"):
        metrics += [fit_metrics(rows, pred, False), fit_metrics(rows, pred, True)]
    with (OUT / "cathub_uma_noble_C1_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader(); writer.writerows(metrics)

    colors = {"H": "#4C78A8", "O": "#E45756", "OH": "#F2CF5B", "C": "#111111", "CH": "#54A24B", "CH2": "#B279A2", "CH3": "#FF9DA6"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, pred, title in zip(axes, ("uma_single_point_eV", "uma_fixed_bottom_relax_eV"), ("UMA single point on DFT geometry", "UMA constrained relaxation")):
        for adsorbate in colors:
            subset = [r for r in rows if r["adsorbate"] == adsorbate]
            ax.scatter([r["cathub_dft_eV"] for r in subset], [r[pred] for r in subset], s=54, color=colors[adsorbate], edgecolor="white", linewidth=.7, label=adsorbate, zorder=3)
        x = np.array([r["cathub_dft_eV"] for r in rows]); y = np.array([r[pred] for r in rows])
        lo, hi = min(x.min(), y.min()) - .35, max(x.max(), y.max()) + .35
        ax.plot([lo, hi], [lo, hi], "--", color="#777777", lw=1.2, label="y = x")
        m = next(item for item in metrics if item["mode"] == ("single_point" if "single" in pred else "fixed_bottom_relax") and item["scope"].startswith("all"))
        xx = np.linspace(lo, hi, 100); ax.plot(xx, m["slope"] * xx + m["intercept_eV"], color="#2F5597", lw=1.8, label="linear fit")
        ax.text(.04, .96, f"N = {m['n']}\n$R^2$ = {m['r2']:.3f}\nMAE = {m['mae_eV']:.3f} eV\ny = {m['slope']:.3f}x {m['intercept_eV']:+.3f}", transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#CCCCCC", alpha=.92))
        ax.set_title(title); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(alpha=.18); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Catalysis-Hub DFT reaction energy (eV)")
    axes[0].set_ylabel("UMA workflow reaction energy (eV)")
    handles, labels_ = axes[1].get_legend_handles_labels(); axes[1].legend(handles, labels_, loc="lower right", fontsize=8, ncol=2)
    fig.suptitle("MamunHighT2019 noble-metal benchmark: CatHub DFT vs UMA", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "cathub_vs_uma_parity.png", dpi=220, bbox_inches="tight"); fig.savefig(OUT / "cathub_vs_uma_parity.pdf", bbox_inches="tight")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
