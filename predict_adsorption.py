#!/usr/bin/env python3
"""Build metal(111) adsorption candidates with ASE and evaluate them with UMA.

This is the prediction backend.  It deliberately stays separate from the
CatHub-backed benchmark backend: the energies produced here are UMA predictions
on ASE-generated structures and do not acquire a DFT reference automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip


ROOT = Path(__file__).resolve().parent
METALS = ("Cu", "Ag", "Au", "Pt", "Pd")
ADSORBATES = ("CO", "CHO", "COH", "CHOH", "CH2OH")
SITES = ("ontop", "bridge", "fcc", "hcp")

# Approximate, intentionally non-equilibrium starting geometries.  Atom 0 is
# the default surface anchor.  Relaxation, not these coordinates, determines
# the final geometry.  CHO and COH are kept as distinct bonding isomers.
ADSORBATE_TEMPLATES = {
    "CO": ("CO", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.16)], [(0, 1)]),
    "CHO": ("COH", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.22), (0.92, 0.0, 0.28)], [(0, 1), (0, 2)]),
    "COH": ("COH", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.25), (0.82, 0.0, 1.78)], [(0, 1), (1, 2)]),
    "CHOH": ("COH2", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.35), (0.92, 0.0, 0.30), (-0.80, 0.0, 1.90)], [(0, 1), (0, 2), (1, 3)]),
    "CH2OH": ("COH3", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.40), (0.92, 0.0, 0.30), (-0.46, 0.80, 0.30), (-0.78, 0.0, 1.96)], [(0, 1), (0, 2), (0, 3), (1, 4)]),
}


@dataclass
class CandidateResult:
    candidate: str
    site: str
    anchor: str
    azimuth_deg: int
    single_point_total_eV: float | None
    single_point_adsorption_eV: float | None
    relaxed_total_eV: float | None
    relaxed_adsorption_eV: float | None
    steps: int | None
    converged: bool
    geometry_status: str
    min_surface_distance_A: float | None
    max_internal_bond_ratio: float | None
    max_surface_displacement_A: float | None
    error: str


def molecule_template(species: str, anchor: str = "C", azimuth_deg: int = 0) -> tuple[Atoms, list[tuple[int, int]]]:
    formula, positions, bonds = ADSORBATE_TEMPLATES[species]
    atoms = Atoms(formula, positions=positions)
    if species == "CO" and anchor == "O":
        atoms.positions[:] = [(0.0, 0.0, 0.0), (0.0, 0.0, -1.16)]
        atoms = atoms[[1, 0]]  # anchor atom remains index 0; symbols become O,C
        bonds = [(0, 1)]
    theta = math.radians(azimuth_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                         [math.sin(theta), math.cos(theta), 0.0],
                         [0.0, 0.0, 1.0]])
    atoms.positions[:] = atoms.positions @ rotation.T
    atoms.set_tags(np.full(len(atoms), 2, dtype=int))
    return atoms, bonds


def gas_reference(species: str) -> Atoms:
    # A gas reference uses the same molecular isomer and internal connectivity,
    # centred in a non-periodic box.  It is relaxed independently with UMA.
    atoms, _ = molecule_template(species, "C", 0)
    atoms.center(vacuum=6.0)
    atoms.pbc = False
    atoms.set_constraint()
    return atoms


def layer_indices(atoms: Atoms, tolerance: float = 0.25) -> list[list[int]]:
    order = np.argsort(atoms.positions[:, 2])
    layers: list[list[int]] = []
    for index in order:
        if not layers or abs(atoms.positions[index, 2] - np.mean(atoms.positions[layers[-1], 2])) > tolerance:
            layers.append([int(index)])
        else:
            layers[-1].append(int(index))
    return layers


def build_slab(metal: str, size: tuple[int, int, int], vacuum: float, fixed_layers: int) -> tuple[Atoms, list[int]]:
    slab = fcc111(metal, size=size, vacuum=vacuum, periodic=True)
    slab.pbc = (True, True, False)
    layers = layer_indices(slab)
    if fixed_layers < 0 or fixed_layers >= len(layers):
        raise ValueError(f"fixed_layers must be between 0 and {len(layers) - 1}")
    fixed = sorted(i for layer in layers[:fixed_layers] for i in layer)
    if fixed:
        slab.set_constraint(FixAtoms(indices=fixed))
    tags = np.ones(len(slab), dtype=int)
    tags[fixed] = 0
    slab.set_tags(tags)
    return slab, fixed


def build_candidate(slab: Atoms, species: str, site: str, anchor: str, azimuth: int, height: float) -> tuple[Atoms, list[int], list[tuple[int, int]]]:
    candidate = slab.copy()
    adsorbate, bonds = molecule_template(species, anchor, azimuth)
    first = len(candidate)
    add_adsorbate(candidate, adsorbate, height=height, position=site, mol_index=0)
    ads_indices = list(range(first, len(candidate)))
    tags = candidate.get_tags()
    tags[ads_indices] = 2
    candidate.set_tags(tags)
    return candidate, ads_indices, [(ads_indices[i], ads_indices[j]) for i, j in bonds]


def attach(calc: FAIRChemCalculator, atoms: Atoms) -> Atoms:
    atoms.calc = calc
    return atoms


def relax(atoms: Atoms, out_prefix: Path, fmax: float, max_steps: int) -> tuple[float, int, bool]:
    optimizer = LBFGS(atoms, trajectory=str(out_prefix.with_suffix(".traj")), logfile=str(out_prefix.with_suffix(".log")))
    optimizer.run(fmax=fmax, steps=max_steps)
    return float(atoms.get_potential_energy()), int(optimizer.nsteps), bool(optimizer.converged())


def geometry_check(initial: Atoms, final: Atoms, ads_indices: list[int], bonds: list[tuple[int, int]]) -> tuple[str, float, float, float]:
    substrate = [i for i in range(len(final)) if i not in set(ads_indices)]
    d = final.get_all_distances(mic=True)
    min_surface = min(float(d[i, j]) for i in ads_indices for j in substrate)
    ratios = []
    for i, j in bonds:
        initial_length = initial.get_distance(i, j, mic=True)
        final_length = final.get_distance(i, j, mic=True)
        ratios.append(float(final_length / initial_length))
    max_ratio = max(ratios, default=1.0)
    surface_disp = np.linalg.norm(final.positions[substrate] - initial.positions[substrate], axis=1)
    max_surface_disp = float(surface_disp.max(initial=0.0))
    top_z = float(final.positions[substrate, 2].max())
    ads_min_z = float(final.positions[ads_indices, 2].min())
    flags = []
    if min_surface > 3.0:
        flags.append("desorbed")
    if ads_min_z < top_z - 1.0:
        flags.append("penetrated_surface")
    if max_ratio > 1.8:
        flags.append("internal_bond_broken")
    if max_surface_disp > 1.5:
        flags.append("large_surface_reconstruction")
    return ("accepted" if not flags else ";".join(flags), min_surface, max_ratio, max_surface_disp)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_csv(path: Path, rows: list[CandidateResult]) -> None:
    fields = list(CandidateResult.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASE automatic adsorption modelling with FAIR-Chem UMA")
    parser.add_argument("--metal", choices=METALS, required=True)
    parser.add_argument("--facet", default="111", choices=("111",))
    parser.add_argument("--adsorbate", choices=ADSORBATES, required=True)
    parser.add_argument("--sites", nargs="+", choices=SITES, default=list(SITES))
    parser.add_argument("--anchors", nargs="+", choices=("C", "O"))
    parser.add_argument("--azimuths", nargs="+", type=int)
    parser.add_argument("--size", nargs=3, type=int, metavar=("NX", "NY", "NLAYERS"), default=(3, 3, 4))
    parser.add_argument("--vacuum", type=float, default=10.0)
    parser.add_argument("--fixed-layers", type=int, default=2)
    parser.add_argument("--height", type=float, default=1.85)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model", default="uma-s-1p2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single-point-only", action="store_true")
    args = parser.parse_args()

    anchors = args.anchors or (["C", "O"] if args.adsorbate == "CO" else ["C"])
    azimuths = args.azimuths or ([0] if args.adsorbate == "CO" else [0, 120, 240])
    if args.adsorbate != "CO" and "O" in anchors:
        raise ValueError("O anchoring is currently validated only for CO; use C anchoring for this intermediate")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "results" / f"prediction_{args.adsorbate}_{args.metal}{args.facet}_{stamp}").resolve()
    structures = output / "structures"
    structures.mkdir(parents=True, exist_ok=False)

    plan = vars(args).copy()
    plan.update({"output": str(output), "anchors": anchors, "azimuths": azimuths,
                 "energy_definition": "E_ads = E_relaxed(slab+X) - E_relaxed(slab) - E_relaxed(X_gas)",
                 "provenance": "ASE-generated prediction; no DFT reference", "created_utc": stamp})
    (output / "plan.json").write_text(json.dumps(plan, indent=2))

    print(f"Loading {args.model} on {args.device} ...")
    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    calc = FAIRChemCalculator(predictor, task_name="oc20")

    slab, fixed = build_slab(args.metal, tuple(args.size), args.vacuum, args.fixed_layers)
    slab_initial = slab.copy()
    attach(calc, slab)
    slab_sp = float(slab.get_potential_energy())
    if args.single_point_only:
        slab_relaxed, slab_steps, slab_converged = slab_sp, 0, True
    else:
        slab_relaxed, slab_steps, slab_converged = relax(slab, output / "clean_slab", args.fmax, args.max_steps)
    write(output / "clean_slab_initial.extxyz", slab_initial)
    write(output / "clean_slab_relaxed.extxyz", slab)

    gas = gas_reference(args.adsorbate)
    gas_initial = gas.copy()
    attach(calc, gas)
    gas_sp = float(gas.get_potential_energy())
    if args.single_point_only:
        gas_relaxed, gas_steps, gas_converged = gas_sp, 0, True
    else:
        gas_relaxed, gas_steps, gas_converged = relax(gas, output / "gas_reference", args.fmax, args.max_steps)
    write(output / "gas_initial.extxyz", gas_initial)
    write(output / "gas_relaxed.extxyz", gas)

    rows: list[CandidateResult] = []
    for site in args.sites:
        for anchor in anchors:
            for azimuth in azimuths:
                name = safe_name(f"{site}_{anchor}down_rot{azimuth}")
                print(f"Running {name} ...")
                initial, ads_indices, bonds = build_candidate(slab.copy(), args.adsorbate, site, anchor, azimuth, args.height)
                final = initial.copy()
                attach(calc, final)
                try:
                    total_sp = float(final.get_potential_energy())
                    ads_sp = total_sp - slab_relaxed - gas_sp
                    if args.single_point_only:
                        total_relaxed, steps, converged = total_sp, 0, True
                    else:
                        total_relaxed, steps, converged = relax(final, structures / f"{name}_relax", args.fmax, args.max_steps)
                    ads_relaxed = total_relaxed - slab_relaxed - gas_relaxed
                    status, min_distance, max_ratio, surface_disp = geometry_check(initial, final, ads_indices, bonds)
                    error = ""
                except Exception as exc:  # one failed candidate must not discard the entire site search
                    total_sp = ads_sp = total_relaxed = ads_relaxed = None
                    steps = None
                    converged = False
                    status = "calculation_failed"
                    min_distance = max_ratio = surface_disp = None
                    error = f"{type(exc).__name__}: {exc}"
                write(structures / f"{name}_initial.extxyz", initial)
                write(structures / f"{name}_final.extxyz", final)
                rows.append(CandidateResult(name, site, anchor, azimuth, total_sp, ads_sp,
                                            total_relaxed, ads_relaxed, steps, converged, status,
                                            min_distance, max_ratio, surface_disp, error))
                write_csv(output / "candidates.csv", rows)

    accepted = [row for row in rows if row.geometry_status == "accepted" and row.relaxed_adsorption_eV is not None]
    best = min(accepted, key=lambda row: row.relaxed_adsorption_eV) if accepted else None
    summary = {
        "status": "complete" if best else "no_accepted_candidate",
        "metal": args.metal, "facet": args.facet, "adsorbate": args.adsorbate,
        "n_candidates": len(rows), "n_accepted": len(accepted),
        "fixed_layers": args.fixed_layers, "fixed_atom_indices": fixed,
        "clean_slab": {"single_point_eV": slab_sp, "relaxed_eV": slab_relaxed, "steps": slab_steps, "converged": slab_converged},
        "gas_reference": {"isomer": args.adsorbate, "single_point_eV": gas_sp, "relaxed_eV": gas_relaxed, "steps": gas_steps, "converged": gas_converged},
        "best_candidate": asdict(best) if best else None,
        "scientific_label": "UMA prediction on ASE-generated candidates; not a Catalysis-Hub DFT benchmark",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if best:
        source = structures / f"{best.candidate}_final.extxyz"
        (output / "best_structure.extxyz").write_bytes(source.read_bytes())
    print(json.dumps(summary, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
