#!/usr/bin/env python3
"""Build elemental-metal adsorption candidates with ASE and evaluate with UMA.

This is the prediction backend.  It deliberately stays separate from the
CatHub-backed benchmark backend: the energies produced here are UMA predictions
on ASE-generated structures and do not acquire a DFT reference automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import (
    add_adsorbate,
    bcc100,
    bcc110,
    bcc111,
    fcc100,
    fcc110,
    fcc111,
    hcp0001,
    hcp10m10,
)
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, reference_states
from ase.io import read, write
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip


ROOT = Path(__file__).resolve().parent
ADSORBATES = ("CO", "CHO", "COH", "CHOH", "CH2OH")
CRYSTAL_STRUCTURES = ("fcc", "bcc", "hcp")
NONMETALLIC_REFERENCE_ELEMENTS = {"Ne", "Ar", "Kr", "Xe", "Se", "Te"}
SURFACE_BUILDERS = {
    ("fcc", "111"): fcc111,
    ("fcc", "100"): fcc100,
    ("fcc", "110"): fcc110,
    ("bcc", "100"): bcc100,
    ("bcc", "110"): bcc110,
    ("bcc", "111"): bcc111,
    ("hcp", "0001"): hcp0001,
    ("hcp", "10m10"): hcp10m10,
}

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


def hydrogen_reference() -> Atoms:
    """Gas-phase H2 reference for computational-hydrogen-electrode energies."""
    atoms = Atoms("H2", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
    atoms.center(vacuum=6.0)
    atoms.pbc = False
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


def normalize_facet(facet: str) -> str:
    value = facet.lower().replace("−", "-").replace("_", "").replace(" ", "")
    value = value.strip("()[]{}").replace(",", "")
    if value in {"10-10", "1010", "10m10"}:
        return "10m10"
    if value in {"0001", "111", "100", "110"}:
        return value
    raise ValueError(f"Unsupported low-index facet: {facet}")


def detect_crystal_structure(metal: str, override: str | None = None) -> tuple[str, dict]:
    symbol = metal[0].upper() + metal[1:].lower()
    if symbol not in atomic_numbers:
        raise ValueError(f"Unknown element symbol: {metal}")
    reference = reference_states[atomic_numbers[symbol]]
    if not reference or reference.get("symmetry") not in CRYSTAL_STRUCTURES:
        raise ValueError(f"ASE has no fcc/bcc/hcp reference state for elemental {symbol}")
    if symbol in NONMETALLIC_REFERENCE_ELEMENTS:
        raise ValueError(f"{symbol} has an ASE close-packed reference but is not treated as an elemental metal")
    crystal = override or reference["symmetry"]
    if crystal not in CRYSTAL_STRUCTURES:
        raise ValueError(f"Unsupported crystal structure: {crystal}")
    return crystal, dict(reference)


def default_size(crystal: str, facet: str) -> tuple[int, int, int]:
    return (3, 4, 4) if (crystal, facet) == ("hcp", "10m10") else (3, 3, 4)


def build_slab(
    metal: str,
    facet: str,
    size: tuple[int, int, int] | None,
    vacuum: float,
    fixed_layers: int,
    crystal_structure: str | None = None,
    lattice_a: float | None = None,
    lattice_c: float | None = None,
) -> tuple[Atoms, list[int], dict, list[str]]:
    symbol = metal[0].upper() + metal[1:].lower()
    crystal, reference = detect_crystal_structure(symbol, crystal_structure)
    if crystal_structure and crystal_structure != reference["symmetry"] and lattice_a is None:
        raise ValueError(
            f"{symbol} reference state is {reference['symmetry']}; --crystal-structure {crystal_structure} "
            "requires --lattice-a (and --lattice-c for hcp if needed)"
        )
    facet = normalize_facet(facet)
    key = (crystal, facet)
    if key not in SURFACE_BUILDERS:
        available = ", ".join(f"{c}({f.replace('m', '-')})" for c, f in SURFACE_BUILDERS if c == crystal)
        raise ValueError(f"Unsupported {crystal} facet {facet}; available surfaces: {available}")
    resolved_size = tuple(size or default_size(crystal, facet))
    if min(resolved_size) < 1:
        raise ValueError("All slab size values must be positive")
    if key == ("hcp", "10m10") and resolved_size[1] % 2:
        raise ValueError("hcp(10-10) requires an even NY; use e.g. --size 3 4 4")
    kwargs = {"size": resolved_size, "vacuum": vacuum, "periodic": True}
    if lattice_a is not None:
        kwargs["a"] = lattice_a
    if crystal == "hcp" and lattice_c is not None:
        kwargs["c"] = lattice_c
    slab = SURFACE_BUILDERS[key](symbol, **kwargs)
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
    sites = sorted(slab.info.get("adsorbate_info", {}).get("sites", {}))
    if not sites:
        raise RuntimeError(f"ASE returned no named adsorption sites for {crystal}({facet})")
    metadata = {
        "element": symbol,
        "crystal_structure": crystal,
        "facet": facet,
        "size": list(resolved_size),
        "ase_reference_state": reference,
        "lattice_a_override_A": lattice_a,
        "lattice_c_override_A": lattice_c,
    }
    return slab, fixed, metadata, sites


def constraint_indices(atoms: Atoms) -> list[int]:
    indices = set()
    for constraint in atoms.constraints:
        getter = getattr(constraint, "get_indices", None)
        if getter is not None:
            indices.update(int(i) for i in getter())
    return sorted(indices)


def apply_bottom_constraints(atoms: Atoms, fixed_layers: int, preserve: bool = True) -> list[int]:
    existing = constraint_indices(atoms)
    if preserve and atoms.constraints:
        fixed = existing
    else:
        layers = layer_indices(atoms)
        if fixed_layers < 0 or fixed_layers >= len(layers):
            raise ValueError(f"fixed_layers must be between 0 and {len(layers) - 1}")
        fixed = sorted(i for layer in layers[:fixed_layers] for i in layer)
        atoms.set_constraint(FixAtoms(indices=fixed) if fixed else [])
    tags = np.ones(len(atoms), dtype=int)
    tags[fixed] = 0
    atoms.set_tags(tags)
    return fixed


def _fractional_xy(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    cell_xy = np.asarray(atoms.cell[:2, :2], dtype=float)
    if abs(np.linalg.det(cell_xy)) < 1e-8:
        raise ValueError("Uploaded slab must have two independent in-plane cell vectors aligned with x/y")
    fractional = atoms.positions[:, :2] @ np.linalg.inv(cell_xy)
    return fractional, cell_xy


def discover_custom_sites(
    slab: Atoms,
    site_types: list[str],
    top_tolerance: float = 0.6,
    max_per_type: int = 6,
    active_atom_indices: list[int] | None = None,
) -> dict[str, tuple[float, float]]:
    """Find sites from geometry; every input atom remains part of the catalyst."""
    if max_per_type < 1:
        raise ValueError("max_sites_per_type must be positive")
    fractional, cell_xy = _fractional_xy(slab)
    if active_atom_indices:
        invalid = sorted({i for i in active_atom_indices if i < 0 or i >= len(slab)})
        if invalid:
            raise ValueError(f"Active atom indices out of range for {len(slab)} atoms: {invalid}")
        top = list(dict.fromkeys(active_atom_indices))
    else:
        top_z = float(slab.positions[:, 2].max())
        top = [int(i) for i, z in enumerate(slab.positions[:, 2]) if z >= top_z - top_tolerance]
    if not top:
        raise ValueError("No top-layer atoms detected in uploaded slab")

    def delta(i: int, j: int) -> tuple[np.ndarray, float]:
        df = fractional[j] - fractional[i]
        df -= np.round(df)
        cart = df @ cell_xy
        return df, float(np.linalg.norm(cart))

    pair_distances = [delta(i, j)[1] for n, i in enumerate(top) for j in top[n + 1:] if delta(i, j)[1] > 0.2]
    nearest = min(pair_distances) if pair_distances else None
    raw: dict[str, list[np.ndarray]] = {kind: [] for kind in site_types}
    if "ontop" in raw:
        raw["ontop"] = [fractional[i] % 1.0 for i in top]
    if nearest is not None and "bridge" in raw:
        for n, i in enumerate(top):
            for j in top[n + 1:]:
                df, distance = delta(i, j)
                if distance <= nearest * 1.20:
                    raw["bridge"].append((fractional[i] + 0.5 * df) % 1.0)
    if nearest is not None and "hollow" in raw:
        for a, i in enumerate(top):
            for b, j in enumerate(top[a + 1:], start=a + 1):
                dij_vec, dij = delta(i, j)
                for k in top[b + 1:]:
                    dik_vec, dik = delta(i, k)
                    _, djk = delta(j, k)
                    distances = (dij, dik, djk)
                    if max(distances) <= nearest * 1.25 and min(distances) >= nearest * 0.75:
                        raw["hollow"].append((fractional[i] + (dij_vec + dik_vec) / 3.0) % 1.0)

    sites: dict[str, tuple[float, float]] = {}
    for kind in site_types:
        unique: list[np.ndarray] = []
        for point in raw[kind]:
            if any(np.linalg.norm(((point - prior) - np.round(point - prior)) @ cell_xy) < 0.20 for prior in unique):
                continue
            unique.append(point)
            if len(unique) >= max_per_type:
                break
        for number, point in enumerate(unique, start=1):
            xy = point @ cell_xy
            sites[f"{kind}_{number:02d}"] = (float(xy[0]), float(xy[1]))
    if not sites:
        raise ValueError(f"No requested adsorption sites found; requested types: {site_types}")
    return sites


def load_custom_slab(
    path: Path,
    fixed_layers: int,
    preserve_constraints: bool,
    site_types: list[str],
    top_tolerance: float,
    max_sites_per_type: int,
    active_atom_indices: list[int] | None = None,
) -> tuple[Atoms, list[int], dict, dict[str, tuple[float, float]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Uploaded structure not found: {path}")
    slab = read(path, index=-1)
    if len(slab) < 2 or slab.cell.rank < 3:
        raise ValueError("Uploaded structure must contain a 3D cell and at least two atoms")
    slab.pbc = (True, True, False)
    had_input_constraints = bool(slab.constraints)
    fixed = apply_bottom_constraints(slab, fixed_layers, preserve_constraints)
    sites = discover_custom_sites(
        slab, site_types, top_tolerance, max_sites_per_type, active_atom_indices
    )
    z_span = float(np.ptp(slab.positions[:, 2]))
    vacuum_estimate = float(slab.cell.lengths()[2] - z_span)
    warnings = []
    if vacuum_estimate < 6.0:
        warnings.append(f"estimated vacuum is only {vacuum_estimate:.2f} A; 8-12 A is usually safer")
    metadata = {
        "source": "uploaded_structure",
        "source_file": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_format": path.name if path.name.upper() in {"POSCAR", "CONTCAR"} else path.suffix.lower().lstrip("."),
        "formula": slab.get_chemical_formula(),
        "crystal_structure": "uploaded",
        "facet": "custom",
        "size": None,
        "n_atoms": len(slab),
        "catalyst_atom_policy": "all atoms in the uploaded structure are catalyst atoms",
        "active_atom_indices_zero_based": active_atom_indices,
        "preserved_input_constraints": bool(preserve_constraints and had_input_constraints),
        "estimated_vacuum_A": vacuum_estimate,
        "warnings": warnings,
    }
    return slab, fixed, metadata, sites


def build_candidate(slab: Atoms, species: str, site: str | tuple[float, float], anchor: str, azimuth: int, height: float) -> tuple[Atoms, list[int], list[tuple[int, int]]]:
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metal", help="Element symbol with an ASE fcc/bcc/hcp reference state")
    source.add_argument("--structure", type=Path, help="Uploaded catalyst slab/framework: CIF, POSCAR/CONTCAR, XYZ/EXTXYZ, TRAJ, or another ASE-readable file")
    parser.add_argument("--facet", help="111/100/110 for cubic; 0001/10-10 for hcp")
    parser.add_argument("--crystal-structure", choices=CRYSTAL_STRUCTURES)
    parser.add_argument("--lattice-a", type=float)
    parser.add_argument("--lattice-c", type=float)
    parser.add_argument("--adsorbate", choices=ADSORBATES, required=True)
    parser.add_argument("--sites", nargs="+", help="ASE site names; default is every site available for the surface")
    parser.add_argument("--site-types", nargs="+", choices=("ontop", "bridge", "hollow"), default=["ontop", "bridge", "hollow"], help="Site types discovered on an uploaded slab")
    parser.add_argument("--site-xy", nargs=2, type=float, action="append", metavar=("X", "Y"), help="Explicit Cartesian adsorption coordinate; repeat for multiple sites")
    parser.add_argument("--active-atom-indices", nargs="+", type=int, help="Zero-based catalyst atom indices used to construct ontop/bridge/hollow sites; useful for COF/MOF/oxide surfaces")
    parser.add_argument("--top-layer-tolerance", type=float, default=0.6)
    parser.add_argument("--max-sites-per-type", type=int, default=6)
    parser.add_argument("--anchors", nargs="+", choices=("C", "O"))
    parser.add_argument("--azimuths", nargs="+", type=int)
    parser.add_argument("--size", nargs=3, type=int, metavar=("NX", "NY", "NLAYERS"))
    parser.add_argument("--vacuum", type=float, default=10.0)
    parser.add_argument("--fixed-layers", type=int, default=2)
    parser.add_argument("--replace-constraints", action="store_true", help="Discard uploaded constraints and fix the requested bottom layers")
    parser.add_argument("--height", type=float, default=1.85)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model", default="uma-s-1p2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single-point-only", action="store_true")
    args = parser.parse_args()

    if args.metal:
        args.metal = args.metal[0].upper() + args.metal[1:].lower()
        args.facet = normalize_facet(args.facet or "111")
    elif args.facet:
        raise ValueError("--facet is for ASE-generated elemental surfaces; omit it with --structure")
    anchors = args.anchors or (["C", "O"] if args.adsorbate == "CO" else ["C"])
    azimuths = args.azimuths or ([0] if args.adsorbate == "CO" else [0, 120, 240])
    if args.adsorbate != "CO" and "O" in anchors:
        raise ValueError("O anchoring is currently validated only for CO; use C anchoring for this intermediate")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_label = f"{args.metal}{args.facet.replace('m', '-')}" if args.metal else safe_name(args.structure.stem)
    output = (args.output or ROOT / "results" / f"prediction_{args.adsorbate}_{source_label}_{stamp}").resolve()
    if args.structure:
        slab, fixed, surface_metadata, custom_sites = load_custom_slab(
            args.structure, args.fixed_layers, not args.replace_constraints,
            args.site_types, args.top_layer_tolerance, args.max_sites_per_type,
            args.active_atom_indices,
        )
        if args.site_xy:
            custom_sites = {f"custom_{i:02d}": tuple(xy) for i, xy in enumerate(args.site_xy, start=1)}
        available_sites = list(custom_sites)
        sites = args.sites or available_sites
        site_positions: dict[str, str | tuple[float, float]] = custom_sites
    else:
        slab, fixed, surface_metadata, available_sites = build_slab(
            args.metal, args.facet, tuple(args.size) if args.size else None,
            args.vacuum, args.fixed_layers, args.crystal_structure,
            args.lattice_a, args.lattice_c,
        )
        sites = args.sites or available_sites
        site_positions = {site: site for site in available_sites}
    unknown_sites = sorted(set(sites) - set(available_sites))
    if unknown_sites:
        raise ValueError(
            f"Unknown sites for {surface_metadata['crystal_structure']}({surface_metadata['facet']}): {unknown_sites}; "
            f"available sites: {available_sites}"
        )
    structures = output / "structures"
    structures.mkdir(parents=True, exist_ok=False)
    plan = vars(args).copy()
    plan["structure"] = str(args.structure.resolve()) if args.structure else None
    plan.update({"output": str(output), "anchors": anchors, "azimuths": azimuths,
                 "sites": sites, "available_sites": available_sites, "surface": surface_metadata,
                 "energy_definition": "E_ads = E_relaxed(slab+X) - E_relaxed(slab) - E_relaxed(X_gas)",
                 "provenance": "ASE-generated prediction; no DFT reference", "created_utc": stamp})
    (output / "plan.json").write_text(json.dumps(plan, indent=2))

    print(f"Loading {args.model} on {args.device} ...")
    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    calc = FAIRChemCalculator(predictor, task_name="oc20")

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

    h2 = hydrogen_reference()
    h2_initial = h2.copy()
    attach(calc, h2)
    h2_sp = float(h2.get_potential_energy())
    if args.single_point_only:
        h2_relaxed, h2_steps, h2_converged = h2_sp, 0, True
    else:
        h2_relaxed, h2_steps, h2_converged = relax(
            h2, output / "h2_che_reference", args.fmax, args.max_steps
        )
    write(output / "h2_che_initial.extxyz", h2_initial)
    write(output / "h2_che_relaxed.extxyz", h2)

    rows: list[CandidateResult] = []
    for site in sites:
        for anchor in anchors:
            for azimuth in azimuths:
                name = safe_name(f"{site}_{anchor}down_rot{azimuth}")
                print(f"Running {name} ...")
                initial, ads_indices, bonds = build_candidate(slab.copy(), args.adsorbate, site_positions[site], anchor, azimuth, args.height)
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
        "metal": args.metal or surface_metadata["formula"], "crystal_structure": surface_metadata["crystal_structure"],
        "facet": surface_metadata["facet"], "slab_size": surface_metadata["size"], "adsorbate": args.adsorbate,
        "structure_source": surface_metadata,
        "available_sites": available_sites, "evaluated_sites": sites,
        "n_candidates": len(rows), "n_accepted": len(accepted),
        "fixed_layers": args.fixed_layers, "fixed_atom_indices": fixed,
        "clean_slab": {"single_point_eV": slab_sp, "relaxed_eV": slab_relaxed, "steps": slab_steps, "converged": slab_converged},
        "gas_reference": {"isomer": args.adsorbate, "single_point_eV": gas_sp, "relaxed_eV": gas_relaxed, "steps": gas_steps, "converged": gas_converged},
        "che_h2_reference": {"formula": "H2", "single_point_eV": h2_sp, "relaxed_eV": h2_relaxed, "steps": h2_steps, "converged": h2_converged,
                             "definition": "mu(H+ + e-) = 1/2 E(H2) - eU at the electronic-energy level"},
        "best_candidate": asdict(best) if best else None,
        "scientific_label": ("UMA prediction on a user-supplied catalyst structure; not a Catalysis-Hub DFT benchmark"
                             if args.structure else
                             "UMA prediction on ASE-generated candidates; not a Catalysis-Hub DFT benchmark"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if best:
        source = structures / f"{best.candidate}_final.extxyz"
        (output / "best_structure.extxyz").write_bytes(source.read_bytes())
        try:
            from visualize_results import render_single_job
            image_path = render_single_job(output)
            summary["visualization"] = str(image_path)
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
        except Exception as exc:
            summary["visualization_error"] = f"{type(exc).__name__}: {exc}"
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
