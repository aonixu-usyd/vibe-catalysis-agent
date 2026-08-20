#!/usr/bin/env python3
"""Construct atom-identical explicit-water hydrogen-transfer endpoints.

This module only builds geometries. It does not assign electronic charge or claim
constant-potential electrochemical energetics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


def _unit(vector):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot define a hydrogen-transfer direction from coincident atoms")
    return vector / norm


def _mic_vector(atoms, start, end):
    return atoms.get_distance(int(start), int(end), mic=True, vector=True)


def validate_water_groups(atoms, water_groups):
    groups = [tuple(map(int, group)) for group in water_groups]
    if not groups:
        raise ValueError("At least one explicit water group is required")
    used = set()
    for group in groups:
        if len(group) != 3 or atoms[group[0]].symbol != "O" or any(atoms[i].symbol != "H" for i in group[1:]):
            raise ValueError("Each pristine water group must be [O, H, H]")
        if used.intersection(group):
            raise ValueError("Water groups overlap")
        used.update(group)
    return groups


def build_hydrogenation_endpoints(template, target_index, water_groups,
                                   product_bond_length=1.02):
    """Move H from the nearest water to a surface/adsorbate target.

    IS: target + H2O network. FS: target-H + OH-minus-like donor in the same
    six-oxygen solvent network. Atom count, order, cell, PBC, and constraints
    are preserved.
    """
    initial = template.copy()
    groups = validate_water_groups(initial, water_groups)
    target_index = int(target_index)
    candidates = [(initial.get_distance(target_index, h, mic=True), o, h)
                  for o, h1, h2 in groups for h in (h1, h2)]
    _, donor_o, transferred_h = min(candidates)
    final = initial.copy()
    direction = _unit(_mic_vector(final, target_index, transferred_h))
    final.positions[transferred_h] = final.positions[target_index] + product_bond_length * direction
    return initial, final, {
        "kind": "aqueous_hydrogenation",
        "reaction_semantics": "nearest water H -> catalyst surface or adsorbate; donor becomes OH-minus-like",
        "target_index": target_index,
        "donor_oxygen_index": donor_o,
        "transferred_hydrogen_index": transferred_h,
        "water_oxygen_count": len(groups),
        "water_groups_from_shared_pristine_template": [list(group) for group in groups],
    }


def build_dehydrogenation_endpoints(template, donor_hydrogen_index, water_groups,
                                     hydronium_bond_length=.99):
    """Move an adsorbate/surface H to the nearest water oxygen.

    IS: hydrogenated adsorbate + pristine H2O network. FS: dehydrogenated
    adsorbate + H3O-plus-like acceptor in the same six-oxygen solvent network.
    """
    initial = template.copy()
    groups = validate_water_groups(initial, water_groups)
    donor_hydrogen_index = int(donor_hydrogen_index)
    if initial[donor_hydrogen_index].symbol != "H":
        raise ValueError("The dehydrogenation donor index must identify H")
    acceptor_o = min((group[0] for group in groups),
                     key=lambda o: initial.get_distance(donor_hydrogen_index, o, mic=True))
    final = initial.copy()
    direction = _unit(_mic_vector(final, acceptor_o, donor_hydrogen_index))
    final.positions[donor_hydrogen_index] = final.positions[acceptor_o] + hydronium_bond_length * direction
    return initial, final, {
        "kind": "aqueous_dehydrogenation",
        "reaction_semantics": "surface/adsorbate H -> nearest water; acceptor becomes H3O-plus-like",
        "donor_hydrogen_index": donor_hydrogen_index,
        "acceptor_oxygen_index": acceptor_o,
        "transferred_hydrogen_index": donor_hydrogen_index,
        "water_oxygen_count": len(groups),
        "water_groups_from_shared_pristine_template": [list(group) for group in groups],
    }


def build_independent_steps(shared_templates, step_specs, water_groups):
    """Build steps independently; never chain a previous final into a new IS.

    `shared_templates` maps stable-state IDs to structures constructed from the
    same pristine solvent template. Every step explicitly selects its reactant
    template by ID. Returned audit data records that no endpoint chaining occurs.
    """
    results = []
    for spec in step_specs:
        state_id = spec["reactant_template"]
        if state_id not in shared_templates:
            raise ValueError(f"Unknown reactant template: {state_id}")
        pristine_reactant = shared_templates[state_id].copy()
        if spec["kind"] == "hydrogenation":
            initial, final, manifest = build_hydrogenation_endpoints(
                pristine_reactant, spec["target_index"], water_groups)
        elif spec["kind"] == "dehydrogenation":
            initial, final, manifest = build_dehydrogenation_endpoints(
                pristine_reactant, spec["donor_hydrogen_index"], water_groups)
        else:
            raise ValueError("Step kind must be hydrogenation or dehydrogenation")
        manifest.update({
            "step_id": spec["id"],
            "reactant_template": state_id,
            "initial_source": "shared_pristine_water_template",
            "inherited_previous_final_water_coordinates": False,
        })
        results.append((initial, final, manifest))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON containing water_atom_indices_zero_based")
    parser.add_argument("--kind", choices=("hydrogenation", "dehydrogenation"), required=True)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--donor-hydrogen-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atoms = read(args.template.resolve(), -1)
    source_manifest = json.loads(args.manifest.read_text())
    groups = source_manifest["water_atom_indices_zero_based"]
    if args.kind == "hydrogenation":
        if args.target_index is None:
            parser.error("--target-index is required for hydrogenation")
        initial, final, manifest = build_hydrogenation_endpoints(atoms, args.target_index, groups)
    else:
        if args.donor_hydrogen_index is None:
            parser.error("--donor-hydrogen-index is required for dehydrogenation")
        initial, final, manifest = build_dehydrogenation_endpoints(atoms, args.donor_hydrogen_index, groups)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "initial.extxyz", initial)
    write(output / "final.extxyz", final)
    (output / "endpoint_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
