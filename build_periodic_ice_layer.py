#!/usr/bin/env python3
"""Build an unrelaxed periodic H-down hexagonal water layer on fcc(111)."""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from ase import Atom
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, reference_states
from ase.io import write


def unit(vector):
    vector = np.asarray(vector, float)
    return vector / np.linalg.norm(vector)


def build_periodic_ice_layer(metal="Pt", lattice_a=None, layers=4, vacuum=10.0,
                             fixed_layers=2):
    """Return a 3x3 fcc(111) slab with six periodic H-bonded waters.

    The two oxygen sublattices form the conventional 2/3-ML
    (sqrt(3)xsqrt(3))R30-degree honeycomb motif.  It is a constructed initial
    structure, not an optimized liquid/electrochemical interface.
    """
    reference = reference_states[atomic_numbers[metal]]
    if not reference or reference.get("symmetry") != "fcc":
        raise ValueError("The automatic six-water ice builder currently requires an fcc(111) metal")
    lattice_a = float(lattice_a or reference["a"])
    slab = fcc111(metal, size=(3, 3, layers), a=lattice_a,
                  vacuum=vacuum, orthogonal=False)
    levels = sorted(set(np.round(slab.positions[:, 2], 6)))
    fixed = [i for i, z in enumerate(slab.positions[:, 2])
             if any(abs(z - level) < .05 for level in levels[:fixed_layers])]
    top_z = max(slab.positions[:, 2])
    top = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - top_z) < .05]

    waters = []
    scaled = slab.get_scaled_positions()
    for pt_index in top:
        i = int(round(3 * scaled[pt_index, 0])) % 3
        j = int(round(3 * scaled[pt_index, 1])) % 3
        sublattice = (i - j) % 3
        if sublattice == 0:
            continue
        height = 2.40 if sublattice == 1 else 2.82
        waters.append({
            "sublattice": sublattice,
            "oxygen": np.array([*slab.positions[pt_index, :2], top_z + height]),
        })

    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    edges = []
    for left in range(6):
        for right in range(left + 1, 6):
            candidates = [
                waters[right]["oxygen"] - waters[left]["oxygen"]
                + sx * slab.cell[0] + sy * slab.cell[1]
                for sx in (-1, 0, 1) for sy in (-1, 0, 1)
            ]
            displacement = min(candidates, key=np.linalg.norm)
            if np.linalg.norm(displacement) < 3.05:
                graph.add_edge(left, right)
                edges.append((left, right, displacement))
    if len(waters) != 6 or any(graph.degree[node] != 3 for node in graph):
        raise RuntimeError("Could not construct a three-connected periodic six-water honeycomb")

    matching = {frozenset(edge) for edge in nx.max_weight_matching(graph, maxcardinality=True)}
    donors = {node: [] for node in graph}
    for left, right, displacement in edges:
        if frozenset((left, right)) in matching:
            donor = left if waters[left]["sublattice"] == 2 else right
        else:
            donor = left if waters[left]["sublattice"] == 1 else right
        donors[donor].append(unit(displacement if donor == left else -displacement))

    atoms = slab.copy()
    water_indices = []
    angle = np.deg2rad(104.5)
    for node, water in enumerate(waters):
        oxygen = water["oxygen"]
        directions = donors[node]
        if water["sublattice"] == 1:
            bisector = unit(directions[0] + directions[1])
            transverse = unit(directions[0] - directions[1])
            hydrogen_directions = [
                np.cos(angle / 2) * bisector + np.sin(angle / 2) * transverse,
                np.cos(angle / 2) * bisector - np.sin(angle / 2) * transverse,
            ]
        else:
            network = directions[0]
            perpendicular_down = unit(np.array([0., 0., -1.])
                                      - np.dot([0., 0., -1.], network) * network)
            hydrogen_directions = [
                network,
                np.cos(angle) * network + np.sin(angle) * perpendicular_down,
            ]
        start = len(atoms)
        atoms += Atom("O", oxygen)
        atoms += Atom("H", oxygen + .97 * unit(hydrogen_directions[0]))
        atoms += Atom("H", oxygen + .97 * unit(hydrogen_directions[1]))
        water_indices.append([start, start + 1, start + 2])

    atoms.set_constraint(FixAtoms(indices=fixed))
    atoms.set_tags([1] * len(slab) + [2] * 18)
    metadata = {
        "status": "constructed_only_no_UMA_no_relaxation",
        "surface": f"{metal}(111)",
        "surface_cell": "3x3 primitive fcc(111)",
        "water_layer": "periodic (sqrt(3)xsqrt(3))R30-degree H-down honeycomb",
        "coverage_ML": 2 / 3,
        "n_water": 6,
        "oxygen_puckering_A": .42,
        "periodic_oxygen_coordination": [graph.degree[node] for node in graph],
        "fixed_atom_indices_zero_based": fixed,
        "water_atom_indices_zero_based": water_indices,
        "warning": "Constructed ice-like initial model; relax and inspect before interpreting energies.",
    }
    return atoms, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", default="Pt")
    parser.add_argument("--facet", default="111")
    parser.add_argument("--lattice-a", type=float)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--vacuum", type=float, default=10.)
    parser.add_argument("--fixed-layers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if str(args.facet).replace("(", "").replace(")", "") != "111":
        raise ValueError("The automatic six-water ice builder currently requires facet 111")
    atoms, metadata = build_periodic_ice_layer(
        args.metal, args.lattice_a, args.layers, args.vacuum, args.fixed_layers)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "periodic_ice_interface.extxyz", atoms)
    write(output / "periodic_ice_interface.traj", atoms)
    write(output / "periodic_ice_interface.cif", atoms)
    write(output / "POSCAR", atoms, format="vasp", direct=True, sort=False)
    (output / "structure_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(output)


if __name__ == "__main__":
    main()
