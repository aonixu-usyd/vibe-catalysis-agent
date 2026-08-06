---
name: vibe-catalysis
description: Build and run local heterogeneous-catalysis adsorption calculations with ASE and FAIR-Chem UMA. Use when a user asks Codex to model, relax, rank, inspect, or estimate adsorption energies for CO, CHO/HCO, COH, CHOH, or CH2OH on a generated elemental fcc/bcc/hcp surface or a user-supplied catalyst slab/framework such as a COF, MOF, oxide, nitride, supported catalyst, CIF, POSCAR/CONTCAR, XYZ/EXTXYZ, or TRAJ; also use for the Vibe Catalysis ten-step workflow, adsorption-site enumeration, active-atom selection, fixed slab layers, UMA relaxation, data processing, plots, or generated structure/result files.
---

# Vibe Catalysis

Run the deterministic local backend; do not generate ad hoc ASE calculation
code when the validated backend can perform the request.

## Resolve the local runtime

Use the backend bundled with this installed skill:

```text
Python: a Python 3.12 environment containing ASE, fairchem-core, NumPy, and Matplotlib
Backend: scripts/predict_adsorption.py
Launcher: scripts/run_local.py
```

Run `python3 scripts/run_local.py --check` before the first calculation in a task.
The launcher must choose the working runtime reported by that check; it searches
`VIBE_CATALYSIS_PYTHON` and common `cathub-uma` Conda locations, so the Python
used to start the launcher does not itself need ASE. Stop and explain the missing
dependency only if this launcher check fails after its search. Never request or print the
Hugging Face token; FAIR-Chem reads the existing local login and cached model.
The user must separately obtain access to the gated `facebook/UMA` model and
authenticate through Hugging Face. Access and model weights are not bundled.

## Translate the request

Extract:

- structure source: either an attached/local catalyst structure readable by ASE, or an
  elemental metal whose ASE reference state is `fcc`, `bcc`, or `hcp`;
- facet for generated slabs: fcc `111`/`100`/`110`, bcc `111`/`100`/`110`, or
  hcp `0001`/`10-10`;
- adsorbate: `CO`, `CHO`, `COH`, `CHOH`, or `CH2OH`;
- optional slab size, sites, anchors, vacuum, fixed layers, `fmax`, and steps.

Treat `CHO`/`HCO` as the formyl isomer. Keep `COH` distinct. If the user omits
settings, use 3×3×4 (3×4×4 for hcp(10-10)), 10 Å vacuum, bottom two
layers fixed, `fmax=0.05 eV/Å`, and at most 100 LBFGS steps. Infer the stable
crystal structure from ASE. If the facet is omitted, use fcc(111), bcc(110),
or hcp(0001). Run one metal per job; multiple requested adsorbates may run as
separate child calculations under one comparison directory.

When the user supplies a structure, resolve its local absolute path and pass it
with `--structure`; do not require a metal or facet. Treat **every atom already
in the file as part of the catalyst/framework**, even C, O, N, or H. Never infer
adsorbates from element identity. Only the species explicitly requested for
addition is the new adsorbate. Never modify or overwrite the uploaded source. CIF, POSCAR/CONTCAR,
XYZ/EXTXYZ, TRAJ, and any other format readable by ASE are acceptable, but CIF
usually cannot preserve VASP selective-dynamics constraints. Do not attempt to
decompose a pre-existing combined structure into catalyst and adsorbate unless
the user explicitly supplies atom membership or paired-state inputs.

## Execute the ten-step workflow

1. Build a supported elemental slab, or read the user-supplied catalyst/framework structure without changing the source or reclassifying its atoms.
2. Validate its 3D cell, in-plane periodicity, and approximate vacuum; for generated structures, add vacuum and surface-plane periodicity.
3. Preserve constraints read by ASE; if none exist, apply `FixAtoms` to the bottom two layers. Replace uploaded constraints only when the user explicitly asks.
4. Build and independently relax the gas-phase molecular/isomer reference and H₂ for CHE hydrogenation energies.
5. For generated slabs, enumerate ASE's named sites for that exact surface. For uploaded slabs, discover indexed ontop, bridge, and threefold-hollow coordinates from top-layer atoms. For COFs, MOFs, porous/multicomponent materials, defects, and supported catalysts, prefer user-selected zero-based `--active-atom-indices` or explicit `--site-xy X Y` coordinates over blind top-layer discovery.
6. Enumerate C-down/O-down for CO; enumerate 0/120/240° azimuths for larger intermediates.
7. Calculate UMA single-point energies with the `oc20` task.
8. Run constrained ASE LBFGS relaxation for the clean slab and every candidate.
9. Flag desorption, surface penetration, internal bond breaking, large reconstruction, non-convergence, and calculation failure.
10. Rank only accepted candidates and save a candidate table, summary, initial/final structures, logs, trajectories, best structure, and automatic energy/top-view visualization.

Write each job to a new directory under the active task's `outputs/`. Use an
explicit absolute `--output` path. Never overwrite an existing result directory.

Invoke:

```bash
<python-with-ase-and-fairchem> scripts/run_local.py \
  --metal Fe --facet 110 --adsorbate CO --output /absolute/task/outputs/CO_Fe110
```

For an uploaded structure:

```bash
<python-with-ase-and-fairchem> scripts/run_local.py \
  --structure /absolute/path/POSCAR --adsorbate CO \
  --output /absolute/task/outputs/CO_uploaded
```

For a framework with known active atoms:

```bash
<python-with-ase-and-fairchem> scripts/run_local.py \
  --structure /absolute/path/cof.cif --adsorbate CO \
  --active-atom-indices 18 24 31 \
  --output /absolute/task/outputs/CO_COF
```

Pass supported backend overrides after the launcher arguments, for example
`--size 2 2 4`, `--sites ontop bridge`, or `--single-point-only`.

## Interpret and report

Use the backend definition:

```text
E_ads(X) = E_UMA(slab + X) - E_UMA(clean slab) - E_UMA(X gas)
```

For proton-electron hydrogenation such as `CO* + H+ + e- -> CHO*`, do not
subtract independently referenced adsorption energies. Use the best relaxed
total energies from the same slab model and the bundled relaxed H₂ reference:

```text
DeltaE_CHE(CO* -> CHO*) = E(CHO*) - E(CO*) - 1/2 E(H2)
DeltaG_CHE_approx(U,pH) = DeltaE_CHE + eU + kB*T*ln(10)*pH
```

Potential defaults to `U=0 V vs SHE`, `pH=0`, and `T=298.15 K`. For multiple
members of the CO/CHO/COH/CHOH/CH2OH family, run `scripts/visualize_results.py`
on the completed job directories; it automatically writes `che_energies.csv`,
a CHE JSON record, and the energy profile. Pass `--potential-v`, `--ph`, and
`--temperature-k` when the user specifies electrochemical conditions.

Report runtime, candidate count, accepted/rejected count, rejection reasons,
lowest accepted site/orientation, adsorption energy, convergence, and clickable
links to `summary.json`, `candidates.csv`, `best_structure.extxyz`, and
`energy_and_topviews.png`.
For uploaded structures, also report the source filename, SHA-256 provenance,
whether input constraints were preserved, estimated vacuum, discovered site
count, active-atom selection, and validation warnings from `summary.json`.

Use the backend's visualization rule: one accepted value becomes a numerical
energy card; multiple sites become a lowest-energy-per-site bar chart; multiple
requested adsorbates become a step-style adsorption-energy profile. Include
ASE-native relaxed-structure top views with standard element colours, radii,
and the periodic unit cell. Label a multi-adsorbate profile as independently
referenced adsorption energies, not a balanced reaction or free-energy diagram.
A CHE electronic-energy diagram uses a consistent H₂ chemical potential, but
is not a full free-energy diagram unless ZPE, entropy, solvation, field and
other requested corrections are supplied.

Always label the result:

> UMA prediction on an ASE-generated or user-supplied structure; not a Catalysis-Hub DFT benchmark.

ASE reference-state support means the structure can be generated; it does not
show that UMA is accurate for that element, magnetic state, or surface. Do not
present a high-symmetry site ranking as experimentally validated. Mention
that finite site/orientation enumeration can miss lower-energy structures and
that solvent, potential, defects, coverage variation, co-adsorbates, and
transition states are outside this workflow.
Automatic site discovery on uploaded reconstructed, stepped, porous, defective,
or multicomponent slabs is a screening heuristic. Require visual inspection of
the ASE-native top views before interpreting its ranking.

For a strict DFT comparison, use the repository's separate Catalysis-Hub
benchmark path only when a matched structure and consistent reference are
available; never compare unlike energy definitions.
