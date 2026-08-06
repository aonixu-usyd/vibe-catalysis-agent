---
name: vibe-catalysis
description: Build and run local heterogeneous-catalysis adsorption calculations with ASE and FAIR-Chem UMA. Use when a user asks Codex in natural language to model, relax, rank, inspect, or estimate adsorption energies for CO, CHO/HCO, COH, CHOH, or CH2OH on an elemental fcc, bcc, or hcp metal and a supported low-index surface, or asks for the Vibe Catalysis ten-step workflow, adsorption-site enumeration, fixed slab layers, UMA relaxation, or generated structure/result files.
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

Run `<python> scripts/run_local.py --check` before the first calculation in a task. Stop
and explain the missing dependency if it fails. Never request or print the
Hugging Face token; FAIR-Chem reads the existing local login and cached model.
The user must separately obtain access to the gated `facebook/UMA` model and
authenticate through Hugging Face. Access and model weights are not bundled.

## Translate the request

Extract:

- metal: an elemental metal whose ASE reference state is `fcc`, `bcc`, or `hcp`;
- facet: fcc `111`/`100`/`110`, bcc `111`/`100`/`110`, or hcp
  `0001`/`10-10`;
- adsorbate: `CO`, `CHO`, `COH`, `CHOH`, or `CH2OH`;
- optional slab size, sites, anchors, vacuum, fixed layers, `fmax`, and steps.

Treat `CHO`/`HCO` as the formyl isomer. Keep `COH` distinct. If the user omits
settings, use 3×3×4 (3×4×4 for hcp(10-10)), 10 Å vacuum, bottom two
layers fixed, `fmax=0.05 eV/Å`, and at most 100 LBFGS steps. Infer the stable
crystal structure from ASE. If the facet is omitted, use fcc(111), bcc(110),
or hcp(0001). Run one metal per job; multiple requested adsorbates may run as
separate child calculations under one comparison directory.

## Execute the ten-step workflow

1. Detect the elemental ASE reference structure and build the requested supported low-index slab.
2. Add vacuum and periodicity only in the surface plane.
3. apply `FixAtoms` to the bottom two layers.
4. Build and independently relax the gas-phase molecular/isomer reference.
5. Read and enumerate the named sites supplied by ASE for that exact surface.
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

Pass supported backend overrides after the launcher arguments, for example
`--size 2 2 4`, `--sites ontop bridge`, or `--single-point-only`.

## Interpret and report

Use the backend definition:

```text
E_ads(X) = E_UMA(slab + X) - E_UMA(clean slab) - E_UMA(X gas)
```

Report runtime, candidate count, accepted/rejected count, rejection reasons,
lowest accepted site/orientation, adsorption energy, convergence, and clickable
links to `summary.json`, `candidates.csv`, `best_structure.extxyz`, and
`energy_and_topviews.png`.

Use the backend's visualization rule: one accepted value becomes a numerical
energy card; multiple sites become a lowest-energy-per-site bar chart; multiple
requested adsorbates become a step-style adsorption-energy profile. Include
ASE-native relaxed-structure top views with standard element colours, radii,
and the periodic unit cell. Label a multi-adsorbate profile as independently
referenced adsorption energies, not a balanced reaction or free-energy diagram.
A strict reaction diagram requires consistent chemical potentials and
thermodynamic corrections.

Always label the result:

> UMA prediction on ASE-generated structures; not a Catalysis-Hub DFT benchmark.

ASE reference-state support means the structure can be generated; it does not
show that UMA is accurate for that element, magnetic state, or surface. Do not
present a high-symmetry site ranking as experimentally validated. Mention
that finite site/orientation enumeration can miss lower-energy structures and
that solvent, potential, defects, coverage variation, co-adsorbates, and
transition states are outside this workflow.

For a strict DFT comparison, use the repository's separate Catalysis-Hub
benchmark path only when a matched structure and consistent reference are
available; never compare unlike energy definitions.
