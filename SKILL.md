---
name: vibe-catalysis
description: Natural-language ASE and FAIR-Chem UMA workflows for general heterogeneous-catalysis surfaces, arbitrary adsorbates/intermediates, Miller-index slabs, adsorption screening, PCET and coupled/decoupled elementary paths, NEB/CI-NEB barriers, steady-state mean-field microkinetics, and Python Nature/CatMAP-style figures.
---

# Vibe Catalysis

Translate the user's natural-language catalytic question into validated structures,
elementary states and steps, calculations, kinetics, and figures. Do not require the user
to write JSON, Python, or shell commands; generate those inputs internally.

## Model surfaces and intermediates

Use `predict_adsorption.py`. Generate fcc/bcc low-index or arbitrary three-index Miller
surfaces such as Pt(211). Retain the named ASE sites on supported low-index builders; for
general Miller surfaces, discover indexed ontop, bridge, and hollow candidates from the
surface geometry. Accept prepared catalysts in any ASE-readable format.

Use built-in CO, CHO/HCO, COH, CHOH, CH2OH, N, N2, NH, NH2, and NH3 structures. For any
other radical, isomer, coadsorbate, or intermediate, create/read an ASE structure and pass
it with `--adsorbate-file`; select its anchor with zero-based `--anchor-indices`.

Run UMA `oc20` single points and constrained ASE relaxations, reject invalid geometries,
rank accepted candidates, and preserve structured CSV/JSON and relaxed structures.

## Build mechanisms and barriers

Convert natural language into a reaction plan for `reaction_workflow.py`. Every edge must
be an elementary `pcet`, `decoupled-proton`, `decoupled-electron`, `coupled`, or `chemical`
step. Split multi-proton/electron transformations.

Use `calculate_barrier.py` for NEB/CI-NEB. Require endpoints with identical atoms, order,
cell and PBC. For PCET, include the transferred H, proton donor/acceptor, solvent/interface
atoms in both endpoints. Report forward/reverse barriers, reaction energy, images,
convergence, and the TS candidate. Do not call the highest UMA image a validated saddle
without CI-NEB/dimer refinement and a one-imaginary-mode frequency check.

Every completed NEB/CI-NEB calculation must automatically call `plot_barrier.py` and
export a CatMAP-style diagram with horizontal initial/final state levels and a smooth
transition-state peak. Export SVG, PDF, 300 dpi PNG, 600 dpi TIFF, source CSV, and caption
text beside `barrier.json`; use chemically meaningful endpoint labels when known. This
diagram represents the optimized state and barrier energies, while `neb_energies.csv`
preserves the discrete image energies for convergence inspection.
Also export a three-panel top-view figure of the relaxed initial state, transition-state
candidate, and relaxed final state, using ASE standard element colours/radii and showing
the periodic unit cell. Save this structure figure as SVG, PDF, 300 dpi PNG, and 600 dpi
TIFF with a JSON structure manifest and caption text.

## Run microkinetics

Use `microkinetics.py` with calculated barriers and explicit reaction stoichiometry,
activities and site balance. It evaluates transition-state-theory rate constants and
solves the ideal single-site mean-field steady state. Report coverages, elementary net
rates, TOF and residual. Do not imply lateral interactions, multiple site types, diffusion
correlations, kMC or transport unless separately modeled.

## Plot through Nature Figure

Use the installed `nature-figure` skill with Python. `plot_reaction_path.py` is the
deterministic bridge: sequential mechanisms use CatMAP-style state levels and TS peaks;
competing branches use grouped reaction/activation energies. Export SVG, PDF, 600 dpi
TIFF, 300 dpi PNG, source CSV and caption text. Never connect branches as a false sequence.

Always label values as UMA predictions, not Catalysis-Hub DFT benchmarks. State when ZPE,
entropy, solvent, potential/field, coverage and constant-potential corrections are absent.
