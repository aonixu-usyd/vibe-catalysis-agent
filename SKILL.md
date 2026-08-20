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

When a request combines a water/aqueous environment with an electrochemical
hydrogenation or dehydrogenation barrier, default to an explicit periodic ice-like
interface. Run `build_periodic_ice_layer.py`. Hexagonal surfaces are matched to an
Ice-Ih(0001)-like honeycomb using integer coincidence supercells; near-square fcc(100)
surfaces default to the compact buckled 4x3/8-water (2/3 ML) periodic motif. Never force
all water oxygens into one plane. Determine
the water count from the matched surface area, cell lengths/angle, ice O-O spacing, and
reported strain; never assume six waters. If a fixed catalyst cell has no acceptable
match, enlarge the catalyst supercell or stop rather than distort the layer silently.
Prefer the smallest validated periodic cell. Require the generated manifest to pass
periodic O-network and directional hydrogen-bond checks before starting UMA or NEB.
The matched water layer and catalyst must both be periodic in-plane, and all water atoms
must remain present with identical ordering within each endpoint pair and NEB. Construct
and present top/side views before energy calculations. Do not replace
the layer with a finite water cluster. Treat this as the default initial interface, not a
room-temperature liquid ensemble; sample proton orderings when quantitative conclusions
depend on the water network. For non-fcc(111) surfaces, do not silently force this
commensurability: build and validate a suitable periodic coincidence cell first.

Use `build_aqueous_h_transfer.py` for explicit-water H-transfer endpoints. Hydrogenation
means transferring H from the nearest water to the catalyst surface or adsorbate, leaving
an OH-minus-like donor in the water network. Dehydrogenation means transferring H from a
surface/adsorbate species to the nearest water oxygen, forming an H3O-plus-like acceptor.
Never silently replace either reaction with H adsorption on a spare metal site. If the
ionic endpoint is unstable without charge/potential, report that limitation instead of
changing the reaction.

For consecutive aqueous hydrogenation/dehydrogenation paths, construct every elementary
step independently from stable-state structures that share the same matched pristine
periodic water template. Keep the water count and ordering derived for that catalyst
coincidence cell consistent within the pathway; this count depends on the surface model.
In particular, a dehydrogenation step's initial water coordinates must come from the
shared pristine template, never from the previous step's final hydronium/OH-like water
structure. Only reaction energies and barriers are accumulated into the continuous plot;
endpoint solvent coordinates are not chained between steps.

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
The primary single-step deliverable must combine the energy diagram and the three aligned
top views in one figure. For a sequential multi-step mechanism, render one continuous
cumulative-energy path with every state level and TS peak; annotate every elementary
forward barrier and reaction energy, and align top views of every stable state and
available TS candidate beneath their reaction-coordinate positions. Never connect
independent branches as though they were consecutive steps.

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
