# Vibe Catalysis Agent

A local, multilingual natural-language interface for reproducible
Catalysis-Hub × ASE × FAIR-Chem UMA heterogeneous-catalysis calculations.

The agent converts a request such as:

```text
比较 Cu、Ag 和 Pd(111) 上 C、CH、CH2、CH3 的吸附
```

into a validated calculation plan, preserves the deposited slab constraints,
and runs UMA single-point energies plus constrained ASE relaxation.

It has two deliberately separated modes:

1. **Database-backed benchmark:** H/O/OH/C/CH/CH2/CH3 structures from
   `MamunHighT2019` are compared directly with deposited DFT references.
2. **ASE automatic prediction:** CO/CHO/COH/CHOH/CH2OH structures are generated
   from scratch, relaxed with UMA, screened for failed geometries, and ranked.
   These are predictions, not DFT benchmark values.

## What is included

- Multilingual local parser: Chinese, English, Spanish, French, German, and
  chemical-formula-led Japanese input.
- No paid LLM or ChatGPT subscription required.
- A validated JSON plan is shown before calculation.
- CatBench retains the deposited ASE `FixAtoms` constraints: the bottom 8 of
  12 slab atoms are fixed for this dataset.
- FAIR-Chem `uma-s-1p2` with the `oc20` task.
- Offline parser tests and a real Cu(111)/H integration test.
- A 31-case Cu/Ag/Au/Pt/Pd benchmark and parity plot.
- Automatic fcc(111) slab construction for Cu/Ag/Au/Pt/Pd.
- Automatic `ontop`, `bridge`, `fcc`, and `hcp` enumeration.
- CO C-down/O-down enumeration; three azimuths for larger C/O/H intermediates.
- Bottom-layer constraints, geometry checks, candidate table, and best structure.

## Requirements

- macOS or Linux; Windows users should use WSL.
- Conda/Miniforge.
- About 5 GB free disk space.
- A Hugging Face account with access granted for
  [`facebook/UMA`](https://huggingface.co/facebook/UMA).
- CPU works. NVIDIA CUDA is faster. FAIR-Chem 2.21 does not use Apple MPS.

Do not share Hugging Face access tokens or commit them to this repository.

## Installation

```bash
git clone https://github.com/aonixu-usyd/vibe-catalysis-agent.git
cd vibe-catalysis-agent
conda env create -f environment.yml
conda activate cathub-uma
hf auth login
```

The UMA checkpoint downloads on the first real calculation after model access
has been granted.

## Test the installation

### 1. Fast test without loading UMA

```bash
python test_agent.py
```

Expected output:

```text
PASS: 6 dataset parser cases, 5 automatic parser cases, builder constraints/orientations, and 1 safety rejection
SKIP: UMA integration test (run with --integration)
```

### 2. Full end-to-end UMA test

```bash
python test_agent.py --integration
```

This performs a real CatHub-matched Cu(111)/H calculation. Approximate values:

```text
PASS: real UMA integration test
CatHub DFT:       -0.0536 eV
UMA single point: -0.0362 eV
UMA relaxed:      -0.0370 eV
```

Small numerical differences may occur across software versions.

## Natural-language usage

Preview the validated plan without calculating:

```bash
python vibe_agent.py "Benchmark CHx on Cu, Ag and Pd(111)"
```

Execute after reviewing the plan:

```bash
python vibe_agent.py \
  "比较 Cu、Ag 和 Pd(111) 上 C、CH、CH2、CH3 的吸附" \
  --execute
```

Skip the confirmation prompt only for trusted scripted runs:

```bash
python vibe_agent.py "Calculate H adsorption on Cu(111)" --execute --yes
```

More examples are in [`examples_multilingual.md`](examples_multilingual.md).

### Automatic modelling without a database structure

Preview the plan:

```bash
python vibe_agent.py "计算CO在Cu(111)上的吸附能"
```

Build, relax, screen, and rank all candidates:

```bash
python vibe_agent.py "计算CO在Cu(111)上的吸附能" --execute --yes
```

The automatic backend can also be called directly for full control:

```bash
python predict_adsorption.py \
  --metal Cu --facet 111 --adsorbate CO \
  --size 3 3 4 --fixed-layers 2 \
  --sites ontop bridge fcc hcp
```

Outputs are written to a new results directory:

- `plan.json`: exact modelling and optimization settings;
- `candidates.csv`: every site/orientation and its status;
- `summary.json`: references, accepted count, and lowest-energy candidate;
- `structures/*_initial.extxyz` and `*_final.extxyz`;
- `best_structure.extxyz`;
- ASE optimizer trajectories and logs.

For CO, the default search is 4 sites × C-down/O-down = 8 candidates. For
CHO, COH, CHOH, and CH2OH, C anchoring and 0/120/240° azimuths are enumerated.

## Scientific definitions

The workflow evaluates the balanced Catalysis-Hub reaction definitions with
the same UMA calculator for all surface and gas structures:

| Species | Reaction definition |
|---|---|
| H | `0.5 H2(g) + * -> H*` |
| O | `H2O(g) - H2(g) + * -> O*` |
| OH | `H2O(g) - 0.5 H2(g) + * -> OH*` |
| C | `CH4(g) - 2 H2(g) + * -> C*` |
| CH | `CH4(g) - 1.5 H2(g) + * -> CH*` |
| CH2 | `CH4(g) - H2(g) + * -> CH2*` |
| CH3 | `CH4(g) - 0.5 H2(g) + * -> CH3*` |

Relaxation uses ASE LBFGS, `fmax = 0.05 eV/Å`, and at most 100 steps.

For automatically generated structures, the direct UMA adsorption energy is

```text
E_ads(X) = E_UMA(slab + X) - E_UMA(clean slab) - E_UMA(X gas)
```

All three terms use the same UMA `oc20` task. Clean slab, gas reference, and
adsorbed structures are relaxed independently. CHO and COH are treated as
different bonded isomers. The result is labelled as an UMA prediction unless a
matched Catalysis-Hub DFT record is added separately.

## Automatic CO/Cu(111) demonstration

On the tested Apple M4 Pro CPU, the cached-model 3×3×4 calculation took about
two minutes. Eight candidates completed; all four O-down candidates desorbed
and were excluded by the geometry checker. The lowest accepted UMA candidate
was hcp C-down with `E_ads = -0.4747 eV`. See
[`results/demo_CO_Cu111_auto`](results/demo_CO_Cu111_auto).

This site preference should not be interpreted as validated chemistry. CO on
Cu(111) is a sensitive test and the UMA ranking must be compared with a matched
DFT dataset. The disagreement itself is useful benchmark evidence.

## Baseline result

For the 31 available noble-metal cases:

| Mode | N | MAE (eV) | RMSE (eV) | R² |
|---|---:|---:|---:|---:|
| UMA single point on DFT geometry | 31 | 0.1287 | 0.1511 | 0.9904 |
| UMA constrained relaxation | 31 | 0.1270 | 0.1490 | 0.9907 |

![CatHub DFT versus UMA parity plot](results/cathub_vs_uma_parity.png)

High R² is partly driven by the broad reaction-energy range. Report MAE and
per-case errors alongside R². This is an in-domain baseline and is not evidence
of transferability to arbitrary catalysts, coverages, solvents, electrochemical
potentials, or transition states.

## Data provenance

- Catalysis-Hub publication: `MamunHighT2019`
- Dataset paper: <https://doi.org/10.1038/s41597-019-0080-z>
- Materials Cloud archive: <https://doi.org/10.24435/materialscloud:2019.0015/v1>
- CatBench compact dataset: <https://doi.org/10.5281/zenodo.17157085>
- FAIR-Chem: <https://github.com/facebookresearch/fairchem>
- ASE: <https://ase-lib.org/>

Please cite the underlying dataset, CatBench, FAIR-Chem/UMA, and ASE as
appropriate when using the results.

## Limitations

- The packaged DFT reference subset does not contain CO/CHO/COH/CHOH/CH2OH;
  those species currently use automatic prediction mode.
- Unsupported facets are rejected rather than silently reconstructed.
- The automatic builder currently supports fcc(111), one adsorbate per cell,
  standard high-symmetry sites, and vacuum calculations. It does not include
  solvent, electrode potential, co-adsorbates, defects, or transition states.
- Polyatomic site/orientation enumeration is finite and may miss a lower-energy
  configuration. Geometry checks flag desorption, penetration, bond breaking,
  and large surface reconstruction but do not prove chemical correctness.
- The local language parser uses constrained rules; an LLM is not required.
- UMA predictions are a fast MLIP baseline and should not be presented as
  independently converged DFT results.

See [`SHARING.md`](SHARING.md) for a shorter recipient setup guide.
