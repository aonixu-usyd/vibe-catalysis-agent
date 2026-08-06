# Vibe Catalysis Agent

A local, multilingual natural-language interface for a reproducible
Catalysis-Hub × ASE × FAIR-Chem UMA heterogeneous-catalysis benchmark.

The agent converts a request such as:

```text
比较 Cu、Ag 和 Pd(111) 上 C、CH、CH2、CH3 的吸附
```

into a validated calculation plan, preserves the deposited slab constraints,
and runs UMA single-point energies plus constrained ASE relaxation.

> **Scope:** this release is a dataset-backed baseline, not a general-purpose
> replacement for DFT. It supports Cu/Ag/Au/Pt/Pd(111) and
> H/O/OH/C/CH/CH2/CH3 from `MamunHighT2019`.

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
PASS: 6 multilingual parser cases and 1 safety rejection
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

- The packaged reference subset does not contain CO, CHO, COH, or other C/O/H
  intermediates beyond the species listed above.
- Unsupported facets are rejected rather than silently reconstructed.
- The local language parser uses constrained rules; an LLM is not required.
- UMA predictions are a fast MLIP baseline and should not be presented as
  independently converged DFT results.

See [`SHARING.md`](SHARING.md) for a shorter recipient setup guide.
