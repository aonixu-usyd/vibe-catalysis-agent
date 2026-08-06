# Sharing and testing Vibe Catalysis

## What the recipient needs

- macOS or Linux; Windows via WSL is recommended.
- Approximately 5 GB free disk space.
- A Hugging Face account with access granted for `facebook/UMA`.
- A personal read-only Hugging Face token. Never share somebody else's token.
- CPU works; an NVIDIA CUDA GPU is faster. FAIR-Chem 2.21 does not use Apple MPS.

## Install

```bash
unzip vibe-catalysis-agent.zip
cd cathub-uma-baseline
conda env create -f environment.yml
conda activate cathub-uma
huggingface-cli login
```

The UMA model license/access request is at:
https://huggingface.co/facebook/UMA

## Test without downloading UMA

```bash
python test_agent.py
```

Expected final lines:

```text
PASS: 6 multilingual parser cases and 1 safety rejection
SKIP: UMA integration test (run with --integration)
```

## Full end-to-end test

```bash
python test_agent.py --integration
```

This loads/downloads `uma-s-1p2`, then performs a real CatHub-matched
Cu(111)/H single point and constrained relaxation. Expected outcome:

```text
PASS: real UMA integration test
```

Approximate reference results may vary slightly with software versions:

- CatHub DFT: about -0.0536 eV
- UMA single point: about -0.0362 eV
- UMA relaxed: about -0.0370 eV

## Try natural language

Preview a validated plan without computing:

```bash
python vibe_agent.py "Benchmark CHx on Cu, Ag and Pd(111)"
```

Execute after confirmation:

```bash
python vibe_agent.py "比较 Cu、Ag 和 Pd(111) 上 C、CH、CH2、CH3 的吸附" --execute
```

## Current boundary

This release is a dataset-backed baseline for Cu/Ag/Au/Pt/Pd(111) and
H/O/OH/C/CH/CH2/CH3. It deliberately rejects unsupported facets instead of
inventing structures or reference energies.
