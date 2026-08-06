# Automatic CO/Cu(111) UMA demonstration

Natural-language request:

```text
计算CO在Cu(111)上的吸附能
```

Settings: ASE fcc(111) 3×3×4 slab, 10 Å vacuum, bottom two layers fixed,
`ontop/bridge/fcc/hcp`, C-down and O-down, UMA `uma-s-1p2` `oc20`, ASE LBFGS,
`fmax=0.05 eV/Å`, maximum 100 steps.

Result: 8/8 calculations completed and 4/8 passed geometry checks. All O-down
structures desorbed. The lowest accepted prediction was hcp C-down:

```text
E_ads = E(slab+CO) - E(clean slab) - E(CO gas) = -0.4747117 eV
```

This is an UMA prediction on automatically generated structures, not a
Catalysis-Hub DFT reference. See `candidates.csv` for every candidate and
`summary.json` for machine-readable metadata.
