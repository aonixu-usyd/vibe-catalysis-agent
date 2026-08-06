#!/usr/bin/env python3
"""Run a reproducible CatHub/MamunHighT2019 subset with UMA via CatBench.

The subset contains the lowest-DFT-energy accepted site for each combination of
Cu/Ag/Au/Pt/Pd and H/O/OH/C/CH/CH2/CH3.  CatBench preserves the deposited ASE
constraints (or its documented inferred constraints) and evaluates both a
single point and a constrained relaxation.
"""

import argparse
import json
import re
from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from fairchem.core import FAIRChemCalculator, pretrained_mlip


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "raw_data/Mamun_noble_C1_subset_adsorption.json"
METALS = ("Cu", "Ag", "Au", "Pt", "Pd")
ADSORBATES = ("H", "O", "OH", "C", "CH", "CH2", "CH3")


def product_from_key(key: str) -> str | None:
    match = re.search(r"->\s*([A-Za-z0-9]+)\*", key)
    return match.group(1) if match else None


def select_lowest(source: Path, metals=METALS, adsorbates=ADSORBATES) -> tuple[dict, list[dict]]:
    data = json.loads(source.read_text())
    selected = {}
    manifest = []
    for metal in metals:
        for adsorbate in adsorbates:
            prefix = f"{metal}12_"
            candidates = [
                (key, value) for key, value in data.items()
                if key.startswith(prefix) and isinstance(value, dict)
                and product_from_key(key) == adsorbate
            ]
            if not candidates:
                manifest.append({"metal": metal, "adsorbate": adsorbate, "status": "missing"})
                continue
            key, value = min(candidates, key=lambda item: item[1]["ref_ads_eng"])
            selected[key] = value
            manifest.append({
                "metal": metal,
                "adsorbate": adsorbate,
                "status": "selected",
                "reaction_key": key,
                "cathub_dft_eV": value["ref_ads_eng"],
                "constraint_source": value.get("constraint_source", "unknown"),
                "n_candidates": len(candidates),
            })

    refs = {
        item["ref"]
        for value in selected.values()
        for item in value["raw"].values()
        if "ref" in item
    }
    selected["_structures"] = {ref: data["_structures"][ref] for ref in refs}
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", default="uma-s-1p2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--metals", nargs="+", choices=METALS, default=list(METALS))
    parser.add_argument("--adsorbates", nargs="+", choices=ADSORBATES, default=list(ADSORBATES))
    parser.add_argument("--benchmark-name", default="Mamun_noble_C1_subset")
    parser.add_argument("--mlip-name", default="UMA-s-1p2-OC20")
    args = parser.parse_args()

    subset, manifest = select_lowest(args.source, args.metals, args.adsorbates)
    raw_dir = ROOT / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    subset_path = raw_dir / f"{args.benchmark_name}_adsorption.json"
    subset_path.write_text(json.dumps(subset))
    (ROOT / "results/subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    n_selected = sum(row["status"] == "selected" for row in manifest)
    print(f"Prepared {n_selected} reactions at {subset_path}")
    if args.prepare_only:
        return

    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    calculator = FAIRChemCalculator(predictor, task_name="oc20")
    AdsorptionCalculation(
        [calculator],
        mlip_name=args.mlip_name,
        benchmark=args.benchmark_name,
        save_files=False,
        f_crit_relax=0.05,
        n_crit_relax=100,
    ).run()


if __name__ == "__main__":
    main()
