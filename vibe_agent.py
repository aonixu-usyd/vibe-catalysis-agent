#!/usr/bin/env python3
"""Multilingual natural-language front end for the CatHub × UMA workflow.

The local parser is deterministic and free.  It converts natural language into
a validated calculation plan; only the validated plan is allowed to reach the
scientific backend.  An optional OpenAI-compatible parser can be added later
without changing that backend contract.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUPPORTED_METALS = ("Cu", "Ag", "Au", "Pt", "Pd")
SUPPORTED_ADSORBATES = ("H", "O", "OH", "C", "CH", "CH2", "CH3")

METAL_ALIASES = {
    "Cu": ("cu", "copper", "铜", "銅", "cobre", "cuivre", "kupfer", "銅"),
    "Ag": ("ag", "silver", "银", "銀", "plata", "argent", "silber"),
    "Au": ("au", "gold", "金", "oro", "or", "gold"),
    "Pt": ("pt", "platinum", "铂", "鉑", "platino", "platine", "platin"),
    "Pd": ("pd", "palladium", "钯", "鈀", "paladio", "palladium"),
}
ADS_ALIASES = {
    "CH3": ("ch3", "ch₃", "methyl", "甲基", "metilo", "méthyle"),
    "CH2": ("ch2", "ch₂", "methylene", "亚甲基", "metileno", "méthylène"),
    "CH": ("ch", "methylidyne", "次甲基", "metilidino", "méthylidyne"),
    "OH": ("oh", "hydroxyl", "羟基", "氢氧根", "hidroxilo", "hydroxyle"),
    "C": ("c", "carbon", "atomic carbon", "碳", "carbone", "carbono", "kohlenstoff"),
    "H": ("h", "hydrogen", "atomic hydrogen", "氢", "氫", "hidrógeno", "hydrogène", "wasserstoff"),
    "O": ("o", "oxygen", "atomic oxygen", "氧", "oxígeno", "oxygène", "sauerstoff"),
}


def contains_alias(text: str, alias: str) -> bool:
    alias = alias.lower()
    if re.fullmatch(r"[a-z0-9]+", alias):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text))
    return alias in text


def parse_prompt(prompt: str) -> dict:
    text = prompt.lower().replace("（", "(").replace("）", ")")
    metals = [symbol for symbol, aliases in METAL_ALIASES.items() if any(contains_alias(text, a) for a in aliases)]
    adsorbates = []
    masked = text
    for species in ("CH3", "CH2", "OH", "CH"):
        if any(contains_alias(masked, a) for a in ADS_ALIASES[species]):
            adsorbates.append(species)
            for alias in ADS_ALIASES[species]: masked = masked.replace(alias.lower(), " ")
    for species in ("C", "H", "O"):
        if any(contains_alias(masked, a) for a in ADS_ALIASES[species]): adsorbates.append(species)
    if any(term in text for term in ("chx", "chₓ", "碳氢", "hydrocarbon fragments")):
        adsorbates.extend(("C", "CH", "CH2", "CH3"))
    if any(term in text for term in ("all noble", "all supported", "所有贵金属", "全部贵金属")):
        metals = list(SUPPORTED_METALS)
    if not metals: metals = list(SUPPORTED_METALS)
    if not adsorbates: adsorbates = ["C", "CH", "CH2", "CH3"]
    metals = [x for x in SUPPORTED_METALS if x in metals]
    adsorbates = [x for x in SUPPORTED_ADSORBATES if x in set(adsorbates)]
    facet_match = re.search(r"(?:facet|surface|晶面|面)?\s*\(?\s*(\d)\s*[, ]?\s*(\d)\s*[, ]?\s*(\d)\s*\)?", text)
    facet = "".join(facet_match.groups()) if facet_match else "111"
    if facet != "111": raise ValueError(f"This dataset-backed baseline currently supports only facet 111, not {facet}.")
    return {
        "metals": metals, "facet": facet, "adsorbates": adsorbates,
        "reference_dataset": "MamunHighT2019", "calculator": "uma-s-1p2",
        "task": "oc20", "calculations": ["single_point", "constrained_relaxation"],
        "constraints": "deposited FixAtoms (bottom 8 of 12 slab atoms)",
        "optimizer": {"name": "LBFGS", "fmax_eV_per_A": 0.05, "max_steps": 100},
        "outputs": ["json", "csv", "metrics", "parity_plot"],
    }


def main():
    parser = argparse.ArgumentParser(description="Multilingual natural-language UMA/CatHub agent")
    parser.add_argument("prompt", nargs="+", help="Natural-language calculation request")
    parser.add_argument("--execute", action="store_true", help="Run after printing the validated plan")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    args = parser.parse_args()
    prompt = " ".join(args.prompt)
    plan = parse_prompt(prompt)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_name = f"nl_job_{stamp}"
    payload = {"schema_version": 1, "original_prompt": prompt, "parser": "local_multilingual_rules_v1", **plan}
    plan_path = ROOT / "results" / f"{job_name}_plan.json"
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nValidated plan saved to: {plan_path}")
    if not args.execute: return
    if not args.yes and input("Run this plan? [y/N] ").strip().lower() not in {"y", "yes", "是"}: return
    command = [sys.executable, str(ROOT / "run_catbench_subset.py"), "--source", str(ROOT / "raw_data/Mamun_noble_C1_subset_adsorption.json"), "--benchmark-name", job_name, "--mlip-name", f"UMA-{job_name}", "--metals", *plan["metals"], "--adsorbates", *plan["adsorbates"]]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"\nCalculation complete. Raw CatBench result: {ROOT / 'result' / f'UMA-{job_name}'}")


if __name__ == "__main__":
    main()
