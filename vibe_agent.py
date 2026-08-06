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

from ase.data import atomic_names, atomic_numbers, chemical_symbols, reference_states

ROOT = Path(__file__).resolve().parent
DATASET_METALS = ("Cu", "Ag", "Au", "Pt", "Pd")
NONMETALLIC_REFERENCE_ELEMENTS = {"Ne", "Ar", "Kr", "Xe", "Se", "Te"}
PREDICTION_METALS = tuple(
    symbol for symbol in chemical_symbols[1:]
    if reference_states[atomic_numbers[symbol]]
    and reference_states[atomic_numbers[symbol]].get("symmetry") in {"fcc", "bcc", "hcp"}
    and symbol not in NONMETALLIC_REFERENCE_ELEMENTS
)
DATASET_ADSORBATES = ("H", "O", "OH", "C", "CH", "CH2", "CH3")
PREDICTION_ADSORBATES = ("CO", "CHO", "COH", "CHOH", "CH2OH")
SUPPORTED_ADSORBATES = DATASET_ADSORBATES + PREDICTION_ADSORBATES

CHINESE_METAL_NAMES = {
    "Li": ("锂",), "Be": ("铍",), "Na": ("钠",), "Mg": ("镁",), "Al": ("铝",),
    "K": ("钾",), "Ca": ("钙",), "Sc": ("钪",), "Ti": ("钛",), "V": ("钒",),
    "Cr": ("铬",), "Fe": ("铁",), "Co": ("钴",), "Ni": ("镍",), "Cu": ("铜", "銅"),
    "Zn": ("锌",), "Y": ("钇",), "Zr": ("锆",), "Nb": ("铌",), "Mo": ("钼",),
    "Tc": ("锝",), "Ru": ("钌",), "Rh": ("铑",), "Pd": ("钯",), "Ag": ("银", "銀"),
    "Cd": ("镉",), "Hf": ("铪",), "Ta": ("钽",), "W": ("钨",), "Re": ("铼",),
    "Os": ("锇",), "Ir": ("铱",), "Pt": ("铂", "鉑"), "Au": ("金",), "Pb": ("铅",),
}
LANGUAGE_METAL_NAMES = {
    "Cu": ("cobre", "cuivre", "kupfer"),
    "Ag": ("plata", "argent", "silber"),
    "Au": ("oro",),
    "Pt": ("platino", "platine", "platin"),
    "Pd": ("paladio",),
}
METAL_ALIASES = {
    symbol: ((atomic_names[atomic_numbers[symbol]].lower(),)
             + CHINESE_METAL_NAMES.get(symbol, ())
             + LANGUAGE_METAL_NAMES.get(symbol, ()))
    for symbol in PREDICTION_METALS
}
ADS_ALIASES = {
    "CH2OH": ("ch2oh", "ch₂oh", "hydroxymethyl", "羟甲基"),
    "CHOH": ("choh", "hydroxymethylene", "羟基亚甲基"),
    "COH": ("coh", "hydroxycarbyne", "羟基碳"),
    "CHO": ("cho", "hco", "formyl", "甲酰基"),
    "CO": ("co", "carbon monoxide", "一氧化碳"),
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
    original = prompt.replace("（", "(").replace("）", ")")
    text = prompt.lower().replace("（", "(").replace("）", ")")
    if contains_alias(text, "co2"):
        raise ValueError("CO2 is not yet supported by the automatic builder; no calculation was started.")
    metals = []
    for symbol, aliases in METAL_ALIASES.items():
        exact_symbol = re.search(rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])", original)
        facet_adjacent = re.search(rf"(?<![a-z]){symbol.lower()}\s*\(?\s*(?:0001|10\s*[-−m]\s*10|111|110|100)", text)
        named = any(contains_alias(text, alias) for alias in aliases)
        if exact_symbol or facet_adjacent or named:
            metals.append(symbol)
    adsorbates = []
    masked = text
    for symbol in metals:
        masked = re.sub(
            rf"(?<![a-z]){symbol.lower()}\s*(?=\(?\s*(?:0001|10\s*[-−m]\s*10|111|110|100))",
            " ", masked,
        )
        for alias in METAL_ALIASES[symbol]:
            masked = masked.replace(alias.lower(), " ")
    for species in ("CH2OH", "CHOH", "COH", "CHO", "CO", "CH3", "CH2", "OH", "CH"):
        if any(contains_alias(masked, a) for a in ADS_ALIASES[species]):
            adsorbates.append(species)
            for alias in ADS_ALIASES[species]: masked = masked.replace(alias.lower(), " ")
    for species in ("C", "H", "O"):
        if any(contains_alias(masked, a) for a in ADS_ALIASES[species]): adsorbates.append(species)
    if any(term in text for term in ("chx", "chₓ", "碳氢", "hydrocarbon fragments")):
        adsorbates.extend(("C", "CH", "CH2", "CH3"))
    if any(term in text for term in ("all noble", "all supported", "所有贵金属", "全部贵金属")):
        metals = list(DATASET_METALS)
    if not metals:
        metals = list(DATASET_METALS)
    if not adsorbates: adsorbates = ["C", "CH", "CH2", "CH3"]
    metals = [x for x in PREDICTION_METALS if x in metals]
    adsorbates = [x for x in SUPPORTED_ADSORBATES if x in set(adsorbates)]
    prediction_species = [x for x in adsorbates if x in PREDICTION_ADSORBATES]
    dataset_species = [x for x in adsorbates if x in DATASET_ADSORBATES]
    if dataset_species and not prediction_species:
        metals = [metal for metal in DATASET_METALS if metal in metals]
    if prediction_species and dataset_species:
        raise ValueError("A single job cannot mix database-backed and automatically modelled adsorbates yet; split it into two requests.")
    if len(metals) != 1 and prediction_species:
        raise ValueError("Automatic modelling currently runs one metal per job; submit separate requests for each metal.")
    crystal_structure = None
    if prediction_species:
        crystal_structure = reference_states[atomic_numbers[metals[0]]]["symmetry"]
    facet_match = re.search(r"(?<!\d)(0001|10\s*[-−m]\s*10|111|110|100)(?!\d)", text)
    if facet_match:
        facet = facet_match.group(1).replace(" ", "").replace("−", "-").replace("10-10", "10m10")
    elif crystal_structure == "bcc":
        facet = "110"
    elif crystal_structure == "hcp":
        facet = "0001"
    else:
        facet = "111"
    allowed = {"fcc": {"111", "100", "110"}, "bcc": {"111", "100", "110"}, "hcp": {"0001", "10m10"}}
    if prediction_species and facet not in allowed[crystal_structure]:
        readable = ", ".join(sorted(x.replace("m", "-") for x in allowed[crystal_structure]))
        raise ValueError(f"Unsupported {crystal_structure} facet {facet}; available facets: {readable}.")
    if dataset_species and facet != "111":
        raise ValueError(f"The packaged Catalysis-Hub baseline supports only facet 111, not {facet}.")
    mode = "ase_automatic_prediction" if prediction_species else "cathub_dataset_benchmark"
    return {
        "metals": metals, "facet": facet, "adsorbates": adsorbates,
        "crystal_structure": crystal_structure,
        "mode": mode,
        "reference_dataset": None if prediction_species else "MamunHighT2019", "calculator": "uma-s-1p2",
        "task": "oc20", "calculations": ["single_point", "constrained_relaxation"],
        "constraints": ("ASE FixAtoms on bottom 2 of 4 generated slab layers" if prediction_species
                        else "deposited FixAtoms (bottom 8 of 12 slab atoms)"),
        "optimizer": {"name": "LBFGS", "fmax_eV_per_A": 0.05, "max_steps": 100},
        "site_enumeration": ("all ASE named high-symmetry sites available for the selected surface" if prediction_species else None),
        "orientation_enumeration": ("CO: C-down and O-down; larger intermediates: 0/120/240 degree azimuths" if prediction_species else None),
        "scientific_label": ("UMA prediction on ASE-generated structures; no DFT reference" if prediction_species
                             else "Catalysis-Hub DFT benchmark"),
        "outputs": (["plan", "candidate_csv", "summary_json", "initial_and_final_structures", "best_structure"]
                    if prediction_species else ["json", "csv", "metrics", "parity_plot"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Multilingual natural-language UMA/CatHub agent")
    parser.add_argument("prompt", nargs="+", help="Natural-language calculation request")
    parser.add_argument("--execute", action="store_true", help="Run after printing the validated plan")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    args = parser.parse_args()
    prompt = " ".join(args.prompt)
    try:
        plan = parse_prompt(prompt)
    except ValueError as error:
        print(f"REQUEST REJECTED: {error}", file=sys.stderr)
        raise SystemExit(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_name = f"nl_job_{stamp}"
    payload = {"schema_version": 1, "original_prompt": prompt, "parser": "local_multilingual_rules_v1", **plan}
    plan_path = ROOT / "results" / f"{job_name}_plan.json"
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nValidated plan saved to: {plan_path}")
    if not args.execute: return
    if not args.yes and input("Run this plan? [y/N] ").strip().lower() not in {"y", "yes", "是"}: return
    if plan["mode"] == "ase_automatic_prediction":
        command = [
            sys.executable, str(ROOT / "predict_adsorption.py"),
            "--metal", plan["metals"][0], "--facet", plan["facet"],
            "--adsorbate", plan["adsorbates"][0],
            "--output", str(ROOT / "results" / job_name),
        ]
    else:
        command = [sys.executable, str(ROOT / "run_catbench_subset.py"), "--source", str(ROOT / "raw_data/Mamun_noble_C1_subset_adsorption.json"), "--benchmark-name", job_name, "--mlip-name", f"UMA-{job_name}", "--metals", *plan["metals"], "--adsorbates", *plan["adsorbates"]]
    subprocess.run(command, cwd=ROOT, check=True)
    if plan["mode"] == "ase_automatic_prediction":
        print(f"\nCalculation complete. Automatic prediction result: {ROOT / 'results' / job_name}")
    else:
        print(f"\nCalculation complete. Raw CatBench result: {ROOT / 'result' / f'UMA-{job_name}'}")


if __name__ == "__main__":
    main()
