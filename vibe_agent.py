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
PREDICTION_ADSORBATES = ("CO", "CHO", "COH", "CHOH", "CH2OH", "N", "N2", "NH", "NH2", "NH3")
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
    "NH3": ("nh3", "nh₃", "ammonia", "氨"),
    "NH2": ("nh2", "nh₂", "amino", "氨基"),
    "NH": ("nh", "imide", "亚氨基"),
    "N2": ("n2", "n₂", "nitrogen", "氮气"),
    "N": ("atomic nitrogen", "adsorbed nitrogen", "吸附氮", "氮原子"),
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


def parse_prompt(prompt: str, uploaded_structure: str | None = None) -> dict:
    original = prompt.replace("（", "(").replace("）", ")")
    text = prompt.lower().replace("（", "(").replace("）", ")")
    water_environment = any(term in text for term in (
        "水环境", "水相", "显式水", "有水", "含水", "水层", "冰层",
        "aqueous", "in water", "water environment", "explicit water",
    ))
    electrochemical_h_transfer = any(term in text for term in (
        "加氢", "脱氢", "volmer", "hydrogenation", "dehydrogenation",
        "proton transfer", "pcet",
    ))
    barrier_request = any(term in text for term in (
        "能垒", "势垒", "过渡态", "反应路径", "barrier", "neb", "transition state",
    ))
    if contains_alias(text, "co2"):
        raise ValueError("CO2 is not yet supported by the automatic builder; no calculation was started.")
    metals = []
    for symbol, aliases in METAL_ALIASES.items():
        exact_symbol = re.search(rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])", original)
        facet_adjacent = re.search(rf"(?<![a-z]){symbol.lower()}\s*\(?\s*(?:0001|10\s*[-−m]\s*10|[0-9]{{3}})", text)
        named = any(contains_alias(text, alias) for alias in aliases)
        if exact_symbol or facet_adjacent or named:
            metals.append(symbol)
    adsorbates = []
    masked = text
    for symbol in metals:
        masked = re.sub(
            rf"(?<![a-z]){symbol.lower()}\s*(?=\(?\s*(?:0001|10\s*[-−m]\s*10|[0-9]{{3}}))",
            " ", masked,
        )
        for alias in METAL_ALIASES[symbol]:
            masked = masked.replace(alias.lower(), " ")
    for species in ("CH2OH", "CHOH", "COH", "CHO", "CO", "NH3", "NH2", "NH", "N2", "CH3", "CH2", "OH", "CH"):
        if any(contains_alias(masked, a) for a in ADS_ALIASES[species]):
            adsorbates.append(species)
            for alias in ADS_ALIASES[species]: masked = masked.replace(alias.lower(), " ")
    for species in ("C", "H", "O", "N"):
        if any(contains_alias(masked, a) for a in ADS_ALIASES[species]): adsorbates.append(species)
    if any(term in text for term in ("chx", "chₓ", "碳氢", "hydrocarbon fragments")):
        adsorbates.extend(("C", "CH", "CH2", "CH3"))
    if any(term in text for term in ("all noble", "all supported", "所有贵金属", "全部贵金属")):
        metals = list(DATASET_METALS)
    if not metals and not uploaded_structure:
        metals = list(DATASET_METALS)
    if not adsorbates:
        if uploaded_structure:
            raise ValueError("Name the adsorbate/intermediate to add, or pass its structure file. Existing atoms remain catalyst atoms.")
        adsorbates = ["C", "CH", "CH2", "CH3"]
    metals = [x for x in PREDICTION_METALS if x in metals]
    adsorbates = [x for x in SUPPORTED_ADSORBATES if x in set(adsorbates)]
    prediction_species = [x for x in adsorbates if x in PREDICTION_ADSORBATES]
    dataset_species = [x for x in adsorbates if x in DATASET_ADSORBATES]
    if uploaded_structure and dataset_species:
        raise ValueError("Uploaded-structure modelling currently supports CO, CHO/HCO, COH, CHOH, and CH2OH.")
    if dataset_species and not prediction_species:
        metals = [metal for metal in DATASET_METALS if metal in metals]
    if prediction_species and dataset_species:
        raise ValueError("A single job cannot mix database-backed and automatically modelled adsorbates yet; split it into two requests.")
    if len(metals) != 1 and prediction_species and not uploaded_structure:
        raise ValueError("Automatic modelling currently runs one metal per job; submit separate requests for each metal.")
    crystal_structure = None
    if uploaded_structure:
        metals = []
        crystal_structure = "uploaded"
    elif prediction_species:
        crystal_structure = reference_states[atomic_numbers[metals[0]]]["symmetry"]
    facet_match = re.search(r"(?<!\d)(0001|10\s*[-−m]\s*10|[0-9]{3})(?!\d)", text)
    if uploaded_structure:
        facet = "custom"
    elif facet_match:
        facet = facet_match.group(1).replace(" ", "").replace("−", "-").replace("10-10", "10m10")
    elif crystal_structure == "bcc":
        facet = "110"
    elif crystal_structure == "hcp":
        facet = "0001"
    else:
        facet = "111"
    if water_environment and electrochemical_h_transfer and barrier_request:
        return {
            "metals": metals, "facet": facet, "adsorbates": adsorbates,
            "crystal_structure": crystal_structure,
            "mode": "aqueous_electrochemical_barrier",
            "interface_model": {
                "explicit_water": True,
                "default_builder": "build_periodic_ice_layer.py",
                "water_count": 6,
                "coverage_ML": 2 / 3,
                "periodicity": "catalyst and water layer periodic in-plane",
                "motif": "H-down (sqrt(3)xsqrt(3))R30-degree hexagonal honeycomb",
                "preserve_water_atoms_in_all_neb_images": True,
            },
            "reaction_kind": "electrochemical_hydrogen_transfer",
            "aqueous_hydrogen_transfer_semantics": {
                "hydrogenation": "transfer H from the nearest explicit water to the catalyst surface or adsorbate; leave an OH-minus-like donor",
                "dehydrogenation": "transfer adsorbate/surface H to the nearest explicit water; form an H3O-plus-like acceptor",
                "endpoint_builder": "build_aqueous_h_transfer.py",
                "forbid_Hstar_substitution": True,
            },
            "multistep_solvent_policy": {
                "shared_pristine_water_template": True,
                "build_each_step_independently": True,
                "same_water_oxygen_count_each_step": True,
                "inherit_previous_final_water_coordinates": False,
                "combine_only_reaction_energies_and_barriers": True,
            },
            "calculations": ["construct_interface", "relax_endpoints", "ci_neb"],
            "execution_guard": "Construct and inspect the periodic interface before UMA; charged/constant-potential claims require a compatible method.",
        }
    if prediction_species and not uploaded_structure and crystal_structure == "hcp" and facet not in {"0001", "10m10"}:
        raise ValueError("Arbitrary Miller generation currently supports cubic fcc/bcc crystals; upload prepared hcp high-index surfaces.")
    if dataset_species and facet != "111":
        raise ValueError(f"The packaged Catalysis-Hub baseline supports only facet 111, not {facet}.")
    mode = ("ase_uploaded_prediction" if uploaded_structure else
            "ase_automatic_prediction" if prediction_species else "cathub_dataset_benchmark")
    return {
        "metals": metals, "facet": facet, "adsorbates": adsorbates,
        "crystal_structure": crystal_structure,
        "mode": mode,
        "reference_dataset": None if prediction_species else "MamunHighT2019", "calculator": "uma-s-1p2",
        "task": "oc20", "calculations": ["single_point", "constrained_relaxation"],
        "uploaded_structure": str(Path(uploaded_structure).expanduser().resolve()) if uploaded_structure else None,
        "constraints": ("Preserve uploaded constraints when available; otherwise ASE FixAtoms on bottom 2 layers" if uploaded_structure
                        else "ASE FixAtoms on bottom 2 of 4 generated slab layers" if prediction_species
                        else "deposited FixAtoms (bottom 8 of 12 slab atoms)"),
        "optimizer": {"name": "LBFGS", "fmax_eV_per_A": 0.05, "max_steps": 100},
        "site_enumeration": ("automatic ontop/bridge/hollow coordinate discovery on the uploaded top layer" if uploaded_structure
                             else "all ASE named high-symmetry sites available for the selected surface" if prediction_species else None),
        "orientation_enumeration": ("CO: C-down and O-down; larger intermediates: 0/120/240 degree azimuths" if prediction_species else None),
        "scientific_label": ("UMA prediction on a user-supplied catalyst structure; no DFT reference" if uploaded_structure
                             else "UMA prediction on ASE-generated structures; no DFT reference" if prediction_species
                             else "Catalysis-Hub DFT benchmark"),
        "outputs": (["plan", "candidate_csv", "summary_json", "energy_visualization", "top_views", "initial_and_final_structures", "best_structure"]
                    if prediction_species else ["json", "csv", "metrics", "parity_plot"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Multilingual natural-language UMA/CatHub agent")
    parser.add_argument("prompt", nargs="+", help="Natural-language calculation request")
    parser.add_argument("--structure", type=Path, help="Uploaded catalyst slab/framework (CIF, POSCAR/CONTCAR, XYZ, EXTXYZ, TRAJ, or another ASE-readable file)")
    parser.add_argument("--active-atom-indices", nargs="+", type=int, help="Zero-based catalyst atom indices defining the candidate active region")
    parser.add_argument("--site-xy", nargs=2, type=float, action="append", metavar=("X", "Y"), help="Explicit Cartesian adsorption coordinate; repeat for multiple sites")
    parser.add_argument("--potential-v", type=float, default=0.0, help="CHE potential vs SHE")
    parser.add_argument("--ph", type=float, default=0.0)
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--execute", action="store_true", help="Run after printing the validated plan")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    args = parser.parse_args()
    prompt = " ".join(args.prompt)
    try:
        plan = parse_prompt(prompt, str(args.structure) if args.structure else None)
    except ValueError as error:
        print(f"REQUEST REJECTED: {error}", file=sys.stderr)
        raise SystemExit(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_name = f"nl_job_{stamp}"
    payload = {"schema_version": 2, "original_prompt": prompt, "parser": "local_multilingual_rules_v2", **plan}
    plan_path = ROOT / "results" / f"{job_name}_plan.json"
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nValidated plan saved to: {plan_path}")
    if not args.execute: return
    if not args.yes and input("Run this plan? [y/N] ").strip().lower() not in {"y", "yes", "是"}: return
    if plan["mode"] == "aqueous_electrochemical_barrier":
        if not plan["metals"]:
            raise ValueError("Automatic periodic ice construction currently requires a generated fcc(111) metal slab")
        interface_output = ROOT / "results" / job_name / "periodic_ice_interface"
        subprocess.run([
            sys.executable, str(ROOT / "build_periodic_ice_layer.py"),
            "--metal", plan["metals"][0], "--facet", plan["facet"],
            "--output", str(interface_output),
        ], cwd=ROOT, check=True)
        print(f"\nPeriodic six-water interface constructed: {interface_output}")
        print("No UMA energy was started by the natural-language planner; define atom-identical reaction endpoints before CI-NEB.")
        return
    if plan["mode"] in {"ase_automatic_prediction", "ase_uploaded_prediction"}:
        prediction_root = ROOT / "results" / job_name
        commands = []
        for adsorbate in plan["adsorbates"]:
            command = [sys.executable, str(ROOT / "predict_adsorption.py")]
            if plan["mode"] == "ase_uploaded_prediction":
                command += ["--structure", plan["uploaded_structure"]]
                if args.active_atom_indices:
                    command += ["--active-atom-indices", *map(str, args.active_atom_indices)]
                for xy in args.site_xy or []:
                    command += ["--site-xy", *map(str, xy)]
            else:
                command += ["--metal", plan["metals"][0], "--facet", plan["facet"]]
            command += ["--adsorbate", adsorbate, "--output", str(prediction_root / adsorbate)]
            commands.append(command)
    else:
        command = [sys.executable, str(ROOT / "run_catbench_subset.py"), "--source", str(ROOT / "raw_data/Mamun_noble_C1_subset_adsorption.json"), "--benchmark-name", job_name, "--mlip-name", f"UMA-{job_name}", "--metals", *plan["metals"], "--adsorbates", *plan["adsorbates"]]
    if plan["mode"] in {"ase_automatic_prediction", "ase_uploaded_prediction"}:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
        if len(commands) > 1:
            subprocess.run([
                sys.executable, str(ROOT / "visualize_results.py"),
                *[str(prediction_root / species) for species in plan["adsorbates"]],
                "--output", str(prediction_root / "adsorption_energy_profile.png"),
                "--potential-v", str(args.potential_v), "--ph", str(args.ph),
                "--temperature-k", str(args.temperature_k),
            ], cwd=ROOT, check=True)
    else:
        subprocess.run(command, cwd=ROOT, check=True)
    if plan["mode"] in {"ase_automatic_prediction", "ase_uploaded_prediction"}:
        print(f"\nCalculation complete. Automatic prediction result: {ROOT / 'results' / job_name}")
    else:
        print(f"\nCalculation complete. Raw CatBench result: {ROOT / 'result' / f'UMA-{job_name}'}")


if __name__ == "__main__":
    main()
