#!/usr/bin/env python3
"""Offline parser tests and an optional real UMA integration test."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from vibe_agent import parse_prompt


ROOT = Path(__file__).resolve().parent

CASES = [
    ("比较 Cu、Ag 和 Pt(111) 上 C、CH、CH2、CH3 的吸附", ["Cu", "Ag", "Pt"], ["C", "CH", "CH2", "CH3"]),
    ("Benchmark CHx on Cu, Ag and Pd(111)", ["Cu", "Ag", "Pd"], ["C", "CH", "CH2", "CH3"]),
    ("Calcular C y CH3 sobre cobre y paladio (111)", ["Cu", "Pd"], ["C", "CH3"]),
    ("Comparer C et CH sur cuivre et argent (111)", ["Cu", "Ag"], ["C", "CH"]),
    ("Berechne C und CH3 auf Kupfer und Platin (111)", ["Cu", "Pt"], ["C", "CH3"]),
    ("Cu、Ag、Au(111)で C、CH2、CH3 を比較", ["Cu", "Ag", "Au"], ["C", "CH2", "CH3"]),
]


def offline_tests():
    for prompt, metals, adsorbates in CASES:
        plan = parse_prompt(prompt)
        assert plan["metals"] == metals, (prompt, plan["metals"], metals)
        assert plan["adsorbates"] == adsorbates, (prompt, plan["adsorbates"], adsorbates)
        assert plan["facet"] == "111"
    try:
        parse_prompt("Calculate C on Cu(100)")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported Cu(100) request was not rejected")
    print(f"PASS: {len(CASES)} multilingual parser cases and 1 safety rejection")


def integration_test():
    name = "agent_integration_test"
    command = [
        sys.executable, str(ROOT / "run_catbench_subset.py"),
        "--source", str(ROOT / "raw_data/Mamun_noble_C1_subset_adsorption.json"),
        "--benchmark-name", name, "--mlip-name", f"UMA-{name}",
        "--metals", "Cu", "--adsorbates", "H",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    result_path = ROOT / "result" / f"UMA-{name}" / f"UMA-{name}_result.json"
    result = json.loads(result_path.read_text())
    key = next(key for key in result if key != "calculation_settings")
    record = result[key]
    assert "single_calculation" in record and "final" in record
    assert record["final"]["steps_total_adslab"] >= 0
    print("PASS: real UMA integration test")
    print(f"  case: {key}")
    print(f"  CatHub DFT: {record['reference']['ads_eng']:.4f} eV")
    print(f"  UMA single point: {record['single_calculation']['ads_eng']:.4f} eV")
    print(f"  UMA relaxed: {record['final']['ads_eng_median']:.4f} eV")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true", help="Also load UMA and run Cu(111)/H")
    args = parser.parse_args()
    offline_tests()
    if args.integration:
        integration_test()
    else:
        print("SKIP: UMA integration test (run with --integration)")


if __name__ == "__main__":
    main()
