#!/usr/bin/env python3
"""Offline parser tests and an optional real UMA integration test."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write

from vibe_agent import parse_prompt
from predict_adsorption import (
    build_candidate, build_slab, discover_custom_sites, gas_reference,
    hydrogen_reference, molecule_template,
)
from visualize_results import classify_che_comparison, compute_che_states, reaction_rows
from plot_barrier import plot_barrier, plot_top_views, plot_combined


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
    auto_cases = [
        ("计算CO在Cu111上的吸附能", "Cu", "CO"),
        ("Calculate CHO adsorption on Pt(111)", "Pt", "CHO"),
        ("计算COH在Pd(111)上的吸附", "Pd", "COH"),
        ("计算CHOH在Ag(111)上的吸附", "Ag", "CHOH"),
        ("计算CH2OH在Au(111)上的吸附", "Au", "CH2OH"),
    ]
    for prompt, metal, adsorbate in auto_cases:
        plan = parse_prompt(prompt)
        assert plan["mode"] == "ase_automatic_prediction"
        assert plan["metals"] == [metal]
        assert plan["adsorbates"] == [adsorbate]
    multi_plan = parse_prompt("计算CO和CHO在Cu(111)上的吸附能")
    assert multi_plan["adsorbates"] == ["CO", "CHO"]
    uploaded_plan = parse_prompt(
        "用上传的 POSCAR 计算 CO 吸附能", "/tmp/example/POSCAR"
    )
    assert uploaded_plan["mode"] == "ase_uploaded_prediction"
    assert uploaded_plan["metals"] == [] and uploaded_plan["facet"] == "custom"
    assert uploaded_plan["adsorbates"] == ["CO"]
    assert uploaded_plan["uploaded_structure"].endswith("/tmp/example/POSCAR")
    expanded_cases = [
        ("计算CO在Ni(100)上的吸附能", "Ni", "fcc", "100"),
        ("Calculate CO adsorption on Fe(110)", "Fe", "bcc", "110"),
        ("计算CHO在Co(0001)上的吸附能", "Co", "hcp", "0001"),
        ("Calculate CO on Mo(100)", "Mo", "bcc", "100"),
        ("计算CO在Ru(10-10)上的吸附能", "Ru", "hcp", "10m10"),
        ("Calculate NH2 adsorption on Pt(211)", "Pt", "fcc", "211"),
    ]
    for prompt, metal, crystal, facet in expanded_cases:
        plan = parse_prompt(prompt)
        assert plan["metals"] == [metal], (prompt, plan["metals"])
        assert plan["crystal_structure"] == crystal
        assert plan["facet"] == facet
    slab, fixed, metadata, sites = build_slab("Cu", "111", (2, 2, 4), 8.0, 2)
    assert len(slab) == 16 and len(fixed) == 8
    assert metadata["crystal_structure"] == "fcc" and set(sites) == {"ontop", "bridge", "fcc", "hcp"}
    custom_sites = discover_custom_sites(slab, ("ontop", "bridge", "hollow"), 0.6, 6)
    assert any(name.startswith("ontop_") for name in custom_sites)
    assert any(name.startswith("bridge_") for name in custom_sites)
    assert any(name.startswith("hollow_") for name in custom_sites)
    framework_sites = discover_custom_sites(slab, ("ontop",), 0.6, 6, [0, 3])
    assert len(framework_sites) == 2
    cof = Atoms(
        "CONH", positions=[(1, 1, 3), (3, 1, 3), (1, 3, 4), (3, 3, 5)],
        cell=[6, 6, 14], pbc=[True, True, False],
    )
    cof_sites = discover_custom_sites(cof, ("ontop", "bridge"), 0.6, 6, [0, 1])
    assert len([name for name in cof_sites if name.startswith("ontop_")]) == 2
    assert any(name.startswith("bridge_") for name in cof_sites)
    try:
        discover_custom_sites(slab, ("ontop",), 0.6, 6, [len(slab)])
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range active atom index was not rejected")
    fe, fe_fixed, fe_metadata, fe_sites = build_slab("Fe", "110", (2, 2, 4), 8.0, 2)
    assert fe_metadata["crystal_structure"] == "bcc" and "hollow" in fe_sites and fe_fixed
    co_slab, co_fixed, co_metadata, co_sites = build_slab("Co", "0001", (2, 2, 4), 8.0, 2)
    assert co_metadata["crystal_structure"] == "hcp" and "fcc" in co_sites and co_fixed
    co, bonds = molecule_template("CO", "C", 0)
    assert co.get_chemical_symbols() == ["C", "O"] and bonds == [(0, 1)]
    oh_down, _ = molecule_template("CO", "O", 0)
    assert oh_down.get_chemical_symbols() == ["C", "O"] and oh_down.positions[0, 2] > 0
    pt211, pt211_fixed, pt211_metadata, pt211_sites = build_slab("Pt", "211", (1, 1, 4), 8.0, 2)
    assert pt211_metadata["general_miller_builder"] and pt211_metadata["facet"] == "211"
    assert pt211_fixed and any(name.startswith("ontop") for name in pt211_sites)
    for species in ("N", "N2", "NH", "NH2", "NH3"):
        atoms, _ = molecule_template(species, "N", 0)
        assert atoms[0].symbol == "N"
    candidate, ads_indices, _ = build_candidate(slab, "CO", "ontop", "C", 0, 1.85)
    assert len(candidate) == 18 and len(ads_indices) == 2
    assert set(candidate.get_tags()[ads_indices]) == {2}
    assert gas_reference("CH2OH").pbc.tolist() == [False, False, False]
    assert hydrogen_reference().get_chemical_formula() == "H2"
    ref, che_energies, che_rows = compute_che_states(
        ["CO", "CHO"], [-100.0, -104.0], -6.0
    )
    assert ref == 0 and abs(che_energies[1] - (-1.0)) < 1e-12
    assert che_rows[1]["delta_E_CHE_eV"] == che_energies[1]
    assert classify_che_comparison(["CO", "CHO"]) == "single_reaction"
    assert classify_che_comparison(["CO", "CHO", "COH"]) == "branch_comparison"
    assert classify_che_comparison(["CO", "CHO", "CHOH"]) == "sequential_path"
    branches = reaction_rows(["CO", "CHO", "COH"], [0.0, -0.4, 0.2], 0, "branch_comparison")
    assert [row["reaction"] for row in branches] == ["CO* -> CHO*", "CO* -> COH*"]
    steps = reaction_rows(["CO", "CHO", "CHOH"], [0.0, -0.4, -0.7], 0, "sequential_path")
    assert abs(steps[0]["reaction_energy_eV"] + 0.4) < 1e-12
    assert abs(steps[1]["reaction_energy_eV"] + 0.3) < 1e-12
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        result = temporary / "barrier.json"
        result.write_text(json.dumps({
            "method": "CI-NEB", "forward_barrier_eV": 1.2,
            "reverse_barrier_eV": 1.6, "reaction_energy_eV": -0.4,
        }))
        output = temporary / "barrier_diagram"
        plot_barrier(result, output, "2N*", r"N$_2$* + *")
        for suffix in ("svg", "pdf", "png", "tiff"):
            assert output.with_suffix("." + suffix).is_file()
        assert output.with_name("barrier_diagram_source_data.csv").is_file()
        assert output.with_name("barrier_diagram_caption.txt").is_file()
        structures = []
        for name, distance in (("initial", 2.2), ("transition_state", 1.6), ("final", 1.1)):
            path = temporary / f"{name}.extxyz"
            write(path, Atoms("Pt2N2", positions=[(0, 0, 0), (2, 2, 0),
                                                   (0.8, 0.8, 1.5),
                                                   (0.8 + distance, 0.8, 1.5)],
                              cell=[5, 5, 10], pbc=[True, True, False]))
            structures.append(path)
        top_views = temporary / "structure_top_views"
        plot_top_views(*structures, top_views, "2N*", r"N$_2$* + *")
        for suffix in ("svg", "pdf", "png", "tiff"):
            assert top_views.with_suffix("." + suffix).is_file()
        assert top_views.with_name("structure_top_views_manifest.json").is_file()
        assert top_views.with_name("structure_top_views_caption.txt").is_file()
        combined = temporary / "barrier_and_top_views"
        plot_combined(result, *structures, combined, "2N*", r"N$_2$* + *")
        for suffix in ("svg", "pdf", "png", "tiff"):
            assert combined.with_suffix("." + suffix).is_file()
        pathway = temporary / "pathway.json"
        pathway.write_text(json.dumps({"title":"Test pathway","states":[
            {"id":"a","label":"A*","structure":str(structures[0])},
            {"id":"b","label":"B*","structure":str(structures[1])},
            {"id":"c","label":"C*","structure":str(structures[2])}],"steps":[
            {"id":"ab","reactant":"a","product":"b","mechanism":"chemical","delta_G_approx_eV":-0.4,"forward_barrier_eV":1.2,"transition_state_structure":str(structures[1])},
            {"id":"bc","reactant":"b","product":"c","mechanism":"chemical","delta_G_approx_eV":-0.3,"forward_barrier_eV":0.8,"transition_state_structure":str(structures[1])}]}))
        subprocess.run([sys.executable,str(ROOT/"plot_reaction_path.py"),str(pathway),"--output",str(temporary/"pathway")],check=True)
        assert (temporary/"pathway.png").is_file()
    print(f"PASS: {len(CASES)} dataset parser cases, {len(auto_cases) + len(expanded_cases)} generated-surface parser cases, uploaded-structure planning/site discovery, fcc/bcc/hcp builders, constraints/orientations, and 1 safety rejection")


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
