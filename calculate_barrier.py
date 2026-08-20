#!/usr/bin/env python3
"""ASE + FAIR-Chem UMA NEB/CI-NEB activation-barrier workflow."""

import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from fairchem.core import FAIRChemCalculator, pretrained_mlip


def validate_endpoints(initial, final):
    if len(initial) != len(final): raise ValueError("NEB endpoints must contain the same atoms")
    if initial.get_chemical_symbols() != final.get_chemical_symbols(): raise ValueError("NEB endpoint atom ordering differs")
    if initial.pbc.tolist() != final.pbc.tolist(): raise ValueError("NEB endpoint PBC differs")
    if not np.allclose(initial.cell.array, final.cell.array, atol=1e-5): raise ValueError("NEB endpoint cells differ")


def main():
    p = argparse.ArgumentParser(description="Calculate an elementary barrier with UMA NEB/CI-NEB")
    p.add_argument("--initial", type=Path, required=True); p.add_argument("--final", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--images", type=int, default=7)
    p.add_argument("--fmax", type=float, default=0.05); p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--method", choices=("aseneb", "improvedtangent", "eb", "spline", "string"), default="aseneb")
    p.add_argument("--interpolation", choices=("idpp", "linear"), default="idpp")
    p.add_argument("--climb", action="store_true"); p.add_argument("--model", default="uma-s-1p2"); p.add_argument("--device", default="cpu")
    p.add_argument("--initial-label", default="Initial state"); p.add_argument("--final-label", default="Final state")
    a = p.parse_args()
    if a.images < 3: raise ValueError("NEB needs at least three total images")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    initial, final = read(a.initial.resolve(), -1), read(a.final.resolve(), -1); validate_endpoints(initial, final)
    images = [initial] + [initial.copy() for _ in range(a.images - 2)] + [final]
    neb = NEB(images, climb=a.climb, method=a.method); neb.interpolate(method=a.interpolation, mic=True)
    predictor = pretrained_mlip.get_predict_unit(a.model, device=a.device)
    for image in images: image.calc = FAIRChemCalculator(predictor, task_name="oc20")
    opt = FIRE(neb, trajectory=str(out / "neb.traj"), logfile=str(out / "neb.log")); opt.run(fmax=a.fmax, steps=a.max_steps)
    energies = [float(image.get_potential_energy()) for image in images]; ts = int(np.argmax(energies))
    rows = []
    for i, (image, energy) in enumerate(zip(images, energies)):
        path = out / f"image_{i:02d}.extxyz"; write(path, image)
        rows.append({"image": i, "energy_eV": energy, "relative_energy_eV": energy - energies[0], "structure": str(path)})
    with (out / "neb_energies.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = {"status": "converged" if opt.converged() else "not_converged", "method": "CI-NEB" if a.climb else "NEB",
              "transition_state_image": ts, "forward_barrier_eV": energies[ts] - energies[0],
              "reverse_barrier_eV": energies[ts] - energies[-1], "reaction_energy_eV": energies[-1] - energies[0],
              "optimizer_steps": int(opt.nsteps), "energies": rows,
              "scientific_warning": "UMA NEB prediction; validate important saddles with consistent DFT and frequencies."}
    result_path = out / "barrier.json"; result_path.write_text(json.dumps(result, indent=2)); ts_path = out / "transition_state_candidate.extxyz"; write(ts_path, images[ts])
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_barrier.py")), str(result_path),
                    "--output", str(out / "barrier_diagram"), "--initial-label", a.initial_label,
                    "--final-label", a.final_label, "--initial-structure", str(a.initial.resolve()),
                    "--transition-state-structure", str(ts_path), "--final-structure", str(a.final.resolve())], check=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
