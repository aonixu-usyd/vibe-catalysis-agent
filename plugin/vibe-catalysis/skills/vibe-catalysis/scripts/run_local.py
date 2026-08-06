#!/usr/bin/env python3
"""Execute the Vibe Catalysis backend bundled with this skill."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def resolve() -> tuple[Path, Path]:
    python = Path(os.environ.get("VIBE_CATALYSIS_PYTHON", sys.executable)).expanduser().resolve()
    backend = Path(__file__).with_name("predict_adsorption.py").resolve()
    missing = [str(path) for path in (python, backend) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local Vibe Catalysis dependency: " + ", ".join(missing))
    return python, backend


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check", action="store_true")
    known, remaining = parser.parse_known_args()
    python, backend = resolve()
    if known.check:
        probe = subprocess.run(
            [str(python), "-c", "import ase, fairchem; print('ASE and FAIR-Chem available')"],
            check=True,
            text=True,
            capture_output=True,
        )
        print(probe.stdout.strip())
        print(f"Python: {python}")
        print(f"Backend: {backend}")
        return
    if not remaining:
        raise SystemExit("Pass backend arguments, e.g. --metal Cu --adsorbate CO --output /absolute/path")
    raise SystemExit(subprocess.run([str(python), str(backend), *remaining]).returncode)


if __name__ == "__main__":
    main()
