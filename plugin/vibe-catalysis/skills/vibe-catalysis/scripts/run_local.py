#!/usr/bin/env python3
"""Execute the Vibe Catalysis backend bundled with this skill."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _works(python: Path) -> bool:
    if not python.is_file():
        return False
    return subprocess.run(
        [str(python), "-c", "import ase, fairchem, numpy, matplotlib"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def resolve_python() -> Path:
    """Find a usable runtime even when Codex launches this script with system Python."""
    explicit = os.environ.get("VIBE_CATALYSIS_PYTHON")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path(sys.executable),
        Path("/opt/anaconda3/envs/cathub-uma/bin/python"),
        Path.home() / "anaconda3/envs/cathub-uma/bin/python",
        Path.home() / "miniconda3/envs/cathub-uma/bin/python",
        Path.home() / "mambaforge/envs/cathub-uma/bin/python",
    ]
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.expanduser().resolve()
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if _works(candidate):
            return candidate
    raise RuntimeError(
        "No Python with ASE, fairchem-core, NumPy, and Matplotlib was found. "
        "Checked: " + ", ".join(checked) + ". Set VIBE_CATALYSIS_PYTHON to override."
    )


def resolve() -> tuple[Path, Path]:
    python = resolve_python()
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
            [str(python), "-c", "import ase, fairchem, numpy, matplotlib; print('ASE, FAIR-Chem, NumPy, and Matplotlib available')"],
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
