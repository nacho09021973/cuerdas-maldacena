# tests/test_basico.py
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command and raise a useful error if it fails."""
    p = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        raise AssertionError(
            "Command failed\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  cwd: {cwd or REPO_ROOT}\n"
            f"  exit: {p.returncode}\n"
            "---- output ----\n"
            f"{p.stdout}\n"
            "--------------\n"
        )
    return p


def test_imports() -> None:
    """
    Sanity import test: these are commonly required by the pipeline/tests.
    Adjust the list if your repo has a different minimal set.
    """
    modules = ["numpy", "h5py", "pytest"]
    missing: list[str] = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    assert not missing, f"Missing or broken imports: {missing}"


def test_estructura_carpetas() -> None:
    """
    Basic repo structure sanity. Keep this conservative (few must-have paths).
    """
    must_exist = [
        REPO_ROOT / "run_pipeline.py",
        REPO_ROOT / "00_validate_io_contracts.py",
        REPO_ROOT / "01_generate_sandbox_geometries.py",
        REPO_ROOT / "02_emergent_geometry_engine.py",
        REPO_ROOT / "03_discover_bulk_equations.py",
        REPO_ROOT / "04_geometry_physics_contracts.py",
        REPO_ROOT / "05_analyze_bulk_equations.py",
        REPO_ROOT / "06_build_bulk_eigenmodes_dataset.py",
        REPO_ROOT / "07_emergent_lambda_sl_dictionary.py",
        REPO_ROOT / "08_build_holographic_dictionary.py",
        REPO_ROOT / "09_real_data_and_dictionary_contracts.py",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in must_exist if not p.exists()]
    assert not missing, f"Missing required paths: {missing}"


def test_py_compile_entrypoints() -> None:
    """
    Fast check: all entrypoints compile.
    """
    scripts = [
        "00_validate_io_contracts.py",
        "01_generate_sandbox_geometries.py",
        "02_emergent_geometry_engine.py",
        "03_discover_bulk_equations.py",
        "04_geometry_physics_contracts.py",
        "05_analyze_bulk_equations.py",
        "06_build_bulk_eigenmodes_dataset.py",
        "07_emergent_lambda_sl_dictionary.py",
        "08_build_holographic_dictionary.py",
        "09_real_data_and_dictionary_contracts.py",
        "run_pipeline.py",
    ]
    cmd = [sys.executable, "-m", "py_compile", *scripts]
    _run(cmd, cwd=REPO_ROOT)


def test_formato_h5_ejemplo() -> None:
    """
    Conservative: if an example H5 exists, verify it opens and has at least 1 key.
    If it doesn't exist (e.g., not committed), we skip rather than fail.
    """
    import h5py  # type: ignore

    candidate_paths = [
        REPO_ROOT / "fase12_data_boundary" / "ising_3d.h5",
        REPO_ROOT / "data" / "ising_3d.h5",
        REPO_ROOT / "data" / "fase12_data_boundary" / "ising_3d.h5",
    ]
    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        # Not failing because example data may be generated, not committed.
        return

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
    assert len(keys) > 0, f"H5 file has no top-level keys: {path}"


def test_solver_sanity() -> None:
    """
    Minimal 'solver sanity' without assuming scientific correctness:
    just check that running a lightweight stage/help command does not crash.

    If your scripts support '--help', this is safe and fast.
    """
    # Prefer '--help' to avoid heavy computation.
    # If any script doesn't implement it, remove that script from this list.
    scripts = [
        "02_emergent_geometry_engine.py",
        "03_discover_bulk_equations.py",
    ]
    for s in scripts:
        p = subprocess.run(
            [sys.executable, str(REPO_ROOT / s), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert p.returncode == 0, f"{s} --help failed:\n{p.stdout}"
        assert len(p.stdout) > 0, f"{s} --help produced no output"
