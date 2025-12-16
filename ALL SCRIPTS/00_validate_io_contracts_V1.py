#!/usr/bin/env python3
"""00_validate_io_contracts.py

Validador read-only del contrato IO_CONTRACTS_V1 para CUERDAS-Maldacena.

Objetivo:
- Detectar temprano inconsistencias de interfaces (metadatos, layouts HDF5, shapes) que
  contaminan el pipeline downstream.

Diseño:
- Read-only por defecto.
- Genera un JSON con PASS/WARN/FAIL por archivo.
- Exit code != 0 si existe cualquier FAIL (o WARN si --fail-on-warn).

Uso típico:
  python 00_validate_io_contracts.py --sandbox-dir runs/sandbox_geometries --output runs/io_contracts_report.json

Nota:
  Este script NO impone física; impone consistencia de formatos.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


D_FROM_NAME_RE = re.compile(r"_d(\d+)_")


@dataclass
class Issue:
    level: str  # "WARN" | "FAIL"
    code: str
    message: str


@dataclass
class FileReport:
    path: str
    kind: str  # "sandbox" | "emergent" | "unknown"
    status: str  # "PASS" | "WARN" | "FAIL"
    attrs: Dict[str, Any]
    issues: List[Issue]


def _safe_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, np.ndarray):
            x = x.ravel()[0]
        return int(x)
    except Exception:
        return None


def _safe_str(x: Any) -> Optional[str]:
    try:
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", errors="replace")
        return str(x)
    except Exception:
        return None


def _infer_d_from_filename(path: Path) -> Optional[int]:
    m = D_FROM_NAME_RE.search(path.name)
    return int(m.group(1)) if m else None


def _read_attrs_root(f: h5py.File) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in f.attrs.items():
        # Convert numpy scalars to python
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="replace")
        out[str(k)] = v
    return out


def _check_monotonic_1d(arr: np.ndarray) -> bool:
    if arr.ndim != 1 or len(arr) < 2:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    return bool(np.all(np.diff(arr) > 0))


def _get_dataset(f: h5py.File, candidates: List[str]) -> Optional[np.ndarray]:
    for key in candidates:
        if key in f:
            return np.array(f[key])
    return None


def _has_path(f: h5py.File, p: str) -> bool:
    return p in f


def detect_kind(f: h5py.File, path: Path) -> str:
    # Strong signals
    if _has_path(f, "bulk_truth") or _has_path(f, "boundary"):
        # likely sandbox container
        if _has_path(f, "bulk_truth/z_grid") or _has_path(f, "bulk_truth/A_truth"):
            return "sandbox"
    # emergent: canonical datasets
    if _has_path(f, "A_emergent") or _has_path(f, "f_emergent") or _has_path(f, "R_emergent"):
        return "emergent"
    if _has_path(f, "A_of_z") or _has_path(f, "f_of_z") or _has_path(f, "R_of_z"):
        return "emergent"
    # heuristic by filename
    if path.name.endswith("_emergent.h5"):
        return "emergent"
    return "unknown"


def validate_common(path: Path, attrs: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    # Required attrs
    d = _safe_int(attrs.get("d"))
    family = _safe_str(attrs.get("family"))
    sys_name = _safe_str(attrs.get("system_name")) or _safe_str(attrs.get("name"))

    if sys_name is None or sys_name.strip() == "":
        issues.append(Issue("FAIL", "ATTR_SYSTEM_NAME_MISSING", "Falta root attr 'system_name' o 'name'."))
    if family is None or family.strip() == "":
        # some emergent files use family_pred only; allow in kind-specific checks
        issues.append(Issue("WARN", "ATTR_FAMILY_MISSING", "Falta root attr 'family'. (Puede ser OK si es emergent y solo existe family_pred.)"))
    if d is None:
        issues.append(Issue("FAIL", "ATTR_D_MISSING", "Falta root attr 'd' (int)."))

    # d coherence with filename
    d_infer = _infer_d_from_filename(path)
    if d_infer is not None and d is not None and d != d_infer:
        issues.append(Issue("FAIL", "ATTR_D_MISMATCH_FILENAME", f"d={d} pero nombre sugiere d={d_infer} (patrón _d<k>_)."))

    return issues


def validate_sandbox(f: h5py.File, path: Path, attrs: Dict[str, Any]) -> List[Issue]:
    issues = []

    # MUST groups
    if "boundary" not in f:
        issues.append(Issue("FAIL", "SANDBOX_BOUNDARY_GROUP_MISSING", "Falta grupo 'boundary/' (layout canónico sandbox)."))
    else:
        b = f["boundary"]
        bd = _safe_int(b.attrs.get("d"))
        bfam = _safe_str(b.attrs.get("family"))
        rd = _safe_int(attrs.get("d"))
        rfam = _safe_str(attrs.get("family"))
        if bd is None:
            issues.append(Issue("FAIL", "BOUNDARY_ATTR_D_MISSING", "Falta boundary.attrs['d']."))
        elif rd is not None and bd != rd:
            issues.append(Issue("FAIL", "BOUNDARY_ATTR_D_MISMATCH", f"boundary d={bd} != root d={rd}."))
        if bfam is None or bfam.strip() == "":
            issues.append(Issue("FAIL", "BOUNDARY_ATTR_FAMILY_MISSING", "Falta boundary.attrs['family']."))
        elif rfam is not None and rfam.strip() != "" and bfam != rfam:
            issues.append(Issue("FAIL", "BOUNDARY_ATTR_FAMILY_MISMATCH", f"boundary family={bfam} != root family={rfam}."))

    if "bulk_truth" not in f:
        issues.append(Issue("FAIL", "SANDBOX_BULK_TRUTH_GROUP_MISSING", "Falta grupo 'bulk_truth/' en sandbox."))
    else:
        z = _get_dataset(f, ["bulk_truth/z_grid"])
        A = _get_dataset(f, ["bulk_truth/A_truth"])
        ff = _get_dataset(f, ["bulk_truth/f_truth"])
        if z is None:
            issues.append(Issue("FAIL", "BULK_TRUTH_Z_MISSING", "Falta dataset bulk_truth/z_grid."))
        else:
            if not _check_monotonic_1d(z):
                issues.append(Issue("FAIL", "BULK_TRUTH_Z_BAD", "bulk_truth/z_grid no es 1D monótono creciente/finito."))
        for nm, arr in [("A_truth", A), ("f_truth", ff)]:
            if arr is None:
                issues.append(Issue("FAIL", f"BULK_TRUTH_{nm.upper()}_MISSING", f"Falta dataset bulk_truth/{nm}."))
            else:
                if arr.ndim != 1 or (z is not None and len(arr) != len(z)):
                    issues.append(Issue("FAIL", f"BULK_TRUTH_{nm.upper()}_SHAPE", f"bulk_truth/{nm} no es 1D o no coincide con z_grid."))
                if not np.all(np.isfinite(arr)):
                    issues.append(Issue("WARN", f"BULK_TRUTH_{nm.upper()}_NONFINITE", f"bulk_truth/{nm} contiene NaN/inf."))

        # R_truth optional
        R = _get_dataset(f, ["bulk_truth/R_truth"])
        if R is not None:
            if R.ndim != 1 or (z is not None and len(R) != len(z)):
                issues.append(Issue("FAIL", "BULK_TRUTH_R_TRUTH_SHAPE", "bulk_truth/R_truth no es 1D o no coincide con z_grid."))

    return issues


def validate_emergent(f: h5py.File, path: Path, attrs: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    # MUST attrs
    sys_name = _safe_str(attrs.get("system_name"))
    if sys_name is None or sys_name.strip() == "":
        issues.append(Issue("FAIL", "EMERGENT_ATTR_SYSTEM_NAME_MISSING", "Falta root attr 'system_name'."))

    fam_pred = _safe_str(attrs.get("family_pred"))
    if fam_pred is None or fam_pred.strip() == "":
        issues.append(Issue("FAIL", "EMERGENT_ATTR_FAMILY_PRED_MISSING", "Falta root attr 'family_pred'."))

    prov = _safe_str(attrs.get("provenance"))
    if prov is None or prov.strip() == "":
        issues.append(Issue("WARN", "EMERGENT_ATTR_PROVENANCE_MISSING", "Falta root attr 'provenance'."))

    # MUST datasets
    z = _get_dataset(f, ["z_grid", "bulk_truth/z_grid"])
    if z is None:
        issues.append(Issue("FAIL", "EMERGENT_Z_GRID_MISSING", "Falta dataset z_grid."))
    else:
        if not _check_monotonic_1d(z):
            issues.append(Issue("FAIL", "EMERGENT_Z_GRID_BAD", "z_grid no es 1D monótono creciente/finito."))

    A = _get_dataset(f, ["A_emergent", "A_of_z"])
    ff = _get_dataset(f, ["f_emergent", "f_of_z"])
    R = _get_dataset(f, ["R_emergent", "R_of_z"])

    for nm, arr in [("A", A), ("f", ff), ("R", R)]:
        if arr is None:
            issues.append(Issue("FAIL", f"EMERGENT_{nm}_MISSING", f"Falta dataset {nm}_emergent (o alias {nm}_of_z)."))
        else:
            if arr.ndim != 1 or (z is not None and len(arr) != len(z)):
                issues.append(Issue("FAIL", f"EMERGENT_{nm}_SHAPE", f"Dataset {nm} no es 1D o no coincide con z_grid."))
            if not np.all(np.isfinite(arr)):
                issues.append(Issue("WARN", f"EMERGENT_{nm}_NONFINITE", f"Dataset {nm} contiene NaN/inf."))

    return issues


def compute_status(issues: List[Issue]) -> str:
    if any(i.level == "FAIL" for i in issues):
        return "FAIL"
    if any(i.level == "WARN" for i in issues):
        return "WARN"
    return "PASS"


def scan_h5_files(dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        if d.is_file() and d.suffix == ".h5":
            files.append(d)
        elif d.is_dir():
            files.extend(sorted(d.rglob("*.h5")))
    # de-dup
    uniq: Dict[str, Path] = {}
    for p in files:
        uniq[str(p.resolve())] = p
    return sorted(uniq.values())


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate IO_CONTRACTS_V1 formats.")
    ap.add_argument("--sandbox-dir", type=str, default="runs/sandbox_geometries", help="Directorio con .h5 de sandbox (layout canónico).")
    ap.add_argument("--emergent-dir", type=str, default="runs", help="Directorio(s) donde buscar geometrías emergentes (*_emergent.h5).")
    ap.add_argument("--output", type=str, default="runs/io_contracts_report.json", help="Ruta del JSON de reporte.")
    ap.add_argument("--fail-on-warn", action="store_true", help="Exit != 0 si hay WARN.")
    ap.add_argument("--max-files", type=int, default=0, help="Limitar número de ficheros (0=sin límite).")
    args = ap.parse_args()

    sandbox_dir = Path(args.sandbox_dir)
    emergent_dir = Path(args.emergent_dir)

    # Escaneo:
    # - sandbox: solo en sandbox_dir
    # - emergent: en emergent_dir, filtrando por *_emergent.h5 para no tragar TODO
    sandbox_files = scan_h5_files([sandbox_dir])

    emergent_candidates = scan_h5_files([emergent_dir])
    emergent_files = [p for p in emergent_candidates if p.name.endswith("_emergent.h5")]

    files = sandbox_files + [p for p in emergent_files if p not in sandbox_files]
    if args.max_files and len(files) > args.max_files:
        files = files[: args.max_files]

    reports: List[FileReport] = []

    for p in files:
        try:
            with h5py.File(p, "r") as f:
                attrs = _read_attrs_root(f)
                kind = detect_kind(f, p)
                issues = []

                issues.extend(validate_common(p, attrs))

                if kind == "sandbox":
                    issues.extend(validate_sandbox(f, p, attrs))
                elif kind == "emergent":
                    issues.extend(validate_emergent(f, p, attrs))
                else:
                    # unknown: minimal checks already applied; add warning
                    issues.append(Issue("WARN", "KIND_UNKNOWN", "No se pudo clasificar (sandbox/emergent)."))

                status = compute_status(issues)
                reports.append(FileReport(path=str(p), kind=kind, status=status, attrs={
                    "d": _safe_int(attrs.get("d")),
                    "family": _safe_str(attrs.get("family")),
                    "family_pred": _safe_str(attrs.get("family_pred")),
                    "system_name": _safe_str(attrs.get("system_name")) or _safe_str(attrs.get("name")),
                }, issues=issues))

        except OSError as e:
            reports.append(FileReport(
                path=str(p),
                kind="unknown",
                status="FAIL",
                attrs={},
                issues=[Issue("FAIL", "H5_OPEN_ERROR", f"No se pudo abrir HDF5: {e}")],
            ))

    # resumen
    n_pass = sum(r.status == "PASS" for r in reports)
    n_warn = sum(r.status == "WARN" for r in reports)
    n_fail = sum(r.status == "FAIL" for r in reports)

    out = {
        "contract": "IO_CONTRACTS_V1",
        "cwd": os.getcwd(),
        "sandbox_dir": str(sandbox_dir),
        "emergent_dir": str(emergent_dir),
        "n_files": len(reports),
        "summary": {"pass": n_pass, "warn": n_warn, "fail": n_fail},
        "files": [
            {
                **{k: v for k, v in asdict(r).items() if k != "issues"},
                "issues": [asdict(i) for i in r.issues],
            }
            for r in reports
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # console
    print("=" * 70)
    print("IO_CONTRACTS_V1 — VALIDATION REPORT")
    print(f"Files scanned: {len(reports)}")
    print(f"PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}")
    print(f"JSON: {output_path}")
    print("=" * 70)

    # exit code
    if n_fail > 0:
        return 2
    if args.fail_on_warn and n_warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
