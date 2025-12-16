#!/usr/bin/env python3
# 06_build_bulk_eigenmodes_dataset.py
# CUERDAS — Bloque B: Espectro escalar (dataset de modos bulk)
#
# OBJETIVO
#   Recorrer las geometrías emergentes y construir un dataset honesto de modos bulk:
#     - Llamar a bulk_scalar_solver.py para cada sistema.
#     - Recopilar pares (Delta_UV, lambda_sl) con metadatos (familia, d, ...).
#
# ENTRADAS
#   - runs/**/geometry_emergent/*.h5
#   - Módulo bulk_scalar_solver.py (o bulk_scalar_solver_v2 si existe)
#
# SALIDAS (IO_CONTRACTS_V1)
#   runs/bulk_eigenmodes/
#     bulk_modes_dataset.csv
#     bulk_modes_meta.json
#   (Opcional / compat)
#     --output-json: JSON agregador (por-sistema / por-family-d), útil para stubs Fase XII.
#
# HONESTIDAD
#   - No se aplica ninguna fórmula teórica Delta(Delta-d).
#   - lambda_sl son autovalores Sturm–Liouville, NO masas holográficas por defecto.
#
# HISTÓRICO
#   - Anteriormente conocido como: make_fase11_bulk_for_fase12c_v2.py

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np

# Importar el solver v2 con nomenclatura honesta
try:
    import bulk_scalar_solver_v2 as bss  # type: ignore
except ImportError:
    import bulk_scalar_solver as bss  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Construir dataset de modos escalares (CSV canónico) a partir de geometrías emergentes. "
            "Nomenclatura honesta: lambda_sl (autovalores Sturm–Liouville)."
        )
    )
    p.add_argument(
        "--geometry-dir",
        type=str,
        required=True,
        help="Directorio con .h5 de geometría emergente (p.ej. runs/emergent_geometry/geometry_emergent)",
    )

    # Salidas canónicas
    p.add_argument(
        "--output-csv",
        type=str,
        default="runs/bulk_eigenmodes/bulk_modes_dataset.csv",
        help="CSV canónico (default: runs/bulk_eigenmodes/bulk_modes_dataset.csv)",
    )
    p.add_argument(
        "--output-meta",
        type=str,
        default="runs/bulk_eigenmodes/bulk_modes_meta.json",
        help="JSON meta del dataset (default: runs/bulk_eigenmodes/bulk_modes_meta.json)",
    )

    # Compat / usos auxiliares (Ising, etc.)
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "(Opcional) JSON agregador por sistema/family-d (compat/aux). "
            "Recomendado para pipelines legacy o stubs Fase XII."
        ),
    )

    p.add_argument(
        "--n-eigs",
        type=int,
        default=4,
        help="Número de autovalores/autovectores por geometría (default: 4)",
    )

    # Datasets dentro del HDF5: por defecto, layout emergent (02)
    p.add_argument(
        "--z-dataset",
        type=str,
        default="z_grid",
        help="Ruta al dataset z dentro del HDF5 (default: z_grid)",
    )
    p.add_argument(
        "--A-dataset",
        type=str,
        default="A_emergent",
        help="Ruta al dataset A(z) dentro del HDF5 (default: A_emergent)",
    )
    p.add_argument(
        "--f-dataset",
        type=str,
        default="f_emergent",
        help="Ruta al dataset f(z) dentro del HDF5 (default: f_emergent)",
    )

    return p.parse_args()


def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return x


def read_required_attrs(h5_path: Path) -> Dict[str, Any]:
    """Lee metadatos críticos. No hace defaults silenciosos."""
    with h5py.File(h5_path, "r") as f:
        # system_name / name
        system_name = _decode_if_bytes(f.attrs.get("system_name", f.attrs.get("name", h5_path.stem)))
        family = _decode_if_bytes(f.attrs.get("family", f.attrs.get("family_pred", None)))
        d = f.attrs.get("d", f.attrs.get("d_pred", None))
        z_dyn = f.attrs.get("z_dyn", np.nan)
        theta = f.attrs.get("theta", np.nan)

    if family is None:
        raise ValueError(f"[{h5_path.name}] Falta attrs 'family' (o 'family_pred')")
    if d is None:
        raise ValueError(f"[{h5_path.name}] Falta attrs 'd' (o 'd_pred')")
    try:
        d_int = int(d)
    except Exception as e:
        raise ValueError(f"[{h5_path.name}] Attr 'd' no es int: {d!r}") from e

    return {
        "system_name": str(system_name),
        "family": str(family),
        "d": d_int,
        "z_dyn": float(z_dyn) if z_dyn is not None else float("nan"),
        "theta": float(theta) if theta is not None else float("nan"),
    }


def pick_existing_dataset(h5_path: Path, candidates: Sequence[str]) -> str:
    """Devuelve el primer dataset existente en el HDF5."""
    with h5py.File(h5_path, "r") as f:
        for key in candidates:
            if key in f:
                return key
    raise KeyError(f"[{h5_path.name}] No existe ninguno de los datasets: {list(candidates)}")


@dataclass
class Row:
    system_name: str
    family: str
    d: int
    z_dyn: float
    theta: float
    mode_id: int
    lambda_sl: float
    Delta_UV: Optional[float]
    quality_flag: str
    is_ground_state: bool


def write_csv(rows: List[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system_name",
        "family",
        "d",
        "z_dyn",
        "theta",
        "mode_id",
        "lambda_sl",
        "Delta_UV",
        "quality_flag",
        "is_ground_state",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "system_name": r.system_name,
                    "family": r.family,
                    "d": r.d,
                    "z_dyn": "" if np.isnan(r.z_dyn) else f"{r.z_dyn:.12g}",
                    "theta": "" if np.isnan(r.theta) else f"{r.theta:.12g}",
                    "mode_id": r.mode_id,
                    "lambda_sl": f"{r.lambda_sl:.16g}",
                    "Delta_UV": "" if r.Delta_UV is None else f"{r.Delta_UV:.16g}",
                    "quality_flag": r.quality_flag,
                    "is_ground_state": int(r.is_ground_state),
                }
            )


def build_legacy_json(rows: List[Row]) -> Dict[str, Any]:
    """Construye un JSON agregador compatible con loaders legacy (por-sistema y por family-d)."""
    systems: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.system_name not in systems:
            systems[r.system_name] = {
                "geometry_name": r.system_name,
                "family": r.family,
                "d": r.d,
                "n_modes": 0,
                "Delta_bulk_uv": [],
                "lambda_sl_bulk": [],
                "lambda_source": "bulk_eigenmode",
            }
        systems[r.system_name]["n_modes"] += 1
        systems[r.system_name]["lambda_sl_bulk"].append(float(r.lambda_sl))
        systems[r.system_name]["Delta_bulk_uv"].append(None if r.Delta_UV is None else float(r.Delta_UV))

    by_family_d: Dict[str, Dict[str, Any]] = {}
    for sys in systems.values():
        key = f"{sys['family']}_d{sys['d']}"
        if key not in by_family_d:
            by_family_d[key] = {
                "family": sys["family"],
                "d": sys["d"],
                "Delta_bulk_uv": [],
                "lambda_sl_bulk": [],
                "geometries": [],
            }
        by_family_d[key]["geometries"].append(sys["geometry_name"])
        by_family_d[key]["Delta_bulk_uv"].extend(sys["Delta_bulk_uv"])
        by_family_d[key]["lambda_sl_bulk"].extend(sys["lambda_sl_bulk"])

    return {
        "timestamp": datetime.now().isoformat(),
        "source": "06_build_bulk_eigenmodes_dataset",
        "nomenclature_version": "v2_lambda_sl",
        "systems": list(systems.values()),
        "by_family_d": by_family_d,
        "notes": [
            "Compat JSON: agregado por sistema/family-d.",
            "lambda_sl son autovalores Sturm–Liouville, NO masas holográficas por defecto.",
            "Delta_bulk_uv es el exponente UV estimado numéricamente cuando es fiable.",
        ],
    }


def main() -> None:
    args = parse_args()
    geom_dir = Path(args.geometry_dir)
    out_csv = Path(args.output_csv)
    out_meta = Path(args.output_meta)
    out_json = Path(args.output_json) if args.output_json else None

    if not geom_dir.exists() or not geom_dir.is_dir():
        raise FileNotFoundError(f"--geometry-dir no es un directorio válido: {geom_dir}")

    h5_files = sorted(geom_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No se encontraron .h5 en {geom_dir}")

    print("=" * 70)
    print("DATASET BULK EIGENMODES (CSV canónico) — CUERDAS Bloque B")
    print("Nomenclatura honesta: lambda_sl (autovalores Sturm–Liouville)")
    print("=" * 70)
    print(f"Geometrías desde: {geom_dir}")
    print(f"N_max modos por geometría: {args.n_eigs}")
    print(f"Salida CSV:  {out_csv}")
    print(f"Salida META: {out_meta}")
    if out_json:
        print(f"Salida JSON (compat): {out_json}")
    print("=" * 70)

    rows: List[Row] = []
    failed: List[Dict[str, Any]] = []
    compat_used_keys: Set[str] = set()

    for h5_path in h5_files:
        meta = read_required_attrs(h5_path)
        system_name = meta["system_name"]
        family = meta["family"]
        d = meta["d"]
        z_dyn = meta["z_dyn"]
        theta = meta["theta"]

        # Resolver datasets: usar el argumento, con fallbacks típicos
        try:
            z_ds = pick_existing_dataset(h5_path, [args.z_dataset, "z_grid", "bulk_truth/z_grid"])
            A_ds = pick_existing_dataset(h5_path, [args.A_dataset, "A_emergent", "A_of_z", "bulk_truth/A_truth"])
            f_ds = pick_existing_dataset(h5_path, [args.f_dataset, "f_emergent", "f_of_z", "bulk_truth/f_truth"])
        except Exception as e:
            failed.append({"system_name": system_name, "file": str(h5_path), "stage": "dataset_resolve", "error": str(e)})
            print(f"\n>> {system_name}: [FAIL] no se pudieron resolver datasets: {e}")
            continue

        print(f"\n>> Procesando: {system_name} (family={family}, d={d})")
        print(f"   datasets: z={z_ds}, A={A_ds}, f={f_ds}")

        try:
            spec = bss.solve_geometry(
                h5_path=h5_path,
                n_eigs=args.n_eigs,
                z_dataset=z_ds,
                A_dataset=A_ds,
                f_dataset=f_ds,
            )
        except Exception as e:
            failed.append({"system_name": system_name, "file": str(h5_path), "stage": "solver", "error": str(e)})
            print(f"   [WARN] Fallo solver en {system_name}: {e}")
            continue

        # Compatibilidad: lambda_sl (nuevo) o m2L2* (legacy)
        if "lambda_sl" in spec:
            lambda_list = spec["lambda_sl"]
            used_key = "lambda_sl"
        elif "m2L2" in spec:
            lambda_list = spec["m2L2"]
            used_key = "m2L2"
        elif "m2L2_legacy" in spec:
            lambda_list = spec["m2L2_legacy"]
            used_key = "m2L2_legacy"
        else:
            failed.append({"system_name": system_name, "file": str(h5_path), "stage": "parse_spec", "error": "missing lambda_sl/m2L2"})
            print(f"   [WARN] Espectro sin 'lambda_sl' ni 'm2L2(_legacy)' en {system_name}")
            continue

        compat_used_keys.add(used_key)
        Delta_uv_list = spec.get("uv_exponents", [])

        n_added = 0
        for mode_id, lam in enumerate(lambda_list):
            if lam is None:
                continue
            try:
                lam_f = float(lam)
            except Exception:
                continue
            if not np.isfinite(lam_f) or lam_f <= 0:
                continue

            Delta_uv: Optional[float] = None
            quality = "ok"
            if mode_id < len(Delta_uv_list):
                dv = Delta_uv_list[mode_id]
                if dv is None or (isinstance(dv, float) and not np.isfinite(dv)):
                    Delta_uv = None
                    quality = "uv_unreliable"
                else:
                    try:
                        Delta_uv = float(dv)
                    except Exception:
                        Delta_uv = None
                        quality = "uv_unreliable"
            else:
                Delta_uv = None
                quality = "uv_missing"

            rows.append(
                Row(
                    system_name=system_name,
                    family=family,
                    d=d,
                    z_dyn=z_dyn,
                    theta=theta,
                    mode_id=int(mode_id),
                    lambda_sl=lam_f,
                    Delta_UV=Delta_uv,
                    quality_flag=quality,
                    is_ground_state=(mode_id == 0),
                )
            )
            n_added += 1

        if n_added == 0:
            print("   [WARN] Sin modos con lambda_sl>0; se omite sistema.")
        else:
            # Mostrar primer modo no-nulo
            ex = next((r for r in rows if r.system_name == system_name), None)
            if ex is not None:
                dv = "(vacío)" if ex.Delta_UV is None else f"{ex.Delta_UV:.6f}"
                print(f"   OK: {n_added} filas. Ejemplo: lambda_sl={ex.lambda_sl:.6f}, Delta_UV={dv}")

    # Escribir CSV + META
    write_csv(rows, out_csv)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "nomenclature_version": "v2_lambda_sl",
        "geometry_dir": str(geom_dir),
        "n_geometries_scanned": len(h5_files),
        "n_geometries_solved": len(sorted(set(r.system_name for r in rows))),
        "n_rows": len(rows),
        "n_eigs": int(args.n_eigs),
        "datasets_requested": {"z": args.z_dataset, "A": args.A_dataset, "f": args.f_dataset},
        "solver_module": getattr(bss, "__name__", "bulk_scalar_solver"),
        "compat_used_keys": sorted(list(compat_used_keys)),
        "all_v2_clean": compat_used_keys == {"lambda_sl"},
        "failed_systems": failed,
        "notes": [
            "CSV canónico para 07: system_name,family,d,mode_id,lambda_sl,Delta_UV (+ opcionales).",
            "lambda_sl son autovalores Sturm–Liouville (NO masas por defecto).",
            "quality_flag marca fiabilidad de Delta_UV (uv_unreliable/uv_missing).",
        ],
    }

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(build_legacy_json(rows), indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("[OK] Dataset bulk-eigenmodes generado")
    print(f"  CSV :  {out_csv}")
    print(f"  META:  {out_meta}")
    if out_json:
        print(f"  JSON:  {out_json}")
    if failed:
        print(f"  WARN: {len(failed)} sistemas fallaron (ver bulk_modes_meta.json)")
    print("=" * 70)


if __name__ == "__main__":
    main()
