#!/usr/bin/env python3
"""
make_fase11_bulk_for_fase12c.py

Construye el dataset verdaderamente EMERGENTE para Fase XII.c a partir de:

  - Geometrías de Fase XI (ficheros .h5 en, por ejemplo, fase11_output_v2/data)
  - Solver escalar bulk (bulk_scalar_solver.py)

Para cada geometría:
  - Carga la métrica (z, A(z), f(z)) desde los datasets del HDF5.
  - Resuelve el problema de autovalores L phi = m^2 phi (ver bulk_scalar_solver.py).
  - Estima exponentes UV Delta_UV de cada modo.
  - Genera pares (Delta_UV, m^2L^2) etiquetados como "bulk_eigenmode".

Salida:
  - Un JSON pensado para alimentar XII.c, sin usar nunca m^2 L^2 = Delta(Delta-d)
    en el camino de datos. Cualquier comparación con esa fórmula se hará después,
    en los contratos o análisis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

import bulk_scalar_solver as bss  # usa el solver de autovalores que ya probaste


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construir dataset bulk-emergente para Fase XII.c"
    )
    p.add_argument(
        "--geometry-dir",
        type=str,
        required=True,
        help="Directorio con los .h5 de Fase XI (p.ej. fase11_output_v2/data)",
    )
    p.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Ruta del JSON de salida (p.ej. data_processed/fase11_bulk_for_fase12c.json)",
    )
    p.add_argument(
        "--n-eigs",
        type=int,
        default=4,
        help="Número de autovalores/autovectores por geometría",
    )
    p.add_argument(
        "--z-dataset",
        type=str,
        default="bulk_truth/z_grid",
        help="Ruta al dataset z dentro del HDF5",
    )
    p.add_argument(
        "--A-dataset",
        type=str,
        default="bulk_truth/A_truth",
        help="Ruta al dataset A(z) dentro del HDF5",
    )
    p.add_argument(
        "--f-dataset",
        type=str,
        default="bulk_truth/f_truth",
        help="Ruta al dataset f(z) dentro del HDF5",
    )
    return p.parse_args()


def get_family_and_d(h5_path: Path) -> Dict[str, object]:
    """
    Lee atributos 'family' y 'd' del HDF5, con defaults razonables.
    """
    with h5py.File(h5_path, "r") as f:
        family = f.attrs.get("family", "unknown")
        if isinstance(family, bytes):
            family = family.decode("utf-8", errors="ignore")
        d_attr = f.attrs.get("d", 4)
        try:
            d = int(d_attr)
        except Exception:
            d = 4
    return {"family": family, "d": d}


def main():
    args = parse_args()
    geom_dir = Path(args.geometry_dir)
    out_path = Path(args.output_json)

    if not geom_dir.exists() or not geom_dir.is_dir():
        raise FileNotFoundError(f"--geometry-dir no es un directorio valido: {geom_dir}")

    h5_files = sorted(geom_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No se encontraron .h5 en {geom_dir}")

    systems: List[Dict[str, object]] = []

    print("=" * 70)
    print("FASE XI → XII.c  —  DATASET BULK-EMERGENTE (eigenmodes)")
    print("=" * 70)
    print(f"Geometrías desde: {geom_dir}")
    print(f"N_max modos por geometría: {args.n_eigs}")
    print(f"Datasets: z={args.z_dataset}, A={args.A_dataset}, f={args.f_dataset}")
    print("=" * 70)

    for h5_path in h5_files:
        geom_name = h5_path.stem
        meta = get_family_and_d(h5_path)
        family = meta["family"]
        d = meta["d"]

        print(f"\n>> Procesando geometría: {geom_name} (family={family}, d={d})")

        try:
            spec = bss.solve_geometry(
                h5_path=h5_path,
                n_eigs=args.n_eigs,
                z_dataset=args.z_dataset,
                A_dataset=args.A_dataset,
                f_dataset=args.f_dataset,
            )
        except Exception as e:
            print(f"   [WARN] Fallo solver en {geom_name}: {e}")
            continue

        m2L2_list = spec["m2L2"]
        Delta_uv_list = spec["uv_exponents"]

        # Filtrar parejas razonables: necesitamos m2>0 y un exponente UV finito
        m2_clean: List[float] = []
        Delta_clean: List[float] = []
        for m2, Delta_uv in zip(m2L2_list, Delta_uv_list):
            if m2 is None:
                continue
            if m2 <= 0:
                continue
            if Delta_uv is None or not np.isfinite(Delta_uv):
                continue
            m2_clean.append(float(m2))
            Delta_clean.append(float(Delta_uv))

        if not m2_clean:
            print("   [WARN] Sin modos escalares con m^2>0 y Delta_UV fiable; se omite.")
            continue

        system_entry = {
            "geometry_name": geom_name,
            "family": family,
            "d": d,
            "n_modes": len(m2_clean),
            "Delta_bulk_uv": Delta_clean,
            "m2L2_bulk": m2_clean,
            "m2L2_method": "bulk_eigenmode",
        }
        systems.append(system_entry)

        print(
            f"   OK: {len(m2_clean)} modos limpios. "
            f"Ejemplo: m2L2={m2_clean[0]:.4f}, Delta_UV={Delta_clean[0]:.4f}"
        )

    # Agrupar por (family, d) para que XII.c lo tenga fácil
    by_system_key: Dict[str, Dict[str, object]] = {}
    for sys in systems:
        key = f"{sys['family']}_d{sys['d']}"
        if key not in by_system_key:
            by_system_key[key] = {
                "family": sys["family"],
                "d": sys["d"],
                "Delta_bulk_uv": [],
                "m2L2_bulk": [],
                "geometries": [],
            }
        by_system_key[key]["Delta_bulk_uv"].extend(sys["Delta_bulk_uv"])
        by_system_key[key]["m2L2_bulk"].extend(sys["m2L2_bulk"])
        by_system_key[key]["geometries"].append(sys["geometry_name"])

    summary = {
        "source": "fase11_bulk_scalar_solver_v1",
        "geometry_dir": str(geom_dir),
        "n_geometries": len(systems),
        "systems": systems,
        "by_family_d": by_system_key,
        "notes": [
            "Dataset para XII.c construido a partir de modos escalares bulk (eigenmodes).",
            "No se ha usado m^2L^2 = Delta(Delta-d) en el camino de datos.",
            "Delta_bulk_uv es el exponente UV estimado numéricamente desde el modo escalar.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 70)
    print(f"Dataset bulk-emergente guardado en: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
