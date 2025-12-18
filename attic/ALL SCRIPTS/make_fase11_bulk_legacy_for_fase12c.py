#!/usr/bin/env python3
"""
make_fase11_bulk_legacy_for_fase12c.py

Convierte el output emergente del solver escalar bulk
(fase11_bulk_for_fase12c.json) al formato legacy `systems[]`
que espera fase12c_emergent_dictionary_real.py.

No introduce m^2 L^2 = Delta(Delta-d); solo reetiqueta campos
y aplica un filtrado suave de outliers numéricos.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_bulk_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def build_legacy_systems(
    data: Dict[str, Any],
    max_abs_delta: float = 5.0,
    max_log10_m2: float = 6.0,
) -> Dict[str, Any]:
    """
    Construye estructura legacy:
      systems[] -> {name, family, d, source, operators[]}
    a partir de la lista 'systems' del JSON emergente.

    Filtros suaves (no teóricos, solo numéricos):
      - |Delta_bulk_uv| <= max_abs_delta
      - log10(m2L2_bulk) <= max_log10_m2
    """
    legacy: Dict[str, Any] = {
        "source": data.get("source", "fase11_bulk_scalar_solver"),
        "geometry_dir": data.get("geometry_dir", ""),
        "n_geometries": data.get("n_geometries", 0),
        "systems": [],
        "notes": data.get("notes", []),
    }

    systems_in = data.get("systems", [])
    n_ops_total = 0
    n_ops_kept = 0

    for sys_entry in systems_in:
        name = sys_entry["geometry_name"]
        family = sys_entry["family"]
        d = int(sys_entry["d"])
        deltas = np.asarray(sys_entry["Delta_bulk_uv"], dtype=float)
        m2_vals = np.asarray(sys_entry["m2L2_bulk"], dtype=float)
        method = sys_entry.get("m2L2_method", "bulk_eigenmode")

        if deltas.shape != m2_vals.shape:
            print(f"[WARN] {name}: shape mismatch Delta({deltas.shape}) vs m2({m2_vals.shape}), SKIP")
            continue

        operators: List[Dict[str, Any]] = []
        for Delta, m2 in zip(deltas, m2_vals):
            n_ops_total += 1

            # Filtro numérico suave: evita Δ absurdamente grandes/negativos y masas gigantes
            if not np.isfinite(Delta) or not np.isfinite(m2):
                continue
            if abs(Delta) > max_abs_delta:
                continue
            if m2 <= 0:
                continue
            if np.log10(m2) > max_log10_m2:
                continue

            op = {
                "Delta": float(Delta),
                "Delta_error": 0.0,
                "m2L2_emergent": float(m2),
                "m2L2_error": 0.0,
                "m2L2_method": method,
            }
            operators.append(op)
            n_ops_kept += 1

        if not operators:
            print(f"[INFO] {name}: sin operadores tras filtrado, SKIP")
            continue

        legacy_system = {
            "name": name,
            "family": family,
            "d": d,
            "source": legacy["source"],
            "operators": operators,
        }
        legacy["systems"].append(legacy_system)

    print(f"[INFO] Operadores totales (antes de filtro): {n_ops_total}")
    print(f"[INFO] Operadores usados   (despues filtro): {n_ops_kept}")
    print(f"[INFO] Sistemas finales: {len(legacy['systems'])}")

    legacy.setdefault("notes", []).append(
        "Construido desde fase11_bulk_for_fase12c.json con filtros numericos suaves; "
        "no se uso m^2 L^2 = Delta(Delta-d)."
    )
    return legacy


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convierte bulk_for_fase12c.json a formato legacy systems[] para XII.c"
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Archivo JSON de entrada (fase11_bulk_for_fase12c.json)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Archivo JSON de salida en formato legacy systems[]",
    )
    parser.add_argument(
        "--max-abs-delta",
        type=float,
        default=5.0,
        help="Filtro |Delta| <= max_abs_delta",
    )
    parser.add_argument(
        "--max-log10-m2",
        type=float,
        default=6.0,
        help="Filtro log10(m2L2) <= max_log10_m2",
    )

    args = parser.parse_args()

    data = load_bulk_json(args.input_json)
    legacy = build_legacy_systems(
        data,
        max_abs_delta=args.max_abs_delta,
        max_log10_m2=args.max_log10_m2,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(legacy, f, indent=2)
    print(f"[OK] Legacy systems guardado en: {args.output_json}")


if __name__ == "__main__":
    main()
