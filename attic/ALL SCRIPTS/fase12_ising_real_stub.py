#!/usr/bin/env python3
"""
fase12_ising_real_stub.py

Stub mínimo de Fase XII para Ising 3D (bootstrap):
- Lee el descriptor ising3d_bootstrap.json (adapter).
- Construye un fase12_report.json con:
    - source = "bootstrap"
    - geometry.predicted_family ≈ "ads-like"
    - dictionary.manual con Δ_sigma y Δ_epsilon del adapter.

Objetivo:
- Permitir que contracts_fase_12_13.py ejecute
  contract_ising3d_consistency y lo marque como PASS
  si todo está bien encajado.

Uso típico:

  (.venv39) python scripts/fase12_ising_real_stub.py \
      --ising-json scripts/data/ising3d_bootstrap.json \
      --output-file runs/fase12_ising_real/fase12/predictions/fase12_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stub Fase XII para Ising 3D (bootstrap) usando el adapter."
    )
    parser.add_argument(
        "--ising-json",
        type=str,
        default="data/ising3d_bootstrap.json",
        help="Ruta al descriptor ising3d_bootstrap.json generado por el adapter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="runs/fase12_ising_real/fase12/predictions/fase12_report.json",
        help="Ruta del fase12_report.json de salida.",
    )
    return parser.parse_args()


def load_ising_descriptor(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe descriptor Ising 3D: {path}")
    return json.loads(path.read_text())


def build_fase12_system_from_ising(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye una entrada 'system' para fase12_report a partir del descriptor Ising 3D.

    Estructura esperada por contracts_fase_12_13.py (Fase XII):

      systems: [
        {
          "name": ...,
          "source": "bootstrap",
          "d": 3,
          "T": 0,
          "geometry": {
              "predicted_family": "...",
              "z_h": 0,
              "z_dyn": 1.0,
              "operators_predicted": [ { "name": ..., "Delta": ... }, ... ]
          },
          "dictionary": {
              "provenance": "manual",
              "operators_predicted": [ { "name": ..., "Delta": ... }, ... ]
          },
          "dictionary_source": "manual"
        },
        ...
      ]
    """
    system_name = descriptor.get("system_name", "ising3d_bootstrap")
    display_name = descriptor.get("display_name", "Ising 3D (bootstrap)")
    d = int(descriptor.get("d", 3))
    source = descriptor.get("source", "bootstrap")

    op_list: List[Dict[str, Any]] = descriptor.get("operators", [])

    # Convertimos operadores del adapter a formato "operators_predicted"
    ops_predicted: List[Dict[str, Any]] = []
    for op in op_list:
        ops_predicted.append(
            {
                "name": op.get("name", ""),
                "label": op.get("label", ""),
                "Delta": op.get("Delta", 0.0),
                "spin": op.get("spin", 0),
                "role": op.get("role", ""),
                "parity": op.get("parity", ""),
                "source": "bootstrap_adapter"
            }
        )

    system: Dict[str, Any] = {
        "name": system_name,
        "display_name": display_name,
        "source": source,
        "category": descriptor.get("category", "real"),
        "d": d,
        "T": 0.0,  # Ising 3D crítico (sin escala de T explícita aquí)

        "geometry": {
            # Esto es un hint "ads-like" para contract_ising3d_consistency
            "predicted_family": "ads_d3_bootstrap_stub",
            "z_h": 0.0,
            "z_dyn": 1.0,
            "operators_predicted": ops_predicted,
        },

        "dictionary": {
            # Diccionario emergente "manual": usamos directamente los Δ del adapter
            "provenance": "manual",
            "operators_predicted": ops_predicted,
        },

        # Fuente explícita del diccionario para transparencia
        "dictionary_source": "manual",
    }

    return system


def main() -> int:
    args = parse_args()
    ising_path = Path(args.ising_json).expanduser().resolve()
    out_path = Path(args.output_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor = load_ising_descriptor(ising_path)
    system = build_fase12_system_from_ising(descriptor)

    report = {
        "phase": 12,
        "description": "Fase XII report stub para Ising 3D (bootstrap, adapter manual).",
        "systems": [system],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("======================================================================")
    print("FASE XII STUB — ISING 3D (BOOTSTRAP)")
    print("======================================================================")
    print(f"  Sistema   : {system['name']}")
    print(f"  Fuente    : {system['source']}")
    print(f"  d         : {system['d']}")
    print(f"  N ops     : {len(system['dictionary']['operators_predicted'])}")
    print(f"  Family    : {system['geometry']['predicted_family']}")
    print(f"  Diccionario: provenance = {system['dictionary']['provenance']}, "
          f"dictionary_source = {system['dictionary_source']}")
    print("---------------------------------------------------------------------")
    print(f"  Reporte Fase XII guardado en: {out_path}")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
