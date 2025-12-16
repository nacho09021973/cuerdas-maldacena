#!/usr/bin/env python3
"""
fase12_adapter_ising3d_bootstrap.py

Primer adapter "real" para Fase XII:
- Construye un descriptor de sistema para Ising 3D basado en bootstrap.
- No genera bulk ni correladores; solo describe el CFT en términos de Δ y metadatos.
- Pensado como fuente `source="bootstrap"` para el pipeline de Fase XII.

Modo v0:
- Usa valores de Δ_sigma y Δ_epsilon aproximados, coherentes con los contratos de Fase XII.
- Marca `provenance.type = "manual"` para dejar claro que es un stub técnico.

Uso:

  (.venv39) python fase12_adapter_ising3d_bootstrap.py \
      --output-file data/ising3d_bootstrap.json

Más adelante se puede extender para leer de un fichero externo (--input-raw).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def build_ising3d_descriptor() -> Dict[str, Any]:
    """
    Construye el descriptor JSON para Ising 3D (bootstrap) en formato CUERDAS v1.
    """
    # Valores aproximados coherentes con contracts_fase_12_13.py
    Delta_sigma = 0.518
    Delta_epsilon = 1.41

    descriptor: Dict[str, Any] = {
        "system_name": "ising3d_bootstrap",
        "display_name": "Ising 3D (bootstrap)",
        "source": "bootstrap",
        "category": "real",
        "d": 3,
        "global_symmetry": "Z2",
        "family_hint": "ads_like",

        "provenance": {
            "type": "manual",
            "description": (
                "Adapter v0 para Ising 3D: Δ_sigma y Δ_epsilon aproximados, "
                "coherentes con contratos de Fase XII. Sustituir por datos "
                "reales del bootstrap cuando estén disponibles."
            ),
            "notes": "Este fichero permite testear la integración de Fase XII con Ising 3D.",
            "reference": "Conformal bootstrap Ising 3D (literatura estándar)."
        },

        "operators": [
            {
                "name": "sigma",
                "label": "σ",
                "sector": "Z2_odd_scalar",
                "spin": 0,
                "parity": "odd",
                "Delta": Delta_sigma,
                "Delta_error": 0.001,
                "role": "order_parameter",
                "OPE": {
                    # Normalización estándar: <σσ> ≈ 1 a distancias cortas (a efectos de pipeline)
                    "lambda_sigma_sigma_identity": 1.0,
                    "lambda_sigma_sigma_epsilon": 1.0
                }
            },
            {
                "name": "epsilon",
                "label": "ε",
                "sector": "Z2_even_scalar",
                "spin": 0,
                "parity": "even",
                "Delta": Delta_epsilon,
                "Delta_error": 0.01,
                "role": "energy_density",
                "OPE": {
                    "lambda_sigma_sigma_epsilon": 1.0
                }
            }
        ]
    }

    descriptor["spectrum_summary"] = {
        "n_operators": len(descriptor["operators"]),
        "has_sigma": any(op["name"] == "sigma" for op in descriptor["operators"]),
        "has_epsilon": any(op["name"] == "epsilon" for op in descriptor["operators"]),
    }

    return descriptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera el descriptor JSON para Ising 3D (bootstrap) en formato CUERDAS."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="ising3d_bootstrap.json",
        help="Ruta del JSON de salida (por defecto: ising3d_bootstrap.json en el cwd).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor = build_ising3d_descriptor()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(descriptor, f, indent=2, ensure_ascii=False)

    print("======================================================================")
    print("ADAPTER FASE XII — ISING 3D (BOOTSTRAP)")
    print("======================================================================")
    print(f"  Sistema   : {descriptor['system_name']}")
    print(f"  Dimensión : d = {descriptor['d']}")
    print(f"  Operadores: {descriptor['spectrum_summary']['n_operators']}")
    print(f"  Δ_sigma   : {descriptor['operators'][0]['Delta']}")
    print(f"  Δ_epsilon : {descriptor['operators'][1]['Delta']}")
    print("  Fuente    : source = 'bootstrap', provenance.type = 'manual'")
    print("---------------------------------------------------------------------")
    print(f"  JSON guardado en: {out_path}")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
