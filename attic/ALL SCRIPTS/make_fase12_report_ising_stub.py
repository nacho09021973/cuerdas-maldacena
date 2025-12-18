#!/usr/bin/env python3
# make_fase12_report_ising_stub.py
# CUERDAS — Construye un fase12_report.json mínimo para Ising 3D
#
# Uso:
#   (.venv39) python make_fase12_report_ising_stub.py \
#     --descriptor real_data_sandbox/ising3d_descriptor.json \
#     --manifest   fase12_data_boundary/manifest_fase12.json \
#     --output     runs/fase12_ising_real/fase12/predictions/fase12_report.json
#
# Este script:
#   - Lee el descriptor limpio de Ising (operadores, Δ, metadatos).
#   - Lee el manifest_fase12.json del adapter (known_predictions, features).
#   - Construye un fase12_report.json compatible con 09_real_data_and_dictionary_contracts.py.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def build_system_entry(
    descriptor: Dict[str, Any],
    manifest_entry: Dict[str, Any],
) -> Dict[str, Any]:
    # 1) Datos básicos del sistema
    name = descriptor.get("system_name", "ising3d_bootstrap")
    source = descriptor.get("source", "bootstrap")
    d = descriptor.get("d", 3)

    # 2) Operadores desde el descriptor (estos serán el "diccionario manual")
    ops_desc: List[Dict[str, Any]] = descriptor.get("operators", [])
    operators_predicted = []
    for op in ops_desc:
        operators_predicted.append(
            {
                "name": op.get("name", "O"),
                "Delta": float(op.get("Delta", 0.0)),
                "spin": int(op.get("spin", 0)),
                "Delta_error": float(op.get("error", 0.0)),
            }
        )

    # 3) Known predictions y features desde el manifest_fase12
    features = manifest_entry.get("features", {})
    known_pred = manifest_entry.get("known_predictions", {})

    # Temperatura efectiva (si está en features); si no, 0.0 (crítico)
    T_eff = float(features.get("T", 0.0))

    # Familia "esperada" desde known_predictions
    predicted_family = known_pred.get("expected_bulk", "ads_d3_bootstrap_stub")

    system_entry: Dict[str, Any] = {
        "name": name,
        "source": source,
        "d": d,
        "T": T_eff,
        "geometry": {
            "predicted_family": predicted_family,
            "z_h": float(features.get("z_h", 0.0)),
            "z_dyn": float(features.get("z_dyn", 1.0)),
            # Aquí podrías meter modos emergentes en el futuro;
            # por ahora rellenamos con los mismos operadores que el diccionario manual.
            "operators_predicted": [
                {"name": op["name"], "Delta": op["Delta"], "spin": op["spin"]}
                for op in operators_predicted
            ],
        },
        "dictionary": {
            "provenance": "manual",
            "operators_predicted": operators_predicted,
        },
        "dictionary_source": "manual_v0",
        "physics_metadata": {
            "known_predictions": known_pred,
            "central_charge_normalized": descriptor.get("metadata", {}).get(
                "central_charge_normalized", None
            ),
            "reference": descriptor.get("metadata", {}).get("reference", ""),
            "notes": descriptor.get("metadata", {}).get("notes", []),
        },
        "features": features,
    }

    return system_entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construye un fase12_report.json mínimo para Ising 3D (bootstrap)"
    )
    parser.add_argument(
        "--descriptor",
        type=str,
        required=True,
        help="JSON descriptor de Ising 3D (ising3d_descriptor.json)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest generado por fase12_real_data_adapters.py (manifest_fase12.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta de salida para fase12_report.json",
    )
    args = parser.parse_args()

    descriptor_path = Path(args.descriptor)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not descriptor_path.exists():
        raise SystemExit(f"Descriptor no encontrado: {descriptor_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest no encontrado: {manifest_path}")

    descriptor = load_json(descriptor_path)
    manifest = load_json(manifest_path)

    processed = manifest.get("processed", [])
    if not processed:
        raise SystemExit("Manifest sin entradas 'processed'. ¿Has ejecutado el adapter?")

    # En este stub tomamos solo la primera entrada (ising_3d)
    entry0 = processed[0]

    system_entry = build_system_entry(descriptor, entry0)

    report = {
        "phase": 12,
        "description": "Stub inicial Ising 3D (bootstrap sintético, diccionario manual v0)",
        "systems": [system_entry],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print("======================================================================")
    print("FASE XII — REPORTE STUB ISING 3D")
    print("======================================================================")
    print(f"Descriptor: {descriptor_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"Output:     {output_path}")
    print("======================================================================")
    print("Este fase12_report.json es un stub honesto:")
    print("  - Diccionario = manual_v0 (Δ tomados del descriptor).")
    print("  - geometry.predicted_family viene de known_predictions.expected_bulk.")
    print("Listo para contratos con 09_real_data_and_dictionary_contracts.py.")
    print("======================================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
