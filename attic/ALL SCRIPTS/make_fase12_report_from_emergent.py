#!/usr/bin/env python3
# make_fase12_report_from_emergent.py
# CUERDAS — Puente Bloque B/C: Reporte Fase XII desde diccionario EMERGENTE
# Compatible con Python 3.9

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class OperatorPrediction:
    name: str
    lambda_sl: float
    Delta_pred: float
    Delta_uncertainty: Optional[float] = None
    source_lambda: str = "bulk_eigenmode"


def load_emergent_dictionary(report_path: Path) -> Dict[str, Any]:
    data = json.loads(report_path.read_text())
    best = data["discovery_results"]["best_equation"]
    r2 = data["discovery_results"]["test_metrics"].get("r2", None)
    provenance = data.get("provenance", "07_emergent_lambda_sl_dictionary")
    source = data.get("dictionary_source", f"emergent_{report_path.parent.name}")

    print(f"[07] Mejor ecuación descubierta: {best}")
    print(f"[07] R² en test: {r2:.4f}" if r2 else "[07] R² no disponible")
    print(f"[07] provenance → {provenance}")

    return {
        "equation_str": best,
        "r2_test": r2,
        "provenance": provenance,
        "dictionary_source": source,
        "raw_report": data,
    }


def evaluate_pysr_equation(equation_str: str, Delta: float, d: int) -> float:
    x0 = Delta
    x1 = d
    try:
        return eval(equation_str)
    except Exception as e:
        print(f"   [eval fallback] {e} → usando sympy")
        import sympy
        expr = sympy.sympify(equation_str, locals={"x0": x0, "x1": x1})
        return float(expr)


def predict_Delta_from_lambda(
    lambda_sl: float,
    emergent_dict: Dict[str, Any],
    d_boundary: int = 3,
    max_candidates: int = 5
) -> List[OperatorPrediction]:
    eq_str = emergent_dict["equation_str"]
    predictions = []

    Delta_grid = np.linspace(0.01, 6.0, 20000)
    lambda_grid = np.array([evaluate_pysr_equation(eq_str, Delta, d_boundary) for Delta in Delta_grid])
    distances = np.abs(lambda_grid - lambda_sl)
    idx_sorted = np.argsort(distances)

    seen = set()
    for idx in idx_sorted[:max_candidates]:
        Delta = float(Delta_grid[idx])
        if round(Delta, 4) in seen:
            continue
        seen.add(round(Delta, 4))

        pred = OperatorPrediction(
            name="unknown",
            lambda_sl=lambda_sl,
            Delta_pred=Delta,
            Delta_uncertainty=float(distances[idx]),
        )
        predictions.append(pred)
        if len(predictions) >= 3:
            break

    return predictions


def load_ising_descriptor(desc_path: Path) -> Dict[str, Any]:
    data = json.loads(desc_path.read_text())
    print(f"[Ising descriptor] {desc_path.name} → {len(data.get('operators', {}))} operadores")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Genera reporte Fase XII usando diccionario EMERGENTE (no manual)"
    )
    parser.add_argument("--emergent-dict", required=True, help="lambda_sl_dictionary_report.json de 07")
    parser.add_argument("--system-desc", required=True, help="Descriptor del sistema real")
    parser.add_argument("--output-report", required=True, help="fase12_report_emergent.json de salida")
    parser.add_argument("--d-boundary", type=int, default=3, help="Dimensión del CFT (default 3)")

    args = parser.parse_args()

    emergent_path = Path(args.emergent_dict)
    desc_path = Path(args.system_desc)
    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MAKE FASE12 REPORT — VERSIÓN EMERGENTE (Python 3.9 compatible)")
    print("=" * 80)

    emergent = load_emergent_dictionary(emergent_path)
    system_desc = load_ising_descriptor(desc_path)
    d_boundary = system_desc.get("d_boundary", args.d_boundary)

    predictions = []
    for op_name, info in system_desc.get("operators", {}).items():
        lambda_target = info.get("lambda_sl_target")
        if lambda_target is None:
            continue

        print(f"\n>> Prediciendo Δ para '{op_name}' (λ_SL = {lambda_target:.6f})")
        candidates = predict_Delta_from_lambda(lambda_target, emergent, d_boundary)

        if candidates:
            best = min(candidates, key=lambda p: p.Delta_uncertainty)
            best.name = op_name
            predictions.append({
                "name": op_name,
                "lambda_sl": lambda_target,
                "Delta_predicted": best.Delta_pred,
                "Delta_uncertainty": float(best.Delta_uncertainty),
                "all_candidates": [
                    {"Delta": c.Delta_pred, "error": float(c.Delta_uncertainty)} for c in candidates
                ]
            })
            print(f"   → Mejor Δ = {best.Delta_pred:.6f}  (error {best.Delta_uncertainty:.2e})")

    report = {
        "system": system_desc.get("system", "unknown"),
        "d_boundary": d_boundary,
        "dictionary_source": emergent["dictionary_source"],
        "provenance": emergent["provenance"],
        "dictionary_r2_test": emergent["r2_test"],
        "dictionary_equation": emergent["equation_str"],
        "operators_predicted": predictions,
        "notes": [
            "Reporte generado con diccionario 100% emergente (07 → PySR)",
            "NO se ha usado Δ(Δ−d) en ningún punto",
            "Inversión λ_SL → Δ por grid search numérico",
            "Compatible con 09_real_data_and_dictionary_contracts.py"
        ]
    }

    out_path.write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 80)
    print(f"REPORTE EMERGENTE GENERADO → {out_path}")
    print(f"Operadores predichos: {len(predictions)}")
    print("Próximo paso: pasar por 09_real_data_and_dictionary_contracts.py")
    print("=" * 80)


if __name__ == "__main__":
    main()