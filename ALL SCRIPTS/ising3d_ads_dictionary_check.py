#!/usr/bin/env python3
"""
fase12_ising_prediction.py — Motor de predicción para Ising 3D Bootstrap

Este script procesa el JSON generado por fase12_ising_adapter.py y:
1. Aplica el diccionario holográfico (m²L² = Δ(Δ-d))
2. Compara con predicciones conocidas
3. Genera un reporte de validación

FLUJO:
    ising3d_bootstrap_prediction_input.json → este motor → reporte de validación
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


# ============================================================
# DICCIONARIO HOLOGRÁFICO EMERGENTE
# ============================================================

@dataclass
class HolographicDictionary:
    """
    Implementa el diccionario holográfico AdS/CFT.
    
    Relación fundamental: m²L² = Δ(Δ-d)
    
    Para campos escalares en AdS_{d+1}:
    - Δ es la dimensión conforme del operador dual en el CFT_d
    - m es la masa del campo en el bulk
    - L es el radio de AdS
    """
    d: int = 3  # Dimensión del CFT (boundary)
    
    def mass_from_dimension(self, Delta: float) -> float:
        """
        Calcula m²L² dado Δ usando la relación holográfica.
        
        m²L² = Δ(Δ - d)
        
        Para Δ < d/2: modo alternativo (menos común)
        Para Δ > d/2: modo estándar
        """
        return Delta * (Delta - self.d)
    
    def dimension_from_mass(self, m2L2: float) -> Tuple[float, float]:
        """
        Calcula Δ dado m²L² (hay dos soluciones).
        
        Δ_± = d/2 ± √(d²/4 + m²L²)
        
        Returns: (Delta_plus, Delta_minus) donde Delta_plus > d/2
        """
        discriminant = (self.d / 2) ** 2 + m2L2
        if discriminant < 0:
            # Masa taquiónica por debajo del BF bound
            return (np.nan, np.nan)
        
        sqrt_disc = np.sqrt(discriminant)
        Delta_plus = self.d / 2 + sqrt_disc
        Delta_minus = self.d / 2 - sqrt_disc
        
        return (Delta_plus, Delta_minus)
    
    def check_bf_bound(self, m2L2: float) -> Tuple[bool, float]:
        """
        Verifica el Breitenlohner-Freedman bound.
        
        Para AdS_{d+1}: m²L² ≥ -(d/2)²
        
        Returns: (satisfies_bound, margin)
        """
        bf_bound = -(self.d / 2) ** 2
        satisfies = m2L2 >= bf_bound
        margin = m2L2 - bf_bound
        return (satisfies, margin)
    
    def unitarity_bound(self, spin: int = 0) -> float:
        """
        Bound de unitariedad para operadores en CFT_d.
        
        Para escalares: Δ ≥ (d-2)/2
        Para spin ℓ: Δ ≥ d - 2 + ℓ (para ℓ ≥ 1)
        """
        if spin == 0:
            return (self.d - 2) / 2
        else:
            return self.d - 2 + spin


def apply_holographic_dictionary(
    operators: List[Dict],
    d: int = 3
) -> List[Dict]:
    """
    Aplica el diccionario holográfico a una lista de operadores.
    
    Para cada operador con dimensión Δ, calcula:
    - m²L² emergente
    - Verificación del BF bound
    - Verificación del bound de unitariedad
    """
    dictionary = HolographicDictionary(d=d)
    results = []
    
    for op in operators:
        Delta = op["Delta"]
        spin = op.get("spin", 0)
        name = op.get("name", op.get("label", "unknown"))
        
        # Calcular m²L² emergente
        m2L2 = dictionary.mass_from_dimension(Delta)
        
        # Verificar BF bound
        bf_satisfied, bf_margin = dictionary.check_bf_bound(m2L2)
        
        # Verificar unitariedad
        unitarity_bound = dictionary.unitarity_bound(spin)
        unitarity_satisfied = Delta >= unitarity_bound
        
        # Clasificar el operador
        if Delta < d:
            relevance = "relevant"
        elif Delta == d:
            relevance = "marginal"
        else:
            relevance = "irrelevant"
        
        results.append({
            "name": name,
            "Delta": Delta,
            "Delta_error": op.get("Delta_error", 0.0),
            "spin": spin,
            "m2L2_emergent": m2L2,
            "bf_bound_satisfied": bf_satisfied,
            "bf_margin": bf_margin,
            "unitarity_satisfied": unitarity_satisfied,
            "unitarity_bound": unitarity_bound,
            "relevance": relevance,
            "bulk_interpretation": _interpret_bulk_field(m2L2, spin, d)
        })
    
    return results


def _interpret_bulk_field(m2L2: float, spin: int, d: int) -> str:
    """Interpreta físicamente el campo en el bulk."""
    if spin == 0:
        if m2L2 < 0:
            return f"Scalar tachyonic (m²L² = {m2L2:.4f} < 0, but above BF bound)"
        elif m2L2 == 0:
            return "Massless scalar (conformally coupled)"
        else:
            return f"Massive scalar (m²L² = {m2L2:.4f})"
    elif spin == 2:
        if abs(m2L2) < 0.01:
            return "Graviton (massless spin-2)"
        else:
            return f"Massive spin-2 field (m²L² = {m2L2:.4f})"
    else:
        return f"Spin-{spin} field with m²L² = {m2L2:.4f}"


# ============================================================
# VALIDADOR DE PREDICCIONES
# ============================================================

def validate_predictions(
    dictionary_results: List[Dict],
    known_predictions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Valida los resultados del diccionario contra predicciones conocidas.
    """
    validation = {
        "passed": True,
        "checks": [],
        "warnings": [],
        "summary": {}
    }
    
    # 1. Verificar m²L² emergentes
    expected_m2L2 = known_predictions.get("expected_m2L2", {})
    
    for result in dictionary_results:
        name = result["name"]
        m2L2_computed = result["m2L2_emergent"]
        
        if name in expected_m2L2:
            m2L2_expected = expected_m2L2[name]
            diff = abs(m2L2_computed - m2L2_expected)
            rel_diff = diff / abs(m2L2_expected) if m2L2_expected != 0 else diff
            
            check = {
                "name": f"m2L2_{name}",
                "expected": m2L2_expected,
                "computed": m2L2_computed,
                "difference": diff,
                "relative_difference": rel_diff,
                "passed": rel_diff < 0.01  # 1% tolerance
            }
            validation["checks"].append(check)
            
            if not check["passed"]:
                validation["passed"] = False
    
    # 2. Verificar BF bounds
    bf_violations = [r for r in dictionary_results if not r["bf_bound_satisfied"]]
    if bf_violations:
        validation["passed"] = False
        validation["warnings"].append(
            f"BF bound violated for: {[r['name'] for r in bf_violations]}"
        )
    
    # 3. Verificar unitariedad
    unitarity_violations = [r for r in dictionary_results if not r["unitarity_satisfied"]]
    if unitarity_violations:
        validation["passed"] = False
        validation["warnings"].append(
            f"Unitarity bound violated for: {[r['name'] for r in unitarity_violations]}"
        )
    
    # 4. Resumen
    validation["summary"] = {
        "n_operators": len(dictionary_results),
        "n_bf_satisfied": sum(1 for r in dictionary_results if r["bf_bound_satisfied"]),
        "n_unitary": sum(1 for r in dictionary_results if r["unitarity_satisfied"]),
        "n_relevant": sum(1 for r in dictionary_results if r["relevance"] == "relevant"),
        "n_checks_passed": sum(1 for c in validation["checks"] if c["passed"]),
        "n_checks_total": len(validation["checks"])
    }
    
    return validation


# ============================================================
# GENERADOR DE PREDICCIONES NUEVAS
# ============================================================

def generate_new_predictions(
    dictionary_results: List[Dict],
    features: Dict[str, float],
    d: int = 3
) -> List[Dict]:
    """
    Genera predicciones nuevas verificables basadas en los resultados.
    """
    predictions = []
    
    # 1. Predicción de la central charge desde el stress tensor
    # Para CFT_3: c_T ∝ L^2 / G_N (relación holográfica)
    # El operador de estrés tiene Δ = d = 3
    predictions.append({
        "type": "stress_tensor_dimension",
        "description": "El tensor de energía-momento tiene Δ = d",
        "predicted_value": d,
        "known_value": d,
        "status": "confirmed",
        "confidence": "high"
    })
    
    # 2. Predicción de operadores de twist bajo
    # En el límite de large spin, τ = Δ - ℓ → 2Δ_min
    Delta_min = features.get("Delta_min", 0.518)
    predictions.append({
        "type": "minimal_twist",
        "description": "Twist mínimo en el espectro de large spin",
        "predicted_value": 2 * Delta_min,
        "formula": "τ_min = 2Δ_σ",
        "confidence": "high",
        "how_to_verify": "Analytic bootstrap at large spin"
    })
    
    # 3. Predicción de la anomalía conforme (para d par)
    if d % 2 == 0:
        predictions.append({
            "type": "conformal_anomaly",
            "description": f"Anomalía conforme en CFT_{d}",
            "status": "requires_computation",
            "confidence": "medium"
        })
    
    # 4. Predicción de OPE asintótico
    Delta_gap = features.get("Delta_gap", 0.89)
    predictions.append({
        "type": "ope_asymptotic",
        "description": "Comportamiento asintótico de OPE coefficients",
        "predicted_scaling": f"λ² ~ exp(-2π√(Δ/{Delta_gap}))",
        "confidence": "medium",
        "how_to_verify": "Numerical bootstrap spectrum extraction"
    })
    
    # 5. Predicción específica para Ising 3D
    if "Delta_sigma" in features and "Delta_epsilon" in features:
        Delta_sigma = features["Delta_sigma"]
        Delta_epsilon = features["Delta_epsilon"]
        
        # Fusión σ × σ → 1 + ε + ...
        predictions.append({
            "type": "fusion_rule",
            "description": "Regla de fusión σ × σ",
            "content": f"σ × σ → 1 + ε (Δ={Delta_epsilon:.4f}) + T (Δ=3) + ...",
            "confidence": "high"
        })
        
        # Relación entre exponentes críticos
        eta = 2 * Delta_sigma - (d - 2)
        nu = 1 / (d - Delta_epsilon)
        predictions.append({
            "type": "critical_exponents",
            "description": "Exponentes críticos derivados",
            "eta": eta,
            "nu": nu,
            "gamma": nu * (2 - eta),  # Relación de escala
            "formula": "γ = ν(2-η)",
            "confidence": "high"
        })
    
    return predictions


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

def run_ising3d_prediction(
    input_json_path: Path,
    known_predictions_path: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta el motor de predicción completo para Ising 3D.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos
    if verbose:
        print(f"Cargando: {input_json_path}")
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    with open(known_predictions_path, 'r') as f:
        known = json.load(f)
    
    name = data["name"]
    d = data["d"]
    features = data["features"]
    
    # Reconstruir lista de operadores desde observables
    observables = data["observables"]
    operators = []
    for i, Delta in enumerate(observables["Delta_list"]):
        operators.append({
            "name": observables["labels"][i],
            "Delta": Delta,
            "spin": observables["spin"][i],
            "Z2_parity": observables["Z2_parity"][i]
        })
    
    if verbose:
        print(f"  Sistema: {name}")
        print(f"  d = {d}")
        print(f"  Operadores: {len(operators)}")
    
    # 2. Aplicar diccionario holográfico
    if verbose:
        print("\nAplicando diccionario holográfico...")
    
    dictionary_results = apply_holographic_dictionary(operators, d)
    
    if verbose:
        print("  Resultados del diccionario:")
        for r in dictionary_results:
            print(f"    {r['name']}: Δ = {r['Delta']:.6f} → m²L² = {r['m2L2_emergent']:.6f}")
    
    # 3. Validar contra predicciones conocidas
    if verbose:
        print("\nValidando contra predicciones conocidas...")
    
    validation = validate_predictions(dictionary_results, known)
    
    if verbose:
        print(f"  Checks pasados: {validation['summary']['n_checks_passed']}/{validation['summary']['n_checks_total']}")
        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"  ⚠ {w}")
    
    # 4. Generar predicciones nuevas
    if verbose:
        print("\nGenerando predicciones nuevas...")
    
    new_predictions = generate_new_predictions(dictionary_results, features, d)
    
    if verbose:
        print(f"  Predicciones generadas: {len(new_predictions)}")
    
    # 5. Compilar reporte
    report = {
        "metadata": {
            "system": name,
            "phase": "XII",
            "version": "XII.ising3d_v1",
            "timestamp": datetime.now().isoformat(),
            "source": data["source"]
        },
        "input": {
            "d": d,
            "n_operators": len(operators),
            "operators": operators,
            "ope_coefficients": {
                k: v for k, v in observables.items() 
                if k.startswith("lambda_")
            }
        },
        "dictionary_results": dictionary_results,
        "validation": validation,
        "new_predictions": new_predictions,
        "overall_status": "validated" if validation["passed"] else "requires_investigation"
    }
    
    # 6. Guardar reporte
    report_path = output_dir / f"{name}_fase12_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    if verbose:
        print(f"\n✓ Reporte guardado: {report_path}")
    
    # 7. Guardar resumen legible
    summary_path = output_dir / f"{name}_fase12_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"FASE XII — REPORTE ISING 3D BOOTSTRAP\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DICCIONARIO HOLOGRÁFICO EMERGENTE\n")
        f.write("-" * 40 + "\n")
        for r in dictionary_results:
            f.write(f"  {r['name']:10s}: Δ = {r['Delta']:.6f}\n")
            f.write(f"             m²L² = {r['m2L2_emergent']:.6f}\n")
            f.write(f"             {r['bulk_interpretation']}\n")
            f.write(f"             BF bound: {'✓' if r['bf_bound_satisfied'] else '✗'}\n")
            f.write(f"             Unitarity: {'✓' if r['unitarity_satisfied'] else '✗'}\n\n")
        
        f.write("VALIDACIÓN\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Status: {report['overall_status']}\n")
        for check in validation["checks"]:
            status = "✓" if check["passed"] else "✗"
            f.write(f"  {status} {check['name']}: expected={check['expected']:.6f}, ")
            f.write(f"computed={check['computed']:.6f}\n")
        
        f.write("\nPREDICCIONES NUEVAS\n")
        f.write("-" * 40 + "\n")
        for pred in new_predictions:
            f.write(f"  • {pred['type']}: {pred['description']}\n")
            if "predicted_value" in pred:
                f.write(f"    Valor: {pred['predicted_value']}\n")
            if "confidence" in pred:
                f.write(f"    Confianza: {pred['confidence']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    if verbose:
        print(f"✓ Resumen guardado: {summary_path}")
    
    return report


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fase XII: Motor de predicción para Ising 3D"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="JSON de entrada (prediction_input)")
    parser.add_argument("--known", "-k", type=str, required=True,
                        help="JSON con predicciones conocidas")
    parser.add_argument("--output-dir", "-o", type=str, default="fase12_predictions",
                        help="Directorio de salida")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Modo silencioso")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FASE XII — MOTOR DE PREDICCIÓN ISING 3D")
    print("=" * 70)
    
    report = run_ising3d_prediction(
        input_json_path=Path(args.input),
        known_predictions_path=Path(args.known),
        output_dir=Path(args.output_dir),
        verbose=not args.quiet
    )
    
    print("\n" + "=" * 70)
    print(f"STATUS FINAL: {report['overall_status'].upper()}")
    print("=" * 70)
    
    return 0 if report["overall_status"] == "validated" else 1


if __name__ == "__main__":
    exit(main())
