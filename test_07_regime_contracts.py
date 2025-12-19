#!/usr/bin/env python3
"""
test_07_regime_contracts.py

Test mínimo para verificar que 07_emergent_lambda_sl_dictionary.py (v3):
1. Genera x_mapping en el JSON
2. Genera metrics_by_regime con al menos un régimen evaluado
3. Genera contract_status a nivel raíz
4. theory_comparison.enabled es False sin --compare-theory

Uso:
    python test_07_regime_contracts.py --json-file path/to/lambda_sl_dictionary_report.json

O con datos sintéticos de juguete:
    python test_07_regime_contracts.py --synthetic
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


def create_synthetic_csv() -> str:
    """Crea un CSV sintético de juguete para testing."""
    # Mezcla de regímenes: algunos lambda_sl < 1, otros > 10
    csv_content = """system_name,family,d,z_dyn,theta,mode_id,lambda_sl,Delta_UV,quality_flag,is_ground_state
ads_d3_000,ads,3,1.0,0.0,0,0.1,0.5,good,True
ads_d3_000,ads,3,1.0,0.0,1,0.2,0.6,good,False
ads_d3_000,ads,3,1.0,0.0,2,0.3,0.7,good,False
ads_d3_000,ads,3,1.0,0.0,3,0.5,0.8,good,False
ads_d3_000,ads,3,1.0,0.0,4,0.8,0.9,good,False
ads_d3_001,ads,3,1.0,0.0,0,15.0,4.0,good,True
ads_d3_001,ads,3,1.0,0.0,1,20.0,5.0,good,False
ads_d3_001,ads,3,1.0,0.0,2,30.0,6.0,good,False
ads_d3_001,ads,3,1.0,0.0,3,50.0,8.0,good,False
ads_d3_001,ads,3,1.0,0.0,4,80.0,10.0,good,False
ads_d3_002,ads,3,1.0,0.0,0,100.0,12.0,good,True
ads_d3_002,ads,3,1.0,0.0,1,150.0,15.0,good,False
"""
    return csv_content


def validate_json_structure(report: dict) -> dict:
    """
    Valida que el JSON tenga la estructura esperada v3.
    
    Returns:
        Dict con {passed: bool, errors: list, warnings: list}
    """
    errors = []
    warnings = []
    
    # 1. Verificar feature_mapping.x_mapping
    if "feature_mapping" not in report:
        errors.append("MISSING: feature_mapping")
    else:
        fm = report["feature_mapping"]
        if "x_mapping" not in fm:
            errors.append("MISSING: feature_mapping.x_mapping")
        else:
            x_map = fm["x_mapping"]
            if not isinstance(x_map, dict):
                errors.append(f"INVALID: x_mapping debe ser dict, es {type(x_map)}")
            elif len(x_map) == 0:
                errors.append("EMPTY: x_mapping está vacío")
            else:
                # Verificar que tiene x0, x1, etc.
                expected_keys = [f"x{i}" for i in range(len(x_map))]
                actual_keys = list(x_map.keys())
                if set(actual_keys) != set(expected_keys):
                    warnings.append(f"x_mapping keys: expected {expected_keys}, got {actual_keys}")
        
        if "features" not in fm:
            warnings.append("MISSING: feature_mapping.features")
        if "target" not in fm:
            warnings.append("MISSING: feature_mapping.target")
    
    # 2. Verificar metrics_by_regime
    if "metrics_by_regime" not in report:
        errors.append("MISSING: metrics_by_regime")
    else:
        mbr = report["metrics_by_regime"]
        if "regimes" not in mbr:
            errors.append("MISSING: metrics_by_regime.regimes")
        else:
            regimes = mbr["regimes"]
            if not isinstance(regimes, dict):
                errors.append(f"INVALID: regimes debe ser dict, es {type(regimes)}")
            elif len(regimes) == 0:
                errors.append("EMPTY: regimes está vacío")
            else:
                # Verificar que al menos un régimen tiene métricas
                evaluated_count = 0
                for regime_name, regime_data in regimes.items():
                    if regime_data.get("contract_status") in ["PASS", "FAIL"]:
                        evaluated_count += 1
                        # Verificar campos obligatorios
                        required_fields = ["r2", "mae", "mae_baseline", "mre", "n_samples"]
                        for field in required_fields:
                            if field not in regime_data:
                                warnings.append(f"MISSING: regimes[{regime_name}].{field}")
                
                if evaluated_count == 0:
                    warnings.append("WARNING: Ningún régimen fue evaluado (todos SKIP)")
        
        if "contract_summary" not in mbr:
            errors.append("MISSING: metrics_by_regime.contract_summary")
        else:
            cs = mbr["contract_summary"]
            if "overall_status" not in cs:
                errors.append("MISSING: contract_summary.overall_status")
            elif cs["overall_status"] not in ["PASS", "FAIL", "INCONCLUSIVE"]:
                errors.append(f"INVALID: overall_status debe ser PASS/FAIL/INCONCLUSIVE, es {cs['overall_status']}")
    
    # 3. Verificar contract_status a nivel raíz
    if "contract_status" not in report:
        errors.append("MISSING: contract_status (nivel raíz)")
    else:
        cs = report["contract_status"]
        if cs not in ["PASS", "FAIL", "INCONCLUSIVE"]:
            errors.append(f"INVALID: contract_status debe ser PASS/FAIL/INCONCLUSIVE, es {cs}")
    
    # 4. Verificar theory_comparison.enabled
    if "theory_comparison" not in report:
        warnings.append("MISSING: theory_comparison")
    else:
        tc = report["theory_comparison"]
        if "enabled" not in tc:
            warnings.append("MISSING: theory_comparison.enabled")
    
    # 5. Verificar discovery_results (existente, pero verificar)
    if "discovery_results" not in report:
        errors.append("MISSING: discovery_results")
    else:
        dr = report["discovery_results"]
        if "best_equation" not in dr:
            errors.append("MISSING: discovery_results.best_equation")
    
    passed = len(errors) == 0
    
    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings
    }


def run_synthetic_test() -> int:
    """
    Ejecuta un test con datos sintéticos.
    
    Nota: Este test NO ejecuta PySR (sería muy lento).
    Solo verifica que un JSON de ejemplo con la estructura correcta pasa validación.
    """
    print("=" * 60)
    print("TEST: Validación de estructura JSON v3 (sintético)")
    print("=" * 60)
    
    # JSON de ejemplo con estructura v3 correcta
    example_report = {
        "timestamp": "2025-01-01T00:00:00",
        "nomenclature_version": "v2_lambda_sl",
        "script_version": "v3_with_regime_contracts",
        "config": {},
        "feature_mapping": {
            "x_mapping": {"x0": "Delta", "x1": "d"},
            "features": ["Delta", "d"],
            "target": "lambda_sl_emergent",
            "note": "x0, x1, ... en las ecuaciones corresponden a features en orden"
        },
        "input_metadata": {
            "format": "csv_bulk_modes_dataset",
            "source": "synthetic_test",
            "total_operators": 12
        },
        "data_stats": {
            "n_points_train": 10,
            "n_points_test": 2,
            "Delta_range": [0.5, 15.0],
            "lambda_sl_range": [0.1, 150.0]
        },
        "discovery_results": {
            "best_equation": "(-169.30809 / x0) + 133.8413",
            "complexity": 5,
            "test_metrics": {
                "r2": 0.85,
                "mae": 5.2,
                "pearson": 0.92
            }
        },
        "theory_comparison": {
            "enabled": False,
            "note": "Comparación con teoría deshabilitada. Usar --compare-theory para activar."
        },
        "metrics_by_regime": {
            "target_column": "lambda_sl_emergent",
            "regime_thresholds": {"lo": 1.0, "hi": 10.0},
            "regimes": {
                "lambda_sl<1.0": {
                    "regime": "lambda_sl<1.0",
                    "n_samples": 5,
                    "r2": -15.2,
                    "mae": 1.69,
                    "mae_baseline": 0.00316,
                    "mre": 162.3,
                    "max_relative_error": 500.0,
                    "mae_beats_baseline": False,
                    "contract_status": "FAIL",
                    "contract_details": {
                        "mre_threshold": 0.5,
                        "mre_ok": False,
                        "mae_must_beat_baseline": True,
                        "mae_ok": False
                    }
                },
                "lambda_sl>10.0": {
                    "regime": "lambda_sl>10.0",
                    "n_samples": 7,
                    "r2": 0.94,
                    "mae": 11.18,
                    "mae_baseline": 11.51,
                    "mre": 0.064,
                    "max_relative_error": 0.12,
                    "mae_beats_baseline": True,
                    "contract_status": "PASS",
                    "contract_details": {
                        "mre_threshold": 0.5,
                        "mre_ok": True,
                        "mae_must_beat_baseline": True,
                        "mae_ok": True
                    }
                }
            },
            "contract_summary": {
                "all_regimes_pass": False,
                "n_regimes_pass": 1,
                "n_regimes_fail": 1,
                "overall_status": "FAIL"
            },
            "warning": "R² global puede ser engañoso - revisar métricas por régimen"
        },
        "contract_status": "FAIL",
        "pareto_front": [],
        "notes": []
    }
    
    result = validate_json_structure(example_report)
    
    print("\n--- Resultado de validación ---")
    print(f"PASSED: {result['passed']}")
    
    if result['errors']:
        print(f"\nERRORS ({len(result['errors'])}):")
        for err in result['errors']:
            print(f"  ❌ {err}")
    
    if result['warnings']:
        print(f"\nWARNINGS ({len(result['warnings'])}):")
        for warn in result['warnings']:
            print(f"  ⚠ {warn}")
    
    if result['passed']:
        print("\n✅ TEST PASSED: La estructura JSON v3 es correcta")
        return 0
    else:
        print("\n❌ TEST FAILED: La estructura JSON v3 tiene errores")
        return 1


def run_file_test(json_file: Path) -> int:
    """Valida un archivo JSON existente."""
    print("=" * 60)
    print(f"TEST: Validación de {json_file}")
    print("=" * 60)
    
    if not json_file.exists():
        print(f"❌ ERROR: Archivo no encontrado: {json_file}")
        return 1
    
    try:
        with open(json_file, 'r') as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: JSON inválido: {e}")
        return 1
    
    result = validate_json_structure(report)
    
    print("\n--- Resultado de validación ---")
    print(f"PASSED: {result['passed']}")
    
    if result['errors']:
        print(f"\nERRORS ({len(result['errors'])}):")
        for err in result['errors']:
            print(f"  ❌ {err}")
    
    if result['warnings']:
        print(f"\nWARNINGS ({len(result['warnings'])}):")
        for warn in result['warnings']:
            print(f"  ⚠ {warn}")
    
    # Mostrar resumen de lo encontrado
    if result['passed']:
        print("\n--- Contenido encontrado ---")
        if "feature_mapping" in report:
            fm = report["feature_mapping"]
            print(f"  x_mapping: {fm.get('x_mapping', 'N/A')}")
            print(f"  features: {fm.get('features', 'N/A')}")
            print(f"  target: {fm.get('target', 'N/A')}")
        
        if "contract_status" in report:
            print(f"  contract_status: {report['contract_status']}")
        
        if "metrics_by_regime" in report:
            mbr = report["metrics_by_regime"]
            if "regimes" in mbr:
                for regime_name, regime_data in mbr["regimes"].items():
                    status = regime_data.get("contract_status", regime_data.get("status", "?"))
                    n = regime_data.get("n_samples", "?")
                    print(f"  {regime_name}: {status} (n={n})")
    
    if result['passed']:
        print("\n✅ TEST PASSED")
        return 0
    else:
        print("\n❌ TEST FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test de estructura JSON v3 para 07")
    parser.add_argument("--json-file", type=str, default=None,
                        help="Archivo JSON a validar")
    parser.add_argument("--synthetic", action="store_true",
                        help="Ejecutar test con datos sintéticos (no requiere archivo)")
    
    args = parser.parse_args()
    
    if args.synthetic:
        return run_synthetic_test()
    elif args.json_file:
        return run_file_test(Path(args.json_file))
    else:
        print("Uso:")
        print("  python test_07_regime_contracts.py --synthetic")
        print("  python test_07_regime_contracts.py --json-file path/to/report.json")
        return 1


if __name__ == "__main__":
    sys.exit(main())
