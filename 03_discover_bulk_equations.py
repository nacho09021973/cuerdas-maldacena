#!/usr/bin/env python3
# 03_discover_bulk_equations.py
# CUERDAS — Bloque A: Geometría emergente (descubrimiento de ecuaciones de bulk)
#
# OBJETIVO
#   Aplicar regresión simbólica (PySR u otro SR) sobre la geometría emergente
#   para descubrir ecuaciones de campo en el bulk (para A, f, R, etc.).
#
# ENTRADAS
#   - runs/emergent_geometry/geometry_emergent/*.h5
#       * z_grid, A_emergent, f_emergent, R_emergent, ...
#   - Configuración de SR:
#       * Librería de operadores "sobria" (+, -, *, /, exp, log, sqrt, square, cos, ...).
#       * Número de iteraciones, seeds, límites de complejidad, etc.
#
# SALIDAS
#   runs/bulk_equations/
#     equations_raw.csv
#       - Población de ecuaciones con métricas asociadas.
#     equations_pareto.json
#       - Frentes de Pareto (error vs complejidad).
#     pysr_summary.json
#       - Resumen de runs, seeds, métricas globales.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Usa la geometría emergente de 02_emergent_geometry_engine.py.
#   - Sus resultados se validan con:
#       * 04_geometry_physics_contracts.py
#   - Se analizan en:
#       * 05_analyze_bulk_equations.py
#
# HONESTIDAD
#   - No se usan ecuaciones teóricas conocidas como features ni como términos forzados.
#   - Las comparaciones con ecuaciones de Einstein o variantes se realizan más tarde,
#     y se etiquetan explícitamente como "post-hoc".
#
# HISTÓRICO
#   - Anteriormente conocido como: 02_discover_einstein_v2.py

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False
    print("Warning: PySR not available.")

# Import local IO module for run manifest support
try:
    from cuerdas_io import RunContext, resolve_predictions_dir, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False

from tools.stage_utils import (
    EXIT_ERROR,
    EXIT_OK,
    STATUS_ERROR,
    STATUS_OK,
    StageContext,
    add_standard_arguments,
    parse_stage_args,
)


# ============================================================
# CALCULO DE TENSORES GEOMETRICOS
# ============================================================

def compute_geometric_tensors(
    z_grid: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    d: int
) -> Dict[str, np.ndarray]:
    """Calcula tensores geometricos en cada punto z."""
    n_z = len(z_grid)
    D = d + 1
    dz = z_grid[1] - z_grid[0]
    
    # Suavizar para estabilidad numerica
    A_smooth = gaussian_filter1d(A, sigma=1)
    f_smooth = gaussian_filter1d(f, sigma=1)
    f_smooth = np.clip(f_smooth, 1e-6, 2.0)
    
    # Derivadas
    dA = np.gradient(A_smooth, dz)
    d2A = np.gradient(dA, dz)
    df = np.gradient(f_smooth, dz)
    d2f = np.gradient(df, dz)
    
    # Escalar de Ricci para metrica diagonal warped
    # R = -2D A'' - D(D-1)(A')^2 - f'A'/f
    R_scalar = -2 * D * d2A - D * (D - 1) * dA**2 - df * dA / (f_smooth + 1e-10)
    
    # Traza del tensor de Einstein
    G_trace = (1 - D / 2) * R_scalar
    
    return {
        "z": z_grid,
        "R_scalar": R_scalar,
        "G_trace": G_trace,
        "A": A_smooth,
        "f": f_smooth,
        "dA": dA,
        "d2A": d2A,
        "df": df,
        "d2f": d2f,
        "D": D
    }


# ============================================================
# DESCUBRIMIENTO SIMBOLICO CON PYSR (LIMPIO)
# ============================================================

def discover_geometric_relations(
    tensors: Dict[str, np.ndarray],
    d: int,
    output_dir: Path,
    niterations: int = 100,
    maxsize: int = 15,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Usa PySR para descubrir relaciones geometricas SIN asumir Einstein.
    
    El objetivo es encontrar que ecuacion satisface R (o G) en terminos
    de variables geometricas, NO imponer que sea Einstein.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    z = tensors["z"]
    R = tensors["R_scalar"]
    A = tensors["A"]
    f = tensors["f"]
    dA = tensors["dA"]
    d2A = tensors["d2A"]
    df = tensors["df"]
    d2f = tensors["d2f"]
    D = tensors["D"]
    
    results = {}
    
    # === Analisis basico de R ===
    R_mean = np.mean(R)
    R_std = np.std(R)
    
    results["R_statistics"] = {
        "mean": float(R_mean),
        "std": float(R_std),
        "min": float(np.min(R)),
        "max": float(np.max(R)),
        "coefficient_of_variation": float(R_std / (np.abs(R_mean) + 1e-10))
    }
    
    print(f"   R statistics:")
    print(f"     Mean: {R_mean:.4f}")
    print(f"     Std:  {R_std:.4f}")
    print(f"     CV:   {results['R_statistics']['coefficient_of_variation']:.4f}")
    
    if not HAS_PYSR:
        results["pysr_available"] = False
        return results
    
    results["pysr_available"] = True
    
    # === Test 1: R es funcion constante? ===
    print("\n   [1] Testing if R ~ constant...")
    
    # Si R es aproximadamente constante, PySR deberia encontrar R = c
    is_constant = results["R_statistics"]["coefficient_of_variation"] < 0.1
    results["R_is_constant"] = is_constant
    
    if is_constant:
        print(f"     R ~ {R_mean:.4f} (constante)")
        results["R_constant_value"] = float(R_mean)
    else:
        print(f"     R varia significativamente")
    
    # === Test 2: Descubrir R(A, f, derivadas) ===
    print("\n   [2] Discovering R = F(A, f, derivatives)...")
    
    # Features: A, f, dA, d2A, df, d2f
    X = np.column_stack([A, f, dA, d2A, df, d2f])
    y = R
    
    # Filtrar valores extremos
    valid = np.isfinite(y) & (np.abs(y) < 1e6)
    X_valid = X[valid]
    y_valid = y[valid]
    
    if len(y_valid) < 10:
        print("   Insufficient valid data points")
        results["R_equation"] = None
        return results
    
    # PySR LIMPIO - con parametros para deterministic
    model_R = PySRRegressor(
        niterations=niterations,
        populations=8,
        population_size=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "exp", "log", "neg"],
        extra_sympy_mappings={"neg": lambda x: -x},
        elementwise_loss="L2DistLoss()",
        maxsize=maxsize,
        model_selection="best",
        progress=False,
        verbosity=0,
        parallelism="serial",       # antes "multithreading"
        deterministic=True,         # nuevo
        random_state=seed,
    )

    model_R.fit(X_valid, y_valid)
    
    best_R = model_R.get_best()
    R_pred = model_R.predict(X_valid)
    
    ss_res = np.sum((y_valid - R_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r2_R = 1 - ss_res / (ss_tot + 1e-10)
    
    results["R_equation"] = {
        "equation": str(best_R["equation"]),
        "complexity": int(best_R["complexity"]),
        "r2": float(r2_R),
        "variables": ["A", "f", "dA", "d2A", "df", "d2f"]
    }
    
    print(f"     Equation: {best_R['equation']}")
    print(f"     R2: {r2_R:.4f}")
    
    # === Test 3: Buscar forma Einstein (posterior, no impuesta) ===
    print("\n   [3] Checking if discovered equation implies Einstein...")
    
    # Si la ecuacion es aproximadamente R = constante, verificar si es R_AdS
    R_ads_expected = -D * (D - 1)  # Para L=1
    
    if is_constant:
        # Extraer Lambda efectivo de R = -2*Lambda*D/(D-2) -> Lambda = -R(D-2)/(2D)
        Lambda_eff_from_data = -R_mean * (D - 2) / (2 * D)
        Lambda_ads_expected = -(D - 1) * (D - 2) / 2
        
        results["einstein_check"] = {
            "R_found": float(R_mean),
            "R_ads_expected": float(R_ads_expected),
            "R_deviation_from_ads": float(abs(R_mean - R_ads_expected) / abs(R_ads_expected)),
            "Lambda_extracted": float(Lambda_eff_from_data),
            "Lambda_ads_expected": float(Lambda_ads_expected),
            "Lambda_deviation": float(abs(Lambda_eff_from_data - Lambda_ads_expected) / abs(Lambda_ads_expected)),
            "consistent_with_einstein_vacuum": float(abs(R_mean - R_ads_expected) / abs(R_ads_expected) < 0.2)
        }
        
        print(f"     R found:    {R_mean:.4f}")
        print(f"     R_AdS:      {R_ads_expected:.4f}")
        print(f"     Lambda extracted: {Lambda_eff_from_data:.4f}")
        print(f"     Lambda_AdS:      {Lambda_ads_expected:.4f}")
    else:
        results["einstein_check"] = {
            "consistent_with_einstein_vacuum": False,
            "note": "R not constant, may indicate non-Einstein or matter sources"
        }
    
    # === Test 4: Descubrir A(z) ===
    print("\n   [4] Discovering A(z)...")
    
    z_norm = z / z.max()
    X_z = z_norm.reshape(-1, 1)
    
    model_A = PySRRegressor(
        niterations=niterations,
        populations=8,
        population_size=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "square", "sqrt"],
        elementwise_loss="L2DistLoss()",
        maxsize=12,
        model_selection="best",
        progress=False,
        verbosity=0,
        parallelism="serial",
        deterministic=True,
        random_state=seed + 1,
    )

    model_A.fit(X_z, A)
    best_A = model_A.get_best()
    A_pred = model_A.predict(X_z)
    
    ss_res_A = np.sum((A - A_pred) ** 2)
    ss_tot_A = np.sum((A - np.mean(A)) ** 2)
    r2_A = 1 - ss_res_A / (ss_tot_A + 1e-10)
    
    results["A_equation"] = {
        "equation": str(best_A["equation"]),
        "complexity": int(best_A["complexity"]),
        "r2": float(r2_A)
    }
    
    print(f"     A(z) = {best_A['equation']}")
    print(f"     R2: {r2_A:.4f}")
    
    # === Test 5: Descubrir f(z) ===
    print("\n   [5] Discovering f(z)...")
    
    model_f = PySRRegressor(
        niterations=niterations,
        populations=8,
        population_size=50,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["square", "cube", "exp"],
        extra_sympy_mappings={"cube": lambda x: x**3},
        constraints={"^": (-1, 1)},
        elementwise_loss="L2DistLoss()",
        maxsize=12,
        model_selection="best",
        progress=False,
        verbosity=0,
        parallelism="serial",
        deterministic=True,
        random_state=seed + 2,
    )

    model_f.fit(X_z, f)
    best_f = model_f.get_best()
    f_pred_symbolic = model_f.predict(X_z)
    
    ss_res_f = np.sum((f - f_pred_symbolic) ** 2)
    ss_tot_f = np.sum((f - np.mean(f)) ** 2)
    r2_f = 1 - ss_res_f / (ss_tot_f + 1e-10)
    
    results["f_equation"] = {
        "equation": str(best_f["equation"]),
        "complexity": int(best_f["complexity"]),
        "r2": float(r2_f)
    }
    
    print(f"     f(z) = {best_f['equation']}")
    print(f"     R2: {r2_f:.4f}")
    
    return results


def validate_einstein_posterior(
    results: Dict[str, Any],
    d: int
) -> Dict[str, Any]:
    """
    Validacion POSTERIOR de si los resultados son consistentes con Einstein.
    Esta es una verificacion, NO un requisito impuesto.
    """
    D = d + 1
    validation = {}
    
    # R es constante?
    R_is_constant = results.get("R_is_constant", False)
    validation["R_constant"] = R_is_constant
    
    # Consistente con Einstein vacio + Lambda?
    einstein_check = results.get("einstein_check", {})
    validation["einstein_vacuum_compatible"] = einstein_check.get("consistent_with_einstein_vacuum", False)
    
    # A(z) tiene forma logaritmica?
    A_eq = results.get("A_equation", {})
    A_str = A_eq.get("equation", "").lower()
    A_is_log = "log" in A_str
    validation["A_is_logarithmic"] = A_is_log
    
    # Score de "Einstein-likeness"
    score = 0.0
    if R_is_constant:
        score += 0.4
    if validation["einstein_vacuum_compatible"]:
        score += 0.4
    if A_is_log:
        score += 0.2
    
    validation["einstein_score"] = float(score)
    
    # Veredicto
    if score > 0.7:
        validation["verdict"] = "LIKELY_EINSTEIN_VACUUM"
    elif score > 0.4:
        validation["verdict"] = "POSSIBLY_EINSTEIN_WITH_MATTER"
    else:
        validation["verdict"] = "NON_EINSTEIN_OR_DEFORMED"
    
    return validation


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fase XI v2: Descubrimiento genuino de ecuaciones gravitatorias"
    )
    parser.add_argument("--geometry-dir", type=str, default=None,
                        help="Directorio con geometry_emergent/ (legacy)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directorio raíz con run_manifest.json (IO v2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida (default: run-dir/bulk_equations o fase11_einstein_v2)")
    parser.add_argument("--niterations", type=int, default=100)
    parser.add_argument("--maxsize", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=4)

    add_standard_arguments(parser)
    args = parse_stage_args(parser)
    ctx = StageContext.from_args(args, stage_number="03", stage_slug="discover_bulk_equations")

    if args.run_dir is None:
        args.run_dir = str(ctx.run_root)
    if args.output_dir is None:
        args.output_dir = str(ctx.stage_dir)

    status = STATUS_OK
    exit_code = EXIT_OK
    error_message: Optional[str] = None

    try:
        ctx.record_artifact(ctx.stage_dir)
    except Exception:
        pass

    try:
        # === RESOLVER RUTAS ===
        preds_dir = None

        # Prioridad 1: --run-dir con manifest
        if args.run_dir and HAS_CUERDAS_IO:
            run_dir = Path(args.run_dir)
            preds_dir = resolve_predictions_dir(run_dir=run_dir)
            if preds_dir is None:
                # Fallback: buscar directamente en run_dir/predictions
                candidate = run_dir / "predictions"
                if candidate.exists():
                    preds_dir = candidate

        # Prioridad 2: --geometry-dir (legacy)
        if preds_dir is None and args.geometry_dir:
            geometry_dir = Path(args.geometry_dir)
            preds_dir = geometry_dir / "predictions"
            if not preds_dir.exists():
                # Quizás geometry_dir ES el directorio de predictions
                if list(geometry_dir.glob("*.npz")):
                    preds_dir = geometry_dir

        if preds_dir is None or not preds_dir.exists():
            parser.error("Debe proporcionar --run-dir con manifest válido o --geometry-dir con predictions/*.npz")

        # Resolver output_dir
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = ctx.stage_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Prioridad 1: --run-dir con manifest
        if args.run_dir and HAS_CUERDAS_IO:
            run_dir = Path(args.run_dir)
            preds_dir = resolve_predictions_dir(run_dir=run_dir)
            if preds_dir is None:
                # Fallback: buscar directamente en run_dir/predictions
                candidate = run_dir / "predictions"
                if candidate.exists():
                    preds_dir = candidate
        
        # Prioridad 2: --geometry-dir (legacy)
        if preds_dir is None and args.geometry_dir:
            geometry_dir = Path(args.geometry_dir)
            preds_dir = geometry_dir / "predictions"
            if not preds_dir.exists():
                # Quizás geometry_dir ES el directorio de predictions
                if list(geometry_dir.glob("*.npz")):
                    preds_dir = geometry_dir
        
        if preds_dir is None or not preds_dir.exists():
            parser.error("Debe proporcionar --run-dir con manifest válido o --geometry-dir con predictions/*.npz")
        
        # Resolver output_dir
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = ctx.stage_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
    
        print("=" * 70)
        print("FASE XI v2 - DESCUBRIMIENTO GENUINO DE ECUACIONES GRAVITATORIAS")
        print("=" * 70)
        print(f"Geometrias: {preds_dir}")
        print(f"Output:     {output_dir}")
        print(f"d:          {args.d}")
        print("=" * 70)
        print("\nNOTA: Este script NO asume Einstein a priori.")
        print("      Descubre la ecuacion y LUEGO verifica si es Einstein.")
        print("=" * 70)
        
        all_results = {"geometries": []}
        
        npz_files = sorted(preds_dir.glob("*.npz"))
        
        for npz_path in npz_files:
            name = npz_path.stem.replace("_geometry", "")
            print(f"\n>> Procesando {name}...")
            
            data = np.load(npz_path, allow_pickle=True)
            z = data["z"]
            A = data["A_pred"]
            f = data["f_pred"]
            category = str(data.get("category", "unknown"))
            
            # Calcular tensores geometricos
            print("   Calculando tensores geometricos...")
            tensors = compute_geometric_tensors(z, A, f, args.d)
            
            # Descubrir ecuaciones SIN asumir Einstein
            print("   Descubriendo ecuaciones (sin asumir Einstein)...")
            geo_output = output_dir / name
            results = discover_geometric_relations(
                tensors, args.d, geo_output,
                niterations=args.niterations,
                maxsize=args.maxsize,
                seed=args.seed
            )
            
            # Validar SI es Einstein (posterior)
            print("\n   Validacion posterior (es Einstein?)...")
            validation = validate_einstein_posterior(results, args.d)
            
            print(f"\n   Validacion:")
            print(f"     R constante:              {'OK' if validation['R_constant'] else 'NO'}")
            print(f"     Compatible con Einstein:  {'OK' if validation['einstein_vacuum_compatible'] else 'NO'}")
            print(f"     A ~ log(z):               {'OK' if validation['A_is_logarithmic'] else 'NO'}")
            print(f"     Einstein score:           {validation['einstein_score']:.2f}")
            print(f"     Veredicto:                {validation['verdict']}")
            
            geo_results = {
                "name": name,
                "category": category,
                "results": results,
                "validation": validation
            }
            all_results["geometries"].append(geo_results)
        
        # Guardar resultados individuales
        json_path = geo_output / "einstein_discovery.json"
        json_path.parent.mkdir(exist_ok=True)
        json_path.write_text(json.dumps(geo_results, indent=2, default=str))
    
        # Resumen global
        print("\n" + "=" * 70)
        print("RESUMEN GLOBAL")
        print("=" * 70)
        
        verdicts = [g["validation"]["verdict"] for g in all_results["geometries"]]
        n_einstein = sum(1 for v in verdicts if v == "LIKELY_EINSTEIN_VACUUM")
        n_possibly = sum(1 for v in verdicts if v == "POSSIBLY_EINSTEIN_WITH_MATTER")
        n_non = sum(1 for v in verdicts if v == "NON_EINSTEIN_OR_DEFORMED")
        n_total = len(verdicts)
        
        avg_score = np.mean([g["validation"]["einstein_score"] for g in all_results["geometries"]]) if verdicts else 0.0
        
        print(f"  Geometrias procesadas:           {n_total}")
        print(f"  Likely Einstein vacuum:          {n_einstein}/{n_total}")
        print(f"  Possibly Einstein + matter:      {n_possibly}/{n_total}")
        print(f"  Non-Einstein or deformed:        {n_non}/{n_total}")
        print(f"  Einstein score promedio:         {avg_score:.2f}")
        
        all_results["summary"] = {
            "n_geometries": n_total,
            "n_likely_einstein": n_einstein,
            "n_possibly_einstein": n_possibly,
            "n_non_einstein": n_non,
            "average_einstein_score": float(avg_score)
        }
        
        summary_path = output_dir / "einstein_discovery_summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2, default=str))
        
        print(f"\n  Resultados: {summary_path}")
        
        # === ACTUALIZAR RUN_MANIFEST (IO v2) ===
        if args.run_dir and HAS_CUERDAS_IO:
            try:
                update_run_manifest(
                    Path(args.run_dir),
                    {
                        "bulk_equations_dir": str(output_dir.relative_to(Path(args.run_dir)) 
                                                  if output_dir.is_relative_to(Path(args.run_dir)) 
                                                  else output_dir),
                        "bulk_equations_summary": str(summary_path.relative_to(Path(args.run_dir))
                                                      if summary_path.is_relative_to(Path(args.run_dir))
                                                      else summary_path),
                    }
                )
                print(f"  Manifest actualizado: {Path(args.run_dir) / 'run_manifest.json'}")
            except Exception as e:
                print(f"  [WARN] No se pudo actualizar run_manifest.json: {e}")
        
        print("=" * 70)
        
        if n_einstein == n_total:
            print("\n[OK] ECUACIONES DE EINSTEIN REDESCUBIERTAS (todas las geometrias)")
        elif n_einstein > 0:
            print(f"\n[OK] Einstein redescubierto en {n_einstein}/{n_total} geometrias")
        else:
            print("\n[!] Ninguna geometria parece ser Einstein puro")
        
        print("\nProximo paso: 08_build_holographic_dictionary.py")

        ctx.record_artifact(output_dir)
        ctx.record_artifact(summary_path)
        ctx.write_manifest(
            outputs={"bulk_equations_dir": str(output_dir.relative_to(ctx.run_root))},
            metadata={"command": " ".join(sys.argv)},
        )
    except Exception as exc:  # pragma: no cover - infra guardrail
        status = STATUS_ERROR
        exit_code = EXIT_ERROR
        error_message = str(exc)
        raise
    finally:
        summary_stage_path = ctx.stage_dir / "stage_summary.json"
        try:
            ctx.record_artifact(summary_stage_path)
        except Exception:
            pass
        ctx.write_summary(status=status, exit_code=exit_code, error_message=error_message)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
