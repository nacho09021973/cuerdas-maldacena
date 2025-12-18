#!/usr/bin/env python3
# 05_analyze_bulk_equations.py
# CUERDAS — Bloque A: Geometría emergente (análisis de ecuaciones)
#
# OBJETIVO
#   Analizar la familia de ecuaciones de bulk descubiertas:
#     - Patrones universales vs específicos de familia.
#     - Dependencia en dimensión d, z_dyn, theta, etc.
#     - Relación entre complejidad, error y clasificación física.
#
# ENTRADAS
#   - runs/bulk_equations/equations_pareto.json
#   - runs/geometry_contracts/geometry_contracts_summary.json
#
# SALIDAS
#   runs/bulk_equations_analysis/
#     bulk_equations_report.json
#       - Estadísticas globales y por familia.
#       - Listado de ecuaciones representativas.
#       - Mapas de qué ecuaciones funcionan en qué regímenes.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Depende de:
#       * 03_discover_bulk_equations.py
#       * 04_geometry_physics_contracts.py
#   - No afecta a la tubería de entrenamiento; es puramente análisis/diagnóstico.
#
# HISTÓRICO
#   - Anteriormente conocido como: analyze_discovered_equations.py

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Import local IO module for run manifest support
try:
    from cuerdas_io import resolve_bulk_equations_dir, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


def parse_equation_coefficients(eq_str: str) -> Dict[str, float]:
    """
    Extrae coeficientes numéricos de una ecuación de PySR.
    
    Ejemplo: "((-20.000029 * square(x2)) - (9.999996 * x3))"
    â†’ {"square_x2": -20.0, "x3": -10.0}
    """
    coefficients = {}
    
    # Buscar patrones como "número * término"
    patterns = [
        (r'([-]?\d+\.?\d*)\s*\*\s*square\(x(\d)\)', 'square_x'),
        (r'([-]?\d+\.?\d*)\s*\*\s*x(\d)', 'x'),
        (r'([-]?\d+\.?\d*)\s*\*\s*cube\(x(\d)\)', 'cube_x'),
        (r'x(\d)\s*\*\s*([-]?\d+\.?\d*)', 'x'),  # Orden invertido
    ]
    
    for pattern, prefix in patterns:
        matches = re.findall(pattern, eq_str)
        for match in matches:
            if isinstance(match, tuple):
                coef, idx = match[0], match[1]
                try:
                    coefficients[f"{prefix}{idx}"] = float(coef)
                except:
                    pass
    
    # Buscar divisiones
    div_pattern = r'/\s*([-]?\d+\.?\d*)'
    div_matches = re.findall(div_pattern, eq_str)
    for i, coef in enumerate(div_matches):
        try:
            coefficients[f"div_{i}"] = float(coef)
        except:
            pass
    
    return coefficients


def analyze_equation_structure(eq_str: str) -> Dict[str, bool]:
    """Analiza qué términos aparecen en la ecuación."""
    structure = {
        "has_dA_squared": "square(x2)" in eq_str or "x2 * x2" in eq_str,
        "has_d2A": "x3" in eq_str,
        "has_df": "x4" in eq_str,
        "has_d2f": "x5" in eq_str,
        "has_A": "x0" in eq_str and "x0)" not in eq_str.replace("x0 ", ""),
        "has_f": "x1" in eq_str,
        "has_cross_terms": ("x2" in eq_str and "x4" in eq_str) or ("x1" in eq_str and "x4" in eq_str),
        "complexity": eq_str.count("(") + eq_str.count("*") + eq_str.count("/")
    }
    return structure


def load_and_analyze(json_path: Path) -> Dict:
    """Carga resultados y analiza patrones."""
    
    with open(json_path) as f:
        data = json.load(f)
    
    results = {
        "by_family": defaultdict(list),
        "by_geometry": {},
        "patterns": {},
        "coefficients_vs_params": []
    }
    
    for geo in data["geometries"]:
        name = geo["name"]
        
        # Extraer z y theta del nombre
        z_dyn = 1.0
        theta = 0.0
        
        if "lifshitz_z" in name:
            match = re.search(r'z(\d+)p(\d+)', name)
            if match:
                z_dyn = float(f"{match.group(1)}.{match.group(2)}")
        
        if "hvlf" in name:
            z_match = re.search(r'z(\d+)p(\d+)', name)
            theta_match = re.search(r'theta(\d+)p(\d+)', name)
            if z_match:
                z_dyn = float(f"{z_match.group(1)}.{z_match.group(2)}")
            if theta_match:
                theta = float(f"{theta_match.group(1)}.{theta_match.group(2)}")
        
        if name == "ads5":
            z_dyn = 1.0
            theta = 0.0
        
        # Analizar ecuación de R
        if "R_equation" in geo["results"]:
            R_eq = geo["results"]["R_equation"]
            eq_str = R_eq["equation"]
            r2 = R_eq["r2"]
            
            coeffs = parse_equation_coefficients(eq_str)
            structure = analyze_equation_structure(eq_str)
            
            # Determinar familia
            if "ads" in name and "hvlf" not in name and "lifshitz" not in name:
                family = "ads"
            elif "lifshitz" in name and "hvlf" not in name:
                family = "lifshitz"
            else:
                family = "hyperscaling"
            
            geo_result = {
                "name": name,
                "z_dyn": z_dyn,
                "theta": theta,
                "family": family,
                "R_equation": eq_str,
                "R_r2": r2,
                "coefficients": coeffs,
                "structure": structure,
                "validation": geo.get("validation", {})
            }
            
            results["by_geometry"][name] = geo_result
            results["by_family"][family].append(geo_result)
            results["coefficients_vs_params"].append({
                "z": z_dyn,
                "theta": theta,
                "family": family,
                "coeffs": coeffs,
                "r2": r2
            })
    
    return results


def find_universal_structure(results: Dict) -> Dict:
    """Busca estructura universal en las ecuaciones."""
    
    patterns = {
        "universal_terms": set(),
        "family_specific_terms": defaultdict(set),
        "coefficient_trends": {}
    }
    
    # Analizar qué términos aparecen en todas las geometrías
    all_structures = [g["structure"] for g in results["by_geometry"].values()]
    
    if all_structures:
        first = all_structures[0]
        for key in first:
            if key != "complexity":
                values = [s[key] for s in all_structures]
                if all(values):
                    patterns["universal_terms"].add(key)
    
    # Analizar por familia
    for family, geos in results["by_family"].items():
        if geos:
            structures = [g["structure"] for g in geos]
            for key in structures[0]:
                if key != "complexity":
                    values = [s[key] for s in structures]
                    if all(values):
                        patterns["family_specific_terms"][family].add(key)
    
    # Buscar tendencias en coeficientes
    data = results["coefficients_vs_params"]
    
    # Agrupar por familia y buscar dependencia en z
    for family in ["ads", "lifshitz", "hyperscaling"]:
        family_data = [d for d in data if d["family"] == family]
        if len(family_data) >= 2:
            z_values = [d["z"] for d in family_data]
            
            # Buscar coeficientes comunes
            common_coeffs = set(family_data[0]["coeffs"].keys())
            for d in family_data[1:]:
                common_coeffs &= set(d["coeffs"].keys())
            
            for coeff_name in common_coeffs:
                coeff_values = [d["coeffs"][coeff_name] for d in family_data]
                
                # Calcular correlación con z
                if len(set(z_values)) > 1:
                    corr = np.corrcoef(z_values, coeff_values)[0, 1]
                    if not np.isnan(corr):
                        patterns["coefficient_trends"][f"{family}_{coeff_name}"] = {
                            "correlation_with_z": float(corr),
                            "values": list(zip(z_values, coeff_values))
                        }
    
    return patterns


def generate_report(results: Dict, patterns: Dict) -> str:
    """Genera reporte de análisis."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("ANÁLISIS DE ECUACIONES DESCUBIERTAS POR PySR")
    lines.append("=" * 70)
    
    # Resumen por familia
    lines.append("\n## ECUACIONES POR FAMILIA\n")
    
    for family in ["ads", "lifshitz", "hyperscaling"]:
        geos = results["by_family"].get(family, [])
        if geos:
            lines.append(f"\n### {family.upper()} ({len(geos)} geometrías)")
            lines.append("-" * 50)
            
            for g in geos[:3]:  # Mostrar máximo 3
                lines.append(f"\n  {g['name']} (z={g['z_dyn']}, Î¸={g['theta']})")
                lines.append(f"  R = {g['R_equation'][:60]}...")
                lines.append(f"  R² = {g['R_r2']:.6f}")
    
    # Términos universales
    lines.append("\n\n## ESTRUCTURA UNIVERSAL")
    lines.append("-" * 50)
    
    if patterns["universal_terms"]:
        lines.append("\nTérminos presentes en TODAS las geometrías:")
        for term in patterns["universal_terms"]:
            lines.append(f"  âœ“ {term}")
    
    # Términos específicos por familia
    lines.append("\n\n## TRMINOS ESPECÍFICOS POR FAMILIA")
    lines.append("-" * 50)
    
    for family, terms in patterns["family_specific_terms"].items():
        extra_terms = terms - patterns["universal_terms"]
        if extra_terms:
            lines.append(f"\n{family}:")
            for term in extra_terms:
                lines.append(f"  + {term}")
    
    # Tendencias en coeficientes
    if patterns["coefficient_trends"]:
        lines.append("\n\n## TENDENCIAS EN COEFICIENTES")
        lines.append("-" * 50)
        
        for name, trend in patterns["coefficient_trends"].items():
            corr = trend["correlation_with_z"]
            if abs(corr) > 0.5:
                direction = "aumenta" if corr > 0 else "disminuye"
                lines.append(f"\n  {name}: {direction} con z (r={corr:.2f})")
    
    # Conclusiones
    lines.append("\n\n## CONCLUSIONES")
    lines.append("=" * 70)
    
    # Verificar si hay estructura diferente
    ads_geos = results["by_family"].get("ads", [])
    lifshitz_geos = results["by_family"].get("lifshitz", [])
    hvlf_geos = results["by_family"].get("hyperscaling", [])
    
    if ads_geos and lifshitz_geos:
        ads_struct = ads_geos[0]["structure"] if ads_geos else {}
        lif_struct = lifshitz_geos[0]["structure"] if lifshitz_geos else {}
        
        if ads_struct != lif_struct:
            lines.append("\nâœ“ Las ecuaciones para Lifshitz son DIFERENTES a AdS")
            lines.append("  Esto indica física genuinamente distinta")
    
    if hvlf_geos:
        hvlf_with_cross = [g for g in hvlf_geos if g["structure"].get("has_cross_terms")]
        if hvlf_with_cross:
            lines.append(f"\n {len(hvlf_with_cross)} geometrías hyperscaling tienen términos cruzados")
            lines.append("  Esto indica acoplamiento materia-geometría no trivial")
    
    # R² promedio
    all_r2 = [g["R_r2"] for g in results["by_geometry"].values()]
    if all_r2:
        avg_r2 = np.mean(all_r2)
        lines.append(f"\n R² promedio: {avg_r2:.6f}")
        if avg_r2 > 0.999:
            lines.append("  Las ecuaciones descubiertas son muy precisas")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analiza ecuaciones descubiertas")
    parser.add_argument("--input", type=str, default=None,
                        help="Archivo einstein_discovery_summary.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Archivo de salida (.txt)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directorio raíz con run_manifest.json (IO v2)")
    args = parser.parse_args()
    
    # === RESOLVER RUTAS ===
    input_path = None
    
    # Prioridad 1: --run-dir
    if args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir)
        bulk_eq_dir = resolve_bulk_equations_dir(run_dir=run_dir)
        if bulk_eq_dir:
            candidate = bulk_eq_dir / "einstein_discovery_summary.json"
            if candidate.exists():
                input_path = candidate
    
    # Prioridad 2: --input explícito
    if input_path is None and args.input:
        input_path = Path(args.input)
    
    # Default legacy
    if input_path is None:
        input_path = Path("sweep_2d_einstein/einstein_discovery_summary.json")
    
    if not input_path.exists():
        print(f"Error: No existe {input_path}")
        return 1
    
    print(f"Analizando: {input_path}")
    
    results = load_and_analyze(input_path)
    patterns = find_universal_structure(results)
    report = generate_report(results, patterns)
    
    print(report)
    
    # Guardar
    # Resolver output
    if args.output:
        output_path = Path(args.output)
    elif args.run_dir:
        output_dir = Path(args.run_dir) / "bulk_equations_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "bulk_equations_report.txt"
    else:
        output_path = Path("equation_analysis.txt")
    
    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n-> Guardado: {output_path}")
    
    # También guardar JSON con datos estructurados
    json_output = {
        "by_family": {k: v for k, v in results["by_family"].items()},
        "patterns": {
            "universal_terms": list(patterns["universal_terms"]),
            "family_specific_terms": {k: list(v) for k, v in patterns["family_specific_terms"].items()},
            "coefficient_trends": patterns["coefficient_trends"]
        }
    }
    
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(json_output, indent=2, default=str))
    print(f"-> Guardado: {json_path}")
    
    # === ACTUALIZAR RUN_MANIFEST (IO v2) ===
    if args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir)
            update_run_manifest(
                run_dir,
                {
                    "bulk_equations_analysis_dir": str(output_path.parent.relative_to(run_dir)
                                                       if output_path.parent.is_relative_to(run_dir)
                                                       else output_path.parent),
                    "bulk_equations_report": str(json_path.relative_to(run_dir)
                                                 if json_path.is_relative_to(run_dir)
                                                 else json_path),
                }
            )
            print(f"-> Manifest actualizado")
        except Exception as e:
            print(f"[WARN] No se pudo actualizar manifest: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())
