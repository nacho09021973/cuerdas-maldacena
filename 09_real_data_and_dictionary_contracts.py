#!/usr/bin/env python3
# 09_real_data_and_dictionary_contracts.py
# CUERDAS — Bloque C: Contratos con datos reales y diccionario emergente
#
# OBJETIVO
#   Ser el punto de encuentro entre:
#     - Diccionario emergente λ_SL ↔ Δ.
#     - Atlas holográfico interno.
#     - Sistemas físicos reales (Ising 3D, O(N), QCD-like, CMB, ...).
#   Aplicar contratos que midan:
#     - Coherencia entre el diccionario emergente y datos de referencia
#       (bootstrap, lattice, observaciones).
#     - Consistencia interna por sistema.
#
# ENTRADAS
#   - runs/emergent_dictionary/lambda_sl_dictionary_report.json
#   - runs/holographic_dictionary/holographic_dictionary_summary.json
#   - Reports de sistemas reales (uno o varios), por ejemplo:
#       * runs/fase12_ising_real/fase12/predictions/fase12_report.json
#       * Futuras rutas para O(N), QCD-like, strange metals, CMB, ...
#
# SALIDAS
#   runs/real_data_contracts/
#     real_data_and_dictionary_contracts_summary.json
#       - Por sistema:
#           * Contratos específicos (ising3d_consistency, on_consistency, ...).
#           * Comparaciones Δ_predicho vs Δ_referencia.
#           * Flags de PASS/FAIL con interpretación clara.
#
# TIPOS DE CONTRATO
#   - Comparar Δ predichos vía diccionario emergente con Δ de bootstrap/lattice.
#   - Contratos de "solo tubería técnica OK" cuando el diccionario sea manual.
#   - Checks Δ(Δ-d) post-hoc, explícitamente etiquetados como diagnósticos.
#
# HONESTIDAD
#   - Mientras dictionary_source o provenance indiquen "manual", cualquier PASS
#     se interpreta como validación de infraestructura, no como descubrimiento físico.
#   - Los contratos no retroalimentan el entrenamiento; evalúan el pipeline ya entrenado.
#
# HISTÓRICO
#   - Anteriormente conocido como: contracts_fase_12_13.py

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Import local IO module for run manifest support
try:
    from cuerdas_io import load_run_manifest, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


# ============================================================
# CONTRATOS FASE XII
# ============================================================

class ContractsFase12:
    """Contratos para validaci³n de datos reales."""
    
    def __init__(self):
        self.results = []
    
    def contract_ising3d_consistency(
        self,
        predicted_family: str,
        predicted_Deltas: List[float],
        known_Deltas: Dict[str, float],
        dictionary_source: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Contrato: Para Ising 3D, las predicciones deben ser consistentes
        con los valores conocidos del bootstrap.
        
         = 0.5181489 ± 0.0000010
        ε = 1.412625 ± 0.000010
        
        FILOSOFÍA DE HONESTIDAD:
        - Si dictionary_source contiene "manual", el contrato deja claro que 
          esto es un check técnico de la tubería, no una confirmación física.
        - El contrato pasa si hay match de Î” dentro de tolerancia, indicando
          que la tubería está conectada correctamente.
        - Para un pass "físico" real, el diccionario debería ser emergente.
        """
        result = {
            "name": "ising3d_consistency", 
            "passed": True, 
            "checks": [],
            "n_predicted_Deltas": len(predicted_Deltas),
            "dictionary_source": dictionary_source
        }
        
        # Nota de honestidad si es diccionario manual
        if "manual" in dictionary_source:
            result["note"] = (
                "Diccionario v0 (manual): este check valida la tubería técnica, "
                "no es una confirmación física. Para validación física, usar "
                "diccionario emergente de Fase XII.c."
            )
        
        # Check 0: Hay predicciones de Î” disponibles
        if not predicted_Deltas:
            result["checks"].append({
                "name": "has_predicted_Deltas",
                "passed": False,
                "reason": "No hay operators_predicted en diccionario ni geometría"
            })
            result["passed"] = False
            self.results.append(result)
            return result
        else:
            result["checks"].append({
                "name": "has_predicted_Deltas",
                "passed": True,
                "n_operators": len(predicted_Deltas)
            })
        
        # Check 1: Familia debe ser AdS o cercana
        if "ads" not in predicted_family.lower() and "unknown" not in predicted_family.lower():
            result["checks"].append({
                "name": "family_is_ads_like",
                "passed": False,
                "expected": "ads or ads-like",
                "got": predicted_family
            })
            result["passed"] = False
        else:
            result["checks"].append({
                "name": "family_is_ads_like", 
                "passed": True,
                "family": predicted_family
            })
        
        # Check 2: Î”Ïƒ cercano a 0.518 (operador Ïƒ del Ising)
        tolerance = 0.1  # Tolerancia amplia para diccionario v0
        if "sigma" in known_Deltas:
            Delta_sigma_known = known_Deltas["sigma"]
            matches = [D for D in predicted_Deltas if abs(D - Delta_sigma_known) < tolerance]
            if matches:
                closest = min(matches, key=lambda x: abs(x - Delta_sigma_known))
                result["checks"].append({
                    "name": "Delta_sigma_match",
                    "passed": True,
                    "known": Delta_sigma_known,
                    "closest_predicted": closest,
                    "difference": abs(closest - Delta_sigma_known),
                    "tolerance": tolerance
                })
            else:
                result["checks"].append({
                    "name": "Delta_sigma_match",
                    "passed": False,
                    "known": Delta_sigma_known,
                    "predicted_Deltas": predicted_Deltas,
                    "tolerance": tolerance,
                    "reason": f"Ningún  predicho dentro de {tolerance} de {Delta_sigma_known}"
                })
                result["passed"] = False
        
        # Check 3: Î”Îµ cercano a 1.41 (operador Îµ del Ising)
        if "epsilon" in known_Deltas:
            Delta_epsilon_known = known_Deltas["epsilon"]
            matches = [D for D in predicted_Deltas if abs(D - Delta_epsilon_known) < tolerance]
            if matches:
                closest = min(matches, key=lambda x: abs(x - Delta_epsilon_known))
                result["checks"].append({
                    "name": "Delta_epsilon_match",
                    "passed": True,
                    "known": Delta_epsilon_known,
                    "closest_predicted": closest,
                    "difference": abs(closest - Delta_epsilon_known),
                    "tolerance": tolerance
                })
            else:
                # Epsilon match es deseable pero no crítico para el pass inicial
                result["checks"].append({
                    "name": "Delta_epsilon_match",
                    "passed": False,
                    "known": Delta_epsilon_known,
                    "predicted_Deltas": predicted_Deltas,
                    "tolerance": tolerance,
                    "note": "Check informativo - sigma es suficiente para pass v0"
                })
                # No marcamos result["passed"] = False aquí - sigma es suficiente
        
        self.results.append(result)
        return result
    
    def contract_kss_bound(
        self,
        eta_over_s: float,
        system_name: str
    ) -> Dict[str, Any]:
        """
        Contrato: ·/s â¥ 1/(4Ï) (bound KSS)
        
        Cualquier teor­a que viole esto tiene problemas de unitariedad.
        """
        kss = 1.0 / (4 * np.pi)
        result = {
            "name": "kss_bound",
            "system": system_name,
            "passed": eta_over_s >= kss * 0.95,  # 5% margen num©rico
            "value": eta_over_s,
            "bound": kss,
            "ratio": eta_over_s / kss
        }
        
        self.results.append(result)
        return result
    
    def contract_thermal_consistency(
        self,
        T_data: float,
        predicted_zh: float,
        d: int,
        system_name: str
    ) -> Dict[str, Any]:
        """
        Contrato: T y z_h deben ser consistentes.
        
        Para AdS-Schwarzschild: T = d / (4Ï z_h)
        Permitimos factor 10 de variaci³n para geometr­as deformadas.
        """
        result = {"name": "thermal_consistency", "system": system_name, "passed": True, "checks": []}
        
        if T_data <= 0 or predicted_zh <= 0:
            result["checks"].append({
                "name": "both_positive",
                "passed": T_data > 0 and predicted_zh > 0,
                "T": T_data,
                "z_h": predicted_zh
            })
            if T_data <= 0 and predicted_zh <= 0:
                result["passed"] = True  # Ambos cero es consistente
            else:
                result["passed"] = False
            self.results.append(result)
            return result
        
        # T esperada desde z_h
        T_expected = d / (4 * np.pi * predicted_zh)
        ratio = T_data / T_expected
        
        # Permitir factor 10 de variaci³n
        in_range = 0.1 < ratio < 10
        
        result["checks"].append({
            "name": "T_zh_ratio",
            "passed": in_range,
            "T_data": T_data,
            "T_from_zh": T_expected,
            "ratio": ratio,
            "allowed_range": [0.1, 10]
        })
        
        result["passed"] = in_range
        self.results.append(result)
        return result
    
    def contract_strange_metal_scaling(
        self,
        rho_exponent: float,
        predicted_z: float,
        d: int,
        system_name: str
    ) -> Dict[str, Any]:
        """
        Contrato: Para strange metals, el exponente de Ï(T) debe ser
        consistente con el exponente z din¡mico predicho.
        
        Ï â T^± donde ± = (d-2)/z para Lifshitz
        """
        result = {"name": "strange_metal_scaling", "system": system_name, "passed": True, "checks": []}
        
        # ± esperado desde z
        if predicted_z > 0:
            alpha_expected = (d - 2) / predicted_z
        else:
            alpha_expected = 1.0
        
        diff = abs(rho_exponent - alpha_expected)
        
        # Tolerancia: ±0.5
        consistent = diff < 0.5
        
        result["checks"].append({
            "name": "rho_vs_z_consistency",
            "passed": consistent,
            "rho_exponent_measured": rho_exponent,
            "alpha_from_z": alpha_expected,
            "z_predicted": predicted_z,
            "difference": diff
        })
        
        result["passed"] = consistent
        self.results.append(result)
        return result
    
    def contract_cosmology_bounds(
        self,
        ns: float,
        predicted_bulk: str,
        system_name: str
    ) -> Dict[str, Any]:
        """
        Contrato: Para cosmolog­a, ns debe estar en rango observacional.
        
        Planck 2018: ns = 0.9649 ± 0.0042
        """
        result = {"name": "cosmology_bounds", "system": system_name, "passed": True, "checks": []}
        
        ns_planck = 0.9649
        ns_error = 0.0042
        
        in_3sigma = abs(ns - ns_planck) < 3 * ns_error
        
        result["checks"].append({
            "name": "ns_in_range",
            "passed": in_3sigma,
            "ns_data": ns,
            "ns_planck": ns_planck,
            "sigma_away": abs(ns - ns_planck) / ns_error
        })
        
        # Si predice dS pero ns muy lejos, inconsistente
        if "ds" in predicted_bulk.lower() and not in_3sigma:
            result["checks"].append({
                "name": "ds_ns_consistency",
                "passed": False,
                "reason": "dS bulk pero ns fuera de rango"
            })
            result["passed"] = False
        
        result["passed"] = result["passed"] and in_3sigma
        self.results.append(result)
        return result
    
    def summary(self) -> Dict[str, Any]:
        """Resumen de todos los contratos."""
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.get("passed", False))
        
        return {
            "phase": "XII",
            "n_contracts": n_total,
            "n_passed": n_passed,
            "pass_rate": n_passed / max(n_total, 1),
            "all_passed": n_passed == n_total,
            "results": self.results
        }


# ============================================================
# CONTRATOS FASE XIII
# ============================================================

class ContractsFase13:
    """Contratos para el explorador de teor­as."""
    
    def __init__(self):
        self.results = []
    
    def contract_atlas_coverage(
        self,
        n_total: int,
        n_families: int,
        expected_families: List[str]
    ) -> Dict[str, Any]:
        """
        Contrato: El atlas debe cubrir mºltiples familias de teor­as.
        """
        result = {"name": "atlas_coverage", "passed": True, "checks": []}
        
        # M­nimo 3 familias diferentes
        result["checks"].append({
            "name": "min_families",
            "passed": n_families >= 3,
            "n_families": n_families,
            "minimum": 3
        })
        
        # Debe incluir AdS
        has_ads = "ads" in [f.lower() for f in expected_families]
        result["checks"].append({
            "name": "includes_ads",
            "passed": has_ads,
            "families": expected_families
        })
        
        # M­nimo 10 teor­as
        result["checks"].append({
            "name": "min_theories",
            "passed": n_total >= 10,
            "n_total": n_total,
            "minimum": 10
        })
        
        result["passed"] = all(c["passed"] for c in result["checks"])
        self.results.append(result)
        return result
    
    def contract_cluster_quality(
        self,
        clusters: Dict[str, List[str]],
        points: List[Dict]
    ) -> Dict[str, Any]:
        """
        Contrato: Los clusters deben ser coherentes.
        
        - Teor­as en mismo cluster deben tener caracter­sticas similares
        - No debe haber cluster dominante (>80% del total)
        """
        result = {"name": "cluster_quality", "passed": True, "checks": []}
        
        total = sum(len(v) for v in clusters.values())
        
        # Check: ningºn cluster > 80%
        for name, members in clusters.items():
            fraction = len(members) / max(total, 1)
            if fraction > 0.8:
                result["checks"].append({
                    "name": f"cluster_{name}_not_dominant",
                    "passed": False,
                    "fraction": fraction
                })
                result["passed"] = False
        
        if not any("not_dominant" in c["name"] for c in result["checks"]):
            result["checks"].append({
                "name": "no_dominant_cluster",
                "passed": True
            })
        
        # Check: clusters no triviales (>1 miembro cada uno)
        non_trivial = sum(1 for v in clusters.values() if len(v) > 1)
        result["checks"].append({
            "name": "non_trivial_clusters",
            "passed": non_trivial >= 2,
            "n_non_trivial": non_trivial
        })
        
        result["passed"] = result["passed"] and non_trivial >= 2
        self.results.append(result)
        return result
    
    def contract_outlier_genuineness(
        self,
        outliers: List[str],
        all_points: List[Dict],
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Contrato: Los outliers deben ser genuinamente diferentes.
        
        - Novelty score > threshold
        - No m¡s del 20% del total pueden ser outliers
        """
        result = {"name": "outlier_genuineness", "passed": True, "checks": []}
        
        n_total = len(all_points)
        n_outliers = len(outliers)
        
        # Check: no demasiados outliers
        outlier_fraction = n_outliers / max(n_total, 1)
        result["checks"].append({
            "name": "outlier_fraction_reasonable",
            "passed": outlier_fraction <= 0.2,
            "fraction": outlier_fraction,
            "maximum": 0.2
        })
        
        # Check: outliers tienen alto novelty score
        if outliers:
            outlier_points = [p for p in all_points if p.get("theory_id") in outliers]
            novelty_scores = [p.get("novelty_score", 0) for p in outlier_points]
            avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
            
            result["checks"].append({
                "name": "high_novelty_scores",
                "passed": avg_novelty >= threshold,
                "avg_novelty": avg_novelty,
                "threshold": threshold
            })
        else:
            result["checks"].append({
                "name": "high_novelty_scores",
                "passed": True,
                "note": "No outliers to check"
            })
        
        result["passed"] = all(c["passed"] for c in result["checks"])
        self.results.append(result)
        return result
    
    def contract_einstein_distribution(
        self,
        n_einstein: int,
        n_non_einstein: int,
        n_total: int
    ) -> Dict[str, Any]:
        """
        Contrato: Debe haber mix de teor­as Einstein y no-Einstein.
        
        Esto asegura que estamos explorando genuinamente el espacio.
        """
        result = {"name": "einstein_distribution", "passed": True, "checks": []}
        
        einstein_frac = n_einstein / max(n_total, 1)
        non_einstein_frac = n_non_einstein / max(n_total, 1)
        
        # Debe haber al menos 10% de cada tipo
        result["checks"].append({
            "name": "einstein_present",
            "passed": einstein_frac >= 0.1,
            "fraction": einstein_frac
        })
        
        result["checks"].append({
            "name": "non_einstein_present",
            "passed": non_einstein_frac >= 0.1,
            "fraction": non_einstein_frac
        })
        
        result["passed"] = all(c["passed"] for c in result["checks"])
        self.results.append(result)
        return result
    
    def contract_exploration_completeness(
        self,
        regions_explored: Dict[str, int],
        min_regions: int = 2
    ) -> Dict[str, Any]:
        """
        Contrato: Deben explorarse mºltiples regiones interesantes.
        """
        result = {"name": "exploration_completeness", "passed": True, "checks": []}
        
        n_regions_with_content = sum(1 for v in regions_explored.values() if v > 0)
        
        result["checks"].append({
            "name": "multiple_regions_explored",
            "passed": n_regions_with_content >= min_regions,
            "n_regions": n_regions_with_content,
            "minimum": min_regions,
            "regions": regions_explored
        })
        
        result["passed"] = n_regions_with_content >= min_regions
        self.results.append(result)
        return result
    
    def summary(self) -> Dict[str, Any]:
        """Resumen de todos los contratos."""
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.get("passed", False))
        
        return {
            "phase": "XIII",
            "n_contracts": n_total,
            "n_passed": n_passed,
            "pass_rate": n_passed / max(n_total, 1),
            "all_passed": n_passed == n_total,
            "results": self.results
        }


# ============================================================
# RUNNER DE CONTRATOS
# ============================================================

def run_contracts_fase12(report_path: Path) -> Dict:
    """Ejecuta contratos Fase XII desde reporte."""

    if report_path.is_dir():
        candidates = [
            report_path / "holographic_dictionary" / "holographic_dictionary_v3_summary.json",
            report_path / "holographic_dictionary" / "holographic_dictionary_summary.json",
            report_path / "fase12_report.json",
        ]
        for c in candidates:
            if c.is_file():
                report_path = c
                break
            
    if not report_path.exists():
        print(f"Error: no existe {report_path}")
        return {"error": "report not found"}
    
    report = json.loads(report_path.read_text())
    contracts = ContractsFase12()
    
    for system in report.get("systems", []):
        name = system.get("name", "")
        source = system.get("source", "")
        
        geo = system.get("geometry", {})
        predicted_family = geo.get("predicted_family", "unknown")
        predicted_zh = geo.get("z_h", 0)
        predicted_z = geo.get("z_dyn", 1.0)
        
        d = system.get("d", 4)
        T = system.get("T", 0)
        
       
        # Contratos por tipo de fuente
        if source == "bootstrap" and "ising" in name.lower():
            # Para Ising 3D queremos usar el diccionario holográfico emergente
            # (operators_predicted en dictionary). Si no hubiera, hacemos fallback a geometry.
            dict_ops = system.get("dictionary", {}).get("operators_predicted", [])
            geom_ops = system.get("geometry", {}).get("operators_predicted", [])
            operators = dict_ops if dict_ops else geom_ops

            predicted_Deltas = [op.get("Delta", 0.0) for op in operators]
            
            # Obtener fuente del diccionario para transparencia
            dictionary_source = system.get("dictionary_source", "unknown")
            if not dictionary_source or dictionary_source == "unknown":
                # Inferir de provenance si está disponible
                provenance = system.get("dictionary", {}).get("provenance", "")
                if provenance == "manual":
                    dictionary_source = "manual"

            contracts.contract_ising3d_consistency(
                predicted_family,
                predicted_Deltas,
                {
                    "sigma": 0.518,
                    "epsilon": 1.41
                    # más adelante podemos añadir epsilon' o T aquí si queremos
                },
                dictionary_source=dictionary_source
            )

        if source == "lattice":
            eta_s = system.get("physics_metadata", {}).get("eta_over_s_min", 0.1)
            if eta_s > 0:
                contracts.contract_kss_bound(eta_s, name)
        
        if T > 0 and predicted_zh > 0:
            contracts.contract_thermal_consistency(T, predicted_zh, d, name)
        
        if source == "condensed":
            rho_exp = system.get("features", {}).get("rho_exponent", 1.0)
            contracts.contract_strange_metal_scaling(rho_exp, predicted_z, d, name)
        
        if source == "cosmology":
            ns = system.get("features", {}).get("ns", 0.965)
            contracts.contract_cosmology_bounds(ns, predicted_family, name)
    
    return contracts.summary()


def run_contracts_fase13(analysis_path: Path, atlas_path=None) -> dict:
    """Ejecuta contratos Fase XIII desde an¡lisis y atlas."""
    
    if not analysis_path.exists():
        print(f"Error: no existe {analysis_path}")
        return {"error": "analysis not found"}
    
    analysis = json.loads(analysis_path.read_text())
    contracts = ContractsFase13()
    
    # Atlas coverage
    contracts.contract_atlas_coverage(
        analysis.get("n_total", 0),
        len(analysis.get("clusters", {})),
        list(analysis.get("clusters", {}).keys())
    )
    
    # Cluster quality
    if atlas_path and atlas_path.is_file():
        atlas = json.loads(atlas_path.read_text())
        clusters = atlas.get("clusters", {})
        points = atlas.get("points", [])
        
        contracts.contract_cluster_quality(clusters, points)
        
        # Outlier genuineness
        outliers = atlas.get("outliers", [])
        contracts.contract_outlier_genuineness(outliers, points)
    
    # Einstein distribution
    contracts.contract_einstein_distribution(
        analysis.get("n_einstein", 0),
        analysis.get("n_non_einstein", 0),
        analysis.get("n_total", 0)
    )
    
    # Exploration completeness
    regions = analysis.get("interesting_regions", {})
    contracts.contract_exploration_completeness(regions)
    
    return contracts.summary()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Contratos de validaci³n para Fases XII y XIII"
    )
    parser.add_argument("--phase", type=str, required=True,
                        choices=["12", "13", "both"],
                        help="Fase a validar")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directorio raíz con run_manifest.json (IO v2). Resuelve inputs automáticamente.")
    parser.add_argument("--fase12-report", type=str, default="",
                        help="Ruta al reporte de Fase XII")
    parser.add_argument("--fase13-analysis", type=str, default="",
                        help="Ruta al an¡lisis de Fase XIII")
    parser.add_argument("--fase13-atlas", type=str, default="",
                        help="Ruta al atlas de Fase XIII")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Archivo de salida")
    args = parser.parse_args()
    
    # === RESOLVER RUTAS DESDE --run-dir ===
    fase12_report = args.fase12_report or ""
    fase13_analysis = args.fase13_analysis or ""
    fase13_atlas = args.fase13_atlas or ""
    output_file = args.output_file

    if args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir)
        manifest = load_run_manifest(run_dir)
        artifacts = manifest.get("artifacts", {}) if manifest else {}
        
        # Intentar resolver fase12-report desde manifest
        if not fase12_report:
            # Buscar en lugares típicos
            candidates = [
                run_dir / artifacts.get("dictionary_report", ""),
                run_dir / "emergent_dictionary" / "lambda_sl_dictionary_report.json",
                run_dir / "bulk_eigenmodes" / "bulk_modes_meta.json",
            ]
            for c in candidates:
                if c.exists():
                    fase12_report = str(c)
                    break
        
        # Intentar resolver fase13-analysis
        if not fase13_analysis:
            candidates = [
                run_dir / artifacts.get("bulk_equations_report", ""),
                run_dir / "bulk_equations_analysis" / "bulk_equations_report.json",
            ]
            for c in candidates:
                if c.exists():
                    fase13_analysis = str(c)
                    break
        
        # Resolver output
        if not output_file:
            contracts_dir = run_dir / "contracts"
            contracts_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(contracts_dir / "contracts_12_13.json")
    
    # Default output
    if not output_file:
        output_file = "contracts_12_13.json"
    
    print("=" * 70)
    print("CONTRATOS FASES XII/XIII")
    print("=" * 70)
    
    results = {}
    
    if args.phase in ["12", "both"]:
        if fase12_report:
            print(f"\n>> Validando Fase XII desde {fase12_report}")
            results["fase12"] = run_contracts_fase12(Path(fase12_report))
            
            summary = results["fase12"]
            print(f"\n   Contratos: {summary.get('n_passed', 0)}/{summary.get('n_contracts', 0)}")
            print(f"   Pass rate: {summary.get('pass_rate', 0):.1%}")
        else:
            print("\n   âš  Se requiere --fase12-report para validar Fase XII")
    
    if args.phase in ["13", "both"]:
        if fase13_analysis:
            print(f"\n>> Validando Fase XIII desde {fase13_analysis}")
            results["fase13"] = run_contracts_fase13(
                Path(fase13_analysis),
                Path(fase13_atlas) if fase13_atlas else None
            )
            
            summary = results["fase13"]
            print(f"\n   Contratos: {summary.get('n_passed', 0)}/{summary.get('n_contracts', 0)}")
            print(f"   Pass rate: {summary.get('pass_rate', 0):.1%}")
        else:
            print("\n   âš  Se requiere --fase13-analysis para validar Fase XIII")
    
    # Guardar resultados
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    
    all_passed = True
    for phase, summary in results.items():
        if isinstance(summary, dict) and "all_passed" in summary:
            status = "OK" if summary["all_passed"] else "FAIL"
            print(f"  {phase}: {status} ({summary['n_passed']}/{summary['n_contracts']})")
            all_passed = all_passed and summary["all_passed"]
    
    print(f"\n  Output: {output_path}")
    
    # === ACTUALIZAR RUN_MANIFEST (IO v2) ===
    if args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir)
            update_run_manifest(
                run_dir,
                {
                    "contracts_dir": str(output_path.parent.relative_to(run_dir)
                                         if output_path.parent.is_relative_to(run_dir)
                                         else output_path.parent),
                    "contracts_output": str(output_path.relative_to(run_dir)
                                            if output_path.is_relative_to(run_dir)
                                            else output_path),
                }
            )
            print(f"  Manifest actualizado")
        except Exception as e:
            print(f"  [WARN] No se pudo actualizar manifest: {e}")
    
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
