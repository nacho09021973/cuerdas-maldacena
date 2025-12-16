#!/usr/bin/env python3
"""
fase12_prediction_engine.py Ã¢â‚¬â€ Fase XII: Motor de Predicciones

OBJETIVO:
    Usar el motor de Fase XI para:
    1. Procesar datos reales adaptados
    2. Generar predicciones de geometrÃƒÂ­a bulk
    3. Validar contra fÃƒÂ­sica conocida
    4. Proponer nuevas predicciones verificables

FASE XII.a:
    Mapeo sistema real Ã¢â€ â€™ familia de geometrÃƒÂ­a Ã¢â€ â€™ candidato de Fase XI
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import h5py


# ============================================================
# MAPEO SISTEMA REAL Ã¢â€ â€™ FAMILIA GEOMÃƒâ€°TRICA (FASE XII.a)
# ============================================================

SYSTEM_TO_FAMILY = {
    "ising_3d": "ads",
    "O4_model": "ads",
    "lattice_qcd": "deformed",
    "strange_metal_cuprate": "lifshitz",
    "cmb_inflation": "unknown",
}


# ============================================================
# CARGADOR DE GEOMETRÃƒÂAS DE FASE XI
# ============================================================

class Fase11GeometryLoader:
    """
    Carga y filtra geometrÃƒÂ­as de Fase XI para usar como candidatas.
    """
    
    def __init__(self, fase11_dir: Path):
        self.fase11_dir = fase11_dir
        self.geometry_dir = fase11_dir / "geometry"
        self.manifest = self._load_manifest()
        self.geometry_cache = {}
    
    def _load_manifest(self) -> Dict:
        """Carga el manifest de Fase XI si existe.

        Soporta varios layouts posibles:
          - <fase11_dir>/manifest.json          (caso ideal)
          - <fase11_dir>/data/manifest.json     (layout de run_fase_11_v2.py)
          - <fase11_dir>/fase11_manifest.json   (fallback legado)
        """
        candidate_paths = [
            self.fase11_dir / "manifest.json",
            self.fase11_dir / "data" / "manifest.json",
            self.fase11_dir / "fase11_manifest.json",
        ]
        for mpath in candidate_paths:
            if mpath.exists():
                try:
                    return json.loads(mpath.read_text())
                except Exception:
                    # Si un manifest estÃƒÂ¡ corrupto, probamos el siguiente
                    continue
        return {"geometries": []}
  
    def get_universes_by_family(self, family: str) -> List[Dict]:
        """
        Obtiene todos los universos de Fase XI que pertenecen a una familia.
        """
        universes = []
        
        # Buscar en manifest
        for geo in self.manifest.get("geometries", []):
            if geo.get("family", "").lower() == family.lower():
                universes.append(geo)
        
        # Si no hay manifest, buscar por archivos
        if not universes and self.geometry_dir.exists():
            for npz_file in self.geometry_dir.glob("*.npz"):
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    file_family = str(data.get("family", ["unknown"])[0]) if "family" in data else "unknown"
                    if file_family.lower() == family.lower():
                        universes.append({
                            "name": npz_file.stem,
                            "family": file_family,
                            "file": str(npz_file)
                        })
                except Exception:
                    continue
            
            # TambiÃƒÂ©n buscar archivos JSON de resultado
            for json_file in self.geometry_dir.glob("*_geometry_result.json"):
                try:
                    result = json.loads(json_file.read_text())
                    file_family = result.get("predicted_family", result.get("family", "unknown"))
                    if file_family.lower() == family.lower():
                        universes.append({
                            "name": json_file.stem.replace("_geometry_result", ""),
                            "family": file_family,
                            "file": str(json_file),
                            "result": result
                        })
                except Exception:
                    continue
        
        return universes
    
    def select_best_candidate(self, universes: List[Dict]) -> Optional[Dict]:
        """
        Selecciona el mejor candidato de una lista de universos.
        Criterios: mejor A_r2/f_r2, o el primero si no hay mÃƒÂ©tricas.
        """
        if not universes:
            return None
        
        # Intentar ordenar por calidad de reconstrucciÃƒÂ³n
        scored = []
        for u in universes:
            score = 0.0
            
            # Si tiene resultado cargado, usar mÃƒÂ©tricas
            if "result" in u:
                r = u["result"]
                A_r2 = r.get("A_r2", r.get("metrics", {}).get("A_r2", 0))
                f_r2 = r.get("f_r2", r.get("metrics", {}).get("f_r2", 0))
                score = (A_r2 + f_r2) / 2
            
            # Intentar cargar desde archivo
            elif "file" in u:
                file_path = Path(u["file"])
                if file_path.suffix == ".json":
                    try:
                        r = json.loads(file_path.read_text())
                        A_r2 = r.get("A_r2", r.get("metrics", {}).get("A_r2", 0))
                        f_r2 = r.get("f_r2", r.get("metrics", {}).get("f_r2", 0))
                        score = (A_r2 + f_r2) / 2
                        u["result"] = r
                    except Exception:
                        pass
            
            scored.append((score, u))
        
        # Ordenar por score descendente
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return scored[0][1] if scored else universes[0]
    
    def get_geometry_for_family(self, family: str) -> Tuple[Optional[Dict], str]:
        """
        Obtiene la mejor geometrÃƒÂ­a candidata para una familia.
        Retorna: (geometry_result, universe_name) o (None, reason)
        """
        universes = self.get_universes_by_family(family)
        
        if not universes:
            return None, f"No se encontraron universos de familia '{family}' en Fase XI"
        
        best = self.select_best_candidate(universes)
        
        if best is None:
            return None, f"No se pudo seleccionar candidato de {len(universes)} universos"
        
        # Cargar resultado completo si no estÃƒÂ¡ cargado
        if "result" not in best:
            result_file = self.geometry_dir / f"{best['name']}_geometry_result.json"
            if result_file.exists():
                best["result"] = json.loads(result_file.read_text())
            else:
                # Construir resultado mÃƒÂ­nimo desde manifest
                best["result"] = {
                    "name": best["name"],
                    "family": best.get("family", family),
                    "predicted_family": best.get("family", family),
                    "source": "fase11_manifest"
                }
        
        return best["result"], best["name"]


# ============================================================
# VALIDADOR DE COHERENCIA CON FÃƒÂSICA CONOCIDA
# ============================================================

class PhysicsValidator:
    """
    Valida que las predicciones del motor sean coherentes
    con lo que la comunidad ya sabe del sistema.
    """
    
    def __init__(self):
        self.validations = []
        self.warnings = []
        self.predictions_new = []
    
    def validate_ads_cft_basics(
        self,
        source: str,
        predicted_family: str,
        predicted_d: int,
        expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ValidaciÃƒÂ³n bÃƒÂ¡sica de consistencia AdS/CFT."""
        result = {"passed": True, "checks": []}
        
        # Check: familia correcta
        if "expected_bulk" in expected:
            exp_bulk = expected["expected_bulk"].lower()
            if "ads" in exp_bulk and "ads" not in predicted_family.lower():
                result["checks"].append({
                    "name": "bulk_family",
                    "passed": False,
                    "expected": exp_bulk,
                    "got": predicted_family
                })
                result["passed"] = False
            elif "lifshitz" in exp_bulk and "lifshitz" not in predicted_family.lower():
                result["checks"].append({
                    "name": "bulk_family",
                    "passed": False,
                    "expected": exp_bulk,
                    "got": predicted_family
                })
                result["passed"] = False
            else:
                result["checks"].append({
                    "name": "bulk_family",
                    "passed": True
                })
        
        # Check: Einstein vs correcciones
        if "expected_einstein" in expected:
            result["checks"].append({
                "name": "einstein_expectation",
                "expected": expected["expected_einstein"],
                "note": "Verificar en output de discover_einstein"
            })
        
        return result
    
    def validate_thermal_physics(
        self,
        has_horizon: bool,
        predicted_zh: float,
        T_data: float,
        eta_over_s: Optional[float] = None
    ) -> Dict[str, Any]:
        """ValidaciÃƒÂ³n de fÃƒÂ­sica tÃƒÂ©rmica/horizonte."""
        result = {"passed": True, "checks": []}
        
        # Si T > 0, debe haber horizonte
        if T_data > 1e-10 and not has_horizon:
            result["checks"].append({
                "name": "horizon_existence",
                "passed": False,
                "reason": "T > 0 pero no se predijo horizonte"
            })
            result["passed"] = False
        
        # ÃŽÂ·/s bound (KSS)
        if eta_over_s is not None:
            kss_bound = 1.0 / (4 * np.pi)
            if eta_over_s < kss_bound * 0.9:
                result["checks"].append({
                    "name": "kss_bound",
                    "passed": False,
                    "reason": f"ÃŽÂ·/s = {eta_over_s:.4f} < KSS = {kss_bound:.4f}",
                    "note": "Posible violaciÃƒÂ³n de unitariedad"
                })
                self.warnings.append("Posible violaciÃƒÂ³n del bound KSS")
            else:
                result["checks"].append({
                    "name": "kss_bound",
                    "passed": True,
                    "value": eta_over_s,
                    "bound": kss_bound
                })
        
        # RelaciÃƒÂ³n T - z_h
        if T_data > 0 and predicted_zh > 0:
            d = 4
            T_predicted = d / (4 * np.pi * predicted_zh)
            T_ratio = T_data / T_predicted if T_predicted > 0 else 0
            
            result["checks"].append({
                "name": "T_zh_relation",
                "T_data": T_data,
                "T_from_zh": T_predicted,
                "ratio": T_ratio,
                "passed": 0.1 < T_ratio < 10
            })
        
        return result
    
    def validate_bootstrap_cft(
        self,
        operators_predicted: List[Dict],
        operators_known: List[Dict],
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """ValidaciÃƒÂ³n para CFTs del bootstrap."""
        result = {"passed": True, "checks": []}
        
        for op_known in operators_known:
            name = op_known.get("name", "")
            Delta_known = op_known["Delta"]
            error = op_known.get("Delta_error", 0.01)
            
            matched = False
            for op_pred in operators_predicted:
                if abs(op_pred["Delta"] - Delta_known) < max(tolerance, 3 * error):
                    matched = True
                    result["checks"].append({
                        "name": f"Delta_{name}",
                        "passed": True,
                        "known": Delta_known,
                        "predicted": op_pred["Delta"],
                        "diff": abs(op_pred["Delta"] - Delta_known)
                    })
                    break
            
            if not matched:
                result["checks"].append({
                    "name": f"Delta_{name}",
                    "passed": False,
                    "known": Delta_known,
                    "reason": "No match encontrado en predicciÃƒÂ³n"
                })
        
        return result
    
    def validate_scaling_exponents(
        self,
        predicted_z: float,
        predicted_theta: float,
        expected_z: Optional[float] = None,
        expected_theta: Optional[float] = None
    ) -> Dict[str, Any]:
        """ValidaciÃƒÂ³n de exponentes de scaling."""
        result = {"passed": True, "checks": []}
        
        if expected_z is not None:
            z_match = abs(predicted_z - expected_z) < 0.5
            result["checks"].append({
                "name": "z_dynamical",
                "passed": z_match,
                "expected": expected_z,
                "predicted": predicted_z
            })
            if not z_match:
                result["passed"] = False
        
        if expected_theta is not None:
            theta_match = abs(predicted_theta - expected_theta) < 0.5
            result["checks"].append({
                "name": "theta_hyperscaling",
                "passed": theta_match,
                "expected": expected_theta,
                "predicted": predicted_theta
            })
            if not theta_match:
                result["passed"] = False
        
        return result
    
    def generate_new_predictions(
        self,
        source: str,
        geometry_result: Dict,
        einstein_result: Dict,
        dictionary_result: Dict
    ) -> List[Dict]:
        """
        Genera predicciones nuevas verificables.
        """
        predictions = []
        d = geometry_result.get("d", 4)
        
        # 1. PredicciÃƒÂ³n de QNMs desde geometrÃƒÂ­a
        if geometry_result.get("z_h", 0) > 0:
            z_h = geometry_result["z_h"]
            omega_real = d / (4 * np.pi * z_h)
            omega_imag = omega_real
            
            predictions.append({
                "type": "qnm_fundamental",
                "observable": "Quasinormal mode frequency",
                "value_real": omega_real,
                "value_imag": -omega_imag,
                "units": "natural",
                "source": source,
                "confidence": "medium",
                "how_to_verify": "Lattice QCD spectral functions o experiments"
            })
        
        # 2. PredicciÃƒÂ³n de entropÃƒÂ­a desde horizonte
        if geometry_result.get("z_h", 0) > 0:
            z_h = geometry_result["z_h"]
            s_over_T3 = 4 * np.pi / d
            
            predictions.append({
                "type": "entropy_density",
                "observable": "s/T^3 ratio",
                "value": s_over_T3,
                "source": source,
                "confidence": "high" if geometry_result.get("family") == "ads" else "medium"
            })
        
        # 3. PredicciÃƒÂ³n de relaciÃƒÂ³n ÃŽâ€ - masa desde diccionario
        if dictionary_result.get("m2L2_relation"):
            predictions.append({
                "type": "mass_dimension_relation",
                "observable": "mÃ‚Â²LÃ‚Â² = ÃŽâ€(ÃŽâ€-d) o variante",
                "equation": dictionary_result["m2L2_relation"],
                "source": source,
                "confidence": "high"
            })
        
        # 4. Predicciones especÃƒÂ­ficas por fuente
        if source == "lattice":
            predictions.append({
                "type": "transport",
                "observable": "ÃŽÂ·/s ratio",
                "prediction": "Close to KSS bound if strongly coupled",
                "source": source
            })
        
        elif source == "condensed":
            if geometry_result.get("z_dyn", 1.0) != 1.0:
                z = geometry_result["z_dyn"]
                predictions.append({
                    "type": "dc_conductivity",
                    "observable": "ÃÆ’_DC temperature scaling",
                    "prediction": f"ÃÆ’ Ã¢Ë†Â T^{{-2/z}} = T^{{{-2/z:.2f}}}",
                    "z_dyn": z,
                    "source": source
                })
        
        elif source == "cosmology":
            predictions.append({
                "type": "tensor_to_scalar",
                "observable": "r = tensor/scalar ratio",
                "note": "Determinado por geometrÃƒÂ­a durante inflaciÃƒÂ³n",
                "source": source
            })
        
        return predictions


# ============================================================
# MOTOR DE PREDICCIONES FASE XII
# ============================================================

class PredictionEngine:
    """
    Motor principal que:
    1. Lee outputs del motor XI
    2. Usa mapeo sistema Ã¢â€ â€™ familia para encontrar geometrÃƒÂ­a candidata
    3. Valida coherencia fÃƒÂ­sica
    4. Genera reporte de predicciones
    """
    
    def __init__(
        self,
        geometry_dir: Path,
        einstein_dir: Path,
        dictionary_dir: Path,
        fase11_dir: Optional[Path] = None
    ):
        self.geometry_dir = geometry_dir
        self.einstein_dir = einstein_dir
        self.dictionary_dir = dictionary_dir
        
        # Fase XII.a: loader de geometrÃƒÂ­as de Fase XI
        self.fase11_loader = None
        if fase11_dir is not None:
            self.fase11_loader = Fase11GeometryLoader(fase11_dir)
        
        self.validator = PhysicsValidator()
        self.results = []
    
    def load_geometry_result(self, name: str) -> Dict:
        """Carga resultado de emergencia geomÃƒÂ©trica."""
        result_file = self.geometry_dir / f"{name}_geometry_result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        return {}
    
    def load_einstein_result(self, name: str) -> Dict:
        """Carga resultado de descubrimiento de Einstein."""
        result_file = self.einstein_dir / f"{name}_einstein_result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        return {}
    
    def load_dictionary_result(self, name: str) -> Dict:
        """Carga resultado del diccionario hologrÃƒÂ¡fico."""
        result_file = self.dictionary_dir / f"{name}_dictionary_result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        return {}
    
    def get_geometry_from_fase11(self, system_name: str) -> Tuple[Dict, str, str]:
        """
        FASE XII.a: Obtiene geometrÃƒÂ­a candidata de Fase XI basÃƒÂ¡ndose en el mapeo.
        
        Returns:
            (geometry_result, geometry_source, universe_name_or_reason)
        """
        # Obtener familia esperada
        family_hint = SYSTEM_TO_FAMILY.get(system_name, "unknown")
        
        if self.fase11_loader is None:
            return {}, "none", "Fase11GeometryLoader no inicializado"
        
        # Buscar geometrÃƒÂ­a candidata
        geo_result, universe_or_reason = self.fase11_loader.get_geometry_for_family(family_hint)
        
        if geo_result is not None:
            return geo_result, "fase11", universe_or_reason
        else:
            return {}, "none", universe_or_reason
    
    def process_system(
        self,
        data_path: Path,
        known_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Procesa un sistema fÃƒÂ­sico completo."""
        
        # Cargar datos del boundary
        with h5py.File(data_path, "r") as f:
            name = f.attrs["name"]
            source = f.attrs["source"]
            d = int(f.attrs["d"])
            T = float(f.attrs.get("T", 0.0))
            has_horizon = bool(f.attrs.get("has_horizon", False))
            operators = json.loads(f.attrs["operators"])
            physics_meta = json.loads(f.attrs.get("physics_metadata", "{}"))
        
        # Primero: intentar cargar resultado directo de Fase XI
        geo_result = self.load_geometry_result(name)
        einstein_result = self.load_einstein_result(name)
        dict_result = self.load_dictionary_result(name)
        
        # FASE XII.a: si no hay resultado directo, buscar por familia
        geometry_source = "direct"
        universe_name = name
        family_hint = SYSTEM_TO_FAMILY.get(name, "unknown")
        no_geometry_reason = ""
        
        if not geo_result:
            geo_result, geometry_source, universe_name = self.get_geometry_from_fase11(name)
            
            if geometry_source == "none":
                no_geometry_reason = universe_name  # En este caso contiene la razÃ³n
                universe_name = ""
            else:
                # ============================================================
                # FASE XII.b: Carga del diccionario del universo XI asociado
                # ============================================================
                # Si obtuvimos geometrÃ­a de Fase XI por familia, tambiÃ©n cargar
                # el diccionario y einstein del universo asociado.
                # Esto es CRUCIAL para que Ising 3D use el diccionario de ads5_pure
                # (que contiene operators_predicted con Î”, mÂ²LÂ², etc.)
                if not dict_result and universe_name:
                    dict_result = self.load_dictionary_result(universe_name)
                if not einstein_result and universe_name:
                    einstein_result = self.load_einstein_result(universe_name)
        
        # Determinar fuente del diccionario para transparencia
        dictionary_source = "none"
        dictionary_universe = ""
        if dict_result:
            provenance = dict_result.get("provenance", "unknown")
            if geometry_source == "fase11" and universe_name:
                dictionary_source = "fase11"
                dictionary_universe = universe_name
            else:
                dictionary_source = "direct"
                dictionary_universe = name
            # AÃ±adir nota si es manual (v0 tÃ©cnica)
            if provenance == "manual":
                dictionary_source = f"{dictionary_source}_manual"
        
        result = {
            "name": name,
            "source": source,
            "d": d,
            "T": T,
            "has_horizon": has_horizon,
            "family_hint": family_hint,
            "geometry_source": geometry_source,
            "geometry_universe": universe_name if geometry_source == "fase11" else "",
            "dictionary_source": dictionary_source,
            "dictionary_universe": dictionary_universe,
            "geometry": geo_result,
            "einstein": einstein_result,
            "dictionary": dict_result,
            "validations": {},
            "predictions_new": [],
            "overall_status": "unknown"
        }
        
        # Si no hay geometrÃ­a, aÃ±adir razÃ³n
        if geometry_source == "none":
            result["no_geometry_reason"] = no_geometry_reason
        
        # ============ VALIDACIONES ============
        
        # 1. ValidaciÃƒÂ³n bÃƒÂ¡sica AdS/CFT
        if geo_result:
            predicted_family = geo_result.get("predicted_family", geo_result.get("family", "unknown"))
            v1 = self.validator.validate_ads_cft_basics(
                source, predicted_family, d, known_predictions
            )
            result["validations"]["ads_cft_basics"] = v1
        
        # 2. ValidaciÃƒÂ³n tÃƒÂ©rmica
        if T > 0 or has_horizon:
            v2 = self.validator.validate_thermal_physics(
                has_horizon,
                geo_result.get("z_h", 0),
                T,
                physics_meta.get("eta_over_s_min", None)
            )
            result["validations"]["thermal_physics"] = v2
        
        # 3. ValidaciÃƒÂ³n de espectro (para bootstrap)
        if source == "bootstrap" and operators:
            v3 = self.validator.validate_bootstrap_cft(
                dict_result.get("operators_predicted", []),
                operators
            )
            result["validations"]["bootstrap_spectrum"] = v3
        
        # 4. ValidaciÃƒÂ³n de exponentes (para condensed)
        if source == "condensed":
            expected_z = known_predictions.get("expected_z_dyn", None)
            expected_theta = known_predictions.get("expected_theta", None)
            if expected_z is not None or expected_theta is not None:
                v4 = self.validator.validate_scaling_exponents(
                    geo_result.get("z_dyn", 1.0),
                    geo_result.get("theta", 0.0),
                    expected_z,
                    expected_theta
                )
                result["validations"]["scaling_exponents"] = v4
        
        # ============ PREDICCIONES NUEVAS ============
        
        predictions = self.validator.generate_new_predictions(
            source, geo_result, einstein_result, dict_result
        )
        result["predictions_new"] = predictions
        
        # ============ STATUS GLOBAL ============
        
        all_passed = all(
            v.get("passed", True) 
            for v in result["validations"].values()
        )
        
        if not geo_result:
            result["overall_status"] = "no_geometry_result"
        elif geometry_source == "fase11":
            result["overall_status"] = "geometry_from_fase11"
        elif all_passed:
            result["overall_status"] = "validated"
        else:
            result["overall_status"] = "requires_investigation"
        
        self.results.append(result)
        return result
    
    def generate_report(self, output_path: Path):
        """Genera reporte completo de Fase XII."""
        
        report = {
            "phase": "XII",
            "title": "CUERDAS en el Mundo Real",
            "version": "XII.a",
            "n_systems": len(self.results),
            "summary": {
                "validated": sum(1 for r in self.results if r["overall_status"] == "validated"),
                "geometry_from_fase11": sum(1 for r in self.results if r["overall_status"] == "geometry_from_fase11"),
                "requires_investigation": sum(1 for r in self.results if r["overall_status"] == "requires_investigation"),
                "no_result": sum(1 for r in self.results if r["overall_status"] == "no_geometry_result")
            },
            "systems": self.results,
            "all_new_predictions": [],
            "warnings": self.validator.warnings
        }
        
        # Agregar todas las predicciones nuevas
        for r in self.results:
            for pred in r.get("predictions_new", []):
                pred["system"] = r["name"]
                report["all_new_predictions"].append(pred)
        
        output_path.write_text(json.dumps(report, indent=2, default=str))
        return report


# ============================================================
# RUNNER FASE XII
# ============================================================

def run_fase12_on_system(
    data_path: Path,
    fase11_output_dir: Path,
    output_dir: Path,
    known_predictions: Dict[str, Any]
) -> Dict:
    """Ejecuta Fase XII completa en un sistema."""
    
    engine = PredictionEngine(
        geometry_dir=fase11_output_dir / "geometry",
        einstein_dir=fase11_output_dir / "einstein",
        dictionary_dir=fase11_output_dir / "dictionary",
        fase11_dir=fase11_output_dir
    )
    
    result = engine.process_system(data_path, known_predictions)
    
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fase XII: Motor de Predicciones"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directorio con datos adaptados (de fase12_real_data_adapters)")
    parser.add_argument("--fase11-dir", type=str, required=True,
                        help="Directorio con outputs de Fase XI")
    parser.add_argument("--output-dir", type=str, default="fase12_output",
                        help="Directorio de salida")
    parser.add_argument("--system", type=str, default="",
                        help="Sistema especÃƒÂ­fico a procesar (o vacÃƒÂ­o para todos)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    fase11_dir = Path(args.fase11_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar manifest de datos adaptados
    manifest_path = data_dir / "manifest_fase12.json"
    if not manifest_path.exists():
        print(f"Error: no existe {manifest_path}")
        print("  Primero ejecutar: fase12_real_data_adapters.py --mode process")
        return 1
    
    manifest = json.loads(manifest_path.read_text())
    
    print("=" * 70)
    print("FASE XII Ã¢â‚¬â€ MOTOR DE PREDICCIONES (v XII.a)")
    print("=" * 70)
    print(f"\n  Data dir:    {data_dir}")
    print(f"  Fase XI dir: {fase11_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Sistemas:    {len(manifest['processed'])}")
    print("\n  Mapeo sistema Ã¢â€ â€™ familia:")
    for sys_name, family in SYSTEM_TO_FAMILY.items():
        print(f"    {sys_name:25s} Ã¢â€ â€™ {family}")
    print("=" * 70)
    
    engine = PredictionEngine(
        geometry_dir=fase11_dir / "geometry",
        einstein_dir=fase11_dir / "einstein",
        dictionary_dir=fase11_dir / "dictionary",
        fase11_dir=fase11_dir
    )
    
    # Procesar cada sistema
    for item in manifest["processed"]:
        name = item["name"]
        
        if args.system and args.system != name:
            continue
        
        print(f"\n>> Procesando: {name} ({item['source']})")
        
        data_path = Path(item["output_file"])
        if not data_path.exists():
            print(f"   Ã¢Å¡Â   No existe {data_path}")
            continue
        
        known = item.get("known_predictions", {})
        
        result = engine.process_system(data_path, known)
        
        # Mostrar resultado
        family_hint = result.get("family_hint", "?")
        geo_source = result.get("geometry_source", "?")
        geo_universe = result.get("geometry_universe", "")
        
        print(f"   Family hint:     {family_hint}")
        print(f"   Geometry source: {geo_source}", end="")
        if geo_universe:
            print(f" ({geo_universe})")
        else:
            print()
        
        if result.get("no_geometry_reason"):
            print(f"   Reason:          {result['no_geometry_reason']}")
        
        print(f"   Status:          {result['overall_status']}")
        
        if result["validations"]:
            print(f"   Validaciones:")
            for vname, vresult in result["validations"].items():
                status = "Ã¢Å“â€œ" if vresult.get("passed", True) else "Ã¢Å“â€”"
                print(f"     {status} {vname}")
        
        if result["predictions_new"]:
            print(f"   Predicciones nuevas: {len(result['predictions_new'])}")
            for pred in result["predictions_new"][:3]:
                print(f"     - {pred['type']}: {pred.get('observable', '')}")
    
    # Generar reporte final
    report_path = output_dir / "fase12_report.json"
    report = engine.generate_report(report_path)
    
    print("\n" + "=" * 70)
    print("FASE XII Ã¢â‚¬â€ RESUMEN")
    print("=" * 70)
    print(f"\n  Sistemas procesados:     {report['n_systems']}")
    print(f"  Validados (directo):     {report['summary']['validated']}")
    print(f"  Geometry from Fase XI:   {report['summary']['geometry_from_fase11']}")
    print(f"  Requieren revisiÃƒÂ³n:      {report['summary']['requires_investigation']}")
    print(f"  Sin resultado:           {report['summary']['no_result']}")
    print(f"  Predicciones nuevas:     {len(report['all_new_predictions'])}")
    
    if report['warnings']:
        print(f"\n  Ã¢Å¡Â   Warnings:")
        for w in report['warnings']:
            print(f"    - {w}")
    
    print(f"\n  Reporte: {report_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
