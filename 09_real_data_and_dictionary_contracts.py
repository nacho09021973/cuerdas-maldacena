#!/usr/bin/env python3
# 09_real_data_and_dictionary_contracts.py
# CUERDAS — Bloque C: Contratos con datos reales y diccionario emergente
#
# CONTROL NEGATIVO (v3)
#   Métrica corregida: False Positive Rate (FPR) sobre señales holográficas.
#   NO mezcla "contratos que deben fallar" con "señales de autoengaño".
#
#   FPR = (señales holográficas disparadas) / (señales evaluables)
#   
#   Esto mide: "¿El pipeline cree que esto es holográfico cuando NO debería?"

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from cuerdas_io import load_run_manifest, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ============================================================
# SEÑALES HOLOGRÁFICAS (para control negativo)
# ============================================================

class SignalStatus(Enum):
    """Estado de una señal holográfica."""
    TRIGGERED = "triggered"      # Señal indica holografía (falso positivo)
    NOT_TRIGGERED = "not_triggered"  # Señal NO indica holografía (correcto)
    NOT_EVALUABLE = "not_evaluable"  # No hay artefactos para evaluar


@dataclass
class HolographicSignal:
    """
    Una señal holográfica es un check binario que responde:
    "¿El pipeline está actuando COMO SI esto fuera holográfico?"
    
    En un control negativo, queremos que estas señales NO se disparen.
    Si se disparan, es un falso positivo.
    """
    name: str
    description: str
    status: SignalStatus = SignalStatus.NOT_EVALUABLE
    value: Any = None
    threshold: Optional[float] = None
    evidence: str = ""
    
    @property
    def triggered(self) -> bool:
        return self.status == SignalStatus.TRIGGERED
    
    @property
    def evaluable(self) -> bool:
        return self.status != SignalStatus.NOT_EVALUABLE
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.threshold,
            "evidence": self.evidence,
            "triggered": self.triggered,
            "evaluable": self.evaluable,
        }


@dataclass
class ExpectedFailContract:
    """
    Un contrato que DEBE fallar en control negativo.
    No entra en el cálculo de FPR - es informacional.
    """
    name: str
    passed: bool
    note: str = "expected-fail for negative control"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "note": self.note,
        }


def evaluate_holographic_signals(artifacts: Dict[str, Any]) -> List[HolographicSignal]:
    """
    Evalúa todas las señales holográficas a partir de los artefactos del pipeline.
    
    Cada señal responde: "¿El pipeline cree que esto es holográfico?"
    
    En control negativo, lo correcto es que NO se disparen.
    """
    signals = []
    
    # ─────────────────────────────────────────────────────────────
    # SEÑAL 1: Familia geométrica es AdS-like
    # ─────────────────────────────────────────────────────────────
    signal_family = HolographicSignal(
        name="family_ads_like",
        description="La familia geométrica clasificada es AdS o AdS-like"
    )
    
    geometry = artifacts.get("geometry", {})
    if geometry:
        family = geometry.get("family", "").lower()
        is_ads = "ads" in family
        
        signal_family.status = SignalStatus.TRIGGERED if is_ads else SignalStatus.NOT_TRIGGERED
        signal_family.value = family
        signal_family.evidence = f"Familia clasificada: '{family}'"
    else:
        signal_family.status = SignalStatus.NOT_EVALUABLE
        signal_family.evidence = "No se encontraron artefactos de geometría"
    
    signals.append(signal_family)
    
    # ─────────────────────────────────────────────────────────────
    # SEÑAL 2: Einstein score alto
    # ─────────────────────────────────────────────────────────────
    signal_einstein = HolographicSignal(
        name="einstein_score_high",
        description="El score de Einstein equations discovery es alto (>=0.5)",
        threshold=0.5
    )
    
    einstein = artifacts.get("einstein", {})
    score = einstein.get("score")
    
    if score is not None:
        is_high = score >= signal_einstein.threshold
        signal_einstein.status = SignalStatus.TRIGGERED if is_high else SignalStatus.NOT_TRIGGERED
        signal_einstein.value = score
        signal_einstein.evidence = f"Einstein score: {score:.3f}"
    else:
        signal_einstein.status = SignalStatus.NOT_EVALUABLE
        signal_einstein.evidence = "No se encontró einstein_score en artefactos"
    
    signals.append(signal_einstein)
    
    # ─────────────────────────────────────────────────────────────
    # SEÑAL 3: Diccionario λ_SL → Δ convergió
    # ─────────────────────────────────────────────────────────────
    signal_dict = HolographicSignal(
        name="dictionary_converged",
        description="El diccionario holográfico λ_SL → Δ convergió"
    )
    
    dictionary = artifacts.get("dictionary", {})
    converged = dictionary.get("converged")
    
    if converged is not None:
        signal_dict.status = SignalStatus.TRIGGERED if converged else SignalStatus.NOT_TRIGGERED
        signal_dict.value = converged
        signal_dict.evidence = f"Convergencia: {converged}"
    else:
        signal_dict.status = SignalStatus.NOT_EVALUABLE
        signal_dict.evidence = "No se encontró estado de convergencia"
    
    signals.append(signal_dict)
    
    # ─────────────────────────────────────────────────────────────
    # SEÑAL 4: Δ predichos en rango físicamente plausible
    # ─────────────────────────────────────────────────────────────
    signal_deltas = HolographicSignal(
        name="deltas_in_physical_range",
        description="Mayoría de Δ predichos están en rango CFT plausible (0.3-4.0)",
        threshold=0.5  # >50% en rango = señal de holografía
    )
    
    predicted_deltas = dictionary.get("predicted_Deltas", [])
    
    if predicted_deltas:
        n_physical = sum(1 for d in predicted_deltas if 0.3 < d < 4.0)
        ratio = n_physical / len(predicted_deltas)
        
        in_range = ratio > signal_deltas.threshold
        signal_deltas.status = SignalStatus.TRIGGERED if in_range else SignalStatus.NOT_TRIGGERED
        signal_deltas.value = ratio
        signal_deltas.evidence = f"{n_physical}/{len(predicted_deltas)} Deltas en rango físico"
    else:
        signal_deltas.status = SignalStatus.NOT_EVALUABLE
        signal_deltas.evidence = "No hay Deltas predichos"
    
    signals.append(signal_deltas)
    
    # ─────────────────────────────────────────────────────────────
    # SEÑAL 5: Bulk equations limpias (symbolic regression exitosa)
    # ─────────────────────────────────────────────────────────────
    signal_bulk = HolographicSignal(
        name="bulk_equations_clean",
        description="Se encontraron ecuaciones bulk limpias (n_equations > 0 con score alto)"
    )
    
    n_equations = einstein.get("n_equations", 0)
    
    if n_equations is not None:
        has_equations = n_equations > 0 and (score is None or score > 0.3)
        signal_bulk.status = SignalStatus.TRIGGERED if has_equations else SignalStatus.NOT_TRIGGERED
        signal_bulk.value = n_equations
        signal_bulk.evidence = f"{n_equations} ecuaciones encontradas"
    else:
        signal_bulk.status = SignalStatus.NOT_EVALUABLE
        signal_bulk.evidence = "No hay información de bulk equations"
    
    signals.append(signal_bulk)
    
    # ─────────────────────────────────────────────────────────────
    # SEÑAL 6: Match con operadores conocidos (Δσ ≈ 0.518)
    # ─────────────────────────────────────────────────────────────
    signal_sigma = HolographicSignal(
        name="delta_sigma_match",
        description="Algún Δ predicho está cerca de Δσ=0.518 (Ising 3D)",
        threshold=0.1  # tolerancia
    )
    
    if predicted_deltas:
        delta_sigma = 0.518
        matches = [d for d in predicted_deltas if abs(d - delta_sigma) < signal_sigma.threshold]
        
        has_match = len(matches) > 0
        signal_sigma.status = SignalStatus.TRIGGERED if has_match else SignalStatus.NOT_TRIGGERED
        signal_sigma.value = matches[0] if matches else None
        signal_sigma.evidence = f"Matches con Δσ: {matches}" if matches else "Sin match"
    else:
        signal_sigma.status = SignalStatus.NOT_EVALUABLE
        signal_sigma.evidence = "No hay Deltas predichos"
    
    signals.append(signal_sigma)
    
    # ─────────────────────────────────────────────────────────────
    # SEÑAL 7: Match con operador ε (Δε ≈ 1.41)
    # ─────────────────────────────────────────────────────────────
    signal_epsilon = HolographicSignal(
        name="delta_epsilon_match",
        description="Algún Δ predicho está cerca de Δε=1.41 (Ising 3D)",
        threshold=0.15
    )
    
    if predicted_deltas:
        delta_epsilon = 1.41
        matches = [d for d in predicted_deltas if abs(d - delta_epsilon) < signal_epsilon.threshold]
        
        has_match = len(matches) > 0
        signal_epsilon.status = SignalStatus.TRIGGERED if has_match else SignalStatus.NOT_TRIGGERED
        signal_epsilon.value = matches[0] if matches else None
        signal_epsilon.evidence = f"Matches con Δε: {matches}" if matches else "Sin match"
    else:
        signal_epsilon.status = SignalStatus.NOT_EVALUABLE
        signal_epsilon.evidence = "No hay Deltas predichos"
    
    signals.append(signal_epsilon)
    
    return signals


def compute_false_positive_rate(signals: List[HolographicSignal]) -> Tuple[float, float, int, int]:
    """
    Calcula el False Positive Rate sobre señales holográficas.
    
    FPR = (señales disparadas) / (señales evaluables)
    
    Retorna: (fpr, coverage, n_triggered, n_evaluable)
    """
    evaluable = [s for s in signals if s.evaluable]
    triggered = [s for s in evaluable if s.triggered]
    
    n_evaluable = len(evaluable)
    n_triggered = len(triggered)
    n_total = len(signals)
    
    fpr = n_triggered / n_evaluable if n_evaluable > 0 else 0.0
    coverage = n_evaluable / n_total if n_total > 0 else 0.0
    
    return fpr, coverage, n_triggered, n_evaluable


def evaluate_expected_fail_contracts(artifacts: Dict[str, Any]) -> List[ExpectedFailContract]:
    """
    Evalúa contratos que DEBEN fallar en control negativo.
    Estos son informativos, no entran en el FPR.
    """
    contracts = []
    
    # ising3d_consistency debería fallar
    dictionary = artifacts.get("dictionary", {})
    predicted_deltas = dictionary.get("predicted_Deltas", [])
    geometry = artifacts.get("geometry", {})
    family = geometry.get("family", "").lower()
    
    # Simular el resultado de ising3d_consistency
    passed = False
    if predicted_deltas and "ads" in family:
        delta_sigma = 0.518
        tolerance = 0.1
        matches = [d for d in predicted_deltas if abs(d - delta_sigma) < tolerance]
        passed = len(matches) > 0
    
    contracts.append(ExpectedFailContract(
        name="ising3d_consistency",
        passed=passed,
        note="DEBE fallar para control negativo (datos no-CFT)"
    ))
    
    return contracts


# ============================================================
# CONTROL NEGATIVO (orquestador)
# ============================================================

def verify_negative_control_h5(h5_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Verifica que un HDF5 tenga los atributos de control negativo."""
    if not HAS_H5PY:
        return False, {"error": "h5py not available"}
    
    if not h5_path.exists():
        return False, {"error": "file not found", "path": str(h5_path)}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'negative_control' not in f:
                return False, {"error": "no 'negative_control' group in HDF5"}
            
            grp = f['negative_control']
            is_negative = grp.attrs.get('IS_NEGATIVE_CONTROL', 0)
            expected_holo = grp.attrs.get('EXPECTED_HOLOGRAPHIC', 1)
            
            if is_negative != 1:
                return False, {"error": "IS_NEGATIVE_CONTROL != 1", "value": int(is_negative)}
            
            if expected_holo != 0:
                return False, {"error": "EXPECTED_HOLOGRAPHIC != 0", "value": int(expected_holo)}
            
            metadata = {
                "IS_NEGATIVE_CONTROL": int(is_negative),
                "EXPECTED_HOLOGRAPHIC": int(expected_holo),
                "type": grp.attrs.get('type', 'unknown'),
                "mass": float(grp.attrs.get('mass', 0)),
                "lattice_size": int(grp.attrs.get('lattice_size', 0)),
                "dimension": int(grp.attrs.get('dimension', 0)),
                "conformal": bool(grp.attrs.get('conformal', False)),
            }
            
            return True, metadata
            
    except Exception as e:
        return False, {"error": str(e)}


def find_negative_control_h5(run_dir: Path) -> Optional[Path]:
    """Busca el HDF5 de control negativo en un directorio."""
    candidates = list(run_dir.glob("negative_control_*.h5"))
    if candidates:
        return candidates[0]
    
    for subdir in run_dir.iterdir():
        if subdir.is_dir():
            candidates = list(subdir.glob("negative_control_*.h5"))
            if candidates:
                return candidates[0]
    
    return None


def load_negative_control_artifacts(run_dir: Path) -> Dict[str, Any]:
    """Carga los artefactos del pipeline ejecutado sobre control negativo."""
    artifacts = {
        "found": False,
        "geometry": {},
        "einstein": {},
        "dictionary": {},
        "errors": []
    }
    
    # Buscar geometría
    for gdir in [run_dir / "geometry_emergent", run_dir / "predictions", run_dir / "geometry"]:
        if gdir.exists() and gdir.is_dir():
            for sf in list(gdir.glob("*summary*.json")) + list(gdir.glob("*report*.json")):
                try:
                    data = json.loads(sf.read_text())
                    artifacts["geometry"] = {
                        "source": str(sf),
                        "family": data.get("predicted_family", data.get("family", "unknown")),
                        "params": data.get("parameters", {}),
                    }
                    artifacts["found"] = True
                    break
                except:
                    pass
            if artifacts["geometry"]:
                break
    
    # Buscar Einstein
    for epath in [
        run_dir / "bulk_equations" / "einstein_discovery_summary.json",
        run_dir / "bulk_equations" / "pareto_equations.json",
        run_dir / "bulk_equations_analysis" / "bulk_equations_report.json",
    ]:
        if epath.exists():
            try:
                data = json.loads(epath.read_text())
                score = data.get("einstein_score", data.get("best_score"))
                if score is None and "equations" in data:
                    for eq in data.get("equations", []):
                        if "einstein" in eq.get("name", "").lower():
                            score = eq.get("score", eq.get("fitness"))
                            break
                
                artifacts["einstein"] = {
                    "source": str(epath),
                    "score": score,
                    "n_equations": len(data.get("equations", [])),
                }
                artifacts["found"] = True
                break
            except:
                pass
    
    # Buscar diccionario
    for dpath in [
        run_dir / "emergent_dictionary" / "lambda_sl_dictionary_report.json",
        run_dir / "holographic_dictionary" / "holographic_dictionary_v3_summary.json",
        run_dir / "holographic_dictionary" / "holographic_dictionary_summary.json",
    ]:
        if dpath.exists():
            try:
                data = json.loads(dpath.read_text())
                
                deltas = []
                for system in data.get("systems", []):
                    ops = system.get("dictionary", {}).get("operators_predicted", [])
                    if not ops:
                        ops = system.get("geometry", {}).get("operators_predicted", [])
                    for op in ops:
                        if "Delta" in op:
                            deltas.append(op["Delta"])
                
                if not deltas and "predicted_Deltas" in data:
                    deltas = data["predicted_Deltas"]
                
                converged = data.get("converged")
                if converged is None:
                    loss = data.get("final_loss", data.get("loss"))
                    if loss is not None:
                        converged = loss < 0.1
                
                artifacts["dictionary"] = {
                    "source": str(dpath),
                    "predicted_Deltas": deltas,
                    "converged": converged,
                    "n_systems": len(data.get("systems", [])),
                }
                artifacts["found"] = True
                break
            except:
                pass
    
    return artifacts


def run_negative_control_check(
    run_dir: Path,
    h5_path: Optional[Path] = None,
    fpr_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Ejecuta verificación de control negativo usando FPR sobre señales holográficas.
    
    FPR = (señales holográficas disparadas) / (señales evaluables)
    
    Esto mide: "¿El pipeline se autoengaña creyendo que hay holografía?"
    """
    logger.info("=" * 60)
    logger.info("CONTROL NEGATIVO - False Positive Rate sobre señales holográficas")
    logger.info("=" * 60)
    
    result = {
        "status": "INCOMPLETE",
        "false_positive_rate": None,
        "coverage": None,
        "n_signals_triggered": 0,
        "n_signals_evaluable": 0,
        "n_signals_total": 0,
        "fpr_threshold": fpr_threshold,
        "h5_path": None,
        "h5_verified": False,
        "signals": [],
        "expected_fail_contracts": [],
        "rationale": "",
        "errors": []
    }
    
    # Paso 1: Encontrar y verificar HDF5
    if h5_path is None:
        h5_path = find_negative_control_h5(run_dir)
    
    if h5_path is None:
        result["errors"].append("No se encontró HDF5 de control negativo")
        result["rationale"] = "Verificación incompleta: no se encontró el archivo HDF5."
        return result
    
    result["h5_path"] = str(h5_path)
    
    is_valid, h5_meta = verify_negative_control_h5(h5_path)
    if not is_valid:
        result["errors"].append(f"HDF5 inválido: {h5_meta.get('error', 'unknown')}")
        result["rationale"] = f"El HDF5 no tiene atributos correctos: {h5_meta}"
        return result
    
    result["h5_verified"] = True
    result["h5_metadata"] = h5_meta
    
    logger.info(f"  HDF5 verificado: {h5_path}")
    
    # Paso 2: Cargar artefactos
    logger.info("  Cargando artefactos del pipeline...")
    artifacts = load_negative_control_artifacts(run_dir)
    
    if not artifacts["found"]:
        result["errors"].append("No se encontraron artefactos del pipeline")
        result["rationale"] = "No hay artefactos. ¿Se ejecutó el pipeline sobre el control negativo?"
        return result
    
    # Paso 3: Evaluar señales holográficas
    logger.info("  Evaluando señales holográficas...")
    signals = evaluate_holographic_signals(artifacts)
    
    # Paso 4: Calcular FPR
    fpr, coverage, n_triggered, n_evaluable = compute_false_positive_rate(signals)
    
    result["false_positive_rate"] = fpr
    result["coverage"] = coverage
    result["n_signals_triggered"] = n_triggered
    result["n_signals_evaluable"] = n_evaluable
    result["n_signals_total"] = len(signals)
    result["signals"] = [s.to_dict() for s in signals]
    
    logger.info(f"    Señales evaluables: {n_evaluable}/{len(signals)}")
    logger.info(f"    Señales disparadas (falsos positivos): {n_triggered}")
    logger.info(f"    FPR: {fpr:.1%}")
    logger.info(f"    Coverage: {coverage:.1%}")
    
    # Paso 5: Evaluar contratos expected-fail (informativos)
    expected_fail = evaluate_expected_fail_contracts(artifacts)
    result["expected_fail_contracts"] = [c.to_dict() for c in expected_fail]
    
    # Paso 6: Determinar status
    if n_evaluable == 0:
        result["status"] = "INCOMPLETE"
        result["rationale"] = "No hay señales evaluables. Coverage insuficiente."
    elif fpr < fpr_threshold:
        result["status"] = "SUCCESS"
        result["rationale"] = (
            f"FPR={fpr:.1%} < {fpr_threshold:.0%}. "
            f"El pipeline NO se autoengaña: {n_triggered}/{n_evaluable} señales disparadas. "
            f"Esto es evidencia de honestidad científica."
        )
    elif fpr < 0.5:
        result["status"] = "WARNING"
        triggered_names = [s.name for s in signals if s.triggered]
        result["rationale"] = (
            f"FPR={fpr:.1%} (moderado). "
            f"Señales disparadas: {triggered_names}. "
            f"Investigar por qué el pipeline detecta holografía espuria."
        )
    else:
        result["status"] = "ALERT"
        triggered_names = [s.name for s in signals if s.triggered]
        result["rationale"] = (
            f"POSIBLE FALSO POSITIVO SISTEMÁTICO: FPR={fpr:.1%} >= 50%. "
            f"Señales disparadas: {triggered_names}. "
            f"Auditoría urgente necesaria."
        )
    
    logger.info(f"\n  Status: {result['status']}")
    logger.info(f"  Rationale: {result['rationale'][:100]}...")
    
    return result


# ============================================================
# CONTRATOS FASE XII (sin cambios sustanciales)
# ============================================================

class ContractsFase12:
    """Contratos para validación de datos reales."""
    
    def __init__(self):
        self.results = []
    
    def contract_ising3d_consistency(
        self,
        predicted_family: str,
        predicted_Deltas: List[float],
        known_Deltas: Dict[str, float],
        dictionary_source: str = "unknown"
    ) -> Dict[str, Any]:
        """Contrato: Para Ising 3D, predicciones consistentes con bootstrap."""
        result = {
            "name": "ising3d_consistency", 
            "passed": True, 
            "checks": [],
            "n_predicted_Deltas": len(predicted_Deltas),
            "dictionary_source": dictionary_source
        }
        
        if "manual" in dictionary_source:
            result["note"] = "Diccionario v0 (manual): check técnico, no confirmación física."
        
        if not predicted_Deltas:
            result["checks"].append({"name": "has_predicted_Deltas", "passed": False})
            result["passed"] = False
            self.results.append(result)
            return result
        else:
            result["checks"].append({"name": "has_predicted_Deltas", "passed": True})
        
        if "ads" not in predicted_family.lower() and "unknown" not in predicted_family.lower():
            result["checks"].append({"name": "family_is_ads_like", "passed": False, "got": predicted_family})
            result["passed"] = False
        else:
            result["checks"].append({"name": "family_is_ads_like", "passed": True})
        
        tolerance = 0.1
        if "sigma" in known_Deltas:
            Delta_sigma = known_Deltas["sigma"]
            matches = [D for D in predicted_Deltas if abs(D - Delta_sigma) < tolerance]
            if matches:
                result["checks"].append({"name": "Delta_sigma_match", "passed": True})
            else:
                result["checks"].append({"name": "Delta_sigma_match", "passed": False})
                result["passed"] = False
        
        if "epsilon" in known_Deltas:
            Delta_epsilon = known_Deltas["epsilon"]
            matches = [D for D in predicted_Deltas if abs(D - Delta_epsilon) < tolerance]
            result["checks"].append({
                "name": "Delta_epsilon_match", 
                "passed": len(matches) > 0,
                "note": "informativo"
            })
        
        self.results.append(result)
        return result
    
    def contract_kss_bound(self, eta_over_s: float, system_name: str) -> Dict[str, Any]:
        kss = 1.0 / (4 * np.pi)
        result = {
            "name": "kss_bound",
            "system": system_name,
            "passed": eta_over_s >= kss * 0.95,
            "value": eta_over_s,
            "bound": kss
        }
        self.results.append(result)
        return result
    
    def contract_thermal_consistency(self, T_data: float, predicted_zh: float, d: int, system_name: str) -> Dict[str, Any]:
        result = {"name": "thermal_consistency", "system": system_name, "passed": True, "checks": []}
        
        if T_data <= 0 or predicted_zh <= 0:
            result["passed"] = T_data <= 0 and predicted_zh <= 0
            self.results.append(result)
            return result
        
        T_expected = d / (4 * np.pi * predicted_zh)
        ratio = T_data / T_expected
        result["passed"] = 0.1 < ratio < 10
        result["checks"].append({"name": "T_zh_ratio", "ratio": ratio})
        
        self.results.append(result)
        return result
    
    def contract_strange_metal_scaling(self, rho_exponent: float, predicted_z: float, d: int, system_name: str) -> Dict[str, Any]:
        result = {"name": "strange_metal_scaling", "system": system_name, "passed": True}
        alpha_expected = (d - 2) / predicted_z if predicted_z > 0 else 1.0
        result["passed"] = abs(rho_exponent - alpha_expected) < 0.5
        self.results.append(result)
        return result
    
    def contract_cosmology_bounds(self, ns: float, predicted_bulk: str, system_name: str) -> Dict[str, Any]:
        ns_planck, ns_error = 0.9649, 0.0042
        in_3sigma = abs(ns - ns_planck) < 3 * ns_error
        result = {"name": "cosmology_bounds", "system": system_name, "passed": in_3sigma}
        self.results.append(result)
        return result
    
    def summary(self) -> Dict[str, Any]:
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
# CONTRATOS FASE XIII (sin cambios)
# ============================================================

class ContractsFase13:
    def __init__(self):
        self.results = []
    
    def contract_atlas_coverage(self, n_total: int, n_families: int, expected_families: List[str]) -> Dict[str, Any]:
        result = {"name": "atlas_coverage", "passed": True, "checks": []}
        result["checks"].append({"name": "min_families", "passed": n_families >= 3})
        result["checks"].append({"name": "includes_ads", "passed": "ads" in [f.lower() for f in expected_families]})
        result["checks"].append({"name": "min_theories", "passed": n_total >= 10})
        result["passed"] = all(c["passed"] for c in result["checks"])
        self.results.append(result)
        return result
    
    def contract_cluster_quality(self, clusters: Dict[str, List[str]], points: List[Dict]) -> Dict[str, Any]:
        result = {"name": "cluster_quality", "passed": True, "checks": []}
        total = sum(len(v) for v in clusters.values())
        for name, members in clusters.items():
            if len(members) / max(total, 1) > 0.8:
                result["passed"] = False
        non_trivial = sum(1 for v in clusters.values() if len(v) > 1)
        result["passed"] = result["passed"] and non_trivial >= 2
        self.results.append(result)
        return result
    
    def contract_outlier_genuineness(self, outliers: List[str], all_points: List[Dict], threshold: float = 1.5) -> Dict[str, Any]:
        result = {"name": "outlier_genuineness", "passed": True}
        n_total = len(all_points)
        result["passed"] = len(outliers) / max(n_total, 1) <= 0.2
        self.results.append(result)
        return result
    
    def contract_einstein_distribution(self, n_einstein: int, n_non_einstein: int, n_total: int) -> Dict[str, Any]:
        result = {"name": "einstein_distribution", "passed": True}
        result["passed"] = (n_einstein / max(n_total, 1) >= 0.1) and (n_non_einstein / max(n_total, 1) >= 0.1)
        self.results.append(result)
        return result
    
    def contract_exploration_completeness(self, regions_explored: Dict[str, int], min_regions: int = 2) -> Dict[str, Any]:
        result = {"name": "exploration_completeness", "passed": True}
        n_regions = sum(1 for v in regions_explored.values() if v > 0)
        result["passed"] = n_regions >= min_regions
        self.results.append(result)
        return result
    
    def summary(self) -> Dict[str, Any]:
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
# RUNNERS
# ============================================================

def run_contracts_fase12(report_path: Path) -> Dict:
    if report_path.is_dir():
        for c in [
            report_path / "holographic_dictionary" / "holographic_dictionary_v3_summary.json",
            report_path / "holographic_dictionary" / "holographic_dictionary_summary.json",
            report_path / "fase12_report.json",
        ]:
            if c.is_file():
                report_path = c
                break
    
    if not report_path.exists():
        return {"error": "report not found"}
    
    report = json.loads(report_path.read_text())
    contracts = ContractsFase12()
    
    for system in report.get("systems", []):
        name = system.get("name", "")
        source = system.get("source", "")
        geo = system.get("geometry", {})
        predicted_family = geo.get("predicted_family", "unknown")
        
        if source == "bootstrap" and "ising" in name.lower():
            dict_ops = system.get("dictionary", {}).get("operators_predicted", [])
            geom_ops = geo.get("operators_predicted", [])
            operators = dict_ops if dict_ops else geom_ops
            predicted_Deltas = [op.get("Delta", 0.0) for op in operators]
            dictionary_source = system.get("dictionary_source", "unknown")
            
            contracts.contract_ising3d_consistency(
                predicted_family, predicted_Deltas,
                {"sigma": 0.518, "epsilon": 1.41},
                dictionary_source
            )
        
        if source == "lattice":
            eta_s = system.get("physics_metadata", {}).get("eta_over_s_min", 0.1)
            if eta_s > 0:
                contracts.contract_kss_bound(eta_s, name)
    
    return contracts.summary()


def run_contracts_fase13(analysis_path: Path, atlas_path=None) -> dict:
    if not analysis_path.exists():
        return {"error": "analysis not found"}
    
    analysis = json.loads(analysis_path.read_text())
    contracts = ContractsFase13()
    
    contracts.contract_atlas_coverage(
        analysis.get("n_total", 0),
        len(analysis.get("clusters", {})),
        list(analysis.get("clusters", {}).keys())
    )
    
    if atlas_path and atlas_path.is_file():
        atlas = json.loads(atlas_path.read_text())
        contracts.contract_cluster_quality(atlas.get("clusters", {}), atlas.get("points", []))
        contracts.contract_outlier_genuineness(atlas.get("outliers", []), atlas.get("points", []))
    
    contracts.contract_einstein_distribution(
        analysis.get("n_einstein", 0),
        analysis.get("n_non_einstein", 0),
        analysis.get("n_total", 0)
    )
    contracts.contract_exploration_completeness(analysis.get("interesting_regions", {}))
    
    return contracts.summary()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Contratos Fases XII/XIII + Control Negativo (FPR)"
    )
    parser.add_argument("--phase", type=str, required=True, choices=["12", "13", "both"])
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--fase12-report", type=str, default="")
    parser.add_argument("--fase13-analysis", type=str, default="")
    parser.add_argument("--fase13-atlas", type=str, default="")
    parser.add_argument("--output-file", type=str, default=None)
    
    # Control negativo
    parser.add_argument("--negative-control-run-dir", type=str, default=None,
                        help="Directorio del run sobre datos anti-holográficos")
    parser.add_argument("--negative-control-h5", type=str, default=None,
                        help="HDF5 del control negativo")
    parser.add_argument("--require-negative-control", action="store_true",
                        help="Si ALERT → exit 1")
    parser.add_argument("--negative-control-fpr-threshold", type=float, default=0.2,
                        help="Umbral FPR para SUCCESS (default: 0.2)")
    
    args = parser.parse_args()
    
    # Resolver rutas
    fase12_report = args.fase12_report or ""
    fase13_analysis = args.fase13_analysis or ""
    fase13_atlas = args.fase13_atlas or ""
    output_file = args.output_file
    
    if args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir)
        manifest = load_run_manifest(run_dir)
        artifacts = manifest.get("artifacts", {}) if manifest else {}
        
        if not fase12_report:
            for c in [run_dir / "emergent_dictionary" / "lambda_sl_dictionary_report.json"]:
                if c.exists():
                    fase12_report = str(c)
                    break
        
        if not output_file:
            contracts_dir = run_dir / "contracts"
            contracts_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(contracts_dir / "contracts_12_13.json")
    
    if not output_file:
        output_file = "contracts_12_13.json"
    
    print("=" * 70)
    print("CONTRATOS FASES XII/XIII + CONTROL NEGATIVO (FPR)")
    print("=" * 70)
    
    results = {}
    negative_control_alert = False
    
    # Fase XII
    if args.phase in ["12", "both"] and fase12_report:
        print(f"\n>> Validando Fase XII desde {fase12_report}")
        results["fase12"] = run_contracts_fase12(Path(fase12_report))
        summary = results["fase12"]
        print(f"   Contratos: {summary.get('n_passed', 0)}/{summary.get('n_contracts', 0)}")
    
    # Fase XIII
    if args.phase in ["13", "both"] and fase13_analysis:
        print(f"\n>> Validando Fase XIII desde {fase13_analysis}")
        results["fase13"] = run_contracts_fase13(
            Path(fase13_analysis),
            Path(fase13_atlas) if fase13_atlas else None
        )
        summary = results["fase13"]
        print(f"   Contratos: {summary.get('n_passed', 0)}/{summary.get('n_contracts', 0)}")
    
    # Control negativo
    if args.negative_control_run_dir:
        print(f"\n>> Control negativo desde {args.negative_control_run_dir}")
        
        results["negative_control"] = run_negative_control_check(
            run_dir=Path(args.negative_control_run_dir),
            h5_path=Path(args.negative_control_h5) if args.negative_control_h5 else None,
            fpr_threshold=args.negative_control_fpr_threshold
        )
        
        nc = results["negative_control"]
        print(f"\n   Status: {nc['status']}")
        if nc["false_positive_rate"] is not None:
            print(f"   FPR: {nc['false_positive_rate']:.1%} (threshold: {nc['fpr_threshold']:.0%})")
            print(f"   Coverage: {nc['coverage']:.1%}")
            print(f"   Señales: {nc['n_signals_triggered']}/{nc['n_signals_evaluable']} disparadas")
        
        if nc["status"] == "ALERT":
            negative_control_alert = True
    
    # Guardar
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    
    all_passed = True
    for phase, summary in results.items():
        if isinstance(summary, dict):
            if phase == "negative_control":
                status = summary.get("status", "INCOMPLETE")
                fpr = summary.get("false_positive_rate")
                if fpr is not None:
                    print(f"  {phase}: {status} (FPR={fpr:.1%})")
                else:
                    print(f"  {phase}: {status}")
                if status == "ALERT":
                    all_passed = False
            elif "all_passed" in summary:
                status = "OK" if summary["all_passed"] else "FAIL"
                print(f"  {phase}: {status} ({summary['n_passed']}/{summary['n_contracts']})")
                all_passed = all_passed and summary["all_passed"]
    
    print(f"\n  Output: {output_path}")
    print("=" * 70)
    
    if args.require_negative_control and negative_control_alert:
        print("\n⚠ Exit 1: --require-negative-control activo y status=ALERT")
        return 1
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
