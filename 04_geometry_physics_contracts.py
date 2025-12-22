#!/usr/bin/env python3
# 04_geometry_physics_contracts.py
# CUERDAS — Bloque A: Geometría emergente (contratos físicos)
#
# OBJETIVO
#   Evaluar la calidad física y la honestidad del bloque de geometría y ecuaciones:
#     - Geometría emergente vs bulk_truth (cuando exista sandbox).
#     - Ecuaciones de bulk descubiertas vs criterios físicos.
#
# ENTRADAS
#   - runs/emergent_geometry/emergent_geometry_summary.json
#   - runs/bulk_equations/equations_pareto.json (y/o pysr_summary.json)
#   - Opcional: runs/sandbox_geometries/bulk_truth/*.h5 (para contratos sandbox)
#
# SALIDAS
#   runs/geometry_contracts/
#     geometry_contracts_summary.json
#       - Clasificación por sistema:
#           * Einstein-like / non-Einstein / incierto
#           * Umbrales de R², estabilidad en rollouts, etc.
#       - Flags de honestidad (uso indebido de bulk, mezcla incorrecta de d, ...).
#
# TIPOS DE CONTRATO (EJEMPLOS)
#   - R² mínimo en test, estabilidad en evoluciones numéricas.
#   - No mezcla de dimensiones d entre sistemas incompatibles.
#   - No uso de variables de "verdad" en la construcción de la loss.
#   - Coherencia entre familia asignada (ads/lifshitz/hvlf/deformed) y patrones geométricos.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Consume salidas de:
#       * 02_emergent_geometry_engine.py
#       * 03_discover_bulk_equations.py
#   - Sus resultados condicionan el análisis de:
#       * 05_analyze_bulk_equations.py
#
# HISTÓRICO
#   - Anteriormente conocido como: 04_contracts_fase_11_v2.py
#   - V2.1: Añadido soporte para modo inference (sin bulk_truth)
#   - V2.2: Hardening IO - robustez ante rutas None y NPZ malformados
#
# MODOS SOPORTADOS (V2.1):
#   - Modo A (sandbox/train): bulk_truth disponible, A_truth/f_truth en NPZ
#   - Modo B (inference/real): sin bulk_truth, solo predicciones
#     En este modo, métricas R² se marcan como null y contratos genéricos se evalúan

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import h5py

try:
    from run_context import RunContext, add_experiment_args
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_context import RunContext, add_experiment_args

SCRIPT_NAME = "04_geometry_physics_contracts.py"


def write_contracts_summary(
    passed_ids: List[str],
    failed_ids: List[str],
    out_json: Path,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Escribe un resumen mínimo (y estable) de contratos para consumo por herramientas/CI.

    Esquema mínimo garantizado:
      { "passed": [...], "failed": [...] }

    Incluye además metadatos útiles (n_passed/n_failed/n_total/pass_rate) y cualquier campo extra.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)

    passed_ids = list(passed_ids)
    failed_ids = list(failed_ids)
    n_passed = len(passed_ids)
    n_failed = len(failed_ids)
    n_total = n_passed + n_failed
    pass_rate = None if n_total == 0 else (n_passed / n_total)

    payload: Dict[str, Any] = {
        "passed": passed_ids,
        "failed": failed_ids,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_total": n_total,
        "pass_rate": pass_rate,
    }

    if extra:
        payload.update(extra)

    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# Import local IO module for run manifest support
try:
    from cuerdas_io import (
        resolve_predictions_dir as cuerdas_resolve_predictions,
        resolve_geometry_emergent_dir as cuerdas_resolve_geometry_emergent,
        resolve_bulk_equations_dir as cuerdas_resolve_bulk_equations,
        resolve_data_dir as cuerdas_resolve_data,
        resolve_dictionary_file as cuerdas_resolve_dictionary,
        update_run_manifest,
    )
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


# ============================================================
# HELPERS PARA RESOLUCIÓN DE RUTAS (V2.1)
# ============================================================

def resolve_predictions_dir(geometry_dir: Path) -> Path:
    """
    Resuelve el directorio de predicciones NPZ.
    
    Busca en orden:
    1. geometry_dir/predictions/
    2. geometry_dir/ (si ya es el directorio de predictions)
    """
    predictions_subdir = geometry_dir / "predictions"
    if predictions_subdir.exists() and predictions_subdir.is_dir():
        return predictions_subdir
    
    # Comprobar si geometry_dir ya contiene NPZ directamente
    npz_files = list(geometry_dir.glob("*_geometry.npz"))
    if npz_files:
        return geometry_dir
    
    # Fallback: devolver el subdirectorio esperado (puede no existir)
    return predictions_subdir


def resolve_emergent_h5_dir(geometry_dir: Path) -> Path:
    """
    Resuelve el directorio de H5 emergentes.
    
    Busca en orden:
    1. geometry_dir/geometry_emergent/
    2. geometry_dir/ (si ya contiene H5 emergentes)
    """
    emergent_subdir = geometry_dir / "geometry_emergent"
    if emergent_subdir.exists() and emergent_subdir.is_dir():
        return emergent_subdir
    
    # Comprobar si geometry_dir ya contiene H5 directamente
    h5_files = list(geometry_dir.glob("*_emergent.h5"))
    if h5_files:
        return geometry_dir
    
    return emergent_subdir


def resolve_bulk_equations_dir(einstein_dir: Path) -> Path:
    """
    Resuelve el directorio de ecuaciones de bulk.
    
    Busca en orden:
    1. einstein_dir/bulk_equations/
    2. einstein_dir/ (si ya es el directorio correcto)
    """
    bulk_eq_subdir = einstein_dir / "bulk_equations"
    if bulk_eq_subdir.exists() and bulk_eq_subdir.is_dir():
        return bulk_eq_subdir
    
    # Comprobar si einstein_dir ya contiene subdirectorios de sistemas
    # (cada sistema tiene su carpeta con einstein_discovery.json)
    json_files = list(einstein_dir.glob("*/einstein_discovery.json"))
    if json_files:
        return einstein_dir
    
    return bulk_eq_subdir


# ============================================================
# HELPERS PARA ACCESO SEGURO A NPZ Y H5 (V2.2)
# ============================================================

def safe_npz_get(npz_file, key: str, fallback_key: Optional[str] = None):
    """
    Acceso seguro a NpzFile. NpzFile no tiene .get(), hay que usar 'in'.
    
    Args:
        npz_file: NpzFile abierto
        key: clave principal
        fallback_key: clave alternativa si la principal no existe
        
    Returns:
        np.ndarray o None
    """
    if key in npz_file.files:
        return npz_file[key]
    if fallback_key is not None and fallback_key in npz_file.files:
        return npz_file[fallback_key]
    return None


def safe_h5_dataset(h5_group, key: str, fallback_key: Optional[str] = None):
    """
    Acceso seguro a dataset H5. Devuelve array o None.
    
    Args:
        h5_group: grupo H5 (file o subgrupo)
        key: clave principal
        fallback_key: clave alternativa
        
    Returns:
        np.ndarray o None
    """
    if key in h5_group:
        return h5_group[key][:]
    if fallback_key is not None and fallback_key in h5_group:
        return h5_group[fallback_key][:]
    return None


# ============================================================
# CONTRATOS GENÉRICOS (aplican a TODAS las geometrías)
# ============================================================

@dataclass
class GenericRegularityContract:
    """Verifica propiedades básicas de regularidad."""
    A_finite: bool
    f_finite: bool
    no_nan: bool
    smooth: bool  # derivadas no explotan
    
    @property
    def passed(self) -> bool:
        return self.A_finite and self.f_finite and self.no_nan


@dataclass
class GenericCausalityContract:
    """Verifica estructura causal básica (no específica de AdS)."""
    f_non_negative: bool  # f >= 0 (o muy cerca)
    f_bounded: bool  # f no explota
    horizon_if_thermal: bool  # si T > 0, debe haber estructura de horizonte
    skipped: bool = False  # True si no hay datos para evaluar (T/z_h ausentes)
    
    @property
    def passed(self) -> bool:
        if self.skipped:
            return True  # No penalizar si no hay datos
        return self.f_non_negative and self.f_bounded


# ============================================================
# CONTRATOS AdS-ESPECÍFICOS (solo para family="ads")
# ============================================================

@dataclass
class AdSEinsteinContract:
    """Verifica ecuaciones de Einstein en vacío con Λ (SOLO para AdS)."""
    R_is_constant: bool
    R_matches_ads: bool
    Lambda_matches_ads: bool
    R_mean: float
    R_expected: float
    
    @property
    def passed(self) -> bool:
        return self.R_is_constant and self.R_matches_ads


@dataclass
class AdSAsymptoticContract:
    """Verifica comportamiento asintótico tipo AdS."""
    A_logarithmic_uv: bool  # A(z) ~ -log(z) cerca de z=0
    f_to_one_uv: bool  # f → 1 cuando z → 0
    A_monotone: bool  # dA/dz < 0
    
    @property
    def passed(self) -> bool:
        return self.A_logarithmic_uv and self.A_monotone


@dataclass
class HolographicDictionaryContract:
    """Verifica diccionario holográfico (más relevante para AdS)."""
    mass_dimension_ok: bool
    hawking_ok: bool
    conformal_symmetry_ok: bool
    
    @property
    def passed(self) -> bool:
        return self.mass_dimension_ok or self.hawking_ok or self.conformal_symmetry_ok


# ============================================================
# CONTRATO GLOBAL
# ============================================================

@dataclass
class PhaseXIContractV2:
    """Contrato completo de Fase XI v2."""
    name: str
    family: str
    category: str
    d: int
    
    # Contratos genéricos (TODOS deben pasar)
    regularity: GenericRegularityContract
    causality: GenericCausalityContract
    
    # Contratos AdS-específicos (solo relevantes si family="ads")
    ads_einstein: AdSEinsteinContract
    ads_asymptotic: AdSAsymptoticContract
    holographic: HolographicDictionaryContract
    
    # Métricas de reconstrucción (pueden ser None en inference)
    A_r2: Optional[float]
    f_r2: Optional[float]
    R_r2: Optional[float]
    family_accuracy: Optional[float]
    
    # Metadatos de modo (V2.1)
    mode: str = "sandbox"  # "sandbox" o "inference"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def generic_passed(self) -> bool:
        """Contratos que TODA geometría debe pasar."""
        return self.regularity.passed and self.causality.passed
    
    @property
    def ads_specific_passed(self) -> bool:
        """Contratos específicos de AdS (solo relevantes si es AdS)."""
        return self.ads_einstein.passed and self.ads_asymptotic.passed
    
    @property
    def is_ads_family(self) -> bool:
        return self.family == "ads"
    
    @property
    def contract_score(self) -> float:
        """Score ponderado por tipo de geometría."""
        def to_float(val) -> float:
            """Convierte bool o string a float."""
            if isinstance(val, bool):
                return 1.0 if val else 0.0
            if isinstance(val, str):
                return 1.0 if val.lower() == 'true' else 0.0
            return float(val)
        
        # Genéricos: siempre cuentan
        score = 0.3 * to_float(self.regularity.passed) + 0.2 * to_float(self.causality.passed)
        
        # Reconstrucción (solo si hay métricas disponibles)
        if self.A_r2 is not None and self.f_r2 is not None:
            a_r2_safe = max(0.0, min(1.0, self.A_r2))
            f_r2_safe = max(0.0, min(1.0, self.f_r2))
            score += 0.2 * (a_r2_safe + f_r2_safe) / 2
        else:
            # En inference sin truth, dar puntos parciales si genéricos pasan
            score += 0.1 * to_float(self.generic_passed)
        
        # AdS-específicos: solo cuentan si es AdS
        if self.is_ads_family:
            score += 0.2 * to_float(self.ads_einstein.passed)
            score += 0.1 * to_float(self.holographic.passed)
        else:
            # Para no-AdS, damos puntos si pasó genéricos
            score += 0.2 * to_float(self.generic_passed)
            score += 0.1 * to_float(self.holographic.conformal_symmetry_ok)
        
        return score
    
    @property
    def overall_passed(self) -> bool:
        """Pasó la fase."""
        if not self.generic_passed:
            return False
        if self.is_ads_family:
            return self.ads_specific_passed
        return True  # Non-AdS solo necesita genéricos


# ============================================================
# FUNCIONES DE VERIFICACIÓN
# ============================================================

def verify_regularity(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray
) -> GenericRegularityContract:
    """Verifica regularidad básica."""
    A_finite = np.all(np.isfinite(A)) and np.max(np.abs(A)) < 1e6
    f_finite = np.all(np.isfinite(f)) and np.max(np.abs(f)) < 1e6
    no_nan = not (np.any(np.isnan(A)) or np.any(np.isnan(f)))
    
    # Suavidad: derivadas no explotan
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    dA = np.gradient(A, dz)
    smooth = np.all(np.abs(dA) < 1e4)
    
    return GenericRegularityContract(
        A_finite=A_finite,
        f_finite=f_finite,
        no_nan=no_nan,
        smooth=smooth
    )


def verify_generic_causality(
    f: np.ndarray,
    z: np.ndarray,
    T: Optional[float],
    z_h: Optional[float]
) -> GenericCausalityContract:
    """Verifica causalidad básica."""
    f_non_negative = np.min(f) >= -0.1
    f_bounded = np.max(f) < 10.0
    
    # Si no hay datos de T/z_h, marcar como skipped
    if T is None or z_h is None:
        return GenericCausalityContract(
            f_non_negative=f_non_negative,
            f_bounded=f_bounded,
            horizon_if_thermal=True,
            skipped=True
        )
    
    # Si hay temperatura, debe haber estructura de horizonte
    if T > 1e-10 and z_h > 0:
        idx_h = np.argmin(np.abs(z - z_h))
        f_near_h = f[max(0, idx_h-3):min(len(f), idx_h+3)]
        horizon_ok = np.min(f_near_h) < 0.5 if len(f_near_h) > 0 else True
    else:
        horizon_ok = True
    
    return GenericCausalityContract(
        f_non_negative=f_non_negative,
        f_bounded=f_bounded,
        horizon_if_thermal=horizon_ok,
        skipped=False
    )


def verify_ads_einstein(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray,
    d: int,
    einstein_results: Dict[str, Any]
) -> AdSEinsteinContract:
    """Verifica ecuaciones de Einstein (solo para AdS)."""
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        # Fallback sin scipy
        def gaussian_filter1d(x, sigma, mode='nearest'):
            return x
    
    D = d + 1
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    n = len(z)
    
    if n < 10:
        # Datos insuficientes
        return AdSEinsteinContract(
            R_is_constant=False,
            R_matches_ads=False,
            Lambda_matches_ads=False,
            R_mean=float('nan'),
            R_expected=float(-D * (D - 1))
        )
    
    # Suavizar A y f antes de derivar (reduce ruido numérico)
    sigma = 2.0
    A_smooth = gaussian_filter1d(A, sigma=sigma, mode='nearest')
    f_smooth = gaussian_filter1d(f, sigma=sigma, mode='nearest')
    f_smooth = np.clip(f_smooth, 1e-6, None)
    
    # Derivadas con esquema de 5 puntos
    def deriv_5pt(y, dx):
        d = np.zeros_like(y)
        for i in range(2, len(y) - 2):
            d[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12 * dx)
        d[0] = (y[1] - y[0]) / dx
        d[1] = (y[2] - y[0]) / (2 * dx) if len(y) > 2 else d[0]
        d[-1] = (y[-1] - y[-2]) / dx
        d[-2] = (y[-1] - y[-3]) / (2 * dx) if len(y) > 2 else d[-1]
        return d
    
    dA = deriv_5pt(A_smooth, dz)
    d2A = deriv_5pt(dA, dz)
    df = deriv_5pt(f_smooth, dz)
    
    # Calcular R
    R = -2 * D * d2A - D * (D - 1) * dA**2 - df * dA / f_smooth
    
    # Evaluar solo en región interior
    margin = max(5, n // 10)
    R_interior = R[margin:-margin] if margin < n // 2 else R
    
    R_mean = np.mean(R_interior)
    R_std = np.std(R_interior)
    R_expected = -D * (D - 1)
    
    cv = R_std / (np.abs(R_mean) + 1e-10)
    R_is_constant = cv < 0.3
    R_matches_ads = np.abs(R_mean - R_expected) / (np.abs(R_expected) + 1e-10) < 0.5
    
    Lambda_check = einstein_results.get("einstein_check", {})
    Lambda_matches = Lambda_check.get("consistent_with_einstein_vacuum", False)
    
    return AdSEinsteinContract(
        R_is_constant=R_is_constant,
        R_matches_ads=R_matches_ads,
        Lambda_matches_ads=Lambda_matches,
        R_mean=float(R_mean),
        R_expected=float(R_expected)
    )


def verify_ads_asymptotic(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray
) -> AdSAsymptoticContract:
    """Verifica comportamiento asintótico tipo AdS."""
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        def gaussian_filter1d(x, sigma, mode='nearest'):
            return x
    
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    n = len(z)
    
    if n < 5:
        return AdSAsymptoticContract(
            A_logarithmic_uv=False,
            f_to_one_uv=False,
            A_monotone=False
        )
    
    n_uv = min(10, n // 5)
    n_uv = max(3, n_uv)
    
    A_uv = A[:n_uv]
    z_uv = z[:n_uv]
    
    # Evitar log(0)
    z_uv_safe = np.clip(z_uv, 1e-10, None)
    expected_A = -np.log(z_uv_safe)
    
    # Correlación
    if np.std(A_uv) > 1e-10 and np.std(expected_A) > 1e-10:
        corr = np.corrcoef(A_uv, expected_A)[0, 1]
        A_log_uv = corr > 0.95 if not np.isnan(corr) else False
    else:
        A_log_uv = False
    
    f_uv = f[:n_uv]
    f_to_one = np.mean(np.abs(f_uv - 1)) < 0.15
    
    A_smooth = gaussian_filter1d(A, sigma=2, mode='nearest')
    dA = np.gradient(A_smooth, dz)
    
    margin = max(3, n // 20)
    dA_interior = dA[margin:-margin] if margin < n // 2 else dA
    A_monotone = np.mean(dA_interior < 0.01) > 0.8
    
    return AdSAsymptoticContract(
        A_logarithmic_uv=A_log_uv,
        f_to_one_uv=f_to_one,
        A_monotone=A_monotone
    )


def verify_holographic(
    dict_results: Dict[str, Any]
) -> HolographicDictionaryContract:
    """Verifica diccionario holográfico."""
    def to_bool(val) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == 'true'
        return bool(val) if val is not None else False
    
    mass_dim = dict_results.get("mass_dimension", {})
    mass_ok = to_bool(mass_dim.get("comparison_with_holographic", {}).get("likely_holographic", False))
    
    hawking = dict_results.get("hawking", {})
    hawking_ok = to_bool(hawking.get("hawking_check", {}).get("hawking_verified", False))
    
    conformal_ok = False
    for geo in dict_results.get("geometries", []):
        conf = geo.get("conformal", {}).get("summary", {})
        if to_bool(conf.get("conformal_symmetry_present", False)):
            conformal_ok = True
            break
    
    return HolographicDictionaryContract(
        mass_dimension_ok=mass_ok,
        hawking_ok=hawking_ok,
        conformal_symmetry_ok=conformal_ok
    )


# ============================================================
# CARGA DE DATOS CON FALLBACKS (V2.2 - HARDENED)
# ============================================================

def load_geometry_data(
    name: str,
    data_dir: Optional[Path],
    predictions_dir: Path,
    emergent_h5_dir: Optional[Path]
) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Carga datos de geometría con fallbacks para modo inference.
    
    V2.2: Robusto ante data_dir=None y emergent_h5_dir=None.
    
    Returns:
        (data_dict, mode, warnings)
        - data_dict: diccionario con z_grid, A_pred, f_pred, R_pred, y opcionalmente *_truth
        - mode: "sandbox" o "inference"
        - warnings: lista de advertencias
    """
    data = {}
    warnings = []
    mode = "sandbox"
    
    # === 1. Intentar cargar NPZ de predictions ===
    npz_path = predictions_dir / f"{name}_geometry.npz"
    
    # V2.2: Solo construir h5_emergent_path si emergent_h5_dir no es None
    h5_emergent_path = None
    if emergent_h5_dir is not None:
        h5_emergent_path = emergent_h5_dir / f"{name}_emergent.h5"
    
    npz_loaded = False
    if npz_path.exists():
        try:
            geo_data = np.load(npz_path, allow_pickle=True)
            
            # V2.2: Usar safe_npz_get en lugar de .get() que no existe en NpzFile
            data["A_pred"] = safe_npz_get(geo_data, "A_pred", "A_of_z")
            data["f_pred"] = safe_npz_get(geo_data, "f_pred", "f_of_z")
            data["R_pred"] = safe_npz_get(geo_data, "R_pred", "R_of_z")
            data["z_grid"] = safe_npz_get(geo_data, "z", "z_grid")
            
            # Cargar truth si existe
            if "A_truth" in geo_data.files:
                data["A_truth"] = geo_data["A_truth"]
                data["f_truth"] = geo_data["f_truth"]
                data["R_truth"] = safe_npz_get(geo_data, "R_truth")
                data["family_pred"] = safe_npz_get(geo_data, "family_pred")
                data["family_truth"] = safe_npz_get(geo_data, "family_truth")
            else:
                mode = "inference"
                warnings.append(f"NPZ sin *_truth: modo inference")
            
            npz_loaded = True
            geo_data.close()  # Cerrar el archivo NPZ
        except Exception as e:
            warnings.append(f"Error cargando NPZ: {e}")
    
    # === 2. Fallback a H5 emergente si NPZ no tiene lo necesario ===
    if not npz_loaded or data.get("A_pred") is None:
        # V2.2: Solo intentar si h5_emergent_path existe
        if h5_emergent_path is not None and h5_emergent_path.exists():
            try:
                with h5py.File(h5_emergent_path, "r") as f:
                    data["z_grid"] = safe_h5_dataset(f, "z_grid")
                    data["A_pred"] = safe_h5_dataset(f, "A_of_z", "A_pred")
                    data["f_pred"] = safe_h5_dataset(f, "f_of_z", "f_pred")
                    
                    # R_pred con fallback a zeros si no existe
                    R_val = safe_h5_dataset(f, "R_of_z", "R_pred")
                    if R_val is not None:
                        data["R_pred"] = R_val
                    elif data.get("A_pred") is not None:
                        data["R_pred"] = np.zeros_like(data["A_pred"])
                    
                    # Metadatos del H5
                    data["family_from_h5"] = str(f.attrs.get("family", f.attrs.get("family_pred", "unknown")))
                    data["d_from_h5"] = int(f.attrs.get("d", f.attrs.get("d_pred", 4)))
                
                mode = "inference"
                warnings.append(f"Cargado desde H5 emergente (fallback)")
            except Exception as e:
                warnings.append(f"Error cargando H5 emergente: {e}")
        elif h5_emergent_path is not None:
            warnings.append(f"No existe H5 emergente: {h5_emergent_path}")
        else:
            warnings.append("emergent_h5_dir no especificado, omitiendo fallback H5")
    
    # === 3. Cargar metadatos del boundary H5 original ===
    # V2.2: Solo intentar si data_dir no es None
    if data_dir is not None:
        boundary_h5_path = data_dir / f"{name}.h5"
        if boundary_h5_path.exists():
            try:
                with h5py.File(boundary_h5_path, "r") as f:
                    data["family"] = str(f.attrs.get("family", "unknown"))
                    data["category"] = str(f.attrs.get("category", "unknown"))
                    
                    # Boundary data
                    if "boundary" in f:
                        boundary = f["boundary"]
                        # temperature puede venir como attr o como dataset boundary/temperature
                        if "temperature" in boundary.attrs:
                            data["T"] = float(boundary.attrs.get("temperature"))
                        elif "temperature" in boundary and hasattr(boundary["temperature"], "shape"):
                            v = boundary["temperature"][()]
                            data["T"] = float((v.reshape(-1)[0]) if hasattr(v, "reshape") else v)
                        else:
                            data["T"] = None
                    else:
                        data["T"] = None
                    
                    # Bulk truth (solo en sandbox)
                    if "bulk_truth" in f:
                        bulk = f["bulk_truth"]
                        data["z_h"] = float(bulk.attrs.get("z_h", 0))
                        if data.get("z_grid") is None:
                            data["z_grid"] = safe_h5_dataset(bulk, "z_grid")
                    else:
                        data["z_h"] = None
                        if mode != "inference":
                            mode = "inference"
                            warnings.append("Sin bulk_truth en H5: modo inference")
            except Exception as e:
                warnings.append(f"Error cargando boundary H5: {e}")
        else:
            # Sin H5 original, usar metadatos del emergente
            data["family"] = data.get("family_from_h5", "unknown")
            data["category"] = "inference"
            data["T"] = None
            data["z_h"] = None
            mode = "inference"
            warnings.append(f"No existe boundary H5: {boundary_h5_path}")
    else:
        # V2.2: data_dir es None - operar en modo inference sin metadatos de boundary
        data["family"] = data.get("family_from_h5", "unknown")
        data["category"] = "inference"
        data["T"] = None
        data["z_h"] = None
        mode = "inference"
        warnings.append("data_dir no especificado: modo inference sin metadatos de boundary")
    
    return data, mode, warnings


# ============================================================
# PROCESAMIENTO (V2.2 - HARDENED)
# ============================================================

def process_geometry(
    name: str,
    data_dir: Optional[Path],
    predictions_dir: Path,
    emergent_h5_dir: Optional[Path],
    einstein_dir: Optional[Path],
    dictionary_results: Dict[str, Any],
    d: int
) -> PhaseXIContractV2:
    """
    Procesa una geometría y genera su contrato v2.
    
    V2.2: Robusto ante data_dir=None, emergent_h5_dir=None, einstein_dir=None.
    
    Soporta dos modos:
    - sandbox: con bulk_truth y *_truth (métricas R² calculadas)
    - inference: sin truth (métricas R² = None, contratos genéricos evaluados)
    """
    errors = []
    warnings = []
    
    # === CARGAR DATOS ===
    try:
        data, mode, load_warnings = load_geometry_data(
            name, data_dir, predictions_dir, emergent_h5_dir
        )
        warnings.extend(load_warnings)
    except Exception as e:
        errors.append(f"Error fatal cargando datos: {e}")
        # Devolver contrato vacío pero presente
        return PhaseXIContractV2(
            name=name,
            family="unknown",
            category="error",
            d=d,
            regularity=GenericRegularityContract(False, False, False, False),
            causality=GenericCausalityContract(False, False, False, skipped=True),
            ads_einstein=AdSEinsteinContract(False, False, False, float('nan'), float('nan')),
            ads_asymptotic=AdSAsymptoticContract(False, False, False),
            holographic=HolographicDictionaryContract(False, False, False),
            A_r2=None,
            f_r2=None,
            R_r2=None,
            family_accuracy=None,
            mode="error",
            errors=errors,
            warnings=warnings
        )
    
    # Extraer arrays
    z_grid = data.get("z_grid")
    A_pred = data.get("A_pred")
    f_pred = data.get("f_pred")
    R_pred = data.get("R_pred")
    
    family = data.get("family", "unknown")
    category = data.get("category", "unknown")
    T = data.get("T")
    z_h = data.get("z_h")
    
    # Validar datos mínimos
    if z_grid is None or A_pred is None or f_pred is None:
        errors.append("Datos insuficientes: falta z_grid, A_pred o f_pred")
        return PhaseXIContractV2(
            name=name,
            family=family,
            category=category,
            d=d,
            regularity=GenericRegularityContract(False, False, False, False),
            causality=GenericCausalityContract(False, False, False, skipped=True),
            ads_einstein=AdSEinsteinContract(False, False, False, float('nan'), float('nan')),
            ads_asymptotic=AdSAsymptoticContract(False, False, False),
            holographic=HolographicDictionaryContract(False, False, False),
            A_r2=None,
            f_r2=None,
            R_r2=None,
            family_accuracy=None,
            mode=mode,
            errors=errors,
            warnings=warnings
        )
    
    # Asegurar arrays numpy
    z_grid = np.asarray(z_grid)
    A_pred = np.asarray(A_pred)
    f_pred = np.asarray(f_pred)
    if R_pred is not None:
        R_pred = np.asarray(R_pred)
    else:
        R_pred = np.zeros_like(A_pred)
    
    # === MÉTRICAS R² (solo en sandbox) ===
    def r2(y_true, y_pred):
        if y_true is None or y_pred is None:
            return None
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        r2_val = 1 - ss_res / ss_tot
        return float(np.clip(r2_val, -1.0, 1.0))
    
    if mode == "sandbox" and "A_truth" in data:
        A_r2 = r2(data["A_truth"], A_pred)
        f_r2 = r2(data["f_truth"], f_pred)
        R_r2 = r2(data.get("R_truth"), R_pred) if data.get("R_truth") is not None else None
        
        # Family accuracy
        fp = data.get("family_pred")
        ft = data.get("family_truth")
        if fp is not None and ft is not None:
            family_match = float(int(fp) == int(ft))
        else:
            family_match = None
    else:
        A_r2 = None
        f_r2 = None
        R_r2 = None
        family_match = None
        if mode == "inference":
            warnings.append("Modo inference: métricas R² no disponibles")
    
    # === CARGAR RESULTADOS EINSTEIN ===
    # V2.2: Solo intentar si einstein_dir no es None
    einstein_results = {}
    if einstein_dir is not None:
        einstein_path = einstein_dir / name / "einstein_discovery.json"
        if einstein_path.exists():
            try:
                einstein_results = json.loads(einstein_path.read_text()).get("results", {})
            except Exception as e:
                warnings.append(f"Error cargando einstein_discovery.json: {e}")
        # No es un error si no existe, simplemente no hay datos de Einstein
    else:
        warnings.append("einstein_dir no especificado: omitiendo verificación Einstein")
    
    # === VERIFICAR CONTRATOS ===
    regularity = verify_regularity(A_pred, f_pred, z_grid)
    causality = verify_generic_causality(f_pred, z_grid, T, z_h)
    ads_einstein = verify_ads_einstein(A_pred, f_pred, z_grid, d, einstein_results)
    ads_asymptotic = verify_ads_asymptotic(A_pred, f_pred, z_grid)
    holographic = verify_holographic(dictionary_results)
    
    return PhaseXIContractV2(
        name=name,
        family=family,
        category=category,
        d=d,
        regularity=regularity,
        causality=causality,
        ads_einstein=ads_einstein,
        ads_asymptotic=ads_asymptotic,
        holographic=holographic,
        A_r2=A_r2,
        f_r2=f_r2,
        R_r2=R_r2,
        family_accuracy=family_match,
        mode=mode,
        errors=errors,
        warnings=warnings
    )


# ============================================================
# MAIN (V2.2 - HARDENED)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fase XI v2.2: Contratos de validación física HONESTOS (con soporte inference y hardening IO)"
    )
    add_experiment_args(parser)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directorio con H5 de boundary (manifest.json). Opcional en modo inference.")
    parser.add_argument("--geometry-dir", type=str, default=None,
                        help="Directorio raíz de geometría (predictions/, geometry_emergent/)")
    parser.add_argument("--einstein-dir", type=str, default=None,
                        help="Directorio de ecuaciones bulk (o raíz que contenga bulk_equations/). Opcional.")
    parser.add_argument("--dictionary-file", type=str, default=None,
                        help="JSON de diccionario holográfico. Opcional.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directorio raíz con run_manifest.json (IO v2). "
                             "Si se proporciona, resuelve automáticamente las rutas.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Archivo de salida JSON (default: run-dir/geometry_contracts/...)")
    parser.add_argument("--output-json", "--contracts-summary", type=str, default=None,
                        help="Resumen mínimo machine-readable de contratos (JSON con {passed, failed, pass_rate})")
    parser.add_argument("--d", type=int, default=4,
                        help="Dimensión d del boundary")
    args = parser.parse_args()
    
    ctx = RunContext.from_args(args, script_name=SCRIPT_NAME)
    output_dir = ctx.stage_dir()
    
    # === RESOLVER RUTAS (IO v2 con fallback legacy) ===
    predictions_dir = None
    emergent_h5_dir = None
    einstein_dir = None
    data_dir = None
    dictionary_path = None
    
    # Prioridad 1: --run-dir con cuerdas_io
    if args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir)
        predictions_dir = cuerdas_resolve_predictions(run_dir=run_dir)
        emergent_h5_dir = cuerdas_resolve_geometry_emergent(run_dir=run_dir)
        einstein_dir = cuerdas_resolve_bulk_equations(run_dir=run_dir)
        data_dir = cuerdas_resolve_data(run_dir=run_dir, data_dir=args.data_dir)
        dictionary_path = cuerdas_resolve_dictionary(run_dir=run_dir, dictionary_file=args.dictionary_file)
    
    # Prioridad 2: argumentos explícitos (legacy)
    if predictions_dir is None and args.geometry_dir:
        geometry_dir = Path(args.geometry_dir)
        predictions_dir = resolve_predictions_dir(geometry_dir)
        emergent_h5_dir = resolve_emergent_h5_dir(geometry_dir)
    
    # V2.2: einstein_dir es opcional, solo resolver si se proporciona
    if einstein_dir is None and args.einstein_dir:
        einstein_dir = resolve_bulk_equations_dir(Path(args.einstein_dir))
    
    # V2.2: data_dir es opcional, solo resolver si se proporciona
    if data_dir is None and args.data_dir:
        data_dir = Path(args.data_dir)
    
    if dictionary_path is None and args.dictionary_file:
        dictionary_path = Path(args.dictionary_file)
    
    # Validar que tenemos las rutas mínimas necesarias
    if predictions_dir is None:
        parser.error("Debe proporcionar --run-dir con manifest o --geometry-dir")
    
    # Resolver output_file
    if args.output_file:
        output_file = Path(args.output_file)
    elif args.run_dir:
        output_dir = Path(args.run_dir) / "geometry_contracts"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "geometry_contracts_summary.json"
    else:
        output_file = Path("contracts_04.json")
    
    print("=" * 90)
    print("FASE XI v2.2 — CONTRATOS DE VALIDACIÓN FÍSICA HONESTOS (HARDENED)")
    print("=" * 90)
    print(f"\n  run-dir:        {args.run_dir or '(no especificado)'}")
    print(f"  data-dir:       {data_dir or '(no especificado - modo inference)'}")
    print(f"  predictions:    {predictions_dir}")
    print(f"  emergent_h5:    {emergent_h5_dir or '(no especificado)'}")
    print(f"  einstein-dir:   {einstein_dir or '(no especificado - omitiendo Einstein)'}")
    print(f"  dictionary:     {dictionary_path or '(no especificado)'}")
    print(f"  output:         {output_file}")
    print("\nFILOSOFÍA:")
    print("   Contratos GENÉRICOS: todas las geometrías deben pasar")
    print("   Contratos AdS-ESPECÍFICOS: solo relevantes para family='ads'")
    print("   Modo INFERENCE: sin truth, R²=null, contratos genéricos evaluados")
    print("=" * 90)
    
    # === CARGAR MANIFEST ===
    geometries = []
    
    # V2.2: data_dir puede ser None, en cuyo caso auto-descubrimos
    if data_dir is None:
        # Sin data_dir, auto-descubrir desde predictions
        print(f"\n[INFO] Sin data-dir, auto-descubriendo sistemas desde predictions/")
        npz_files = list(predictions_dir.glob("*_geometry.npz"))
        if npz_files:
            print(f"[INFO] Encontrados {len(npz_files)} sistemas")
            geometries = [{"name": f.stem.replace("_geometry", "")} for f in npz_files]
        else:
            print("[WARN] No hay archivos NPZ en predictions/, intentando H5...")
            # Fallback: buscar H5 emergentes
            if emergent_h5_dir is not None:
                h5_files = list(emergent_h5_dir.glob("*_emergent.h5"))
                if h5_files:
                    print(f"[INFO] Encontrados {len(h5_files)} sistemas desde H5")
                    geometries = [{"name": f.stem.replace("_emergent", "")} for f in h5_files]
    else:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"\n[WARN] No existe manifest.json en {data_dir}")
            # Intentar auto-descubrir sistemas desde predictions/
            npz_files = list(predictions_dir.glob("*_geometry.npz"))
            if npz_files:
                print(f"[INFO] Auto-descubriendo {len(npz_files)} sistemas desde predictions/")
                geometries = [{"name": f.stem.replace("_geometry", "")} for f in npz_files]
        else:
            manifest = json.loads(manifest_path.read_text())
            geometries = manifest.get("geometries", [])
    
    # === VALIDAR QUE HAY GEOMETRÍAS ===
    all_contracts = []
    
    if not geometries:
        print("[ERROR] No hay sistemas para procesar")
        # V2.2: Aún así, escribir el JSON de salida si se solicitó
        if args.output_json:
            out_json = Path(args.output_json)
            write_contracts_summary(
                passed_ids=[],
                failed_ids=["_no_systems_found"],
                out_json=out_json,
                extra={
                    "version": "2.2.1",
                    "error": "No hay sistemas para procesar",
                }
            )
            print(f"  Resumen contratos (CI): {out_json}")
        return
    
    # Cargar diccionario
    if dictionary_path and dictionary_path.exists():
        dictionary_results = json.loads(dictionary_path.read_text())
    else:
        dictionary_results = {}
        if dictionary_path:
            print(f"[WARN] No existe diccionario: {dictionary_path}")
        else:
            print("[WARN] No se especificó archivo de diccionario")
    
    for geo_info in geometries:
        name = geo_info["name"] if isinstance(geo_info, dict) else geo_info
        print(f"\n>> {name}")
        
        contract = process_geometry(
            name, data_dir, predictions_dir, emergent_h5_dir,
            einstein_dir, dictionary_results, args.d
        )
        all_contracts.append(contract)
        
        # Mostrar resultado
        if contract.errors:
            print(f"   [ERROR] {contract.errors}")
        else:
            print(f"   Family: {contract.family} | Mode: {contract.mode}")
            print(f"   Genéricos:     {'✓' if contract.generic_passed else '✗'} "
                  f"(reg={contract.regularity.passed}, caus={contract.causality.passed})")
            if contract.is_ads_family:
                print(f"   AdS-specific:  {'✓' if contract.ads_specific_passed else '✗'} "
                      f"(einstein={contract.ads_einstein.passed}, asymp={contract.ads_asymptotic.passed})")
            else:
                print(f"   AdS-specific:  N/A (no es family='ads')")
            
            if contract.A_r2 is not None:
                print(f"   R²: A={contract.A_r2:.3f}, f={contract.f_r2:.3f}")
            else:
                print(f"   R²: (no disponible en inference)")
            
            print(f"   Score: {contract.contract_score:.2f}")
        
        if contract.warnings:
            for w in contract.warnings:
                print(f"   [WARN] {w}")
    
    # ============================================================
    # RESUMEN
    # ============================================================
    
    print("\n" + "=" * 100)
    print("RESUMEN DE CONTRATOS FASE XI v2.2")
    print("=" * 100)
    
    # Tabla
    print(f"\n{'Name':<25} {'Family':<12} {'Mode':<10} {'Generic':^8} {'AdS':^8} {'Score':^8} {'Pass':^6}")
    print("-" * 100)
    
    for c in all_contracts:
        ads_col = '✓' if c.ads_specific_passed else ('✗' if c.is_ads_family else '—')
        mode_short = "sandbox" if c.mode == "sandbox" else "infer"
        print(f"{c.name:<25} {c.family:<12} {mode_short:<10} "
              f"{'✓' if c.generic_passed else '✗':^8} "
              f"{ads_col:^8} "
              f"{c.contract_score:.2f} "
              f"{'✓' if c.overall_passed else '✗':^6}")
    
    print("-" * 100)
    
    # Estadísticas
    n_total = len(all_contracts)
    n_with_errors = sum(1 for c in all_contracts if c.errors)
    n_inference = sum(1 for c in all_contracts if c.mode == "inference")
    n_generic = sum(c.generic_passed for c in all_contracts)
    n_ads_family = sum(c.is_ads_family for c in all_contracts)
    n_ads_passed = sum(c.ads_specific_passed for c in all_contracts if c.is_ads_family)
    n_overall = sum(c.overall_passed for c in all_contracts)
    
    contracts_with_score = [c for c in all_contracts if not c.errors]
    avg_score = np.mean([c.contract_score for c in contracts_with_score]) if contracts_with_score else 0.0
    
    print(f"\nESTADÍSTICAS:")
    print(f"  Total geometrías:      {n_total}")
    print(f"  Con errores:           {n_with_errors}")
    print(f"  Modo inference:        {n_inference}")
    print(f"  Genéricos OK:          {n_generic}/{n_total}")
    print(f"  AdS families:          {n_ads_family}/{n_total}")
    print(f"  AdS-specific OK:       {n_ads_passed}/{n_ads_family} (de las que son AdS)")
    print(f"  Overall passed:        {n_overall}/{n_total}")
    print(f"  Score promedio:        {avg_score:.3f}")
    
    # Por modo
    print("\nPOR MODO:")
    for mode_name in ["sandbox", "inference"]:
        mode_contracts = [c for c in all_contracts if c.mode == mode_name]
        if mode_contracts:
            n_m = len(mode_contracts)
            n_passed = sum(c.overall_passed for c in mode_contracts)
            avg_m = np.mean([c.contract_score for c in mode_contracts])
            print(f"  {mode_name:12}: {n_passed}/{n_m} passed, score={avg_m:.2f}")
    
    # Veredicto final
    print("\n" + "=" * 90)
    
    phase_passed = (n_with_errors == 0) and (n_generic == n_total) and (avg_score > 0.5 or n_total == 0)
    
    if n_total == 0:
        print("⚠ NO HAY GEOMETRÍAS PARA EVALUAR")
    elif phase_passed:
        print("✓ FASE XI v2.2 COMPLETADA — VALIDACIÓN HONESTA EXITOSA")
        print("=" * 90)
        print(f"\n  El sistema CUERDAS ha logrado:")
        print(f"     Todas las geometrías pasan contratos genéricos")
        if n_inference > 0:
            print(f"     {n_inference} geometrías procesadas en modo inference")
    else:
        print("✗ FASE XI v2.2 REQUIERE REFINAMIENTO")
        print("=" * 90)
        if n_with_errors > 0:
            print(f"\n  {n_with_errors} geometrías con errores")
        if n_generic < n_total:
            print(f"\n  {n_total - n_generic} geometrías no pasan contratos genéricos")
        if avg_score <= 0.5:
            print(f"\n  Score promedio ({avg_score:.2f}) demasiado bajo")
    
    # === GUARDAR ===
    def serialize_value(v):
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            if v is None:
                return None
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, dict):
            return {k: serialize_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [serialize_value(item) for item in v]
        if v is None:
            return None
        return v
    
    def clean_contract(c_dict):
        return {k: serialize_value(v) for k, v in c_dict.items()}
    
    def serialize_contract(c: PhaseXIContractV2) -> dict:
        """Serializa contrato incluyendo propiedades calculadas (V2.2.1)."""
        d = clean_contract(asdict(c))
        # Añadir propiedades @property que asdict() no serializa
        d["contract_score"] = serialize_value(c.contract_score)
        d["generic_passed"] = serialize_value(c.generic_passed)
        d["ads_specific_passed"] = serialize_value(c.ads_specific_passed)
        d["overall_passed"] = serialize_value(c.overall_passed)
        d["is_ads_family"] = serialize_value(c.is_ads_family)
        return d
    
    output_data = {
        "version": "2.2.1",
        "n_total": n_total,
        "n_with_errors": n_with_errors,
        "n_inference_mode": n_inference,
        "n_generic_passed": int(n_generic),
        "n_ads_family": int(n_ads_family),
        "n_ads_specific_passed": int(n_ads_passed),
        "n_overall_passed": int(n_overall),
        "avg_score": float(avg_score) if not np.isnan(avg_score) else None,
        "phase_passed": bool(phase_passed),
        "contracts": [serialize_contract(c) for c in all_contracts]
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n  Resultados: {output_file}")


    # === RESUMEN MACHINE-READABLE (para CI / negative-control) ===
    if args.output_json:
        passed_ids: List[str] = []
        failed_ids: List[str] = []

        for c in all_contracts:
            geo = getattr(c, "name", "unknown")

            # Si hubo errores fatales en el contrato, marcar como fail explícito
            if getattr(c, "errors", None):
                failed_ids.append(f"{geo}/fatal_error")
                continue

            # Contratos genéricos (siempre cuentan)
            (passed_ids if c.regularity.passed else failed_ids).append(f"{geo}/regularity")
            (passed_ids if c.causality.passed else failed_ids).append(f"{geo}/causality")

            # Contratos AdS-específicos (solo si family='ads')
            if c.is_ads_family:
                (passed_ids if c.ads_einstein.passed else failed_ids).append(f"{geo}/ads_einstein")
                (passed_ids if c.ads_asymptotic.passed else failed_ids).append(f"{geo}/ads_asymptotic")

        out_json = Path(args.output_json)
        write_contracts_summary(
            passed_ids=passed_ids,
            failed_ids=failed_ids,
            out_json=out_json,
            extra={
                "version": "2.2.1",
                "source_summary": str(output_file),
            }
        )
        print(f"  Resumen contratos (CI): {out_json}")

    # === ACTUALIZAR RUN_MANIFEST (IO v2) ===
    if args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir)
            update_run_manifest(
                run_dir,
                {
                    "geometry_contracts_dir": str(output_file.parent.relative_to(run_dir)
                                                  if output_file.parent.is_relative_to(run_dir)
                                                  else output_file.parent),
                    "geometry_contracts_summary": str(output_file.relative_to(run_dir)
                                                      if output_file.is_relative_to(run_dir)
                                                      else output_file),
                }
            )
            print(f"  Manifest actualizado: {run_dir / 'run_manifest.json'}")
        except Exception as e:
            print(f"  [WARN] No se pudo actualizar run_manifest.json: {e}")
    
    print("=" * 90)


    
    # === V3: Registrar outputs ===
    ctx.register_outputs({"geometry_contracts_summary": "geometry_contracts_summary.json"})
    ctx.create_aliases()
    ctx.save_manifest()

if __name__ == "__main__":
    main()
