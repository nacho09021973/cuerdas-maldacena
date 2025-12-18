#!/usr/bin/env python3
# 07_emergent_lambda_sl_dictionary.py
# CUERDAS — Bloque C: Diccionario emergente λ_SL ↔ Δ
#
# OBJETIVO
#   Aprender una relación emergente entre el espectro escalar en el bulk (λ_SL)
#   y los exponentes UV/Δ, utilizando:
#     - Un modelo suave (p.ej. KAN) para aproximar la relación.
#     - PySR (u otro SR) para destilar una forma simbólica compacta.
#
# ENTRADAS
#   - runs/bulk_eigenmodes/bulk_modes_dataset.csv
#
# SALIDAS
#   runs/emergent_dictionary/
#     lambda_sl_dictionary_pareto.csv
#       - Conjunto de expresiones candidatas (frente de Pareto).
#     lambda_sl_dictionary_report.json
#       - Resumen con métricas, selección de modelos, etc.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Consume el dataset generado por:
#       * 06_build_bulk_eigenmodes_dataset.py
#   - Sus resultados se usan en:
#       * 09_real_data_and_dictionary_contracts.py
#
# HONESTIDAD
#   - No se fuerza la fórmula Δ(Δ-d) ni se inyectan diccionarios conocidos.
#   - Cualquier comparación con fórmulas teóricas se realiza posteriormente,
#     en scripts de análisis/contratos.
#
# HISTÓRICO
#   - Anteriormente conocido como: fase12c_emergent_dictionary_v2.py

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

# Import local IO module for run manifest support
try:
    from cuerdas_io import update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


@dataclass
class DiscoveryConfig:
    """Configuración para el descubrimiento de relaciones emergentes."""
    niterations: int = 200
    populations: int = 30
    ncyclesperiteration: int = 1000
    maxsize: int = 30
    features: Tuple[str, ...] = ("Delta", "d")
    target: str = "lambda_sl_emergent"  # CAMBIADO: antes era m2L2_emergent
    binary_operators: Tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: Tuple[str, ...] = ("square", "sqrt", "exp", "log")
    binary_ops_minimal: Tuple[str, ...] = ("+", "-", "*", "/")
    unary_ops_minimal: Tuple[str, ...] = ("square", "sqrt")
    complexity_of_constants: float = 1.0
    complexity_of_variables: float = 1.0
    parsimony: float = 0.003
    annealing: bool = True
    early_stop_condition: str = "1e-6"
    random_state: int = 42
    deterministic: bool = True
    parallelism: str = "serial"
    test_split_ratio: float = 0.2
    evaluation_metrics: Tuple[str, ...] = ("r2", "mae", "pearson", "complexity")
    save_pareto_front: bool = True
    save_raw_equations: bool = True


# Métodos que podrían contaminar los datos con conocimiento teórico
SUSPICIOUS_METHODS: Set[str] = {
    "theoretical_formula", "ads_formula", "manual_delta_d",
    "delta_formula", "analytic_formula", "exact_formula",
    "input_formula", "adscft_formula", "breitenlohner_freedman",
    "bf_bound", "m2_calculated", "calculated_from_delta"
}

# Métodos válidos para extracción holográfica
VALID_HOLOGRAPHIC_METHODS: Set[str] = {
    "qnm_fitting", "green_function_poles", "bulk_scalar_fit",
    "spectral_peaks", "effective_action", "quasinormal_modes",
    "propagator_poles", "correlator_fit", "bulk_wave_equation",
    "laplacian_spectrum", "tower_of_states", "holographic_reconstruction",
    "from_extracted_delta", "emergent", "hdf5", "bulk_eigenmode"
}


def detect_input_format(data: Dict[str, Any]) -> str:
    """Detecta el formato del archivo de entrada."""
    # Formato v2: buscar lambda_sl_*
    if "nomenclature_version" in data and "lambda_sl" in data.get("nomenclature_version", ""):
        return "v2_lambda_sl"
    
    # Formato v3 dictionary
    if "mass_dimension" in data and "by_system" in data.get("mass_dimension", {}):
        return "dictionary_v3"
    
    # Formato legacy con systems[]
    if "systems" in data:
        # Detectar si es v2 o legacy basándose en las claves
        first_sys = data["systems"][0] if data["systems"] else {}
        if "lambda_sl_bulk" in first_sys:
            return "v2_lambda_sl"
        elif "m2L2_bulk" in first_sys:
            return "legacy_systems"
    
    # by_family_d format
    if "by_family_d" in data:
        first_family = list(data["by_family_d"].values())[0] if data["by_family_d"] else {}
        if "lambda_sl_bulk" in first_family:
            return "v2_lambda_sl"
        elif "m2L2_bulk" in first_family:
            return "legacy_systems"
    
    raise ValueError(
        "Formato de archivo no reconocido. "
        "Se espera formato v2 (lambda_sl_*), dictionary_v3, o legacy (m2L2_*)."
    )


def load_from_v2_format(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga datos desde formato v2 con nomenclatura lambda_sl."""
    records = []
    metadata = {
        "format": "v2_lambda_sl",
        "source": data.get("source", "unknown"),
        "nomenclature_version": data.get("nomenclature_version", "v2_lambda_sl"),
        "systems": [],
        "total_operators": 0,
        "methods_lambda_sl": set(),
        "suspicious_methods_found": set(),
        "valid_methods_found": set()
    }
    
    # Procesar systems[] si existe
    for system in data.get("systems", []):
        sys_name = system.get("geometry_name", "unknown")
        family = system.get("family", "unknown")
        d = system.get("d", 4)
        source = system.get("lambda_source", "bulk_eigenmode")
        
        lambda_list = system.get("lambda_sl_bulk", [])
        Delta_list = system.get("Delta_bulk_uv", [])
        
        if len(lambda_list) != len(Delta_list):
            warnings.warn(f"Sistema {sys_name}: Delta ({len(Delta_list)}) y lambda_sl ({len(lambda_list)}) tienen diferente longitud")
            continue
        
        # Validar método
        method = source.lower()
        is_suspicious = any(sus in method for sus in SUSPICIOUS_METHODS)
        is_valid = any(val in method for val in VALID_HOLOGRAPHIC_METHODS)
        
        if is_suspicious:
            metadata["suspicious_methods_found"].add(method)
        if is_valid:
            metadata["valid_methods_found"].add(method)
        
        metadata["methods_lambda_sl"].add(method)
        
        system_metadata = {
            "name": sys_name,
            "family": family,
            "d": d,
            "source": source,
            "n_operators": len(lambda_list),
            "has_suspicious_methods": is_suspicious
        }
        metadata["systems"].append(system_metadata)
        
        # Crear registros
        for i, (Delta, lam) in enumerate(zip(Delta_list, lambda_list)):
            record = {
                "system": sys_name,
                "family": family,
                "d": d,
                "source": source,
                "operator": f"{sys_name}_mode{i}",
                "Delta": float(Delta),
                "Delta_error": 0.0,
                "lambda_sl_emergent": float(lam),
                "lambda_sl_error": 0.0,
                "lambda_source": method,
                "method_is_suspicious": is_suspicious,
                "method_is_valid_holographic": is_valid
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    metadata["total_operators"] = len(df)
    metadata["methods_lambda_sl"] = list(metadata["methods_lambda_sl"])
    metadata["suspicious_methods_found"] = list(metadata["suspicious_methods_found"])
    metadata["valid_methods_found"] = list(metadata["valid_methods_found"])
    
    return df, metadata


def load_from_dictionary_v3(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga datos desde formato diccionario v3, convirtiendo a nomenclatura lambda_sl."""
    records = []
    metadata = {
        "format": "dictionary_v3_converted",
        "version": data.get("version", "unknown"),
        "mass_source": data.get("mass_source", "unknown"),
        "systems": [],
        "total_operators": 0,
        "methods_lambda_sl": set(),
        "suspicious_methods_found": set(),
        "valid_methods_found": set()
    }
    
    by_system = data.get("mass_dimension", {}).get("by_system", {})
    
    if not by_system:
        raise ValueError("No se encontraron datos en mass_dimension.by_system")
    
    for sys_key, sys_data in by_system.items():
        family = sys_data.get("family", "unknown")
        d = sys_data.get("d", 4)
        source = sys_data.get("source", "emergent")
        n_points = sys_data.get("n_points", 0)
        
        deltas = sys_data.get("Delta", [])
        # Aceptar tanto m2L2_emergent (legacy) como lambda_sl_emergent (v2)
        lambda_values = sys_data.get("lambda_sl_emergent", sys_data.get("m2L2_emergent", []))
        
        if len(deltas) != len(lambda_values):
            warnings.warn(f"Sistema {sys_key}: Delta ({len(deltas)}) y lambda ({len(lambda_values)}) tienen diferente longitud")
            continue
        
        # Determinar método basado en source
        method = source.lower()
        is_suspicious = any(sus in method for sus in SUSPICIOUS_METHODS)
        is_valid = any(val in method for val in VALID_HOLOGRAPHIC_METHODS)
        
        if is_suspicious:
            metadata["suspicious_methods_found"].add(method)
        if is_valid:
            metadata["valid_methods_found"].add(method)
        
        metadata["methods_lambda_sl"].add(method)
        
        system_metadata = {
            "name": sys_key,
            "family": family,
            "d": d,
            "source": source,
            "n_operators": n_points,
            "has_suspicious_methods": is_suspicious
        }
        metadata["systems"].append(system_metadata)
        
        # Crear registros con nomenclatura v2
        for i, (delta, lam) in enumerate(zip(deltas, lambda_values)):
            record = {
                "system": sys_key,
                "family": family,
                "d": d,
                "source": source,
                "operator": f"{sys_key}_op{i}",
                "Delta": float(delta),
                "Delta_error": 0.0,
                "lambda_sl_emergent": float(lam),  # CONVERTIDO a nomenclatura v2
                "lambda_sl_error": 0.0,
                "lambda_source": method,
                "method_is_suspicious": is_suspicious,
                "method_is_valid_holographic": is_valid
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    metadata["total_operators"] = len(df)
    metadata["methods_lambda_sl"] = list(metadata["methods_lambda_sl"])
    metadata["suspicious_methods_found"] = list(metadata["suspicious_methods_found"])
    metadata["valid_methods_found"] = list(metadata["valid_methods_found"])
    
    return df, metadata


def load_from_legacy_systems(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga datos desde formato legacy con m2L2_*, convirtiendo a nomenclatura lambda_sl."""
    records = []
    metadata = {
        "format": "legacy_systems_converted",
        "systems": [],
        "total_operators": 0,
        "methods_lambda_sl": set(),
        "suspicious_methods_found": set(),
        "valid_methods_found": set()
    }
    
    for system in data.get("systems", []):
        sys_name = system.get("geometry_name", system.get("name", "unknown"))
        family = system.get("family", "unknown")
        d = system.get("d", 4)
        source = system.get("m2L2_method", system.get("lambda_source", "unknown"))
        
        # Aceptar tanto m2L2_bulk (legacy) como lambda_sl_bulk (v2)
        lambda_list = system.get("lambda_sl_bulk", system.get("m2L2_bulk", []))
        Delta_list = system.get("Delta_bulk_uv", [])
        
        if len(lambda_list) != len(Delta_list):
            warnings.warn(f"Sistema {sys_name}: longitudes no coinciden")
            continue
        
        method = source.lower()
        is_suspicious = any(sus in method for sus in SUSPICIOUS_METHODS)
        is_valid = any(val in method for val in VALID_HOLOGRAPHIC_METHODS)
        
        if is_suspicious:
            metadata["suspicious_methods_found"].add(method)
        if is_valid:
            metadata["valid_methods_found"].add(method)
        
        metadata["methods_lambda_sl"].add(method)
        
        system_metadata = {
            "name": sys_name,
            "family": family,
            "d": d,
            "source": source,
            "n_operators": len(lambda_list),
            "has_suspicious_methods": is_suspicious
        }
        metadata["systems"].append(system_metadata)
        
        # Crear registros con nomenclatura v2
        for i, (Delta, lam) in enumerate(zip(Delta_list, lambda_list)):
            record = {
                "system": sys_name,
                "family": family,
                "d": d,
                "source": source,
                "operator": f"{sys_name}_mode{i}",
                "Delta": float(Delta),
                "Delta_error": 0.0,
                "lambda_sl_emergent": float(lam),  # CONVERTIDO a nomenclatura v2
                "lambda_sl_error": 0.0,
                "lambda_source": method,
                "method_is_suspicious": is_suspicious,
                "method_is_valid_holographic": is_valid
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    metadata["total_operators"] = len(df)
    metadata["methods_lambda_sl"] = list(metadata["methods_lambda_sl"])
    metadata["suspicious_methods_found"] = list(metadata["suspicious_methods_found"])
    metadata["valid_methods_found"] = list(metadata["valid_methods_found"])
    
    return df, metadata


def load_emergent_data(input_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga datos emergentes del pipeline Fase XI con detección automática de formato."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    input_format = detect_input_format(data)
    print(f"\n   Formato detectado: {input_format}")
    
    if input_format == "v2_lambda_sl":
        df, metadata = load_from_v2_format(data)
    elif input_format == "dictionary_v3":
        df, metadata = load_from_dictionary_v3(data)
    else:
        df, metadata = load_from_legacy_systems(data)
    
    # Validaciones comunes
    if len(df) == 0:
        raise ValueError("No se cargaron datos. Verificar archivo de entrada.")
    
    n_suspicious = df["method_is_suspicious"].sum() if "method_is_suspicious" in df.columns else 0
    n_valid = df["method_is_valid_holographic"].sum() if "method_is_valid_holographic" in df.columns else 0
    
    print(f"\n   ANÁLISIS DE CALIDAD DE DATOS:")
    print(f"   - Operadores totales: {len(df)}")
    print(f"   - Sistemas: {len(metadata['systems'])}")
    print(f"   - Métodos únicos encontrados: {metadata['methods_lambda_sl']}")
    print(f"   - Operadores con métodos válidos (holográficos): {n_valid}")
    print(f"   - Operadores con métodos sospechosos: {n_suspicious}")
    
    if n_suspicious > 0:
        warnings.warn(
            f"   ALERTA DE CONTAMINACIÓN: Se encontraron {n_suspicious} operadores "
            f"con métodos sospechosos: {metadata['suspicious_methods_found']}",
            UserWarning
        )
    
    return df, metadata


def prepare_training_data(
    df: pd.DataFrame, 
    config: DiscoveryConfig,
    test_split: float = 0.2,
    filter_suspicious: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepara datos para PySR."""
    required_cols = list(config.features) + [config.target]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
    
    if filter_suspicious and "method_is_suspicious" in df.columns:
        original_len = len(df)
        df = df[~df["method_is_suspicious"]].copy()
        filtered_len = len(df)
        if filtered_len < original_len:
            print(f"   - Filtrados {original_len - filtered_len} puntos con métodos sospechosos")
    
    df_clean = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df_clean) < 6:
        raise ValueError(f"Datos insuficientes después de limpieza: {len(df_clean)} muestras")
    
    df_clean = df_clean.sample(frac=1, random_state=config.random_state).reset_index(drop=True)
    
    split_idx = int(len(df_clean) * (1 - test_split))
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[list(config.features)].values.astype(float)
    y_train = train_df[config.target].values.astype(float)
    X_test = test_df[list(config.features)].values.astype(float)
    y_test = test_df[config.target].values.astype(float)
    
    print(f"\n   DATOS PREPARADOS PARA PYSR:")
    print(f"   - Entrenamiento: {len(X_train)} puntos")
    print(f"   - Test: {len(X_test)} puntos")
    print(f"   - Características: {config.features}")
    print(f"   - Target: {config.target} (autovalores Sturm–Liouville)")
    print(f"   - Rango Delta: [{df_clean['Delta'].min():.3f}, {df_clean['Delta'].max():.3f}]")
    print(f"   - Rango λ_SL: [{df_clean[config.target].min():.3f}, {df_clean[config.target].max():.3f}]")
    
    return X_train, y_train, X_test, y_test, test_df


def discover_emergent_relation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: DiscoveryConfig,
    use_minimal_ops: bool = False
) -> Optional[PySRRegressor]:
    """Usa PySR para descubrir relación simbólica emergente."""
    if not HAS_PYSR:
        warnings.warn("PySR no disponible. Instalar con: pip install pysr")
        return None
    
    if use_minimal_ops:
        binary_ops = list(config.binary_ops_minimal)
        unary_ops = list(config.unary_ops_minimal)
    else:
        binary_ops = list(config.binary_operators)
        unary_ops = list(config.unary_operators)
    
    print(f"\n   CONFIGURANDO PYSR:")
    print(f"   - Iteraciones: {config.niterations}")
    print(f"   - Poblaciones: {config.populations}")
    print(f"   - Operadores binarios: {binary_ops}")
    print(f"   - Operadores unarios: {unary_ops}")
    
    model = PySRRegressor(
        niterations=config.niterations,
        populations=config.populations,
        ncyclesperiteration=config.ncyclesperiteration,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        maxsize=config.maxsize,
        elementwise_loss="L2DistLoss()",
        complexity_of_constants=config.complexity_of_constants,
        complexity_of_variables=config.complexity_of_variables,
        parsimony=config.parsimony,
        annealing=config.annealing,
        early_stop_condition=config.early_stop_condition,
        random_state=config.random_state,
        deterministic=config.deterministic,
        parallelism=config.parallelism,
        progress=True,
        verbosity=1,
        temp_equation_file=True,
        model_selection="best",
        extra_sympy_mappings={"square": lambda x: x**2},
    )
    
    print("\n   ENTRENANDO PYSR (descubrimiento emergente)...")
    print("   NOTA: Buscando relación Δ ↔ λ_SL (sin asumir m²L² = Δ(Δ-d))")
    model.fit(X_train, y_train)
    
    return model


def evaluate_equation(
    equation_str: str,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Tuple[str, ...]
) -> Dict[str, float]:
    """Evalúa una ecuación descubierta."""
    try:
        y_pred = []
        for row in X:
            env = {f'x{i}': val for i, val in enumerate(row)}
            import math
            env.update({
                'square': lambda x: x**2,
                'sqrt': math.sqrt,
                'exp': math.exp,
                'log': math.log,
            })
            y_pred.append(eval(equation_str, {"__builtins__": {}}, env))
        
        y_pred = np.array(y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        
        return {
            "r2": float(r2),
            "mae": float(mae),
            "pearson": float(pearson_corr),
            "success": True
        }
    except Exception as e:
        return {"r2": -np.inf, "mae": np.inf, "pearson": 0.0, "success": False, "error": str(e)}


def evaluate_against_maldacena(
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Tuple[str, ...],
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Evalúa los datos contra la fórmula teórica de Maldacena m²L² = Δ(Δ-d).
    
    IMPORTANTE: Esta evaluación es SOLO para comparación a posteriori.
    NO afecta al ajuste de PySR. Es una forma de validar si los autovalores
    λ_SL descubiertos tienen interpretación como masas holográficas.
    
    Args:
        X: Features (Delta, d)
        y_true: Valores λ_SL observados
        feature_names: Nombres de features
        tolerance: Tolerancia para considerar "compatible" (default 0.1 = 10% en R²)
    
    Returns:
        Dict con métricas de comparación
    """
    theoretical_predictions = []
    for row in X:
        Delta = row[0]
        d = row[1]
        theoretical_predictions.append(Delta * (Delta - d))
    
    theoretical_predictions = np.array(theoretical_predictions)
    
    # Métricas de la fórmula teórica
    theory_r2 = r2_score(y_true, theoretical_predictions)
    theory_mae = mean_absolute_error(y_true, theoretical_predictions)
    
    # Error relativo máximo
    nonzero_mask = np.abs(y_true) > 1e-10
    if nonzero_mask.sum() > 0:
        rel_errors = np.abs(theoretical_predictions[nonzero_mask] - y_true[nonzero_mask]) / np.abs(y_true[nonzero_mask])
        max_rel_error = float(np.max(rel_errors))
        mean_rel_error = float(np.mean(rel_errors))
    else:
        max_rel_error = np.inf
        mean_rel_error = np.inf
    
    return {
        "enabled": True,
        "formula": "λ_theory = Δ(Δ - d)",
        "note": "Esta comparación es A POSTERIORI y NO afecta al ajuste de PySR",
        "r2": float(theory_r2),
        "mae": float(theory_mae),
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "tolerance_used": tolerance,
        "is_compatible": theory_r2 > (1.0 - tolerance),
        "interpretation": (
            "Si is_compatible=True, los λ_SL pueden interpretarse como m²L² holográficas"
            if theory_r2 > (1.0 - tolerance) else
            "La relación λ_SL ↔ Δ difiere de Maldacena; puede indicar física más allá de AdS/CFT"
        )
    }


def compare_with_theory(
    best_eq: Dict[str, Any],
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Tuple[str, ...],
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Comparación A POSTERIORI con fórmula teórica m²L² = Δ(Δ-d).
    
    NOTA: Esta comparación es DESPUÉS del descubrimiento, no antes.
    Si la ecuación descubierta coincide con Δ(Δ-d), entonces HEMOS DESCUBIERTO
    que los autovalores λ_SL tienen interpretación como masas holográficas.
    """
    # Evaluar fórmula teórica
    maldacena_eval = evaluate_against_maldacena(X, y_true, feature_names, tolerance)
    
    # Evaluar ecuación descubierta
    eq_metrics = evaluate_equation(best_eq["equation"], X, y_true, feature_names)
    
    return {
        "theoretical_formula": "λ_SL ≈ Δ(Δ - d)  [si fuera masa holográfica m²L²]",
        "discovered_formula": best_eq["equation"],
        "theory_r2": maldacena_eval["r2"],
        "theory_mae": maldacena_eval["mae"],
        "theory_max_rel_error": maldacena_eval["max_rel_error"],
        "discovered_r2": eq_metrics.get("r2", -np.inf),
        "discovered_mae": eq_metrics.get("mae", np.inf),
        "compatible_with_maldacena": maldacena_eval["is_compatible"] and eq_metrics.get("success", False),
        "maldacena_comparison": maldacena_eval,
        "note": "Esta comparación se hace a posteriori y NO afecta al ajuste de PySR"
    }


def save_results(
    model: PySRRegressor,
    config: DiscoveryConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: Path,
    use_minimal_ops: bool
) -> Dict[str, Any]:
    """Guarda resultados del descubrimiento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    equations_df = model.equations_
    best_eq = model.get_best()
    
    # Evaluar en test
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_pearson, _ = pearsonr(y_test, y_pred_test)
    
    test_metrics = {
        "r2": float(test_r2),
        "mae": float(test_mae),
        "pearson": float(test_pearson)
    }
    
    # Comparación con teoría
    theory_comparison = compare_with_theory(
        {"equation": str(best_eq["equation"])},
        X_test, y_test, config.features
    )
    
    # Construir resumen
    summary = {
        "timestamp": datetime.now().isoformat(),
        "nomenclature_version": "v2_lambda_sl",
        "config": asdict(config),
        "input_metadata": {
            "format": metadata.get("format", "unknown"),
            "source": metadata.get("source", "unknown"),
            "total_operators": metadata.get("total_operators", 0),
            "systems_count": len(metadata.get("systems", [])),
            "methods_found": metadata.get("methods_lambda_sl", []),
            "suspicious_methods": metadata.get("suspicious_methods_found", []),
            "valid_methods": metadata.get("valid_methods_found", [])
        },
        "data_stats": {
            "n_points_train": int(len(X_train)),
            "n_points_test": int(len(X_test)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "Delta_range": [float(X_test[:, 0].min()), float(X_test[:, 0].max())],
            "lambda_sl_range": [float(y_test.min()), float(y_test.max())]
        },
        "discovery_results": {
            "best_equation": str(best_eq["equation"]),
            "complexity": int(best_eq.get("complexity", 0)),
            "train_metrics": {
                "r2": float(r2_score(y_train, model.predict(X_train)))
            },
            "test_metrics": test_metrics,
            "use_minimal_ops": use_minimal_ops
        },
        "theory_comparison": theory_comparison,
        "pareto_front": [],
        "notes": [
            "Los valores λ_SL son autovalores Sturm–Liouville, NO asumidos como m²L².",
            "La comparación con Δ(Δ-d) es A POSTERIORI, no una suposición.",
            "Si compatible_with_maldacena=True, el descubrimiento valida la interpretación física."
        ]
    }
    
    # Pareto front
    if config.save_pareto_front and equations_df is not None:
        for _, row in equations_df.iterrows():
            summary["pareto_front"].append({
                "equation": str(row.get("equation", "")),
                "complexity": int(row.get("complexity", 0)),
                "loss": float(row.get("loss", np.inf))
            })
    
    # Guardar
    summary_file = output_dir / "lambda_sl_dictionary_report.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Compatibilidad: mantenemos el nombre histórico si alguien lo consumía
    legacy_file = output_dir / "fase12c_discovery_summary_v2.json"
    try:
        with open(legacy_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception:
        pass
    
    if config.save_raw_equations and equations_df is not None:
        eq_file = output_dir / "fase12c_all_equations_v2.csv"
        equations_df.to_csv(eq_file, index=False)
    
    print(f"\n   RESULTADOS GUARDADOS:")
    print(f"   - Directorio: {output_dir}")
    print(f"   - Mejor ecuación: {best_eq['equation']}")
    print(f"   - R² en test: {test_metrics.get('r2', 'N/A'):.4f}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="FASE XII.c v2: Diccionario Emergente (nomenclatura λ_SL)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Archivo JSON de entrada (v2 lambda_sl, dictionary_v3, o legacy m2L2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida para resultados")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directorio raíz con run_manifest.json (IO v2). Resuelve input/output automáticamente.")
    parser.add_argument("--ops-minimal", action="store_true",
                        help="Usar operadores mínimos (+,-,*,/,square,sqrt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Número de iteraciones de PySR")
    parser.add_argument("--no-filter-suspicious", action="store_true",
                        help="No filtrar operadores con métodos sospechosos")
    parser.add_argument("--drop-suspicious", action="store_true",
                        help="Descartar filas con method_is_suspicious==True antes de entrenar")
    parser.add_argument("--force-continue", action="store_true",
                        help="Continuar aunque se detecten métodos sospechosos")
    
    args = parser.parse_args()
    
    # === RESOLVER RUTAS ===
    input_path = None
    output_dir = None
    
    # Prioridad 1: --run-dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
        # Buscar input en bulk_eigenmodes/
        bulk_modes_json = run_dir / "bulk_eigenmodes" / "bulk_modes_dataset_v2.json"
        if not bulk_modes_json.exists():
            bulk_modes_json = run_dir / "bulk_eigenmodes" / "bulk_modes_dataset.json"
        if bulk_modes_json.exists():
            input_path = bulk_modes_json
        # Output en emergent_dictionary/
        output_dir = run_dir / "emergent_dictionary"
    
    # Prioridad 2: argumentos explícitos
    if input_path is None and args.input_file:
        input_path = Path(args.input_file)
    
    if output_dir is None and args.output_dir:
        output_dir = Path(args.output_dir)
    
    if input_path is None or output_dir is None:
        parser.error("Debe proporcionar --run-dir o ambos --input-file y --output-dir")
    
    config = DiscoveryConfig(random_state=args.seed, niterations=args.iterations)
    
    print("=" * 80)
    print("FASE XII.c v2 - DICCIONARIO HOLOGRÁFICO EMERGENTE")
    print("Nomenclatura honesta: λ_SL (autovalores Sturm–Liouville)")
    print("=" * 80)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    
    print(f"\n   Archivo de entrada: {input_path}")
    
    df, metadata = load_emergent_data(input_path)
    
    # Filtrar métodos sospechosos si se solicitó
    if args.drop_suspicious and "method_is_suspicious" in df.columns:
        n_before = len(df)
        df = df[~df["method_is_suspicious"]].copy()
        n_after = len(df)
        if n_before > n_after:
            print(f"\n   >> Drop suspicious: {n_before - n_after} operadores eliminados, quedan {n_after}.")
            metadata["dropped_suspicious"] = n_before - n_after
    
    n_suspicious = len(metadata.get("suspicious_methods_found", []))
    if n_suspicious > 0 and not args.force_continue:
        print(f"\n   Se encontraron {n_suspicious} métodos sospechosos.")
        response = input("   ¿Continuar? (s/N): ")
        if response.lower() != 's':
            print("   Abortado por el usuario.")
            return 1
    
    X_train, y_train, X_test, y_test, test_df = prepare_training_data(
        df, config, test_split=config.test_split_ratio,
        filter_suspicious=not args.no_filter_suspicious
    )
    
    if not HAS_PYSR:
        print("   ERROR: PySR no disponible. Instalar con: pip install pysr")
        return 1
    
    model = discover_emergent_relation(X_train, y_train, config, use_minimal_ops=args.ops_minimal)
    
    if model is None:
        return 1
    
    summary = save_results(model, config, X_train, y_train, X_test, y_test,
                          test_df, metadata, output_dir, args.ops_minimal)
    
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    best_eq = summary["discovery_results"]["best_equation"]
    test_metrics = summary["discovery_results"]["test_metrics"]
    theory_comp = summary["theory_comparison"]
    
    print(f"\n   ECUACIÓN DESCUBIERTA: {best_eq}")
    print(f"\n   MÉTRICAS EN TEST:")
    print(f"   - R²: {test_metrics.get('r2', 'N/A'):.4f}")
    print(f"   - MAE: {test_metrics.get('mae', 'N/A'):.4f}")
    print(f"   - Pearson: {test_metrics.get('pearson', 'N/A'):.4f}")
    print(f"\n   COMPARACIÓN A POSTERIORI CON TEORÍA:")
    print(f"   - Fórmula teórica (si fuera m²L²): λ_SL = Δ(Δ - d)")
    print(f"   - R² teórico: {theory_comp.get('theory_r2', 'N/A'):.4f}")
    print(f"   - Compatible con Maldacena: {theory_comp.get('compatible_with_maldacena', 'N/A')}")
    
    if theory_comp.get('compatible_with_maldacena'):
        print(f"\n   ✓ DESCUBRIMIENTO: Los λ_SL pueden interpretarse como m²L² holográficas!")
    else:
        print(f"\n   ⚠ NOTA: La relación descubierta difiere de Δ(Δ-d).")
        print(f"          Esto puede indicar física más allá de AdS/CFT estándar.")
    
    print(f"\n   Resultados en: {output_dir.absolute()}")
    
    # === ACTUALIZAR RUN_MANIFEST (IO v2) ===
    if args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir)
            report_file = output_dir / "lambda_sl_dictionary_report.json"
            update_run_manifest(
                run_dir,
                {
                    "emergent_dictionary_dir": str(output_dir.relative_to(run_dir)
                                                   if output_dir.is_relative_to(run_dir)
                                                   else output_dir),
                    "dictionary_report": str(report_file.relative_to(run_dir)
                                             if report_file.is_relative_to(run_dir)
                                             else report_file),
                }
            )
            print(f"   Manifest actualizado")
        except Exception as e:
            print(f"   [WARN] No se pudo actualizar manifest: {e}")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
