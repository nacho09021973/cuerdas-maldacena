#!/usr/bin/env python3
"""
fase12c_emergent_dictionary_real.py - Diccionario Holografico Emergente (Version Honesta)

FASE XII.c del proyecto CUERDAS (version realista)

OBJETIVO:
    Descubrir la relacion Delta <-> m²L² emergente usando solo datos del pipeline Fase XI.
    NO asume forma funcional ni inyecta conocimiento teorico.
    Los m²L² provienen del motor holografico (Fase XI), no de formulas teoricas.

FILOSOFIA:
    - Input: Pares (Delta, m²L²_emergent) donde m²L²_emergent viene de la reconstruccion bulk
    - PySR busca relacion simbolica sin conocimiento previo
    - Solo evaluacion a posteriori contra formula teorica (si existe)

FORMATOS SOPORTADOS:
    1. Diccionario v3 (holographic_dictionary_v3_summary.json):
       - mass_dimension.by_system con claves tipo "ads_d4"
       - Cada sistema tiene Delta[], m2L2_emergent[], d
    
    2. Formato legacy con systems[]:
       - Lista de sistemas con operators[]
       - Cada operador tiene Delta, m2L2_emergent

USO:
    python fase12c_emergent_dictionary_real.py \
        --input-file fase11_output/dictionary/holographic_dictionary_v3_summary.json \
        --output-dir results/fase12c_real \
        [--ops-minimal] [--seed 42]
"""

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


@dataclass
class DiscoveryConfig:
    """Configuracion para el descubrimiento de relaciones emergentes."""
    niterations: int = 200
    populations: int = 30
    ncyclesperiteration: int = 1000
    maxsize: int = 30
    features: Tuple[str, ...] = ("Delta", "d")
    target: str = "m2L2_emergent"
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


# Metodos que podrian contaminar los datos con conocimiento teorico
SUSPICIOUS_METHODS: Set[str] = {
    "theoretical_formula", "ads_formula", "manual_delta_d",
    "delta_formula", "analytic_formula", "exact_formula",
    "input_formula", "adscft_formula", "breitenlohner_freedman",
    "bf_bound", "m2_calculated", "calculated_from_delta"
}

# Metodos validos para extraccion holografica
VALID_HOLOGRAPHIC_METHODS: Set[str] = {
    "qnm_fitting", "green_function_poles", "bulk_scalar_fit",
    "spectral_peaks", "effective_action", "quasinormal_modes",
    "propagator_poles", "correlator_fit", "bulk_wave_equation",
    "laplacian_spectrum", "tower_of_states", "holographic_reconstruction",
    "from_extracted_delta", "emergent", "hdf5"
}


def detect_input_format(data: Dict[str, Any]) -> str:
    """Detecta el formato del archivo de entrada."""
    if "mass_dimension" in data and "by_system" in data.get("mass_dimension", {}):
        return "dictionary_v3"
    elif "systems" in data:
        return "legacy_systems"
    else:
        raise ValueError(
            "Formato de archivo no reconocido. "
            "Se espera 'mass_dimension.by_system' (diccionario v3) o 'systems[]' (legacy)."
        )


def load_from_dictionary_v3(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga datos desde formato diccionario v3 (holographic_dictionary_v3_summary.json)."""
    records = []
    metadata = {
        "format": "dictionary_v3",
        "version": data.get("version", "unknown"),
        "mass_source": data.get("mass_source", "unknown"),
        "systems": [],
        "total_operators": 0,
        "methods_m2L2": set(),
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
        m2L2_values = sys_data.get("m2L2_emergent", [])
        
        if len(deltas) != len(m2L2_values):
            warnings.warn(f"Sistema {sys_key}: Delta ({len(deltas)}) y m2L2 ({len(m2L2_values)}) tienen diferente longitud")
            continue
        
        # Determinar metodo basado en source
        m2_method = source.lower()
        is_suspicious = any(sus in m2_method for sus in SUSPICIOUS_METHODS)
        is_valid = any(val in m2_method for val in VALID_HOLOGRAPHIC_METHODS)
        
        if is_suspicious:
            metadata["suspicious_methods_found"].add(m2_method)
        if is_valid:
            metadata["valid_methods_found"].add(m2_method)
        
        metadata["methods_m2L2"].add(m2_method)
        
        system_metadata = {
            "name": sys_key,
            "family": family,
            "d": d,
            "source": source,
            "n_operators": n_points,
            "has_suspicious_methods": is_suspicious
        }
        metadata["systems"].append(system_metadata)
        
        # Crear registros
        for i, (delta, m2) in enumerate(zip(deltas, m2L2_values)):
            record = {
                "system": sys_key,
                "family": family,
                "d": d,
                "source": source,
                "operator": f"{sys_key}_op{i}",
                "Delta": float(delta),
                "Delta_error": 0.0,
                "m2L2_emergent": float(m2),
                "m2L2_error": 0.0,
                "m2L2_method": m2_method,
                "method_is_suspicious": is_suspicious,
                "method_is_valid_holographic": is_valid
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    metadata["total_operators"] = len(df)
    metadata["methods_m2L2"] = list(metadata["methods_m2L2"])
    metadata["suspicious_methods_found"] = list(metadata["suspicious_methods_found"])
    metadata["valid_methods_found"] = list(metadata["valid_methods_found"])
    
    return df, metadata


def load_from_legacy_systems(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga datos desde formato legacy con systems[]."""
    records = []
    metadata = {
        "format": "legacy_systems",
        "systems": [],
        "total_operators": 0,
        "methods_m2L2": set(),
        "suspicious_methods_found": set(),
        "valid_methods_found": set()
    }
    
    for system in data.get("systems", []):
        sys_name = system["name"]
        d = system["d"]
        source = system.get("source", "unknown")
        
        system_metadata = {
            "name": sys_name,
            "d": d,
            "source": source,
            "n_operators": len(system.get("operators", [])),
            "has_suspicious_methods": False
        }
        
        for op in system.get("operators", []):
            m2_method = op.get("m2L2_method", "unknown").lower()
            
            is_suspicious = any(sus in m2_method for sus in SUSPICIOUS_METHODS)
            is_valid = any(val in m2_method for val in VALID_HOLOGRAPHIC_METHODS)
            
            if is_suspicious:
                metadata["suspicious_methods_found"].add(m2_method)
                system_metadata["has_suspicious_methods"] = True
            
            if is_valid:
                metadata["valid_methods_found"].add(m2_method)
            
            record = {
                "system": sys_name,
                "d": d,
                "source": source,
                "operator": op.get("name", "unknown"),
                "Delta": float(op["Delta"]),
                "Delta_error": float(op.get("Delta_error", 0.0)),
                "m2L2_emergent": float(op["m2L2_emergent"]),
                "m2L2_error": float(op.get("m2L2_error", 0.0)),
                "m2L2_method": m2_method,
                "method_is_suspicious": is_suspicious,
                "method_is_valid_holographic": is_valid
            }
            records.append(record)
            metadata["methods_m2L2"].add(m2_method)
        
        metadata["systems"].append(system_metadata)
    
    df = pd.DataFrame(records)
    metadata["total_operators"] = len(df)
    metadata["methods_m2L2"] = list(metadata["methods_m2L2"])
    metadata["suspicious_methods_found"] = list(metadata["suspicious_methods_found"])
    metadata["valid_methods_found"] = list(metadata["valid_methods_found"])
    
    return df, metadata


def load_emergent_data(input_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga datos emergentes del pipeline Fase XI con deteccion automatica de formato."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    input_format = detect_input_format(data)
    print(f"\n   Formato detectado: {input_format}")
    
    if input_format == "dictionary_v3":
        df, metadata = load_from_dictionary_v3(data)
    else:
        df, metadata = load_from_legacy_systems(data)
    
    # Validaciones comunes
    if len(df) == 0:
        raise ValueError("No se cargaron datos. Verificar archivo de entrada.")
    
    n_suspicious = df["method_is_suspicious"].sum() if "method_is_suspicious" in df.columns else 0
    n_valid = df["method_is_valid_holographic"].sum() if "method_is_valid_holographic" in df.columns else 0
    
    print(f"\n   ANALISIS DE CALIDAD DE DATOS:")
    print(f"   - Operadores totales: {len(df)}")
    print(f"   - Sistemas: {len(metadata['systems'])}")
    print(f"   - Metodos unicos encontrados: {metadata['methods_m2L2']}")
    print(f"   - Operadores con metodos validos (holograficos): {n_valid}")
    print(f"   - Operadores con metodos sospechosos: {n_suspicious}")
    
    if n_suspicious > 0:
        warnings.warn(
            f"   ALERTA DE CONTAMINACION: Se encontraron {n_suspicious} operadores "
            f"con metodos sospechosos: {metadata['suspicious_methods_found']}",
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
            print(f"   - Filtrados {original_len - filtered_len} puntos con metodos sospechosos")
    
    df_clean = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df_clean) < 6:
        raise ValueError(f"Datos insuficientes despues de limpieza: {len(df_clean)} muestras")
    
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
    print(f"   - Caracteristicas: {config.features}")
    print(f"   - Rango Delta: [{df_clean['Delta'].min():.3f}, {df_clean['Delta'].max():.3f}]")
    print(f"   - Rango m²L²: [{df_clean['m2L2_emergent'].min():.3f}, {df_clean['m2L2_emergent'].max():.3f}]")
    
    return X_train, y_train, X_test, y_test, test_df


def discover_emergent_relation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: DiscoveryConfig,
    use_minimal_ops: bool = False
) -> Optional[PySRRegressor]:
    """Usa PySR para descubrir relacion simbolica emergente."""
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
    model.fit(X_train, y_train)
    
    return model


def evaluate_equation(
    equation_str: str,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Tuple[str, ...]
) -> Dict[str, float]:
    """Evalua una ecuacion descubierta."""
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


def compare_with_theory(
    best_eq: Dict[str, Any],
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Tuple[str, ...]
) -> Dict[str, Any]:
    """Comparacion a posteriori con formula teorica."""
    theoretical_predictions = []
    for row in X:
        Delta = row[0]
        d = row[1]
        theoretical_predictions.append(Delta * (Delta - d))
    
    theoretical_predictions = np.array(theoretical_predictions)
    theory_r2 = r2_score(y_true, theoretical_predictions)
    theory_mae = mean_absolute_error(y_true, theoretical_predictions)
    
    eq_metrics = evaluate_equation(best_eq["equation"], X, y_true, feature_names)
    
    return {
        "theoretical_formula": "m²L² = Delta(Delta - d)",
        "discovered_formula": best_eq["equation"],
        "theory_r2": float(theory_r2),
        "theory_mae": float(theory_mae),
        "discovered_r2": eq_metrics.get("r2", -np.inf),
        "compatible": abs(theory_r2 - eq_metrics.get("r2", -np.inf)) < 0.1 if eq_metrics.get("success") else False,
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
    
    # Comparacion con teoria
    theory_comparison = compare_with_theory(
        {"equation": str(best_eq["equation"])},
        X_test, y_test, config.features
    )
    
    # Construir resumen
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "input_metadata": {
            "format": metadata.get("format", "unknown"),
            "version": metadata.get("version", "unknown"),
            "mass_source": metadata.get("mass_source", "unknown"),
            "total_operators": metadata.get("total_operators", 0),
            "systems_count": len(metadata.get("systems", [])),
            "methods_found": metadata.get("methods_m2L2", []),
            "suspicious_methods": metadata.get("suspicious_methods_found", []),
            "valid_methods": metadata.get("valid_methods_found", [])
        },
        "data_stats": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "Delta_range": [float(X_test[:, 0].min()), float(X_test[:, 0].max())],
            "m2L2_range": [float(y_test.min()), float(y_test.max())]
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
        "pareto_front": []
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
    summary_file = output_dir / "fase12c_discovery_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    if config.save_raw_equations and equations_df is not None:
        eq_file = output_dir / "fase12c_all_equations.csv"
        equations_df.to_csv(eq_file, index=False)
    
    print(f"\n   RESULTADOS GUARDADOS:")
    print(f"   - Directorio: {output_dir}")
    print(f"   - Mejor ecuacion: {best_eq['equation']}")
    print(f"   - R² en test: {test_metrics.get('r2', 'N/A'):.4f}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="FASE XII.c: Diccionario Emergente")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Archivo JSON de entrada (diccionario v3 o formato legacy)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directorio de salida para resultados")
    parser.add_argument("--ops-minimal", action="store_true",
                        help="Usar operadores minimos (+,-,*,/,square,sqrt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Numero de iteraciones de PySR")
    parser.add_argument("--no-filter-suspicious", action="store_true",
                        help="No filtrar operadores con metodos sospechosos")
    parser.add_argument("--force-continue", action="store_true",
                        help="Continuar aunque se detecten metodos sospechosos")
    
    args = parser.parse_args()
    
    config = DiscoveryConfig(random_state=args.seed, niterations=args.iterations)
    
    print("=" * 80)
    print("FASE XII.c - DICCIONARIO HOLOGRAFICO EMERGENTE")
    print("=" * 80)
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    
    print(f"\n   Archivo de entrada: {input_path}")
    
    df, metadata = load_emergent_data(input_path)
    
    n_suspicious = len(metadata.get("suspicious_methods_found", []))
    if n_suspicious > 0 and not args.force_continue:
        print(f"\n   Se encontraron {n_suspicious} metodos sospechosos.")
        response = input("   Continuar? (s/N): ")
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
    
    output_dir = Path(args.output_dir)
    summary = save_results(model, config, X_train, y_train, X_test, y_test,
                          test_df, metadata, output_dir, args.ops_minimal)
    
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    best_eq = summary["discovery_results"]["best_equation"]
    test_metrics = summary["discovery_results"]["test_metrics"]
    theory_comp = summary["theory_comparison"]
    
    print(f"\n   ECUACION DESCUBIERTA: {best_eq}")
    print(f"\n   METRICAS EN TEST:")
    print(f"   - R²: {test_metrics.get('r2', 'N/A'):.4f}")
    print(f"   - MAE: {test_metrics.get('mae', 'N/A'):.4f}")
    print(f"   - Pearson: {test_metrics.get('pearson', 'N/A'):.4f}")
    print(f"\n   COMPARACION CON TEORIA:")
    print(f"   - Formula teorica: {theory_comp.get('theoretical_formula', 'N/A')}")
    print(f"   - R² teorico: {theory_comp.get('theory_r2', 'N/A'):.4f}")
    print(f"   - Compatible: {theory_comp.get('compatible', 'N/A')}")
    
    print(f"\n   Resultados en: {output_dir.absolute()}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
