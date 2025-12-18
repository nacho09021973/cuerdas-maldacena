#!/usr/bin/env python3
"""
fase12_ising_adapter.py — Adaptador específico para Ising 3D Bootstrap

Procesa datos del paper "Precision Islands in the Ising and O(N) Models"
(arXiv:1603.04436) para usar en el pipeline CUERDAS.

FLUJO:
    ising3d_bootstrap_raw.json → este adaptador → BoundaryDataStandard → Motor XI/XII
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import h5py


@dataclass
class Ising3DBootstrapData:
    """Datos del bootstrap del Ising 3D en formato interno CUERDAS."""
    
    # Identificación
    name: str = "ising3d_bootstrap"
    family: str = "cft_bootstrap"
    source: str = "arXiv:1603.04436"
    d: int = 3  # Dimensión del CFT (boundary)
    
    # Operadores primarios
    operators: List[Dict] = field(default_factory=list)
    
    # OPE coefficients
    ope_coefficients: Dict[str, float] = field(default_factory=dict)
    
    # Exponentes críticos derivados
    critical_exponents: Dict[str, float] = field(default_factory=dict)
    
    # Central charge (si disponible)
    c_T: Optional[float] = None
    
    # Features para el motor de predicción
    features: Dict[str, float] = field(default_factory=dict)
    
    # Correladores sintéticos (generados desde espectro)
    x_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    G2_data: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Metadata original
    raw_metadata: Dict = field(default_factory=dict)


def load_ising3d_bootstrap_raw(json_path: Path) -> Dict:
    """
    Carga el JSON raw del bootstrap Ising 3D.
    
    Formato esperado (Precision Islands):
    {
        "name": "ising3d_bootstrap",
        "d": 3,
        "operators": [
            {"label": "sigma", "Delta": 0.5181489, ...},
            {"label": "epsilon", "Delta": 1.412625, ...}
        ],
        "ope_coefficients": [
            {"structure": "sigma-sigma-epsilon", "lambda": 1.0518537, ...}
        ],
        ...
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def to_internal_ising3d(raw_data: Dict) -> Ising3DBootstrapData:
    """
    Convierte datos raw del bootstrap al formato interno CUERDAS.
    
    Este es el corazón del adaptador: traduce el esquema del paper
    al esquema que entiende el motor de predicción.
    """
    result = Ising3DBootstrapData()
    
    # Metadatos básicos
    result.name = raw_data.get("name", "ising3d_bootstrap")
    result.source = raw_data.get("reference", "arXiv:1603.04436")
    result.d = raw_data.get("d", 3)
    result.raw_metadata = raw_data
    
    # Procesar operadores
    for op in raw_data.get("operators", []):
        result.operators.append({
            "name": op.get("label", op.get("name", "unknown")),
            "Delta": op["Delta"],
            "Delta_error": op.get("Delta_error", 0.0),
            "spin": op.get("spin", 0),
            "Z2_parity": op.get("Z2_parity", 1),
            "is_relevant": op.get("is_relevant", op["Delta"] < result.d)
        })
    
    # Procesar OPE coefficients
    for ope in raw_data.get("ope_coefficients", []):
        key = ope.get("notation", ope.get("structure", "unknown"))
        result.ope_coefficients[key] = ope["lambda"]
    
    # Exponentes críticos
    if "critical_exponents" in raw_data:
        for name, exp_data in raw_data["critical_exponents"].items():
            if isinstance(exp_data, dict):
                result.critical_exponents[name] = exp_data["value"]
            else:
                result.critical_exponents[name] = exp_data
    
    # Central charge
    if "central_charge" in raw_data:
        cc = raw_data["central_charge"]
        if isinstance(cc, dict):
            result.c_T = cc.get("c_T_normalized")
        else:
            result.c_T = cc
    
    # Extraer features para el motor de predicción
    result.features = _extract_features(result)
    
    # Generar correladores sintéticos desde el espectro
    result.x_grid, result.G2_data = _generate_correlators(result)
    
    return result


def _extract_features(data: Ising3DBootstrapData) -> Dict[str, float]:
    """
    Extrae features numéricos para el motor de predicción.
    
    Estos features son los que el motor XI/XII usa para
    clasificar geometrías y hacer predicciones.
    """
    features = {}
    
    # Dimensiones de los operadores principales
    if data.operators:
        Deltas = sorted([op["Delta"] for op in data.operators])
        features["Delta_min"] = Deltas[0]
        if len(Deltas) > 1:
            features["Delta_gap"] = Deltas[1] - Deltas[0]
            features["Delta_ratio"] = Deltas[1] / Deltas[0] if Deltas[0] > 0 else 0
        
        # Operadores relevantes (Delta < d)
        features["n_relevant"] = sum(1 for D in Deltas if D < data.d)
        features["n_irrelevant"] = len(Deltas) - features["n_relevant"]
    
    # Para Ising 3D: Delta_sigma y Delta_epsilon específicos
    for op in data.operators:
        if op["name"] == "sigma":
            features["Delta_sigma"] = op["Delta"]
        elif op["name"] == "epsilon":
            features["Delta_epsilon"] = op["Delta"]
    
    # OPE coefficients
    if data.ope_coefficients:
        ope_vals = list(data.ope_coefficients.values())
        features["n_ope"] = len(ope_vals)
        if len(ope_vals) >= 2:
            features["ope_ratio"] = ope_vals[0] / ope_vals[1] if ope_vals[1] > 0 else 0
        
        # Guardar OPEs específicos
        for key, val in data.ope_coefficients.items():
            features[f"ope_{key}"] = val
    
    # Exponentes críticos
    for name, val in data.critical_exponents.items():
        features[f"exp_{name}"] = val
    
    # Central charge
    if data.c_T is not None:
        features["c_T"] = data.c_T
    
    # Dimensión del espacio
    features["d_boundary"] = data.d
    features["d_bulk"] = data.d + 1  # Bulk es d+1 dimensional
    
    return features


def _generate_correlators(data: Ising3DBootstrapData) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Genera correladores de 2 puntos sintéticos a partir del espectro.
    
    Para un operador primario de dimensión Δ en un CFT:
        G_2(x) ∝ 1/|x|^{2Δ}
    
    Esto da al motor algo con qué trabajar aunque no tengamos
    los correladores exactos.
    """
    # Grid en distancia (unidades del cutoff UV)
    x_grid = np.logspace(-1, 2, 200)  # De 0.1 a 100
    
    G2_data = {}
    
    for op in data.operators:
        Delta = op["Delta"]
        name = op["name"]
        
        # Correlador CFT estándar
        # Añadimos regularización UV en x pequeños
        epsilon_uv = 0.01
        G2 = 1.0 / ((x_grid**2 + epsilon_uv**2) ** Delta)
        
        # Normalización: G2(1) = 1
        G2 = G2 / G2[np.argmin(np.abs(x_grid - 1.0))]
        
        G2_data[f"G2_{name}"] = G2
    
    # También generamos el correlador mezclado σ-ε si tenemos el OPE
    if "lambda_sigma_sigma_epsilon" in data.ope_coefficients:
        Delta_sigma = next((op["Delta"] for op in data.operators if op["name"] == "sigma"), 0.518)
        Delta_epsilon = next((op["Delta"] for op in data.operators if op["name"] == "epsilon"), 1.41)
        lambda_sse = data.ope_coefficients["lambda_sigma_sigma_epsilon"]
        
        # Contribución dominante al 4-punto en canal s
        # Simplificación: usamos estructura conforme básica
        z = x_grid / (x_grid + 1)  # Cross-ratio aproximado
        G4_approx = (lambda_sse ** 2) * (z ** Delta_epsilon) / ((1-z) ** (2*Delta_sigma))
        G2_data["G4_ssss_s_channel"] = G4_approx / np.nanmax(G4_approx)
    
    return x_grid, G2_data


def to_hdf5(data: Ising3DBootstrapData, output_path: Path) -> None:
    """
    Guarda los datos en formato HDF5 compatible con el motor XI.
    
    Este es el formato que lee fase12_prediction_engine.py
    """
    with h5py.File(output_path, 'w') as f:
        # Atributos globales
        f.attrs["name"] = data.name
        f.attrs["family"] = data.family
        f.attrs["source"] = data.source
        f.attrs["d"] = data.d
        
        # Operadores como JSON en atributo
        f.attrs["operators_json"] = json.dumps(data.operators)
        
        # OPE coefficients
        f.attrs["ope_json"] = json.dumps(data.ope_coefficients)
        
        # Features
        features_grp = f.create_group("features")
        for key, val in data.features.items():
            if val is not None:
                features_grp.attrs[key] = val
        
        # Correladores
        boundary = f.create_group("boundary")
        boundary.create_dataset("x_grid", data=data.x_grid)
        for name, G2 in data.G2_data.items():
            boundary.create_dataset(name, data=G2)
        
        # Metadata completa
        f.attrs["raw_metadata_json"] = json.dumps(data.raw_metadata)


def to_prediction_input(data: Ising3DBootstrapData) -> Dict[str, Any]:
    """
    Convierte a formato de entrada para fase12_prediction_engine.
    
    Este diccionario es lo que el motor espera recibir.
    """
    return {
        "name": data.name,
        "family": data.family,
        "source": data.source,
        "d": data.d,
        "observables": {
            "Delta_list": [op["Delta"] for op in data.operators],
            "labels": [op["name"] for op in data.operators],
            "spin": [op["spin"] for op in data.operators],
            "Z2_parity": [op["Z2_parity"] for op in data.operators],
            **data.ope_coefficients,
            "critical_exponents": data.critical_exponents,
            "c_T": data.c_T
        },
        "features": data.features,
        "correlators": {
            "x_grid": data.x_grid.tolist(),
            "G2": {k: v.tolist() for k, v in data.G2_data.items()}
        }
    }


def get_known_holographic_predictions() -> Dict[str, Any]:
    """
    Devuelve predicciones holográficas conocidas para el Ising 3D.
    
    Estas son las "respuestas correctas" que CUERDAS debería
    aproximar si funciona bien.
    """
    return {
        # Estructura bulk esperada
        "expected_bulk_dim": 4,  # AdS4
        "expected_family": "ads",  # No hyperscaling violation
        
        # Relación masa-dimensión holográfica: m²L² = Δ(Δ-d)
        # Para sigma (Δ=0.518, d=3): m²L² = 0.518*(0.518-3) = -1.286
        # Para epsilon (Δ=1.41, d=3): m²L² = 1.41*(1.41-3) = -2.24
        "expected_m2L2": {
            "sigma": 0.518 * (0.518 - 3),  # ≈ -1.286
            "epsilon": 1.41 * (1.41 - 3)   # ≈ -2.24
        },
        
        # ¿Deberíamos ver bulk gravitacional?
        "has_stress_tensor": True,  # CFT tiene T_μν
        "bulk_has_gravity": True,
        
        # Notas
        "notes": [
            "Ising 3D es un CFT unitario sin parámetros continuos",
            "No hay dual holográfico conocido exacto",
            "CUERDAS debería proponer geometría tipo AdS4 o deformación suave",
            "El test es: ¿m²L² emergente ≈ Δ(Δ-d)?"
        ]
    }


# ============================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================

def load_and_convert(json_path: Path) -> Tuple[Ising3DBootstrapData, Dict]:
    """Carga, convierte y devuelve datos + input para predicción."""
    raw = load_ising3d_bootstrap_raw(json_path)
    data = to_internal_ising3d(raw)
    pred_input = to_prediction_input(data)
    return data, pred_input


def process_ising3d_for_cuerdas(
    json_path: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Pipeline completo: JSON → formato CUERDAS → archivos de salida.
    
    Returns:
        Diccionario con resumen del procesamiento
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar y convertir
    if verbose:
        print(f"Cargando: {json_path}")
    
    raw = load_ising3d_bootstrap_raw(json_path)
    data = to_internal_ising3d(raw)
    
    # 2. Guardar HDF5
    h5_path = output_dir / f"{data.name}.h5"
    to_hdf5(data, h5_path)
    if verbose:
        print(f"  → HDF5: {h5_path}")
    
    # 3. Guardar JSON para prediction engine
    pred_input = to_prediction_input(data)
    json_out_path = output_dir / f"{data.name}_prediction_input.json"
    with open(json_out_path, 'w') as f:
        json.dump(pred_input, f, indent=2, default=str)
    if verbose:
        print(f"  → JSON: {json_out_path}")
    
    # 4. Guardar predicciones conocidas para validación
    known = get_known_holographic_predictions()
    known_path = output_dir / f"{data.name}_known_predictions.json"
    with open(known_path, 'w') as f:
        json.dump(known, f, indent=2)
    if verbose:
        print(f"  → Known: {known_path}")
    
    # 5. Resumen
    summary = {
        "name": data.name,
        "source": data.source,
        "d": data.d,
        "n_operators": len(data.operators),
        "n_ope": len(data.ope_coefficients),
        "n_features": len(data.features),
        "outputs": {
            "h5": str(h5_path),
            "json": str(json_out_path),
            "known": str(known_path)
        },
        "features_summary": {
            "Delta_sigma": data.features.get("Delta_sigma"),
            "Delta_epsilon": data.features.get("Delta_epsilon"),
            "Delta_gap": data.features.get("Delta_gap"),
            "n_relevant": data.features.get("n_relevant")
        }
    }
    
    if verbose:
        print(f"\n  Resumen:")
        print(f"    Operadores: {summary['n_operators']}")
        print(f"    OPEs: {summary['n_ope']}")
        print(f"    Features: {summary['n_features']}")
        print(f"    Δ_σ = {summary['features_summary']['Delta_sigma']}")
        print(f"    Δ_ε = {summary['features_summary']['Delta_epsilon']}")
    
    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Adaptador Ising 3D Bootstrap → CUERDAS"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="JSON de entrada con datos bootstrap")
    parser.add_argument("--output-dir", "-o", type=str, default="fase12_ising",
                        help="Directorio de salida")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Modo silencioso")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FASE XII — ADAPTADOR ISING 3D BOOTSTRAP")
    print("=" * 70)
    
    summary = process_ising3d_for_cuerdas(
        json_path=Path(args.input),
        output_dir=Path(args.output_dir),
        verbose=not args.quiet
    )
    
    # Guardar manifest
    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Procesamiento completo")
    print(f"  Manifest: {manifest_path}")
    print("=" * 70)
