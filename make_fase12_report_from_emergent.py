#!/usr/bin/env python3
"""
make_fase12_report_from_emergent.py

Construye un reporte tipo Fase XII para Ising 3D a partir de:

  - bulk_modes_dataset_ising.json (salida de 06_build_bulk_eigenmodes_dataset.py)
  - summary del diccionario emergente (salida de 07_emergent_lambda_sl_dictionary.py)
  - descriptor real opcional (ising3d_descriptor.json)

Objetivo:
  Generar un JSON con la misma forma lógica que fase12_report.json, pero con
  dictionary_source != "manual" y operators_predicted rellenos usando el
  diccionario emergente λ_SL ↔ Δ.

Este reporte es consumible por:
  09_real_data_and_dictionary_contracts.py --phase 12 --fase12-report ...
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np


# ============================================================
# UTILIDADES
# ============================================================

def safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el fichero JSON: {path}")
    return json.loads(path.read_text())


def pick_system_from_bulk_modes(bulk_data: Dict[str, Any],
                                preferred_geometry: Optional[str] = None) -> Dict[str, Any]:
    """
    Selecciona la entrada relevante de 'systems' en bulk_modes_dataset_ising.json.

    Estrategia:
      - Si preferred_geometry está definido, buscar por geometry_name.
      - Si no, usar el primer sistema.
    """
    systems: List[Dict[str, Any]] = bulk_data.get("systems", [])
    if not systems:
        raise ValueError("bulk_modes_dataset no contiene sistemas en 'systems'")

    if preferred_geometry:
        for sys in systems:
            if sys.get("geometry_name") == preferred_geometry:
                return sys

    # Fallback: primer sistema
    return systems[0]


def extract_lambda_list(sys_entry: Dict[str, Any]) -> List[float]:
    """
    Extrae la lista de λ_SL desde la entrada de sistema del dataset bulk.

    Intentamos varias claves por compatibilidad:
      - 'lambda_sl_bulk' (nomenclatura v2)
      - 'lambda_sl' (alternativo)
      - 'm2L2_emergent' (legacy, por si acaso)
    """
    for key in ["lambda_sl_bulk", "lambda_sl", "lambda_sl_emergent", "m2L2_emergent"]:
        if key in sys_entry:
            values = sys_entry[key]
            if isinstance(values, list) and values:
                return [float(v) for v in values]
    raise ValueError(
        "No se encontraron claves de λ_SL reconocibles en la entrada del sistema "
        "(intentado: lambda_sl_bulk, lambda_sl, lambda_sl_emergent, m2L2_emergent)"
    )


def eval_equation_scalar(equation_str: str, Delta: float, d: float) -> float:
    """
    Evalúa la ecuación descubierta por PySR en un punto (Delta, d).

    NOTA:
      - En 07_emergent_lambda_sl_dictionary.py se usa PySR sin feature_names,
        con lo que la ecuación está en términos de x0, x1, ...
      - Por convención:
          x0 -> Delta
          x1 -> d
    """
    env = {
        "x0": float(Delta),
        "x1": float(d),
        # funciones unarias usadas en 07
        "square": lambda x: x * x,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
    }
    # Desactivar builtins por seguridad
    return float(eval(equation_str, {"__builtins__": {}}, env))


def invert_lambda_to_delta_grid(
    equation_str: str,
    lambda_obs: float,
    d: float,
    delta_min: float,
    delta_max: float,
    n_grid: int = 512,
) -> Tuple[float, float]:
    """
    Inversión numérica simple: dado λ_obs y d, busca Δ que minimiza
    |λ_pred(Δ, d) - λ_obs| en una rejilla uniforme de Δ.

    Devuelve:
      - delta_best: Δ_predicho
      - residual: |λ_pred(delta_best) - λ_obs|
    """
    grid = np.linspace(delta_min, delta_max, n_grid)
    lambdas_pred = np.zeros_like(grid)

    for i, Delta in enumerate(grid):
        try:
            lambdas_pred[i] = eval_equation_scalar(equation_str, float(Delta), float(d))
        except Exception:
            lambdas_pred[i] = np.inf

    diffs = np.abs(lambdas_pred - float(lambda_obs))
    idx = int(np.argmin(diffs))
    return float(grid[idx]), float(diffs[idx])


def guess_delta_range(dictionary_summary: Dict[str, Any]) -> Tuple[float, float]:
    """
    Intenta extraer el rango de Δ desde los data_stats del diccionario.
    Si no lo encuentra, usa un rango razonable por defecto.
    """
    data_stats = dictionary_summary.get("data_stats", {})
    delta_range = data_stats.get("Delta_range")
    if isinstance(delta_range, (list, tuple)) and len(delta_range) == 2:
        return float(delta_range[0]), float(delta_range[1])
    # Fallback conservador
    return 0.1, 5.0


def infer_dimension_from_sources(
    descriptor: Optional[Dict[str, Any]],
    bulk_sys_entry: Dict[str, Any]
) -> int:
    """
    Obtiene d (dimensión CFT) del descriptor si es posible, si no del dataset bulk.
    """
    # 1) descriptor
    if descriptor:
        for key in ["d", "dimension", "spacetime_dimension"]:
            if key in descriptor:
                try:
                    return int(descriptor[key])
                except Exception:
                    pass

    # 2) dataset bulk
    d_val = bulk_sys_entry.get("d", 3)
    try:
        return int(d_val)
    except Exception:
        return 3


def load_real_descriptor(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] Descriptor real no encontrado, se continúa sin él: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[WARN] Error al leer descriptor real {path}: {e}")
        return None


# ============================================================
# CONSTRUCCIÓN DEL REPORTE FASE XII EMERGENTE
# ============================================================

def build_ising3d_emergent_report(
    bulk_modes_path: Path,
    dictionary_summary_path: Path,
    output_path: Path,
    descriptor_path: Optional[Path] = None,
    system_name: str = "ising3d_bootstrap",
    max_modes: int = 4,
) -> Dict[str, Any]:
    """
    Construye el reporte emergente tipo Fase XII para Ising 3D.

    Entradas:
      - bulk_modes_path: JSON de 06 (λ_SL_bulk para Ising).
      - dictionary_summary_path: summary del diccionario emergente.
      - descriptor_path: descriptor real opcional.
      - system_name: nombre lógico del sistema en el reporte.
      - max_modes: número máximo de modos a incluir.

    Salida:
      - Dict con la estructura del reporte, también escrito a output_path.
    """
    bulk_data = safe_load_json(bulk_modes_path)
    dict_summary = safe_load_json(dictionary_summary_path)
    real_desc = load_real_descriptor(descriptor_path)

    # 1. Seleccionar sistema Ising en bulk_modes
    preferred_geom_name = None
    if real_desc and "geometry_name" in real_desc:
        preferred_geom_name = real_desc["geometry_name"]

    sys_entry = pick_system_from_bulk_modes(bulk_data, preferred_geometry=preferred_geom_name)
    lambda_list = extract_lambda_list(sys_entry)

    if max_modes > 0:
        lambda_list = lambda_list[:max_modes]

    # 2. Extraer ecuación y rango de Δ del diccionario emergente
    disc_results = dict_summary.get("discovery_results", {})
    equation_str = disc_results.get("best_equation")
    if not equation_str:
        raise ValueError("El resumen del diccionario no contiene 'discovery_results.best_equation'")

    delta_min, delta_max = guess_delta_range(dict_summary)

    # 3. Determinar d (dimensión CFT) de Ising 3D
    d = infer_dimension_from_sources(real_desc, sys_entry)

    # 4. Invertir λ_SL -> Δ para cada modo
    predicted_operators: List[Dict[str, Any]] = []
    for mode_id, lam in enumerate(lambda_list):
        delta_pred, resid = invert_lambda_to_delta_grid(
            equation_str=equation_str,
            lambda_obs=float(lam),
            d=float(d),
            delta_min=delta_min,
            delta_max=delta_max,
            n_grid=512,
        )
        predicted_operators.append({
            "name": f"mode_{mode_id}",
            "Delta": delta_pred,
            "lambda_sl": float(lam),
            "mode_id": mode_id,
            "residual_lambda": resid,
            "inversion_method": "grid_search_v1",
        })

    # 5. Construir campos principales del sistema
    #    - Si hay descriptor, usar sus metadatos cuando se reconozcan
    src = "bootstrap"
    T = 0.0
    z_h = 0.0
    z_dyn = 1.0
    predicted_family = sys_entry.get("family", "ads_like_unknown")

    if real_desc:
        src = real_desc.get("source", src)
        T = float(real_desc.get("T", T))
        z_h = float(real_desc.get("z_h", z_h))
        z_dyn = float(real_desc.get("z_dyn", z_dyn))
        predicted_family = real_desc.get("predicted_family", predicted_family)

    geometry_block = {
        "predicted_family": predicted_family,
        "z_h": z_h,
        "z_dyn": z_dyn,
        # Podemos dejar operators_predicted vacío: 09 usa primero el diccionario.
        "operators_predicted": [],
    }

    dictionary_block = {
        "provenance": "lambda_sl_emergent_v1",
        "operators_predicted": predicted_operators,
    }

    system_block = {
        "name": system_name,
        "source": src,
        "d": int(d),
        "T": T,
        "geometry": geometry_block,
        "dictionary": dictionary_block,
        "dictionary_source": "emergent_lambda_sl_2025_dec",
    }

    # Podemos añadir algo de trazabilidad extra opcional
    system_block["metadata"] = {
        "bulk_modes_file": str(bulk_modes_path),
        "dictionary_summary_file": str(dictionary_summary_path),
        "real_descriptor_file": str(descriptor_path) if descriptor_path else None,
        "n_modes_used": len(lambda_list),
        "equation_str": equation_str,
        "delta_search_range": [delta_min, delta_max],
    }

    report = {
        "phase": 12,
        "description": (
            "Ising 3D — reporte emergente: λ_SL_bulk de geometría emergente + "
            "diccionario λ_SL↔Δ descubierto en sandbox. "
            "dictionary_source='emergent_lambda_sl_2025_dec'."
        ),
        "systems": [system_block],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 70)
    print("REPORTE FASE XII EMERGENTE (Ising 3D) GENERADO")
    print("=" * 70)
    print(f"  λ_SL bulk     : {bulk_modes_path}")
    print(f"  Diccionario   : {dictionary_summary_path}")
    if descriptor_path:
        print(f"  Descriptor    : {descriptor_path}")
    print(f"  Output report : {output_path}")
    print("=" * 70)

    return report


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construye un reporte Fase XII emergente para Ising 3D "
                    "a partir de λ_SL_bulk y diccionario λ_SL↔Δ."
    )
    p.add_argument(
        "--bulk-modes",
        type=str,
        required=True,
        help="Ruta al JSON de modos bulk (salida de 06_build_bulk_eigenmodes_dataset.py)",
    )
    p.add_argument(
        "--dictionary-summary",
        type=str,
        required=True,
        help="Resumen del diccionario emergente (salida de 07_emergent_lambda_sl_dictionary.py)",
    )
    p.add_argument(
        "--real-descriptor",
        type=str,
        default=None,
        help="Descriptor real opcional (ising3d_descriptor.json u otro)",
    )
    p.add_argument(
        "--system-name",
        type=str,
        default="ising3d_bootstrap",
        help="Nombre lógico del sistema en el reporte",
    )
    p.add_argument(
        "--max-modes",
        type=int,
        default=4,
        help="Número máximo de modos λ_SL a usar para predicciones de Δ",
    )
    p.add_argument(
        "--output-file",
        type=str,
        default="runs/fase12_ising_real/fase12/predictions/ising3d_emergent_report.json",
        help="Ruta del fichero de salida (reporte Fase XII emergente)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bulk_modes_path = Path(args.bulk_modes)
    dictionary_summary_path = Path(args.dictionary_summary)
    output_path = Path(args.output_file)
    descriptor_path = Path(args.real_descriptor) if args.real_descriptor else None

    build_ising3d_emergent_report(
        bulk_modes_path=bulk_modes_path,
        dictionary_summary_path=dictionary_summary_path,
        output_path=output_path,
        descriptor_path=descriptor_path,
        system_name=args.system_name,
        max_modes=args.max_modes,
    )


if __name__ == "__main__":
    main()
