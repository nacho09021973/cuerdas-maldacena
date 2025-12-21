#!/usr/bin/env python3
"""
boundary_delta_extractor.py
CUERDAS — Extracción honesta de Δ desde correladores de frontera

OBJETIVO
    Extraer dimensiones conformes Δ de operadores CFT a partir de
    correladores de dos puntos G2(x) en el límite UV (x pequeño).

FÍSICA
    En una CFT, el correlador de dos puntos de un operador primario O
    con dimensión conforme Δ va como:
    
        <O(x) O(0)> = G2(x) ~ A · x^(-2Δ)
    
    Esto es la DEFINICIÓN del correlador, no una fórmula de diccionario.
    Es una medición directa de Δ desde datos de frontera.

USO
    Este módulo es llamado por 06_build_bulk_eigenmodes_dataset.py
    cuando detecta correladores en el grupo boundary/ del HDF5.

HONESTIDAD
    - No se usa la fórmula m²L² = Δ(Δ-d) aquí.
    - Solo se extrae Δ del decaimiento del correlador.
    - El resultado es una medición, no una imposición teórica.
    - La comparación con valores de referencia (bootstrap, etc.) se hace
      en scripts de análisis/contratos (08/09), NO aquí.

DEPENDENCIAS
    - numpy (requerido)
    - h5py (requerido)
    - NO scipy (eliminado para compliance con contratos de dependencias)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@dataclass
class DeltaExtraction:
    """Resultado de extracción de Δ desde un correlador."""
    operator_name: str
    Delta: float
    Delta_error: float
    amplitude: float
    r_squared: float
    n_points_used: int
    x_range_used: Tuple[float, float]
    quality: str  # "good", "marginal", "unreliable"
    method: str   # "power_law_fit_loglog"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a dict para serialización JSON."""
        return {
            "operator_name": self.operator_name,
            "Delta": self.Delta,
            "Delta_error": self.Delta_error,
            "amplitude": self.amplitude,
            "r_squared": self.r_squared,
            "n_points_used": self.n_points_used,
            "x_range_used": list(self.x_range_used),
            "quality": self.quality,
            "method": self.method,
        }


def extract_delta_from_correlator(
    x: np.ndarray,
    G2: np.ndarray,
    operator_name: str = "unknown",
    x_max_for_fit: Optional[float] = None,
    min_points: int = 5,
    expected_Delta_range: Tuple[float, float] = (0.1, 10.0),
) -> Optional[DeltaExtraction]:
    """
    Extrae Δ ajustando G2(x) ~ A * x^(-2Δ) en la región UV (x pequeño).
    
    Método: ajuste lineal en log-log usando numpy.polyfit (sin scipy).
    
    Args:
        x: Array de posiciones (distancias)
        G2: Array de valores del correlador G2(x)
        operator_name: Nombre del operador (para reporting)
        x_max_for_fit: Usar solo x < x_max_for_fit para el ajuste (default: 30% del rango)
        min_points: Mínimo de puntos para ajuste válido
        expected_Delta_range: Rango esperado de Δ para validación
    
    Returns:
        DeltaExtraction o None si falla
    """
    # Validar inputs
    if len(x) != len(G2):
        return None
    if len(x) < min_points:
        return None
    
    # Ordenar por x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    G2 = G2[sort_idx]
    
    # Filtrar valores no válidos
    valid_mask = (x > 0) & (G2 > 0) & np.isfinite(x) & np.isfinite(G2)
    x = x[valid_mask]
    G2 = G2[valid_mask]
    
    if len(x) < min_points:
        return None
    
    # Determinar región UV para ajuste
    if x_max_for_fit is None:
        # Usar el 30% inferior del rango de x
        x_max_for_fit = x.min() + 0.3 * (x.max() - x.min())
    
    uv_mask = x <= x_max_for_fit
    x_uv = x[uv_mask]
    G2_uv = G2[uv_mask]
    
    if len(x_uv) < min_points:
        # Si no hay suficientes puntos, usar todos
        x_uv = x
        G2_uv = G2
        x_max_for_fit = x.max()
    
    n_points = len(x_uv)
    
    # Ajuste lineal en log-log usando numpy (sin scipy)
    try:
        log_x = np.log(x_uv)
        log_G2 = np.log(G2_uv)
        
        # Ajuste lineal: log(G2) = log(A) - 2*Delta*log(x)
        # Pendiente = -2*Delta
        coeffs = np.polyfit(log_x, log_G2, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        Delta_linear = -slope / 2.0
        A_linear = np.exp(intercept)
        
        # Calcular R² del ajuste lineal
        log_G2_pred = intercept + slope * log_x
        ss_res = np.sum((log_G2 - log_G2_pred) ** 2)
        ss_tot = np.sum((log_G2 - np.mean(log_G2)) ** 2)
        r2_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Error estimado de Delta (propagación simplificada desde residuos)
        residuals = log_G2 - log_G2_pred
        mse = np.mean(residuals ** 2)
        Delta_error = np.sqrt(mse) / 2.0
        
    except Exception:
        return None
    
    # Validar resultado
    Delta = Delta_linear
    amplitude = A_linear
    r_squared = r2_linear
    
    # Determinar calidad
    if r_squared > 0.95 and expected_Delta_range[0] <= Delta <= expected_Delta_range[1]:
        quality = "good"
    elif r_squared > 0.8 and expected_Delta_range[0] <= Delta <= expected_Delta_range[1]:
        quality = "marginal"
    else:
        quality = "unreliable"
    
    return DeltaExtraction(
        operator_name=operator_name,
        Delta=float(Delta),
        Delta_error=float(Delta_error),
        amplitude=float(amplitude),
        r_squared=float(r_squared),
        n_points_used=n_points,
        x_range_used=(float(x_uv.min()), float(x_uv.max())),
        quality=quality,
        method="power_law_fit_loglog",
    )


def extract_deltas_from_hdf5(
    h5_path: Path,
    boundary_group: str = "boundary",
    x_dataset: str = "x_grid",
    correlator_prefix: str = "G2_",
    x_max_for_fit: Optional[float] = None,
) -> Dict[str, DeltaExtraction]:
    """
    Extrae Δ de todos los correladores G2_* en el grupo boundary/ de un HDF5.
    
    Args:
        h5_path: Ruta al archivo HDF5
        boundary_group: Nombre del grupo con datos de frontera
        x_dataset: Nombre del dataset con posiciones x
        correlator_prefix: Prefijo de los datasets de correladores
        x_max_for_fit: x máximo para ajuste UV
    
    Returns:
        Dict mapping operator_name -> DeltaExtraction
    """
    if not HAS_H5PY:
        raise ImportError("h5py no disponible")
    
    results: Dict[str, DeltaExtraction] = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Verificar que existe el grupo boundary
        if boundary_group not in f:
            return results
        
        bnd = f[boundary_group]
        
        # Obtener x_grid
        if x_dataset not in bnd:
            return results
        
        x = np.array(bnd[x_dataset])
        
        # Buscar correladores G2_*
        for key in bnd.keys():
            if key.startswith(correlator_prefix) and key != x_dataset:
                operator_name = key[len(correlator_prefix):]  # "G2_sigma" -> "sigma"
                G2 = np.array(bnd[key])
                
                extraction = extract_delta_from_correlator(
                    x=x,
                    G2=G2,
                    operator_name=operator_name,
                    x_max_for_fit=x_max_for_fit,
                )
                
                if extraction is not None:
                    results[operator_name] = extraction
    
    return results


def get_delta_for_eigenmode(
    extractions: Dict[str, DeltaExtraction],
    mode_id: int,
    operator_order: Optional[List[str]] = None,
) -> Tuple[Optional[float], str]:
    """
    Mapea un mode_id a un Δ extraído.
    
    HEURÍSTICA (documentar en meta):
    Por defecto, ordena operadores por Δ creciente:
    - mode_id=0 → operador con menor Δ (relevante, ej. sigma)
    - mode_id=1 → siguiente operador (ej. epsilon)
    
    Para evitar sesgo, se recomienda pasar operator_order explícito
    cuando se conozca la correspondencia física.
    
    Args:
        extractions: Dict de DeltaExtraction por operador
        mode_id: Índice del modo (0-indexed)
        operator_order: Orden explícito de operadores (opcional)
    
    Returns:
        (Delta, quality_flag) o (None, "no_extraction")
    """
    if not extractions:
        return None, "no_extraction"
    
    # Ordenar por Delta si no se especifica orden
    if operator_order is None:
        sorted_ops = sorted(extractions.items(), key=lambda kv: kv[1].Delta)
    else:
        sorted_ops = [(op, extractions[op]) for op in operator_order if op in extractions]
    
    if mode_id >= len(sorted_ops):
        return None, "mode_id_out_of_range"
    
    op_name, extraction = sorted_ops[mode_id]
    
    if extraction.quality == "unreliable":
        return extraction.Delta, "uv_unreliable"
    elif extraction.quality == "marginal":
        return extraction.Delta, "uv_marginal"
    else:
        return extraction.Delta, "ok"


def get_extraction_metadata(
    extractions: Dict[str, DeltaExtraction],
    operator_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Genera metadata para auditoría del mapping mode_id → operator → Delta.
    
    Incluye:
    - Criterio de ordenación usado
    - Mapping explícito mode_id → operator_name → Delta
    - Calidad de cada extracción
    
    Esta metadata debe guardarse en bulk_modes_meta.json para trazabilidad.
    """
    if operator_order is None:
        sorted_ops = sorted(extractions.items(), key=lambda kv: kv[1].Delta)
        ordering_criterion = "sorted_by_Delta_ascending"
    else:
        sorted_ops = [(op, extractions[op]) for op in operator_order if op in extractions]
        ordering_criterion = "explicit_operator_order"
    
    mapping = {}
    for mode_id, (op_name, ext) in enumerate(sorted_ops):
        mapping[str(mode_id)] = {
            "operator_name": op_name,
            "Delta": ext.Delta,
            "Delta_error": ext.Delta_error,
            "quality": ext.quality,
            "r_squared": ext.r_squared,
        }
    
    return {
        "ordering_criterion": ordering_criterion,
        "operator_order_used": [op for op, _ in sorted_ops],
        "mode_to_operator_mapping": mapping,
        "extraction_method": "power_law_fit_loglog",
        "note": "Mapping heurístico: mode_id asignado por Delta creciente si no se especifica orden",
    }


# =============================================================================
# CLI para testing (diagnóstico, NO para pipeline)
# =============================================================================

def main():
    """
    CLI de diagnóstico. NO es parte del pipeline principal.
    Para comparación con valores de referencia, usar scripts de análisis (08/09).
    """
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="[DIAGNÓSTICO] Extraer Δ desde correladores de frontera"
    )
    parser.add_argument("h5_file", type=str, help="Archivo HDF5 con grupo boundary/")
    parser.add_argument("--x-max", type=float, default=None, help="x máximo para ajuste UV")
    parser.add_argument("--output-json", type=str, default=None, help="Guardar resultados en JSON")
    args = parser.parse_args()
    
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"ERROR: No existe {h5_path}")
        return 1
    
    print("=" * 60)
    print(f"EXTRACCIÓN DE Δ DESDE CORRELADORES DE FRONTERA")
    print(f"Archivo: {h5_path.name}")
    print("=" * 60)
    
    extractions = extract_deltas_from_hdf5(h5_path, x_max_for_fit=args.x_max)
    
    if not extractions:
        print("\nNo se encontraron correladores G2_* en boundary/")
        return 1
    
    print(f"\nOperadores encontrados: {len(extractions)}")
    print("-" * 60)
    
    for op_name, ext in sorted(extractions.items(), key=lambda kv: kv[1].Delta):
        print(f"\n  {op_name}:")
        print(f"    Δ = {ext.Delta:.4f} ± {ext.Delta_error:.4f}")
        print(f"    R² = {ext.r_squared:.4f}")
        print(f"    Calidad: {ext.quality}")
        print(f"    x_range: [{ext.x_range_used[0]:.3f}, {ext.x_range_used[1]:.3f}]")
        print(f"    n_points: {ext.n_points_used}")
    
    # Mostrar metadata de mapping
    meta = get_extraction_metadata(extractions)
    print(f"\n" + "=" * 60)
    print("METADATA DE MAPPING (para auditoría)")
    print("=" * 60)
    print(f"  Criterio: {meta['ordering_criterion']}")
    print(f"  Orden: {meta['operator_order_used']}")
    
    if args.output_json:
        output = {
            "source": str(h5_path),
            "method": "boundary_correlator_fit",
            "operators": {
                op: ext.to_dict() for op, ext in extractions.items()
            },
            "mapping_metadata": meta,
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"\nResultados guardados en: {args.output_json}")
    
    print("\n" + "=" * 60)
    print("NOTA: Para comparar con valores de referencia (bootstrap, etc.),")
    print("usar scripts de análisis/contratos (08/09).")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
