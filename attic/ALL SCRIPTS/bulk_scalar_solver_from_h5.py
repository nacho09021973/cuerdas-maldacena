#!/usr/bin/env python3
"""
bulk_scalar_solver_from_h5.py

Wrapper alrededor de BulkScalarSolver (versión Gork) para:

  - Leer A(z), f(z) desde un .h5 de Fase XI (p.ej. bulk_truth/A_truth, f_truth).
  - Construir funciones A(z), f(z) interpoladas.
  - Ajustar m^2 para un Δ objetivo resolviendo la ecuación de KG en ese fondo.

No se usa la fórmula m^2 L^2 = Δ(Δ-d) en el camino numérico. Solo se pasa
(d, Δ) al solver, que ajusta sigma_UV a partir de la solución a la PDE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Tuple

import h5py
import numpy as np
from scipy.interpolate import interp1d

import bulk_scalar_solver_gork as gsolver  # asegúrate de que el fichero se llama así


def load_metric_from_h5(
    h5_path: str | Path,
    z_dataset: str = "bulk_truth/z_grid",
    A_dataset: str = "bulk_truth/A_truth",
    f_dataset: str = "bulk_truth/f_truth",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Carga z, A(z), f(z) desde un archivo HDF5 de Fase XI.

    Devuelve:
        z_grid (np.ndarray, creciente)
        A (np.ndarray)
        f (np.ndarray)
        d (int)
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        z = np.array(f[z_dataset][:], dtype=float)
        A = np.array(f[A_dataset][:], dtype=float)
        ff = np.array(f[f_dataset][:], dtype=float)

        d_attr = f.attrs.get("d", 4)
        try:
            d = int(d_attr)
        except Exception:
            d = 4

    if z.ndim != 1:
        raise ValueError(f"{z_dataset} en {h5_path} no es 1D")
    if not (len(z) == len(A) == len(ff)):
        raise ValueError(
            f"Tamaños inconsistentes en {h5_path}: "
            f"len(z)={len(z)}, len(A)={len(A)}, len(f)={len(ff)}"
        )

    # Asegurar orden creciente en z
    idx = np.argsort(z)
    z = z[idx]
    A = A[idx]
    ff = ff[idx]

    return z, A, ff, d

def make_interpolated_metric(
    z: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
) -> Tuple[Callable[[float | np.ndarray], np.ndarray],
           Callable[[float | np.ndarray], np.ndarray],
           float,
           float]:
    """
    Construye funciones A(z), f(z) interpoladas y devuelve también z_min/z_max
    que usaremos para el solver.

    Recorta el dominio para evitar la región donde f(z) ~ 0 (horizonte),
    porque el solver de Gork usa factores ~ 1/sqrt(f(z)).
    """
    z = np.asarray(z, dtype=float)
    A = np.asarray(A, dtype=float)
    f = np.asarray(f, dtype=float)

    # Orden creciente
    idx = np.argsort(z)
    z = z[idx]
    A = A[idx]
    f = f[idx]

    # Seleccionar solo región donde f(z) es claramente > 0
    f_min = 1e-4
    mask = f > f_min
    if mask.sum() < 10:
        raise ValueError(
            f"Solo {mask.sum()} puntos con f(z)>{f_min}; "
            "la métrica no es adecuada para este solver."
        )

    z_valid = z[mask]
    A_valid = A[mask]
    f_valid = f[mask]

    z_min_grid = float(z_valid[0])
    z_max_grid = float(z_valid[-1])

    # Evitar z=0 exacto
    if z_min_grid <= 0.0:
        if len(z_valid) > 1:
            z_min = max(float(z_valid[1]), 1e-4 * z_max_grid)
        else:
            z_min = 1e-4 * z_max_grid
    else:
        z_min = z_min_grid

    z_max = z_max_grid

    from scipy.interpolate import interp1d

    A_interp = interp1d(
        z_valid,
        A_valid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    f_interp = interp1d(
        z_valid,
        f_valid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )

    def A_func(z_eval):
        return A_interp(z_eval)

    def f_func(z_eval):
        return f_interp(z_eval)

    return A_func, f_func, z_min, z_max



def solve_m2_for_delta_from_h5(
    h5_path: str | Path,
    target_delta: float,
    d_override: int | None = None,
    L: float = 1.0,
    z_dataset: str = "bulk_truth/z_grid",
    A_dataset: str = "bulk_truth/A_truth",
    f_dataset: str = "bulk_truth/f_truth",
    num_points: int = 1000,
    bracket: Tuple[float, float] | None = None,
) -> dict:
    """
    Alto nivel: dado un .h5 y un Δ objetivo, resuelve m² emergente numéricamente.

    Args:
        h5_path: archivo .h5 de Fase XI
        target_delta: Δ objetivo (por ejemplo, el que viene de Fase XI o bootstrap)
        d_override: si se pasa, fuerza esa d; si no, se toma de attrs['d']
        L: escala de longitud; L=1 normalmente
        z_dataset, A_dataset, f_dataset: paths dentro del HDF5
        num_points: puntos del grid interno del solver
        bracket: intervalo [m2_min, m2_max] para root_scalar; si es None se usa
                 el default del BulkScalarSolver (desde cota BF hasta positivo).

    Returns:
        dict con:
            - geometry_name
            - target_delta
            - d
            - m2L2_bulk
            - bracket_used
            - converged (bool)
            - error_message (si falla)
    """
    h5_path = Path(h5_path)
    z, A, ff, d_attr = load_metric_from_h5(
        h5_path=h5_path,
        z_dataset=z_dataset,
        A_dataset=A_dataset,
        f_dataset=f_dataset,
    )

    d = int(d_override) if d_override is not None else int(d_attr)
    A_func, f_func, z_min, z_max = make_interpolated_metric(z, A, ff)

    # Construimos el solver de Gork con nuestra métrica emergente
    solver = gsolver.BulkScalarSolver(
        d=d,
        L=L,
        A_func=A_func,
        f_func=f_func,
        z_max=z_max,
        z_min=z_min,
        num_points=num_points,
    )

    if bracket is None:
        bracket_used = None  # usará el default interno basado en la BF bound
    else:
        bracket_used = list(bracket)

    try:
        m2 = solver.solve_m2(target_delta, bracket=bracket)
        converged = True
        err_msg = None
    except Exception as e:
        m2 = None
        converged = False
        err_msg = str(e)

    return {
        "geometry_name": h5_path.stem,
        "target_delta": float(target_delta),
        "d": d,
        "L": float(L),
        "m2L2_bulk": float(m2) if m2 is not None else None,
        "bracket_used": bracket_used,
        "converged": converged,
        "error_message": err_msg,
        "z_dataset": z_dataset,
        "A_dataset": A_dataset,
        "f_dataset": f_dataset,
        "num_points": num_points,
    }


def _cli():
    parser = argparse.ArgumentParser(
        description="Resolver m^2_bulk emergente para un Δ dado usando métrica de un .h5"
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        required=True,
        help="Archivo .h5 de Fase XI (p.ej. fase11_output_v2/data/ads_d3_Tfinite_known_000.h5)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help="Δ objetivo (por ejemplo 3.0)",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=None,
        help="Sobrescribir d (si no, se toma de attrs['d'])",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=1.0,
        help="Escala L (por defecto 1.0)",
    )
    parser.add_argument(
        "--z-dataset",
        type=str,
        default="bulk_truth/z_grid",
        help="Dataset del grid radial en el HDF5",
    )
    parser.add_argument(
        "--A-dataset",
        type=str,
        default="bulk_truth/A_truth",
        help="Dataset de A(z) en el HDF5",
    )
    parser.add_argument(
        "--f-dataset",
        type=str,
        default="bulk_truth/f_truth",
        help="Dataset de f(z) en el HDF5",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1000,
        help="Número de puntos internos del solver",
    )
    parser.add_argument(
        "--bracket-low",
        type=float,
        default=None,
        help="Extremo inferior del bracket para m^2 (opcional)",
    )
    parser.add_argument(
        "--bracket-high",
        type=float,
        default=None,
        help="Extremo superior del bracket para m^2 (opcional)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Archivo JSON de salida con el resultado (opcional)",
    )
    args = parser.parse_args()

    if (args.bracket_low is None) != (args.bracket_high is None):
        raise ValueError("Hay que pasar ambos --bracket-low y --bracket-high, o ninguno.")

    bracket = None
    if args.bracket_low is not None:
        bracket = (args.bracket_low, args.bracket_high)

    result = solve_m2_for_delta_from_h5(
        h5_path=args.h5_file,
        target_delta=args.delta,
        d_override=args.d,
        L=args.L,
        z_dataset=args.z_dataset,
        A_dataset=args.A_dataset,
        f_dataset=args.f_dataset,
        num_points=args.num_points,
        bracket=bracket,
    )

    print("\n======================================================")
    print(f"GEOMETRIA: {result['geometry_name']}  (d={result['d']}, L={result['L']})")
    print("======================================================")
    print(f"  Δ objetivo      = {result['target_delta']}")
    print(f"  m^2_bulk (L^2)  = {result['m2L2_bulk']}")
    print(f"  convergencia    = {result['converged']}")
    if result["bracket_used"] is not None:
        print(f"  bracket usado   = {result['bracket_used']}")
    if result["error_message"]:
        print(f"  error solver    = {result['error_message']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nResultado guardado en: {out_path}")


if __name__ == "__main__":
    _cli()
