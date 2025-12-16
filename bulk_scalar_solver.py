#!/usr/bin/env python3
# bulk_scalar_solver.py
# CUERDAS — Bloque B: Espectro escalar (módulo de Sturm–Liouville)
#
# OBJETIVO
#   Resolver el problema escalar tipo Sturm–Liouville en el bulk:
#     - A partir de una geometría (z, A(z), f(z)), construir el operador diferencial.
#     - Calcular autovalores λ_SL y autovectores.
#     - Estimar el comportamiento UV de los modos (Δ_UV).
#
# USO COMO MÓDULO
#   - Diseñado para ser importado desde otros scripts, en particular:
#       * 06_build_bulk_eigenmodes_dataset.py
#
# ENTRADAS (como módulo)
#   - Fichero de geometría emergente (.h5) con:
#       * z_grid, A_emergent, f_emergent, ... (u otros campos necesarios).
#
# SALIDAS (como módulo)
#   - Estructuras Python con:
#       * Lista de modos, λ_SL, Δ_UV, etiquetas de tipo de modo, etc.
#
# OPCIONAL: USO COMO CLI
#   - Puede exponer una pequeña interfaz de línea de comandos para pruebas unitarias:
#       python bulk_scalar_solver.py --geometry-file ... --n-modes ...
#
# HONESTIDAD
#   - Este módulo trabaja exclusivamente con geometrías (emergentes o sandbox).
#   - No contiene ninguna referencia embebida a fórmulas tipo Δ(Δ-d).
#
# HISTÓRICO
#   - Anteriormente conocido como: bulk_scalar_solver_v2.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


@dataclass
class BulkGeometry:
    """Geometría bulk 1D para el solver escalar."""

    name: str
    family: str
    d: int
    z: np.ndarray      # Grid radial (monótono creciente)
    A: np.ndarray      # Warp factor A(z)
    f: np.ndarray      # Blackening factor f(z)


def load_bulk_geometry_from_h5(
    h5_path: str | Path,
    z_dataset: str = "bulk/z_grid",
    A_dataset: str = "bulk/A_of_z",
    f_dataset: str = "bulk/f_of_z",
) -> BulkGeometry:
    """
    Carga una geometría bulk desde un archivo HDF5.

    NOTA: Los nombres de datasets z/A/f dependen de cómo guardaste Fase XI.
    Si tus ficheros usan otros paths, ajusta z_dataset/A_dataset/f_dataset.

    Args:
        h5_path: ruta al archivo .h5
        z_dataset: path dentro del HDF5 para el grid radial
        A_dataset: path para A(z)
        f_dataset: path para f(z)

    Returns:
        BulkGeometry
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        # Ajusta estos paths si tu layout es distinto
        z = np.array(f[z_dataset][:], dtype=float)
        A = np.array(f[A_dataset][:], dtype=float)
        ff = np.array(f[f_dataset][:], dtype=float)

        d_attr = f.attrs.get("d", 4)
        try:
            d = int(d_attr)
        except Exception:
            d = 4

        family = f.attrs.get("family", "unknown")
        if isinstance(family, bytes):
            family = family.decode("utf-8", errors="ignore")

    # Chequeos básicos
    if z.ndim != 1:
        raise ValueError(f"z_grid en {h5_path} no es 1D")
    if not (len(z) == len(A) == len(ff)):
        raise ValueError(
            f"Tamaños inconsistentes en {h5_path}: "
            f"len(z)={len(z)}, len(A)={len(A)}, len(f)={len(ff)}"
        )
    if not np.all(np.diff(z) > 0):
        raise ValueError(f"z_grid en {h5_path} no es estrictamente creciente")

    return BulkGeometry(
        name=h5_path.stem,
        family=family,
        d=d,
        z=z,
        A=A,
        f=ff,
    )


def build_sturm_liouville_operator(geom: BulkGeometry) -> np.ndarray:
    """
    Construye la matriz del operador Sturm–Liouville:

        L phi = - (1/sqrt(-g)) d_z [ sqrt(-g) g^{zz} d_z phi ]

    usando diferencias finitas de segunda orden con paso uniforme.

    Asumimos:
        sqrt(-g) = e^{d A(z)}
        g^{zz}   = f(z)

    Condiciones de contorno:
        phi(z_min) = phi(z_max) = 0  (Dirichlet)

    Devuelve:
        L: matriz NxN tal que L @ phi ≈ lambda * phi
    """
    z = geom.z
    A = geom.A
    fz = geom.f
    d = geom.d

    N = len(z)
    if N < 5:
        raise ValueError("Grid demasiado pequeño para construir el operador")

    # Chequeo de espaciado (asumimos uniforme para simplificar)
    dz = np.diff(z)
    if np.max(np.abs(dz - dz[0])) > 1e-6 * abs(dz[0]):
        raise ValueError(
            "El grid z no parece uniformemente espaciado; "
            "esta implementación asume dz constante."
        )
    dz = float(dz[0])

    # Peso y coeficiente "p(z)" en forma Sturm–Liouville
    # L phi = - (1/w) d_z [ p d_z phi ], con
    #   w(z) = sqrt(-g) = e^{d A(z)}
    #   p(z) = sqrt(-g) g^{zz} = e^{d A(z)} f(z)
    w = np.exp(d * A)
    p = w * fz

    # Construir L como matriz densa NxN (para empezar; si hace falta, se pasa a sparse)
    L = np.zeros((N, N), dtype=float)

    # Condiciones de contorno Dirichlet: phi(0)=phi(N-1)=0
    # Implementamos imponiendo L[0,0]=L[N-1,N-1]=1 y el resto de la fila a 0
    L[0, 0] = 1.0
    L[-1, -1] = 1.0

    # Puntos interiores: esquema de diferencias finitas tipo:
    #   L_i phi ≈ - 1/w_i * [ (p_{i+1/2} (phi_{i+1}-phi_i)/dz
    #                          - p_{i-1/2} (phi_i - phi_{i-1})/dz ) / dz ]
    # donde p_{i+1/2} ≈ (p_i + p_{i+1})/2
    for i in range(1, N - 1):
        w_i = w[i]
        p_iphalf = 0.5 * (p[i] + p[i + 1])
        p_imhalf = 0.5 * (p[i] + p[i - 1])

        L[i, i - 1] = +p_imhalf / (w_i * dz * dz)
        L[i, i + 1] = +p_iphalf / (w_i * dz * dz)
        L[i, i] = -(p_imhalf + p_iphalf) / (w_i * dz * dz)

    return L


def solve_eigenvalue_spectrum(
    geom: BulkGeometry,
    n_eigs: int = 5,
    discard_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resuelve el problema de autovalores:

        L phi_n = lambda_n phi_n

    y devuelve los n_eigs autovalores más bajos (ordenados) y sus autovectores.

    NOTA: Los autovalores son λ_SL (Sturm–Liouville), NO masas holográficas.

    Args:
        geom: geometría bulk
        n_eigs: número de modos a devolver (por defecto 5)
        discard_negative: si True, descarta autovalores negativos
                          (artefactos numéricos) antes de cortar

    Returns:
        lambda_sl: array (k,) de λ_n (k <= n_eigs)
        modes:     matriz (N, k) con los autovectores correspondientes
    """
    L = build_sturm_liouville_operator(geom)

    # Resolver problema simétrico: L es (aprox.) Hermítica real
    evals, evecs = np.linalg.eigh(L)
    evals = np.real(evals)

    if discard_negative:
        mask = evals > 0.0
        evals = evals[mask]
        evecs = evecs[:, mask]

    # Ordenar por autovalor creciente
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    k = min(n_eigs, len(evals))
    return evals[:k], evecs[:, :k]


def estimate_uv_exponent(
    z: np.ndarray,
    phi: np.ndarray,
    frac_uv: float = 0.2,
) -> float | None:
    """
    Estima un exponente UV Delta_UV a partir de un modo phi(z), asumiendo
    un comportamiento de potencia cerca de la frontera:

        |phi(z)| ~ z^{Delta_UV}

    Procedimiento:
        - Toma la fracción inicial del grid (por ejemplo, 20% más cercano a z_min).
        - Ajusta log|phi| = a + Delta_UV * log z mediante mínimos cuadrados.
        - Devuelve Delta_UV.

    Si los datos son demasiado ruidosos o hay signos cambiantes que impiden
    tomar log, devuelve None.
    """
    z = np.array(z, dtype=float)
    phi = np.array(phi, dtype=float)

    N = len(z)
    if N < 10:
        return None

    n_uv = max(5, int(frac_uv * N))
    z_uv = z[:n_uv]
    phi_uv = phi[:n_uv]

    # Evitar ceros y cambios de signo
    abs_phi = np.abs(phi_uv)
    mask = abs_phi > 1e-12
    if mask.sum() < 5:
        return None

    z_fit = z_uv[mask]
    phi_fit = abs_phi[mask]

    logz = np.log(z_fit)
    logphi = np.log(phi_fit)

    A = np.vstack([logz, np.ones_like(logz)]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, logphi, rcond=None)
    slope, _ = coeffs

    Delta_uv = slope  # log |phi| ~ Delta_uv * log z + const
    return float(Delta_uv)


def solve_geometry(
    h5_path: str | Path,
    n_eigs: int = 5,
    z_dataset: str = "bulk/z_grid",
    A_dataset: str = "bulk/A_of_z",
    f_dataset: str = "bulk/f_of_z",
) -> Dict[str, object]:
    """
    Atajo de alto nivel:
        - Carga geometría desde un .h5.
        - Construye L.
        - Resuelve los n_eigs primeros modos.
        - Estima exponentes UV para cada modo.

    Devuelve un diccionario listo para volcar a JSON.
    
    NOMENCLATURA v2:
        - "lambda_sl": autovalores Sturm–Liouville (antes "m2L2")
        - "m2L2_legacy": alias para compatibilidad con scripts antiguos
    """
    geom = load_bulk_geometry_from_h5(
        h5_path=h5_path,
        z_dataset=z_dataset,
        A_dataset=A_dataset,
        f_dataset=f_dataset,
    )
    lambda_sl, modes = solve_eigenvalue_spectrum(geom, n_eigs=n_eigs)

    uv_exponents: List[float | None] = []
    for j in range(modes.shape[1]):
        Delta_uv = estimate_uv_exponent(geom.z, modes[:, j])
        uv_exponents.append(Delta_uv)

    return {
        "geometry_name": geom.name,
        "family": geom.family,
        "d": geom.d,
        "n_eigs": len(lambda_sl),
        # Nueva nomenclatura honesta
        "lambda_sl": lambda_sl.tolist(),
        # Alias legacy para compatibilidad con scripts antiguos (READ-ONLY)
        # NOTA: Código nuevo debe leer SOLO lambda_sl, m2L2_legacy es solo para compat.
        "m2L2_legacy": lambda_sl.tolist(),
        "uv_exponents": uv_exponents,
        # Marca explícita de versión de nomenclatura
        "nomenclature_version": "v2_lambda_sl",
    }


def _cli():
    parser = argparse.ArgumentParser(
        description="Resolver espectro Sturm–Liouville en una geometría bulk emergente."
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        required=True,
        help="Ruta al archivo .h5 de geometría (fase11_output_v2/data/...).",
    )
    parser.add_argument(
        "--n-eigs",
        type=int,
        default=5,
        help="Número de autovalores/autovectores a devolver.",
    )
    parser.add_argument(
        "--z-dataset",
        type=str,
        default="bulk/z_grid",
        help="Ruta al dataset z dentro del HDF5.",
    )
    parser.add_argument(
        "--A-dataset",
        type=str,
        default="bulk/A_of_z",
        help="Ruta al dataset A(z) dentro del HDF5.",
    )
    parser.add_argument(
        "--f-dataset",
        type=str,
        default="bulk/f_of_z",
        help="Ruta al dataset f(z) dentro del HDF5.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="bulk_scalar_spectrum.json",
        help="Archivo JSON de salida con el espectro calculado.",
    )
    args = parser.parse_args()

    import json

    result = solve_geometry(
        h5_path=args.h5_file,
        n_eigs=args.n_eigs,
        z_dataset=args.z_dataset,
        A_dataset=args.A_dataset,
        f_dataset=args.f_dataset,
    )

    print("\n======================================================")
    print(f"GEOMETRIA: {result['geometry_name']}  (family={result['family']}, d={result['d']})")
    print("======================================================")
    print("NOTA: lambda_SL son autovalores Sturm–Liouville, NO masas holográficas.")
    print("------------------------------------------------------")
    for j, (lam, Delta_uv) in enumerate(zip(result["lambda_sl"], result["uv_exponents"])):
        if Delta_uv is None:
            print(f"  modo {j}: λ_SL ≈ {lam:.6f},   Delta_UV ≈ (no fiable)")
        else:
            print(f"  modo {j}: λ_SL ≈ {lam:.6f},   Delta_UV ≈ {Delta_uv:.6f}")

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nEspectro guardado en: {out_path}")


if __name__ == "__main__":
    _cli()
