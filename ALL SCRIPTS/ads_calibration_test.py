#!/usr/bin/env python3
"""
ads_calibration_test.py - Test de Calibración AdS para CUERDAS

OBJETIVO:
    Verificar si bulk_scalar_solver.py reproduce correctamente la relación
    de Maldacena m²L² = Δ(Δ-d) para AdS puro.

MÉTODO:
    1. Crear geometría AdS pura sintética con parámetros conocidos
    2. Resolver el problema de autovalores con el solver
    3. Comparar autovalores numéricos con predicción teórica
    4. Identificar factores de conversión/calibración

RESULTADO ESPERADO SI TODO ESTÁ BIEN:
    Los autovalores numéricos deberían satisfacer m²L² = Δ(Δ-d)
    con Δ extraído del comportamiento UV de los modos.

SI HAY MISMATCH:
    Este script identificará el factor de conversión necesario.

USO:
    python ads_calibration_test.py

AUTOR: Script de diagnóstico para proyecto CUERDAS
"""

import numpy as np
from pathlib import Path
import sys

# ============================================================================
# PARÁMETROS DE CALIBRACIÓN
# ============================================================================
D_VALUES = [3, 4, 5]           # Dimensiones boundary a testear
N_GRID = 200                    # Puntos en el grid radial
Z_MIN = 0.01                    # Cutoff UV (boundary)
Z_MAX = 10.0                    # Cutoff IR
N_EIGS = 10                     # Número de autovalores a calcular

# ============================================================================
# GEOMETRÍA AdS PURA
# ============================================================================

def create_pure_ads_geometry(d: int, n_grid: int, z_min: float, z_max: float):
    """
    Crea geometría AdS pura en coordenadas de Poincaré.
    
    Métrica: ds² = (L²/z²)(-dt² + dx² + dz²)
    
    En la parametrización del solver:
        ds² = e^{2A(z)}(-f(z)dt² + dx²) + dz²/f(z)
    
    Para AdS puro con L=1:
        e^{A(z)} = 1/z  →  A(z) = -log(z)
        f(z) = 1
    
    Args:
        d: dimensión del boundary (el bulk es d+1 dimensional)
        n_grid: número de puntos en el grid z
        z_min: cutoff UV
        z_max: cutoff IR
    
    Returns:
        z, A, f: arrays para el grid radial, warp factor, y blackening factor
    """
    z = np.linspace(z_min, z_max, n_grid)
    
    # AdS puro con L = 1
    A = -np.log(z)      # e^A = 1/z
    f = np.ones_like(z) # f(z) = 1 para AdS puro (sin horizonte)
    
    return z, A, f


# ============================================================================
# SOLVER STURM-LIOUVILLE (copia simplificada de bulk_scalar_solver.py)
# ============================================================================

def build_sturm_liouville_operator(z, A, f, d):
    """
    Construye el operador L tal que L·φ = m²·φ
    
    L φ = - (1/√-g) ∂_z [√-g g^{zz} ∂_z φ]
    
    Con:
        √-g = e^{d·A(z)}
        g^{zz} = f(z)
    """
    N = len(z)
    dz = z[1] - z[0]
    
    # Peso y coeficiente
    w = np.exp(d * A)           # √-g
    p = w * f                   # √-g · g^{zz}
    
    # Matriz del operador
    L = np.zeros((N, N))
    
    # Condiciones Dirichlet en los extremos
    L[0, 0] = 1.0
    L[-1, -1] = 1.0
    
    # Puntos interiores (diferencias finitas)
    for i in range(1, N - 1):
        w_i = w[i]
        p_plus = 0.5 * (p[i] + p[i + 1])
        p_minus = 0.5 * (p[i] + p[i - 1])
        
        L[i, i - 1] = +p_minus / (w_i * dz * dz)
        L[i, i + 1] = +p_plus / (w_i * dz * dz)
        L[i, i] = -(p_minus + p_plus) / (w_i * dz * dz)
    
    return L


def solve_eigenvalues(L, n_eigs=10):
    """Resuelve el problema de autovalores, devolviendo solo positivos."""
    evals, evecs = np.linalg.eigh(L)
    
    # Filtrar positivos
    mask = evals > 1e-10
    evals = evals[mask]
    evecs = evecs[:, mask]
    
    # Ordenar
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    return evals[:n_eigs], evecs[:, :n_eigs]


def estimate_uv_exponent(z, phi, frac_uv=0.2):
    """
    Estima Δ del comportamiento UV: |φ(z)| ~ z^Δ cerca de z→0
    """
    N = len(z)
    n_uv = max(5, int(frac_uv * N))
    
    z_uv = z[:n_uv]
    phi_uv = np.abs(phi[:n_uv])
    
    # Filtrar ceros
    mask = phi_uv > 1e-12
    if mask.sum() < 5:
        return None
    
    z_fit = z_uv[mask]
    phi_fit = phi_uv[mask]
    
    # Ajuste log-log: log|φ| = Δ·log(z) + const
    logz = np.log(z_fit)
    logphi = np.log(phi_fit)
    
    A_mat = np.vstack([logz, np.ones_like(logz)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, logphi, rcond=None)
    
    Delta = coeffs[0]
    return Delta


# ============================================================================
# TEST DE CALIBRACIÓN
# ============================================================================

def run_calibration_test():
    """Ejecuta el test de calibración para todas las dimensiones."""
    
    print("=" * 80)
    print("TEST DE CALIBRACIÓN AdS - PROYECTO CUERDAS")
    print("=" * 80)
    print(f"\nParámetros:")
    print(f"  Grid: N={N_GRID}, z ∈ [{Z_MIN}, {Z_MAX}]")
    print(f"  Dimensiones boundary: d ∈ {D_VALUES}")
    print(f"  Autovalores a calcular: {N_EIGS}")
    
    all_results = []
    
    for d in D_VALUES:
        print(f"\n{'='*80}")
        print(f"DIMENSIÓN BOUNDARY d = {d} (bulk d+1 = {d+1})")
        print("=" * 80)
        
        # Crear geometría AdS pura
        z, A, f = create_pure_ads_geometry(d, N_GRID, Z_MIN, Z_MAX)
        
        # Construir operador y resolver
        L_op = build_sturm_liouville_operator(z, A, f, d)
        eigenvalues, eigenvectors = solve_eigenvalues(L_op, N_EIGS)
        
        print(f"\n{'Modo':<6} {'λ_num':>12} {'Δ_UV':>10} {'m²L²_teo':>12} {'Ratio':>10}")
        print("-" * 60)
        
        for n in range(len(eigenvalues)):
            lambda_n = eigenvalues[n]
            phi_n = eigenvectors[:, n]
            
            # Estimar Δ del comportamiento UV
            Delta_uv = estimate_uv_exponent(z, phi_n)
            
            if Delta_uv is not None:
                # Predicción teórica: m²L² = Δ(Δ - d)
                m2L2_theory = Delta_uv * (Delta_uv - d)
                
                # Ratio entre numérico y teórico
                if abs(m2L2_theory) > 1e-6:
                    ratio = lambda_n / m2L2_theory
                else:
                    ratio = float('nan')
                
                print(f"{n:<6} {lambda_n:>12.4f} {Delta_uv:>10.4f} {m2L2_theory:>12.4f} {ratio:>10.4f}")
                
                all_results.append({
                    'd': d,
                    'mode': n,
                    'lambda_num': lambda_n,
                    'Delta_uv': Delta_uv,
                    'm2L2_theory': m2L2_theory,
                    'ratio': ratio
                })
            else:
                print(f"{n:<6} {lambda_n:>12.4f} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
    
    # Análisis global
    print("\n" + "=" * 80)
    print("ANÁLISIS GLOBAL")
    print("=" * 80)
    
    if all_results:
        ratios = [r['ratio'] for r in all_results if not np.isnan(r['ratio'])]
        if ratios:
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            
            print(f"\nRatio medio λ_num / m²L²_teo: {mean_ratio:.4f} ± {std_ratio:.4f}")
            
            if abs(mean_ratio - 1.0) < 0.1:
                print("\n✓ CALIBRACIÓN OK: El solver reproduce la física de Maldacena")
            else:
                print(f"\n✗ MISMATCH DETECTADO: Factor de corrección necesario ≈ {mean_ratio:.4f}")
                print(f"\n  Interpretación:")
                if mean_ratio > 1:
                    print(f"  Los autovalores numéricos son {mean_ratio:.1f}x MAYORES que la teoría")
                else:
                    print(f"  Los autovalores numéricos son {1/mean_ratio:.1f}x MENORES que la teoría")
                print(f"\n  Para corregir: m²L²_corregido = λ_numérico / {mean_ratio:.4f}")
    
    # Verificar si el problema es el signo
    print("\n" + "-" * 80)
    print("ANÁLISIS DE SIGNOS")
    print("-" * 80)
    
    n_positive_lambda = sum(1 for r in all_results if r['lambda_num'] > 0)
    n_positive_theory = sum(1 for r in all_results if r['m2L2_theory'] > 0)
    n_negative_theory = sum(1 for r in all_results if r['m2L2_theory'] < 0)
    
    print(f"  Autovalores numéricos positivos: {n_positive_lambda}/{len(all_results)}")
    print(f"  m²L² teóricos positivos: {n_positive_theory}/{len(all_results)}")
    print(f"  m²L² teóricos negativos: {n_negative_theory}/{len(all_results)}")
    
    if n_negative_theory > n_positive_theory:
        print("\n  ⚠ NOTA: La mayoría de los Δ_UV extraídos dan m²L² teórico NEGATIVO")
        print("    Esto ocurre cuando Δ < d (región de operadores relevantes)")
        print("    Los autovalores numéricos son SIEMPRE positivos por construcción")
        print("    del operador Sturm-Liouville")
    
    return all_results


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    results = run_calibration_test()
    
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    print("""
Este test crea geometrías AdS puras sintéticas y verifica si el solver
reproduce la relación m²L² = Δ(Δ-d).

Si hay un mismatch sistemático, las posibles causas son:
1. Normalización del grid z (escala de L)
2. Definición del warp factor A(z)
3. Condiciones de contorno
4. El operador discretizado no es exactamente el □_AdS

SIGUIENTE PASO: Si el ratio es consistentemente diferente de 1,
usar ese factor para calibrar los resultados de Fase XI antes
de pasarlos a XII.c.
""")
