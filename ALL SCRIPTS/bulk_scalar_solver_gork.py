#!/usr/bin/env python3
"""
bulk_scalar_solver.py - Solver de campo escalar en el bulk para extraer m²L² experimental

Toma una métrica emergente A(z), f(z) de Fase XI.
Resuelve la ecuación Klein-Gordon (q=0) numéricamente.
Ajusta m² para que el comportamiento UV (exponente sigma) coincida con el Δ target.
Devuelve m²L² "experimental" para entrenar XII.c.

USO:
    from bulk_scalar_solver import solve_for_m2
    m2 = solve_for_m2(target_delta=3.0, d=4, L=1.0)

FILOSOFIA:
    - No inyecta fórmula teórica.
    - Emerge m² a partir de PDE numérica y fit UV.
    - Generalizable a métricas no-AdS (cambiar A, f).
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar

class BulkScalarSolver:
    def __init__(self, d, L=1.0, A_func=None, f_func=None, z_max=100.0, z_min=0.001, num_points=1000):
        self.d = d
        self.L = L
        self.zspan = np.linspace(z_max, z_min, num_points)
        self.A = A_func if A_func else self.default_A
        self.f = f_func if f_func else self.default_f

    def default_A(self, z):
        return np.log(self.L / z)

    def default_f(self, z):
        return self.L**2 / z**2

    def K(self, z):
        return np.exp(self.d * self.A(z)) / np.sqrt(self.f(z))

    def V(self, z):
        return np.exp(self.d * self.A(z)) * np.sqrt(self.f(z))

    def dYdz(self, Y, z, m2):
        psi, u = Y
        dpsi = u / self.K(z)
        du = m2 * self.V(z) * psi
        return [dpsi, du]

    def fitted_sigma(self, m2):
        Y0 = [1.0, 0.0]  # Condición inicial arbitraria en IR (z_max)
        sol = odeint(self.dYdz, Y0, self.zspan, args=(m2,), atol=1e-10, rtol=1e-10)
        i = -1  # Punto en UV (z_min)
        z_small = self.zspan[i]
        psi_small = sol[i, 0]
        u_small = sol[i, 1]
        k_small = self.K(z_small)
        if psi_small == 0:
            return np.nan  # Evitar división por cero
        sigma = z_small * (u_small / k_small) / psi_small
        return sigma

    def fitted_delta(self, m2):
        sigma = self.fitted_sigma(m2)
        if np.isnan(sigma):
            return np.nan
        return self.d - sigma

    def error(self, m2, target_delta):
        return self.fitted_delta(m2) - target_delta

    def solve_m2(self, target_delta, bracket=None):
        if bracket is None:
            bracket = [- (self.d/2)**2 + 0.1, 100.0]  # Desde BF bound hasta positivo
        result = root_scalar(lambda m2: self.error(m2, target_delta), bracket=bracket)
        if result.converged:
            return result.root / self.L**2  # Devuelve m² (normalizado por L² si necesario)
        else:
            raise ValueError(f"No convergió para Δ={target_delta}")

def solve_for_m2(target_delta, d, L=1.0, A_func=None, f_func=None, z_max=100.0, z_min=0.001, num_points=1000, bracket=None):
    solver = BulkScalarSolver(d, L, A_func, f_func, z_max, z_min, num_points)
    return solver.solve_m2(target_delta, bracket)

if __name__ == "__main__":
    # Test con AdS puro
    target_delta = 3.0
    d = 4
    m2 = solve_for_m2(target_delta, d, bracket=[-4, 0])
    print(f"Para Δ={target_delta}, d={d}, m² experimental = {m2}")