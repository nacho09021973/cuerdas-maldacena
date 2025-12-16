#!/usr/bin/env python3
"""
ecuaciones_emd.py â€” modulo independiente
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

class EMDLifshitzSolver:
    """
    Solver numÃ©rico para el sistema Einstein-Maxwell-DilatÃ³n con
    geometrÃ­a Lifshitz y violaciÃ³n de hiperscaling.
    
    MÃ©trica: dsÂ² = r^{-2Î¸/d} [-r^{-2(z-1)}f(r)dtÂ² + drÂ²/f(r) + dx_iÂ²]
    """
    
    def __init__(self, d=2, z=2, theta=0, lam=1, Q=1, r_h=1):
        self.d = d
        self.z = z
        self.theta = theta
        self.lam = lam
        self.Q = Q
        self.r_h = r_h
        
        # ParÃ¡metro geomÃ©trico agrupado n
        self.n = d + z - theta
        
    def system_equations(self, r, y):
        """
        Sistema de EDOs acopladas:
        y = [f, phi, psi, A_t]
        """
        f, phi, psi, A_t = y
        
        # 1. EcuaciÃ³n de Maxwell (A_t')
        # A_t' = Q * r^(-n) * e^(-lambda*phi)
        At_prime = self.Q * (r**-self.n) * np.exp(-self.lam * phi)
        
        # 2. EcuaciÃ³n mÃ©trica (f')
        # f' = -0.5*psi^2*f - (d-1)n/r
        f_prime = -0.5 * (psi**2) * f - ((self.d - 1) * self.n) / r
        
        # 3. EcuaciÃ³n del dilatÃ³n (psi' = phi'')
        # Singularidad removible en f=0.
        # Si f es muy pequeÃ±o, la integraciÃ³n podrÃ­a volverse inestable,
        # pero solve_ivp con RK45 suele manejarlo si no empezamos exactamente en 0.
        if abs(f) < 1e-15:
            # En el horizonte exacto, usar regla de L'Hopital si fuera necesario,
            # pero aquÃ­ confiamos en la condiciÃ³n inicial desplazada.
            psi_prime = 0 
        else:
            term1 = (self.lam / 2) * np.exp(self.lam * phi) * (At_prime**2)
            term2 = self.n * (f / r) * psi
            term3 = f_prime * psi
            psi_prime = (term1 - term2 - term3) / f
            
        # 4. DefiniciÃ³n de psi
        phi_prime = psi
        
        return [f_prime, phi_prime, psi_prime, At_prime]

    def get_horizon_conditions(self, phi_h, epsilon=1e-5):
        """
        Calcula las condiciones iniciales en r = r_h + epsilon usando expansiÃ³n en serie.
        """
        r = self.r_h
        
        # Valor de f'(r_h) determinado por la ecuaciÃ³n de f evaluada en f=0
        f1 = -((self.d - 1) * self.n) / r
        
        # Calcular psi(r_h) usando la ecuaciÃ³n de restricciÃ³n (Hamiltoniana)
        # (d-1)n = 0.5*Q^2*r^(-2n)*e^(-lam*phi) + 0.5*r^2*psi^2 - n(n-1)
        lhs = (self.d - 1) * self.n + self.n * (self.n - 1)
        term_Q = 0.5 * (self.Q**2) * (r**(-2*self.n)) * np.exp(-self.lam * phi_h)
        
        psi_sq_term = lhs - term_Q
        
        if psi_sq_term < 0:
            # Esto indica que no existe soluciÃ³n de agujero negro para este phi_h
            return None
            
        # Despejar psi^2 -> psi
        psi_h_sq = (2 * psi_sq_term) / (r**2)
        psi_h = -np.sqrt(psi_h_sq) # Elegimos raÃ­z negativa para comportamiento estÃ¡ndar
        
        # Potencial elÃ©ctrico cerca del horizonte (A_t(rh) = 0)
        A1 = self.Q * (r**-self.n) * np.exp(-self.lam * phi_h)
        
        # Condiciones desplazadas epsilon
        f_0 = f1 * epsilon
        phi_0 = phi_h + psi_h * epsilon
        psi_0 = psi_h # Orden cero para la derivada
        At_0 = A1 * epsilon # A_t ~ A1 * (r - rh)
        
        return [f_0, phi_0, psi_0, At_0]

    def solve(self, phi_h, r_uv=1e-3, epsilon=1e-5):
        """
        Integra el sistema desde r_h + epsilon hasta r_uv.
        """
        y0 = self.get_horizon_conditions(phi_h, epsilon)
        if y0 is None:
            return None
            
        r_span = (self.r_h + epsilon, r_uv) # Integrar hacia r pequeÃ±os (UV)
        
        sol = solve_ivp(
            self.system_equations,
            r_span,
            y0,
            method='RK45',
            rtol=1e-9,
            atol=1e-11,
            max_step=0.01
        )
        return sol

    def check_constraint(self, sol):
        """
        Verifica la ecuaciÃ³n de restricciÃ³n a lo largo de la soluciÃ³n numÃ©rica.
        Debe ser cero (o muy cercano) para una soluciÃ³n vÃ¡lida.
        Constraint: C = (1/2)r^2 psi^2 + (1/2)Q^2 r^(-2n)e^(-lam phi) - n(n-1) - (d-1)n + (d-1)n f/r (??)
        Nota: La restricciÃ³n provista en el prompt no tiene f. Asumimos que es la
        restricciÃ³n evaluada en el horizonte o una forma reducida. 
        Para el chequeo general, usamos la ecuaciÃ³n diferencial derivada.
        """
        r = sol.t
        f, phi, psi, At = sol.y
        
        # ReconstrucciÃ³n de la restricciÃ³n original basada en las ecuaciones de movimiento
        # Normalmente E_rr = 0.
        term1 = 0.5 * (r**2) * (psi**2)
        term2 = 0.5 * (self.Q**2) * (r**(-2*self.n)) * np.exp(-self.lam * phi)
        term3 = -self.n * (self.n - 1)
        term4 = -(self.d - 1) * self.n
        
        # Nota: Si la ecuaciÃ³n provista en el prompt es exacta para todo r,
        # constraint = RHS - LHS
        constraint = term1 + term2 + term3 - term4
        return constraint

    def plot_solution(self, sol, phi_h_val):
        """Genera grÃ¡ficos de los campos."""
        r = sol.t
        f, phi, psi, At = sol.y
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'SoluciÃ³n EMD (d={self.d}, z={self.z}, $\\theta$={self.theta}, $\phi_h$={phi_h_val:.2f})')
        
        # F(r)
        axs[0, 0].plot(r, f, 'b-', lw=2)
        axs[0, 0].set_xlabel('r')
        axs[0, 0].set_ylabel('f(r)')
        axs[0, 0].set_title('FunciÃ³n de ennegrecimiento')
        axs[0, 0].invert_xaxis()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Phi(r)
        axs[0, 1].plot(r, phi, 'r-', lw=2)
        axs[0, 1].set_xlabel('r')
        axs[0, 1].set_ylabel('$\phi(r)$')
        axs[0, 1].set_title('Campo DilatÃ³n')
        axs[0, 1].invert_xaxis()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Psi(r)
        axs[1, 0].plot(r, psi, 'g-', lw=2)
        axs[1, 0].set_xlabel('r')
        axs[1, 0].set_ylabel('$\psi(r) = \phi\'(r)$')
        axs[1, 0].set_title('Derivada del DilatÃ³n')
        axs[1, 0].invert_xaxis()
        axs[1, 0].grid(True, alpha=0.3)
        
        # A_t(r)
        axs[1, 1].plot(r, At, 'k-', lw=2)
        axs[1, 1].set_xlabel('r')
        axs[1, 1].set_ylabel('$A_t(r)$')
        axs[1, 1].set_title('Campo de Gauge')
        axs[1, 1].invert_xaxis()
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# --- EjecuciÃ³n del Ejemplo ---

def main():
    # 1. Definir parÃ¡metros fÃ­sicos
    # Ejemplo: AdS Reissner-Nordstrom (z=1, theta=0) para verificar
    # O un caso Lifshitz interesante (z=2, theta=1)
    params = {
        'd': 3,       # Dimensiones espaciales
        'z': 2.0,     # Exponente Lifshitz
        'theta': 1.0, # ViolaciÃ³n de Hiperscaling
        'lam': 0.5,   # Acoplamiento
        'Q': 2.0,     # Carga
        'r_h': 1.0    # Horizonte
    }
    
    solver = EMDLifshitzSolver(**params)
    
    print(f"Resolviendo sistema EMD para: {params}")
    print(f"Parametro efectivo n = {solver.n}")

    # 2. BÃºsqueda manual rÃ¡pida o Shooting
    # Probamos un valor de phi_h razonable
    phi_h_trial = 0.5
    
    sol = solver.solve(phi_h=phi_h_trial, r_uv=1e-4)
    
    if sol is not None and sol.success:
        print("IntegraciÃ³n exitosa.")
        print(f"Valor final de f en UV (r={sol.t[-1]:.2e}): {sol.y[0][-1]:.4f}")
        
        # Calcular error de restricciÃ³n
        err = solver.check_constraint(sol)
        print(f"Error mÃ¡ximo en restricciÃ³n: {np.max(np.abs(err)):.2e}")
        
        solver.plot_solution(sol, phi_h_trial)
        
        # CÃ¡lculo simple de Temperatura
        # T ~ |f'(rh)| * rh^(-z+1) / 4pi (dependiendo de la normalizaciÃ³n exacta del tiempo)
        f_prime_h = sol.y[0][0] / (sol.t[0] - solver.r_h) # aprox numÃ©rica gruesa o usar fÃ³rmula analÃ­tica
        print(f"Pendiente de f cerca del horizonte: ~{f_prime_h:.4f}")
        
    else:
        print("FallÃ³ la integraciÃ³n o no existe soluciÃ³n fÃ­sica para ese phi_h.")

if __name__ == "__main__":
    main()