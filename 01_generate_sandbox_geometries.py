#!/usr/bin/env python3
from __future__ import annotations
# 01_generate_sandbox_geometries.py
# CUERDAS — Bloque A: Geometría emergente (generación de sandbox)
#
# OBJETIVO
#   Generar familias de "universos sandbox" controlados a partir de geometrías base
#   (AdS, Lifshitz, hyperscaling, deformed, ...) y producir datasets de:
#       - boundary/: datos CFT en el borde (entrada del learner)
#       - bulk_truth/: geometría de referencia (solo para validación/contratos)
#
# PRINCIPALES CARACTERÍSTICAS
#   - Generación de múltiples universos por geometría base (jitter de parámetros).
#   - Jitter de parámetros físicos (z_h, d, theta, z_dyn, deformation, ...).
#   - CLI escalable: p.ej. --n-known, --n-test, --n-unknown.
#   - Backend opcional EMD para familias tipo Lifshitz / hyperscaling.
#
# ENTRADAS (típicas)
#   - Parámetros de familias por CLI o fichero de configuración:
#       * family ∈ {ads, lifshitz, hyperscaling, deformed, ...}
#       * d, z_dyn, theta, etc.
#
# SALIDAS (estructura esperada)
#   runs/sandbox_geometries/
#     boundary/
#       <system_name>_boundary.h5
#         - x_grid, temperature, G2_<O>, omega_grid, k_grid, G_R_real, G_R_imag, ...
#     bulk_truth/
#       <system_name>_bulk_truth.h5
#         - z_grid, A_truth, f_truth, R_truth
#         - attrs: z_h, family, d, theta, z_dyn, ...
#     manifest.json
#       - Lista de "geometries" generadas y metadatos básicos.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Entrada directa para:
#       * 02_emergent_geometry_engine.py       (reconstruye geometría a partir de boundary/)
#       * 04_geometry_physics_contracts.py    (usa bulk_truth/ para contratos físicos)
#
# HONESTIDAD
#   - El learner NUNCA ve ni la métrica real ni el solver EMD.
#   - Solo se exponen datos CFT de boundary a los modelos de geometría.
#   - El bulk_truth se reserva exclusivamente para validación y contratos físicos.
#
# HISTÓRICO
#   - Anteriormente conocido como: 00_generate_fase_11_v3.py
#
# FIX 2025-12-21: Guardrail de d movido ANTES de generar boundary_data y bulk_truth
#   para asegurar consistencia entre nombre del archivo y datos generados.

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tools.stage_utils import (
    EXIT_ERROR,
    EXIT_OK,
    STATUS_ERROR,
    STATUS_OK,
    StageContext,
    add_standard_arguments,
    parse_stage_args,
)

# Backend opcional: soluciones EMD reales para Lifshitz / hyperscaling
try:
    from ecuaciones_emd import EMDLifshitzSolver  # type: ignore
    HAS_EMD = True
except ImportError:
    EMDLifshitzSolver = None  # type: ignore
    HAS_EMD = False


# ============================================================
#  GEOMETRÍA OCULTA
# ============================================================

@dataclass
class HiddenGeometry:
    """
    Geometría del bulk que genera los datos CFT.
    NO se expone al learner (solo a bulk_truth para validación).
    """
    name: str
    family: str           # "ads", "lifshitz", "hyperscaling", "deformed", "unknown"
    category: str         # "known", "test", "unknown"
    d: int                # dimensión del boundary (CFT_d)
    z_h: Optional[float] = None  # posición del horizonte (si hay BH)
    theta: float = 0.0           # exponente de hyperscaling
    z_dyn: float = 1.0           # exponente dinámico de Lifshitz
    deformation: float = 0.0     # deformación genérica de A(z)
    L: float = 1.0               # escala AdS
    metadata: Dict = field(default_factory=dict)

    # ---------- Warp factor y blackening (toy) ----------

    def warp_factor(self, z: np.ndarray) -> np.ndarray:
        """
        A(z) tal que:
            ds² = e^{2A(z)} [ -f(z) dt² + dx_i² ] + dz² / f(z)
        Implementación toy distinta por familia.
        """
        eps = 1e-6
        z = np.clip(z, eps, None)

        if self.family == "ads":
            # AdS_{d+1} puro
            return -np.log(z / self.L)

        elif self.family == "lifshitz":
            # Lifshitz: parte espacial ~ AdS
            return -np.log(z / self.L)

        elif self.family == "hyperscaling":
            # HV: ds² ~ z^{-2(d-θ)/d}(...)
            return -(1.0 - self.theta / self.d) * np.log(z / self.L)

        elif self.family == "deformed":
            base = -np.log(z / self.L)
            deform = self.deformation * (z / self.L) ** 2
            return base + deform

        else:  # "unknown"
            base = -np.log(z / self.L)
            deform = self.deformation * np.sin(z / (self.L + 1e-3))
            return base + deform

    def blackening_factor(self, z: np.ndarray) -> np.ndarray:
        """
        f(z) que codifica horizonte/temperatura.
        """
        if self.z_h is None or self.z_h <= 0:
            return np.ones_like(z)

        ratio = np.clip(z / self.z_h, 0.0, 1.0)

        if self.family == "ads":
            return np.clip(1.0 - ratio ** self.d, 0.0, 1.0)
        elif self.family == "lifshitz":
            return np.clip(1.0 - ratio ** (self.d + self.z_dyn - 1), 0.0, 1.0)
        elif self.family == "hyperscaling":
            eff_d = max(1.0, self.d - self.theta)
            return np.clip(1.0 - ratio ** eff_d, 0.0, 1.0)
        else:
            return np.clip(1.0 - ratio ** 4, 0.0, 1.0)

    def ricci_scalar(self, z: np.ndarray) -> np.ndarray:
        """
        Escalar de Ricci R(z) aproximado numericamente a partir de A(z), f(z).
        Para AdS puro fijamos el valor constante exacto.
        """
        eps = 1e-4
        z = np.clip(z, eps, None)

        A = self.warp_factor(z)
        f = self.blackening_factor(z)

        if len(z) > 1:
            dz = z[1] - z[0]
        else:
            dz = eps

        dA = np.gradient(A, dz)
        d2A = np.gradient(dA, dz)
        df = np.gradient(f, dz)

        D = self.d + 1
        R = -2 * D * d2A - D * (D - 1) * dA ** 2 - (df * dA) / (f + 1e-10)

        if self.family == "ads":
            R_ads = -self.d * (self.d + 1) / (self.L ** 2)
            R = np.full_like(z, R_ads)

        return R

    def einstein_tensor_trace(self, z: np.ndarray) -> np.ndarray:
        """
        Traza del tensor de Einstein:
        """
        D = self.d + 1
        R = self.ricci_scalar(z)
        return (1.0 - D / 2.0) * R

    def effective_central_charge(self, n_points: int = 256) -> float:
        """
        Observable de borde tipo "central charge" toy.

        Definimos un escalar
            c_eff ∼ ∫ e^{(d-1) A(z)} dz

        integrado desde el UV hasta el horizonte (si existe) o hasta una
        escala IR típica ~ L en geometrías sin horizonte.

        Este observable se usará solo como dato de boundary (visible al learner)
        y condensa información global sobre la forma de A(z) sin exponer A(z)
        punto a punto.
        """
        eps = 1e-4

        if self.z_h is not None and self.z_h > 0:
            z_max = float(self.z_h)
        else:
            # En geometrías sin horizonte usamos una escala IR razonable.
            z_max = float(self.L if self.L > 0 else 1.0)

        z = np.linspace(eps, z_max, n_points)
        A = self.warp_factor(z)

        integrand = np.exp((self.d - 1) * A)
        c_eff = float(np.trapz(integrand, z))
        return c_eff

# ============================================================
#  GEOMETRÍA BASE
# ============================================================

def get_phase11_geometries() -> List[Tuple[HiddenGeometry, str]]:
    """
    Prototipos de geometría (base) para la Fase XI.
    Cada una se clonará con jitter para generar múltiples universos.
    """
    geos: List[Tuple[HiddenGeometry, str]] = []

    # --- known ---
    geos.append((
        HiddenGeometry(
            name="ads_d3_Tfinite",
            family="ads",
            category="known",
            d=3,
            z_h=1.0,
            theta=0.0,
            z_dyn=1.0,
            deformation=0.0,
            L=1.0,
            metadata={"description": "AdS_4-Schwarzschild toy"}
        ),
        "known",
    ))

    # --- test (control positivo AdS) ---
    geos.append((
        HiddenGeometry(
            name="ads_d3_Tfinite_test",
            family="ads",
            category="test",
            d=3,
            z_h=1.0,
            theta=0.0,
            z_dyn=1.0,
            deformation=0.0,
            L=1.0,
            metadata={"description": "AdS_4-Schwarzschild toy (test)"}
        ),
        "test",
    ))

    geos.append((
        HiddenGeometry(
            name="lifshitz_d3_z2",
            family="lifshitz",
            category="known",
            d=3,
            z_h=1.0,
            theta=0.0,
            z_dyn=2.0,
            deformation=0.0,
            L=1.0,
            metadata={"description": "Lifshitz z=2, d=3"}
        ),
        "known",
    ))

    geos.append((
        HiddenGeometry(
            name="hvlf_d3_theta1",
            family="hyperscaling",
            category="known",
            d=3,
            z_h=1.2,
            theta=1.0,
            z_dyn=1.0,
            deformation=0.0,
            L=1.0,
            metadata={"description": "HV-Lifshitz theta=1, d=3"}
        ),
        "known",
    ))

    # --- test ---
    geos.append((
        HiddenGeometry(
            name="ads_deformed_d3",
            family="deformed",
            category="test",
            d=3,
            z_h=0.8,
            theta=0.0,
            z_dyn=1.0,
            deformation=0.2,
            L=1.0,
            metadata={"description": "AdS deformado suave"}
        ),
        "test",
    ))

    geos.append((
        HiddenGeometry(
            name="lifshitz_deformed_d3",
            family="lifshitz",
            category="test",
            d=3,
            z_h=1.1,
            theta=0.3,
            z_dyn=1.5,
            deformation=0.15,
            L=1.0,
            metadata={"description": "Lifshitz deformado, z≈1.5"}
        ),
        "test",
    ))

    # --- unknown ---
    geos.append((
        HiddenGeometry(
            name="unknown_family_1",
            family="unknown",
            category="unknown",
            d=3,
            z_h=1.0,
            theta=0.5,
            z_dyn=1.3,
            deformation=0.3,
            L=1.0,
            metadata={"description": "Familia desconocida 1"}
        ),
        "unknown",
    ))

    geos.append((
        HiddenGeometry(
            name="unknown_family_2",
            family="unknown",
            category="unknown",
            d=4,
            z_h=1.3,
            theta=0.2,
            z_dyn=1.1,
            deformation=0.4,
            L=1.0,
            metadata={"description": "Familia desconocida 2"}
        ),
        "unknown",
    ))

    return geos


def jitter_geometry(
    base: HiddenGeometry,
    rng: np.random.Generator,
    z_h_jitter: float = 0.1,
    theta_jitter: float = 0.2,
    z_dyn_jitter: float = 0.3,
    deformation_jitter: float = 0.2,
) -> HiddenGeometry:
    """
    Copia la geometría base con ligeras variaciones de parámetros.
    Sirve para generar muchos universos por familia sin colapsar al learner
    en unos pocos casos finitos.
    """
    geo = HiddenGeometry(**asdict(base))

    if geo.z_h is not None:
        zh_factor = 1.0 + z_h_jitter * (2 * rng.random() - 1)
        geo.z_h = max(0.3, geo.z_h * zh_factor)

    geo.theta = geo.theta + theta_jitter * (2 * rng.random() - 1)
    geo.z_dyn = max(0.5, geo.z_dyn + z_dyn_jitter * (2 * rng.random() - 1))
    geo.deformation = geo.deformation + deformation_jitter * (2 * rng.random() - 1)

    geo.metadata["jittered"] = True
    return geo


# ============================================================
#  OPERADORES Y CORRELADORES EN EL BOUNDARY
# ============================================================

def generate_operators_for_geometry(
    geo: HiddenGeometry,
    n_ops: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Espectro toy de operadores escalares O_i
    """
    deltas = np.sort(geo.d / 2 + 0.5 + 2.0 * rng.random(n_ops))
    ops: List[Dict] = []

    for i, Delta in enumerate(deltas):
        # Relación masa-dimensión AdS_{d+1}: m²L² = Δ(Δ-d)
        m2L2 = float(Delta * (Delta - geo.d))
        ops.append(
            {
                "name": f"O{i+1}",
                "Delta": float(Delta),
                "m2L2": m2L2,
                "spin": 0,
            }
        )

    return ops


def correlator_2pt_thermal(
    x: np.ndarray,
    Delta: float,
    d: int,
    T: float,
) -> np.ndarray:
    """
    <O(x)O(0)> a temperatura T:
        T = 0: G2 ~ 1/|x|^{2Δ}
        T > 0: G2 ~ (πT / sinh(πT x))^{2Δ}
    """
    x = np.asarray(x)
    x_safe = np.maximum(np.abs(x), 1e-8)
    prefactor = 1.0

    if T < 1e-12:
        return prefactor / (x_safe ** (2 * Delta))
    else:
        arg = np.pi * T * x_safe
        arg = np.clip(arg, 1e-6, 50.0)
        return prefactor * (np.pi * T / np.sinh(arg)) ** (2 * Delta)


def generate_boundary_data(
    geo: HiddenGeometry,
    operators: List[Dict],
    n_samples: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Genera datos del boundary (LO ÚNICO visible al learner).

    Devuelve un dict con:
        - x_grid
        - temperature
        - G2_<O> para cada operador
        - omega_grid, k_grid, G_R_real, G_R_imag
    """
    d = geo.d

    # Temperatura aproximada a partir del horizonte
    if geo.z_h is not None and geo.z_h > 0:
        T = d / (4.0 * np.pi * geo.z_h)
    else:
        T = 0.0

    data: Dict[str, np.ndarray] = {}

    # Grid espacial
    x_grid = np.linspace(0.1, 10.0, n_samples)
    data["x_grid"] = x_grid
    data["temperature"] = np.array([T], dtype=float)
    data["x_grid"] = x_grid
    data["temperature"] = np.array([T], dtype=float)

    # Observable escalar de borde: "central charge" toy
    c_eff = geo.effective_central_charge()
    data["central_charge_eff"] = np.array([c_eff], dtype=float)

    # (opcional pero recomendable) exportar d explícito en boundary
    data["d"] = np.array([geo.d], dtype=np.int32)

    # 2-pt correlators para cada operador
    for op in operators:
        name = op["name"]
        Delta = op["Delta"]
        G2 = correlator_2pt_thermal(x_grid, Delta, d, T)
        data[f"G2_{name}"] = G2.astype(np.float32)

    # Respuesta lineal toy G_R(ω, k)
    omega_grid = np.linspace(0.1, 10.0, 50)
    k_grid = np.linspace(0.0, 5.0, 30)

    if T > 0:
        omega_qnm = 2 * np.pi * T * 0.5 - 1j * np.pi * T
    else:
        omega_qnm = 1.0 - 0.1j

    OMEGA, K = np.meshgrid(omega_grid, k_grid)
    G_R = 1.0 / (OMEGA - np.real(omega_qnm) - 1j * np.abs(np.imag(omega_qnm)) + 1e-3)

    data["omega_grid"] = omega_grid
    data["k_grid"] = k_grid
    data["G_R_real"] = np.real(G_R).astype(np.float32)
    data["G_R_imag"] = np.imag(G_R).astype(np.float32)

    return data


# ============================================================
#  BACKEND EMD (OPCIONAL)
# ============================================================

def _ricci_from_A_f(z: np.ndarray, A: np.ndarray, f: np.ndarray, d: int) -> np.ndarray:
    """
    Calcula el escalar de Ricci R para una métrica estática plano-simétrica,
    usando el mismo esquema que HiddenGeometry.ricci_scalar pero con A,f explícitos.
    """
    D = d + 1
    if len(z) > 1:
        dz = z[1] - z[0]
    else:
        dz = 1e-3

    dA = np.gradient(A, dz)
    d2A = np.gradient(dA, dz)
    df = np.gradient(f, dz)

    R = -2.0 * D * d2A - D * (D - 1) * dA ** 2 - (df * dA) / (f + 1e-10)
    return R


def generate_lifshitz_from_emd(
    d: int,
    z_dyn: float,
    theta: float,
    r_h: float = 1.0,
    lam: float = 1.0,
    Q: float = 1.0,
    phi_h: float = 0.5,
    r_uv: float = 1e-3,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera A(z), f(z) usando soluciones EMD reales.
    Retorna: (z_grid, A_z, f_z) en coordenada z = 1/r.
    """
    if not HAS_EMD or EMDLifshitzSolver is None:
        raise ImportError("EMDLifshitzSolver no disponible")

    solver = EMDLifshitzSolver(d=d, z=z_dyn, theta=theta, lam=lam, Q=Q, r_h=r_h)
    sol = solver.solve(phi_h=phi_h, r_uv=r_uv)

    if sol is None or getattr(sol, "success", True) is False:
        raise ValueError(f"EMD solver failed for d={d}, z={z_dyn}, θ={theta}")

    r_grid = np.asarray(sol.t)
    f_r = np.asarray(sol.y[0])

    # z = 1/r (boundary en z→0)
    z_grid = 1.0 / r_grid
    order = np.argsort(z_grid)
    z_grid = z_grid[order]
    f_z = f_r[order]

    # Warp factor compatible con Lifshitz/HV:
    # g_xx ~ z^{2θ/d} y g_tt ~ z^{-2z_dyn}, así que:
    with np.errstate(divide="ignore"):
        A_z = (theta / d - z_dyn) * np.log(z_grid)

    return z_grid, A_z, f_z


def generate_bulk_truth(
    geo: HiddenGeometry,
    z_grid: np.ndarray,
    use_emd: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Genera la "verdad" del bulk para validación.
    El learner NO tiene acceso a esto durante el entrenamiento.

    Si use_emd=True y HAS_EMD=True y family ∈ {lifshitz,hyperscaling},
    se usa EMDLifshitzSolver como backend. En caso de fallo, hay fallback
    silencioso (con warning) al modo toy analítico.
    """
    # Por defecto: backend analítico toy
    A_truth = geo.warp_factor(z_grid)
    f_truth = geo.blackening_factor(z_grid)

    if use_emd and HAS_EMD and geo.family in ["lifshitz", "hyperscaling"]:
        try:
            z_emd, A_emd, f_emd = generate_lifshitz_from_emd(
                d=geo.d,
                z_dyn=geo.z_dyn,
                theta=geo.theta,
                r_h=geo.z_h if geo.z_h else 1.0,
            )
            # Interpola a la malla de z deseada
            A_truth = np.interp(z_grid, z_emd, A_emd)
            f_truth = np.interp(z_grid, z_emd, f_emd)
        except Exception as e:
            print(f"   [WARN] EMD fallback to toy for {geo.name}: {e}")

    # R y traza de Einstein consistentes con A,f
    R_truth = _ricci_from_A_f(z_grid, A_truth, f_truth, geo.d)
    D = geo.d + 1
    G_trace_truth = (1.0 - D / 2.0) * R_truth

    return {
        "z_grid": z_grid,
        "A_truth": A_truth,
        "f_truth": f_truth,
        "R_truth": R_truth,
        "G_trace_truth": G_trace_truth,
        "family": geo.family,
        "d": geo.d,
        "z_h": geo.z_h if geo.z_h else 0.0,
        "theta": geo.theta,
        "z_dyn": geo.z_dyn,
    }


# ============================================================
#  LOOP PRINCIPAL Y I/O
# ============================================================

def make_geometry_instance(
    base: HiddenGeometry,
    category: str,
    idx: int,
    rng: np.random.Generator,
) -> HiddenGeometry:
    """
    Crea una copia de `base` con nombre único y pequeños jitters en parámetros
    según la familia.

    Jitters aplicados:
    - z_h: factor multiplicativo [0.7, 1.3], clipped a [0.3, 3.0]
    - d: con prob 0.5, cambiar a otro valor en {3, 4, 5}
    - z_dyn (lifshitz): uniforme en [1.5, 3.0]
    - theta (hyperscaling): uniforme en [0.3, min(d-0.5, 2.0)]
    - deformation (deformed): uniforme en [0.05, 0.3]
    """

    # --- CONTROL EINSTEIN PURO (sin jitter) -----------------------------
    # Para las geometrías AdS puras conocidas (ads5_pure, ads4_pure)
    # queremos al menos una instancia EXACTAMENTE igual al prototipo:
    #   - z_h = None  → T = 0
    #   - family = "ads"
    #   - R(z) constante = -D(D-1)/L² (ver HiddenGeometry.ricci_scalar)
    if (
        base.family == "ads"
        and base.name in ("ads5_pure", "ads4_pure")
        and category == "known"
        and idx == 0
        and base.z_h is None
    ):
        params = asdict(base)
        params["name"] = f"{base.name}_{category}_{idx:03d}"
        return HiddenGeometry(**params)

    # --- CÓDIGO ORIGINAL DE JITTER -------------------------------------
    params = asdict(base)

    # Nombre único: base_category_idx
    params["name"] = f"{base.name}_{category}_{idx:03d}"

    # Copia de trabajo
    d = params.get("d", base.d)

    # Jitters por familia
    family = base.family

    # --- Horizonte ---
    z_h = params.get("z_h", base.z_h)
    if z_h is not None and z_h > 0:
        # Perturbación suave multiplicativa
        factor = rng.uniform(0.7, 1.3)
        z_h = float(np.clip(z_h * factor, 0.3, 3.0))
    else:
        # A veces introducimos un pequeño horizonte en geometrías sin él
        if rng.random() < 0.3:
            z_h = float(rng.uniform(0.8, 2.0))
        else:
            z_h = None
    params["z_h"] = z_h

    # --- Dimensión del boundary ---
    # Usamos valores razonables: 3, 4 o 5
    if rng.random() < 0.5:
        params["d"] = int(rng.choice([3, 4, 5]))
    else:
        params["d"] = int(d)

    # --- Jitters específicos por familia ---
    if family == "lifshitz":
        # z_dyn > 1 típico
        params["z_dyn"] = float(rng.uniform(1.5, 3.0))
    elif family == "hyperscaling":
        d_eff = params["d"]
        max_theta = max(0.5, min(d_eff - 0.5, 2.0))
        params["theta"] = float(rng.uniform(0.3, max_theta))
    elif family == "deformed":
        params["deformation"] = float(rng.uniform(0.05, 0.3))
    elif family == "ads":
        # Nada especial extra, ads ya se controla con d y z_h
        pass
    else:
        # unknown: sólo tocamos z_h y d (ya hechos arriba)
        pass

    return HiddenGeometry(**params)


def main():
    parser = argparse.ArgumentParser(
        description="Fase XI v3: Generación de datos para emergencia geométrica (escalable)"
    )
    parser.add_argument("--output-dir", type=str, default="fase11_data")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-operators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--n-z", type=int, default=100)

    # nuevos argumentos v3
    parser.add_argument(
        "--n-known",
        type=int,
        default=20,
        help="Número de universos por geometría base con category='known'",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=10,
        help="Número de universos por geometría base con category='test'",
    )
    parser.add_argument(
        "--n-unknown",
        type=int,
        default=5,
        help="Número de universos por geometría base con category='unknown'",
    )
    parser.add_argument(
        "--use-emd-lifshitz",
        action="store_true",
        help="Usa EMDLifshitzSolver para familias lifshitz/hyperscaling si está disponible",
    )
    
    parser.add_argument(
        "--ads-only",
        action="store_true",
        help=(
            "Si se activa, filtra las geometrías base para quedarse solo con "
            "familia='ads' (control positivo AdS puro)."
        ),
    )
    add_standard_arguments(parser)

    args = parse_stage_args(parser)

    ctx = StageContext.from_args(args, stage_number="01", stage_slug="generate_sandbox_geometries")

    status = STATUS_OK
    exit_code = EXIT_OK
    error_message = None

    try:
        ctx.record_artifact(ctx.stage_dir)
    except Exception:
        pass

    try:
        import h5py  # type: ignore
        global np, gamma_func  # type: ignore
        import numpy as np  # type: ignore
        from scipy.special import gamma as gamma_func  # type: ignore

        rng = np.random.default_rng(args.seed)
        output_dir = ctx.stage_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # malla en z común a todos los universos
        z_grid = np.linspace(0.01, args.z_max, args.n_z)

        base_geometries = get_phase11_geometries()
        
        # Modo control positivo: solo geometrías AdS
        if args.ads_only:
            base_geometries = [
                (geo, cat)
                for (geo, cat) in base_geometries
                if geo.family == "ads"
            ]
            if not base_geometries:
                raise RuntimeError(
                    "ads-only solicitado, pero get_phase11_geometries() no contiene ninguna familia='ads'."
                )
            print("[MODO CONTROL POSITIVO] Filtrando sandbox a familia='ads' únicamente.")
        
        geometries: List[Tuple[HiddenGeometry, str]] = []

        # expandir con jitter
        for base_geo, category in base_geometries:
            if category == "known":
                n_instances = args.n_known
            elif category == "test":
                n_instances = args.n_test
            else:
                n_instances = args.n_unknown

            for k in range(n_instances):
                geo = make_geometry_instance(base_geo, category, k, rng)
                geometries.append((geo, category))

        # resumen inicial
        n_known_total = sum(1 for _, cat in geometries if cat == "known")
        n_test_total = sum(1 for _, cat in geometries if cat == "test")
        n_unknown_total = sum(1 for _, cat in geometries if cat == "unknown")

        print("=" * 70)
        print("FASE XI v3 — EMERGENCIA AUTÓNOMA DE GEOMETRÍA")
        print("=" * 70)
        print(f"Output:       {output_dir}")
        print(f"Prototipos:   {len(base_geometries)}")
        print(f"Geometrías:   {len(geometries)} total")
        print(f"  - known:    {n_known_total}")
        print(f"  - test:     {n_test_total}")
        print(f"  - unknown:  {n_unknown_total}")
        print(f"Operadores:   {args.n_operators}")
        print(f"z grid:       [0.01, {args.z_max}]  {args.n_z}")
        print(f"EMD backend:  {'ON' if args.use_emd_lifshitz and HAS_EMD else 'OFF'}")
        print("=" * 70)

        manifest: Dict = {
            "geometries": [],
            "version": "v3",
            "config": {
                "n_known_per_base": args.n_known,
                "n_test_per_base": args.n_test,
                "n_unknown_per_base": args.n_unknown,
                "n_samples": args.n_samples,
                "n_operators": args.n_operators,
                "z_max": args.z_max,
                "n_z": args.n_z,
                "seed": args.seed,
                "use_emd_lifshitz": args.use_emd_lifshitz,
            },
        }

        # loop principal
        for idx, (geo, category) in enumerate(geometries):
            print(f"[{idx+1:04d}/{len(geometries):04d}] {geo.name} ({geo.family}, {category})")

            # operadores
            operators = generate_operators_for_geometry(geo, args.n_operators, rng)
            deltas_str = ", ".join(f"{op['Delta']:.2f}" for op in operators)
            zh_display = geo.z_h if geo.z_h is not None else 0.0
            print(f"   d={geo.d}, z_h={zh_display:.3f}, θ={geo.theta:.2f}, z_dyn={geo.z_dyn:.2f}")
            print(f"   Δ: [{deltas_str}]")

            # ============================================================
            # FIX 2025-12-21: Guardrail IO v1 ANTES de generar datos
            # ============================================================
            # Si el nombre codifica "_d<k>_", debe coincidir con geo.d
            # IMPORTANTE: esto debe ejecutarse ANTES de generar boundary_data
            # y bulk_truth para que ambos usen el valor correcto de d.
            m_d = re.search(r"_d(\d+)_", geo.name)
            if m_d is not None:
                d_name = int(m_d.group(1))
                if int(geo.d) != d_name:
                    print(
                        f"[IO_CONTRACT][AUTO-FIX] d mismatch: {geo.name}: geo.d={geo.d} -> {d_name} (from name)"
                    )
                    geo.d = d_name

            # boundary (VISIBLE para el learner)
            boundary_data = generate_boundary_data(geo, operators, args.n_samples, rng)

            # bulk (solo para validación/contratos)
            bulk_truth = generate_bulk_truth(geo, z_grid, use_emd=args.use_emd_lifshitz)

            # guardar en HDF5
            output_path = output_dir / f"{geo.name}.h5"
            with h5py.File(output_path, "w") as f:
                # attrs globales
                f.attrs["name"] = geo.name
                f.attrs["system_name"] = geo.name
                f.attrs["family"] = geo.family
                f.attrs["category"] = category
                f.attrs["d"] = geo.d
                f.attrs["z_h"] = geo.z_h if geo.z_h is not None else 0.0
                f.attrs["theta"] = geo.theta
                f.attrs["z_dyn"] = geo.z_dyn
                f.attrs["deformation"] = geo.deformation
                f.attrs["operators"] = json.dumps(operators)

                # boundary
                bgrp = f.create_group("boundary")
                for key, val in boundary_data.items():
                    if isinstance(val, np.ndarray):
                        bgrp.create_dataset(key, data=val)
                    else:
                        bgrp.attrs[key] = val
                bgrp.attrs["d"] = geo.d
                bgrp.attrs["family"] = geo.family

                # bulk_truth
                tgrp = f.create_group("bulk_truth")
                for key, val in bulk_truth.items():
                    if isinstance(val, np.ndarray):
                        tgrp.create_dataset(key, data=val)
                    else:
                        tgrp.attrs[key] = val

            # entrada en manifest
            manifest["geometries"].append(
                {
                    "name": geo.name,
                    "family": geo.family,
                    "category": category,
                    "d": geo.d,
                    "file": str(output_path.name),
                    "operators": [op["name"] for op in operators],
                }
            )

        # escribir manifest
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # resumen final por familia
        families: Dict[str, Dict[str, int]] = {}
        for geo, category in geometries:
            fam = geo.family
            if fam not in families:
                families[fam] = {"known": 0, "test": 0, "unknown": 0}
            families[fam][category] = families[fam].get(category, 0) + 1

        print("\n" + "=" * 70)
        print("RESUMEN POR FAMILIA")
        for fam, counts in sorted(families.items()):
            print(
                f"  {fam:15s}: "
                f"known={counts.get('known', 0):3d}, "
                f"test={counts.get('test', 0):3d}, "
                f"unknown={counts.get('unknown', 0):3d}"
            )

        print("\n" + "=" * 70)
        print("✓ GENERACIÓN FASE XI v3 COMPLETADA")
        print(f"  Manifest: {manifest_path}")
        print(f"  Total:    {len(geometries)} universos")
        print("=" * 70)
        print("\nPróximo paso: 01_emergent_geometry_v2.py")
        print("El learner solo verá datos del boundary — debe descubrir la geometría.")

        ctx.record_artifact(output_dir)
        ctx.record_artifact(manifest_path)
        ctx.write_manifest(
            outputs={"sandbox_dir": str(output_dir.relative_to(ctx.run_root))},
            metadata={"command": " ".join(sys.argv)},
        )
    except Exception as exc:  # pragma: no cover - infra guardrail
        status = STATUS_ERROR
        exit_code = EXIT_ERROR
        error_message = str(exc)
        raise
    finally:
        summary_path = ctx.stage_dir / "stage_summary.json"
        try:
            ctx.record_artifact(summary_path)
        except Exception:
            pass
        ctx.write_summary(status=status, exit_code=exit_code, error_message=error_message)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
