#!/usr/bin/env python3
"""
fase12_real_data_adapters.py â€” Fase XII: Adaptadores para Datos Reales

OBJETIVO:
    Conectar el motor de emergencia geomÃ©trica (Fase XI) con datos fÃ­sicos reales:
    - Conformal Bootstrap (Ising 3D, O(N), etc.)
    - Lattice QCD (ecuaciÃ³n de estado, espectros)
    - Materia condensada (strange metals, cupratos)
    - CosmologÃ­a (CMB, perturbaciones)

ARQUITECTURA:
    Datos Reales â†’ Adapter â†’ Features Boundary EstÃ¡ndar â†’ Motor XI â†’ Bulk Emergente
"""

import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import h5py

# ============================================================
# INTERFAZ BASE PARA ADAPTADORES
# ============================================================

@dataclass
class BoundaryDataStandard:
    """
    Formato estÃ¡ndar de datos del boundary.
    Cualquier adaptador debe producir este formato.
    """
    name: str
    source: str  # "bootstrap", "lattice", "condensed", "cosmology", "synthetic"
    d: int  # dimensiÃ³n del boundary
    
    # Espectro de operadores
    operators: List[Dict] = field(default_factory=list)
    
    # Correladores
    x_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    G2_data: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Temperatura/horizonte
    T: float = 0.0
    has_horizon: bool = False
    
    # Features extraÃ­dos
    features: Dict[str, float] = field(default_factory=dict)
    
    # Metadata del sistema fÃ­sico original
    physics_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_hdf5(self, path: Path):
        """Guarda en formato HDF5 compatible con motor XI."""
        with h5py.File(path, "w") as f:
            f.attrs["name"] = self.name
            f.attrs["source"] = self.source
            f.attrs["d"] = self.d
            f.attrs["T"] = self.T
            f.attrs["has_horizon"] = self.has_horizon
            f.attrs["operators"] = json.dumps(self.operators)
            f.attrs["physics_metadata"] = json.dumps(self.physics_metadata)
            
            boundary = f.create_group("boundary")
            boundary.create_dataset("x_grid", data=self.x_grid)
            for key, val in self.G2_data.items():
                boundary.create_dataset(key, data=val)
            
            features_grp = f.create_group("features")
            for key, val in self.features.items():
                features_grp.attrs[key] = val


class RealDataAdapter(ABC):
    """Clase base abstracta para adaptadores de datos reales."""
    
    @abstractmethod
    def load(self, source_path: Path) -> None:
        """Carga datos desde archivo o API."""
        pass
    
    @abstractmethod
    def to_boundary_standard(self) -> BoundaryDataStandard:
        """Convierte a formato estÃ¡ndar del boundary."""
        pass
    
    @abstractmethod
    def get_known_predictions(self) -> Dict[str, Any]:
        """Devuelve predicciones conocidas para validaciÃ³n."""
        pass


# ============================================================
# ADAPTER: CONFORMAL BOOTSTRAP
# ============================================================

class BootstrapAdapter(RealDataAdapter):
    """
    Adaptador para datos del Conformal Bootstrap.
    
    Fuentes tÃ­picas:
    - Ising 3D: Î”Ïƒ â‰ˆ 0.518, Î”Îµ â‰ˆ 1.41
    - O(N) modelos
    - TeorÃ­as con simetrÃ­a conforme
    """
    
    def __init__(self):
        self.theory_name = ""
        self.d = 3
        self.operators = []
        self.ope_coefficients = {}
        self.central_charge = None
        
    def load(self, source_path: Path) -> None:
        """
        Carga datos de bootstrap.
        Formato esperado: JSON con dimensiones y OPE.
        """
        data = json.loads(source_path.read_text())
        
        raw_theory = data.get("theory") or data.get("name") or "unknown_cft"
        # Normalizamos Ising 3D a un nombre canÃ³nico
        if "ising" in raw_theory.lower():
            self.theory_name = "ising_3d"
        else:
            self.theory_name = raw_theory

        self.d = data.get("d", 3)

        
        # Operadores primarios
        self.operators = []
        for op in data.get("operators", []):
            self.operators.append({
                "name": op["name"],
                "Delta": op["Delta"],
                "spin": op.get("spin", 0),
                "Delta_error": op.get("Delta_error", op.get("error", 0.001))
            })

        # Coeficientes OPE
        self.ope_coefficients = data.get("ope_coefficients", {})
        self.central_charge = data.get("central_charge", None)
    
    def _generate_correlators_from_spectrum(self) -> Tuple[np.ndarray, Dict]:
        """Genera correladores a partir del espectro."""
        x_grid = np.linspace(0.1, 10.0, 100)
        G2_data = {}
        
        for op in self.operators:
            Delta = op["Delta"]
            name = op["name"]
            # G_2(x) âˆ 1/|x|^{2Î”}
            G2 = 1.0 / (x_grid ** (2 * Delta) + 1e-20)
            # NormalizaciÃ³n
            G2 = G2 / G2[0]
            G2_data[f"G2_{name}"] = G2
        
        return x_grid, G2_data
    
    def _extract_bootstrap_features(self) -> Dict[str, float]:
        """Extrae features especÃ­ficos de bootstrap."""
        features = {}
        
        if self.operators:
            Deltas = [op["Delta"] for op in self.operators]
            features["Delta_min"] = min(Deltas)
            features["Delta_gap"] = Deltas[1] - Deltas[0] if len(Deltas) > 1 else 0.0
            features["n_relevant"] = sum(1 for D in Deltas if D < self.d)
        
        if self.central_charge:
            features["central_charge"] = self.central_charge
        
        # Ratios de OPE (importantes para holografÃ­a)
        if self.ope_coefficients:
            ope_vals = list(self.ope_coefficients.values())
            if len(ope_vals) > 1:
                features["ope_ratio_12"] = ope_vals[0] / (ope_vals[1] + 1e-10)
        
        return features
    
    def to_boundary_standard(self) -> BoundaryDataStandard:
        x_grid, G2_data = self._generate_correlators_from_spectrum()
        features = self._extract_bootstrap_features()
        
        return BoundaryDataStandard(
            name=self.theory_name,
            source="bootstrap",
            d=self.d,
            operators=self.operators,
            x_grid=x_grid,
            G2_data=G2_data,
            T=0.0,
            has_horizon=False,
            features=features,
            physics_metadata={
                "ope_coefficients": self.ope_coefficients,
                "central_charge": self.central_charge
            }
        )
    
    def get_known_predictions(self) -> Dict[str, Any]:
        """Predicciones conocidas para CFTs bien entendidas."""
        known = {}
        
        if "ising" in self.theory_name.lower():
            known["expected_bulk"] = "AdS4 or slight deformation"
            known["expected_einstein"] = True
            known["expected_theta"] = 0.0  # No hyperscaling violation
        
        return known


# ============================================================
# ADAPTER: LATTICE QCD
# ============================================================

class LatticeQCDAdapter(RealDataAdapter):
    """
    Adaptador para datos de Lattice QCD.
    
    Datos tÃ­picos:
    - EcuaciÃ³n de estado: p(T), Îµ(T), s(T)
    - Espectros de hadrones
    - Viscosidades: Î·/s
    """
    
    def __init__(self):
        self.T_grid = np.array([])
        self.pressure = np.array([])
        self.energy_density = np.array([])
        self.entropy = np.array([])
        self.eta_over_s = None
        self.Tc = None  # Temperatura crÃ­tica
        
    def load(self, source_path: Path) -> None:
        """Carga datos de lattice."""
        with h5py.File(source_path, "r") as f:
            self.T_grid = f["T"][:]
            self.pressure = f["pressure"][:]
            self.energy_density = f["energy_density"][:]
            self.entropy = f.get("entropy", f["pressure"][:] / self.T_grid)[:]
            
            if "eta_over_s" in f:
                self.eta_over_s = f["eta_over_s"][:]
            
            self.Tc = f.attrs.get("Tc", 0.15)  # GeV
    
    def _map_eos_to_correlators(self) -> Tuple[np.ndarray, Dict]:
        """
        Mapea ecuaciÃ³n de estado a correladores efectivos.
        Usa la relaciÃ³n hologrÃ¡fica Îµ + p âˆ T^{d+1} s
        """
        # Escala espacial efectiva desde temperatura
        x_grid = 1.0 / (self.T_grid + 1e-10)
        
        G2_data = {}
        
        # El correlador del tensor de energÃ­a-momento
        # <T_{00}(x) T_{00}(0)> estÃ¡ relacionado con fluctuaciones
        trace_anomaly = self.energy_density - 3 * self.pressure
        G2_data["G2_Tmunu"] = np.abs(trace_anomaly) / (np.max(np.abs(trace_anomaly)) + 1e-20)
        
        # Speed of sound â†’ Ã­ndice del correlador
        cs2 = np.gradient(self.pressure) / (np.gradient(self.energy_density) + 1e-10)
        G2_data["cs2"] = np.clip(cs2, 0, 1)
        
        return x_grid, G2_data
    
    def _extract_qcd_features(self) -> Dict[str, float]:
        """Features especÃ­ficos de QCD."""
        features = {}
        
        # Punto crÃ­tico
        features["Tc"] = float(self.Tc)
        
        # Velocidad del sonido en diferentes regÃ­menes
        cs2 = np.gradient(self.pressure) / (np.gradient(self.energy_density) + 1e-10)
        features["cs2_max"] = float(np.max(cs2))
        features["cs2_at_Tc"] = float(np.interp(self.Tc, self.T_grid, cs2))
        
        # AnomalÃ­a conforme
        trace = self.energy_density - 3 * self.pressure
        features["trace_anomaly_peak"] = float(np.max(np.abs(trace)))
        
        # Î·/s si disponible (relacionado con horizonte)
        if self.eta_over_s is not None:
            features["eta_over_s_min"] = float(np.min(self.eta_over_s))
            features["has_horizon"] = 1.0
        else:
            features["has_horizon"] = 0.0
        
        return features
    
    def _infer_operators(self) -> List[Dict]:
        """Infiere operadores CFT efectivos desde datos QCD."""
        operators = []
        
        # T^{Î¼Î½} tiene Î” = d = 4 en 4D
        operators.append({"name": "Tmunu", "Delta": 4.0, "spin": 2})
        
        # Operador de quark condensate âˆ ÏˆÌ„Ïˆ
        # DimensiÃ³n canÃ³nica d-1 = 3, con correcciones anÃ³malas
        operators.append({"name": "qqbar", "Delta": 3.0, "spin": 0})
        
        # Gluon condensate âˆ F^2
        operators.append({"name": "F2", "Delta": 4.0, "spin": 0})
        
        return operators
    
    def to_boundary_standard(self) -> BoundaryDataStandard:
        x_grid, G2_data = self._map_eos_to_correlators()
        features = self._extract_qcd_features()
        operators = self._infer_operators()
        
        T_avg = float(np.mean(self.T_grid))
        
        return BoundaryDataStandard(
            name="lattice_qcd",
            source="lattice",
            d=4,
            operators=operators,
            x_grid=x_grid,
            G2_data=G2_data,
            T=T_avg,
            has_horizon=self.eta_over_s is not None,
            features=features,
            physics_metadata={
                "Tc_GeV": self.Tc,
                "T_range_GeV": [float(self.T_grid.min()), float(self.T_grid.max())]
            }
        )
    
    def get_known_predictions(self) -> Dict[str, Any]:
        return {
            "expected_bulk": "AdS5-Schwarzschild deformed",
            "expected_einstein": True,
            "eta_over_s_bound": 1.0 / (4 * np.pi),  # KSS bound
            "expected_horizon": True
        }


# ============================================================
# ADAPTER: MATERIA CONDENSADA (STRANGE METALS)
# ============================================================

class CondensedMatterAdapter(RealDataAdapter):
    """
    Adaptador para datos de materia condensada.
    
    Sistemas tÃ­picos:
    - Strange metals (cupratos)
    - Heavy fermions
    - Sistemas con T-lineal resistividad
    """
    
    def __init__(self):
        self.T_grid = np.array([])
        self.resistivity = np.array([])
        self.conductivity = np.array([])
        self.omega_grid = np.array([])
        self.optical_conductivity = np.array([])
        self.material = ""
        
    def load(self, source_path: Path) -> None:
        """Carga datos de transporte."""
        data = json.loads(source_path.read_text())
        
        self.material = data.get("material", "strange_metal")
        self.T_grid = np.array(data["T_K"])  # Kelvin
        self.resistivity = np.array(data["rho_uOhm_cm"])  # Î¼Î©Â·cm
        
        if "omega_eV" in data:
            self.omega_grid = np.array(data["omega_eV"])
            self.optical_conductivity = np.array(data["sigma_omega"])
    
    def _extract_transport_features(self) -> Dict[str, float]:
        """Extrae features de transporte."""
        features = {}
        
        # Exponente de resistividad: Ï âˆ T^Î±
        log_T = np.log(self.T_grid[self.T_grid > 0])
        log_rho = np.log(self.resistivity[self.T_grid > 0])
        
        if len(log_T) > 2:
            alpha, _ = np.polyfit(log_T, log_rho, 1)
            features["rho_exponent"] = float(alpha)
        else:
            features["rho_exponent"] = 1.0
        
        # Para strange metals: Î± â‰ˆ 1
        features["is_strange_metal"] = float(abs(features["rho_exponent"] - 1.0) < 0.2)
        
        # Rango de T-lineal
        features["T_linear_range_K"] = float(self.T_grid.max() - self.T_grid.min())
        
        return features
    
    def _infer_scaling_exponents(self) -> Dict[str, float]:
        """
        Infiere exponentes de scaling para bulk hologrÃ¡fico.
        
        Strange metal â†’ posible Lifshitz o hyperscaling violating
        Ï âˆ T â†’ z = 1 (relativista) o z â‰  1 (Lifshitz)
        """
        features = self._extract_transport_features()
        
        exponents = {}
        alpha = features.get("rho_exponent", 1.0)
        
        # Para Lifshitz: Ï âˆ T^{(d-2)/z}
        # Si Î± = 1 y d = 2+1: z = 1
        # Si Î± â‰  1: z = (d-2)/Î±
        d = 3  # 2+1 dimensional
        if abs(alpha) > 0.1:
            exponents["z_lifshitz"] = (d - 2) / alpha
        else:
            exponents["z_lifshitz"] = 1.0
        
        # Hyperscaling violating: Î¸
        # s âˆ T^{(d-Î¸)/z}
        exponents["theta_candidate"] = 0.0  # Default, refinar con mÃ¡s datos
        
        return exponents
    
    def _map_to_correlators(self) -> Tuple[np.ndarray, Dict]:
        """Mapea datos de transporte a correladores."""
        # x efectivo desde temperatura
        x_grid = 1.0 / (self.T_grid + 1e-10)
        x_grid = x_grid / x_grid[0]  # Normalizar
        
        G2_data = {}
        
        # Correlador de corriente <J J> desde conductividad
        sigma = 1.0 / (self.resistivity + 1e-10)
        G2_data["G2_JJ"] = sigma / np.max(sigma)
        
        if len(self.optical_conductivity) > 0:
            G2_data["sigma_omega"] = self.optical_conductivity
            G2_data["omega"] = self.omega_grid
        
        return x_grid, G2_data
    
    def _infer_operators(self) -> List[Dict]:
        """Infiere operadores desde propiedades de transporte."""
        exponents = self._infer_scaling_exponents()
        z = exponents.get("z_lifshitz", 1.0)
        
        operators = []
        
        # Corriente J tiene Î” = d - 1 para z = 1
        # Para Lifshitz: Î”_J = d - 2 + z
        d = 3
        Delta_J = d - 2 + z
        operators.append({"name": "J", "Delta": Delta_J, "spin": 1})
        
        # Densidad de energÃ­a
        operators.append({"name": "epsilon", "Delta": d + z - 1, "spin": 0})
        
        return operators
    
    def to_boundary_standard(self) -> BoundaryDataStandard:
        x_grid, G2_data = self._map_to_correlators()
        features = self._extract_transport_features()
        features.update(self._infer_scaling_exponents())
        operators = self._infer_operators()
        
        T_avg = float(np.mean(self.T_grid))
        
        return BoundaryDataStandard(
            name=self.material,
            source="condensed",
            d=3,
            operators=operators,
            x_grid=x_grid,
            G2_data=G2_data,
            T=T_avg,
            has_horizon=True,  # Temperatura finita implica horizonte
            features=features,
            physics_metadata={
                "material": self.material,
                "T_range_K": [float(self.T_grid.min()), float(self.T_grid.max())],
                "rho_exponent": features.get("rho_exponent", 1.0)
            }
        )
    
    def get_known_predictions(self) -> Dict[str, Any]:
        features = self._extract_transport_features()
        
        if features.get("is_strange_metal", 0) > 0.5:
            return {
                "expected_bulk": "Lifshitz zâ‰ˆ1 or AdS2 Ã— R2",
                "expected_einstein": False,  # Puede necesitar correcciones
                "expected_z_dyn": 1.0,
                "expected_theta": 0.0
            }
        else:
            return {
                "expected_bulk": "AdS4 deformed",
                "expected_einstein": True
            }


# ============================================================
# ADAPTER: COSMOLOGÃA (CMB/dS)
# ============================================================

class CosmologyAdapter(RealDataAdapter):
    """
    Adaptador para datos cosmolÃ³gicos.
    
    Interpreta observables cosmolÃ³gicos como "boundary" de dS o FLRW.
    """
    
    def __init__(self):
        self.ell = np.array([])  # Multipolos
        self.Cl_TT = np.array([])  # Espectro de potencia TT
        self.Cl_EE = np.array([])  # PolarizaciÃ³n
        self.ns = None  # Ãndice espectral
        self.As = None  # Amplitud
        self.H0 = None
        
    def load(self, source_path: Path) -> None:
        """Carga datos de CMB o espectro primordial."""
        data = json.loads(source_path.read_text())
        
        self.ell = np.array(data.get("ell", []))
        self.Cl_TT = np.array(data.get("Cl_TT", []))
        self.Cl_EE = np.array(data.get("Cl_EE", []))
        
        # ParÃ¡metros cosmolÃ³gicos
        self.ns = data.get("ns", 0.965)
        self.As = data.get("As", 2.1e-9)
        self.H0 = data.get("H0", 67.4)
    
    def _map_cmb_to_correlators(self) -> Tuple[np.ndarray, Dict]:
        """
        Mapea espectro CMB a correladores.
        El CMB es efectivamente <Î¶ Î¶> donde Î¶ es curvatura.
        """
        if len(self.ell) == 0:
            # Generar mock data
            self.ell = np.arange(2, 2500)
            # Espectro aproximado
            self.Cl_TT = 1e-9 * (self.ell / 200.0) ** (-0.04) * np.exp(-(self.ell / 2000)**2)
        
        # x "angular" desde multipolo
        x_grid = np.pi / (self.ell + 1e-10)
        
        G2_data = {}
        G2_data["G2_zeta"] = self.Cl_TT * self.ell * (self.ell + 1)  # Dl = l(l+1)Cl
        
        if len(self.Cl_EE) > 0:
            G2_data["G2_E"] = self.Cl_EE * self.ell * (self.ell + 1)
        
        return x_grid, G2_data
    
    def _extract_inflationary_features(self) -> Dict[str, float]:
        """Extrae features del perÃ­odo inflacionario."""
        features = {}
        
        # Ãndice espectral: ns - 1 = -2Îµ - Î· donde Îµ, Î· son slow-roll
        features["ns"] = float(self.ns) if self.ns else 0.965
        features["ns_minus_1"] = features["ns"] - 1.0
        
        # Relacionar con dimensiÃ³n del inflaton
        # En dS/CFT: Î” = 3/2 Â± âˆš(9/4 - mÂ²/HÂ²)
        # ns - 1 â‰ˆ -2Îµ se relaciona con masa del inflaton
        features["slow_roll_epsilon"] = -features["ns_minus_1"] / 2.0
        
        # Hubble
        if self.H0:
            features["H0"] = float(self.H0)
        
        return features
    
    def _infer_operators(self) -> List[Dict]:
        """Infiere operadores de la "CFT dual" a dS."""
        operators = []
        
        # El inflaton Ï† tiene Î” relacionado con masa
        ns = self.ns if self.ns else 0.965
        # AproximaciÃ³n: Î” â‰ˆ 3 + (ns - 1)/2
        Delta_phi = 3.0 + (ns - 1.0) / 2.0
        operators.append({"name": "inflaton", "Delta": Delta_phi, "spin": 0})
        
        # Tensor de energÃ­a-momento del graviton
        # En dS4: Î”_T = 4 (marginal)
        operators.append({"name": "Tmunu", "Delta": 4.0, "spin": 2})
        
        # Curvatura Î¶
        operators.append({"name": "zeta", "Delta": 0.0, "spin": 0})  # Marginal en el sentido de dS
        
        return operators
    
    def to_boundary_standard(self) -> BoundaryDataStandard:
        x_grid, G2_data = self._map_cmb_to_correlators()
        features = self._extract_inflationary_features()
        operators = self._infer_operators()
        
        return BoundaryDataStandard(
            name="cmb_inflation",
            source="cosmology",
            d=3,  # Rebanada espacial 3D
            operators=operators,
            x_grid=x_grid,
            G2_data=G2_data,
            T=0.0,  # T=0 efectivamente para primordial
            has_horizon=True,  # Horizonte de Hubble
            features=features,
            physics_metadata={
                "ns": self.ns,
                "As": self.As,
                "H0": self.H0,
                "interpretation": "dS/CFT or cosmological holography"
            }
        )
    
    def get_known_predictions(self) -> Dict[str, Any]:
        return {
            "expected_bulk": "dS4 or FLRW",
            "expected_einstein": True,
            "expected_lambda": "positive",
            "note": "dS/CFT es mÃ¡s especulativo que AdS/CFT"
        }


# ============================================================
# GENERADOR DE DATOS SINTÃ‰TICOS REALISTAS
# ============================================================

def generate_synthetic_bootstrap_data(theory: str, output_path: Path):
    """Genera datos sintÃ©ticos tipo bootstrap para testing."""
    
    if theory == "ising3d":
        data = {
            "theory": "ising_3d",
            "d": 3,
            "operators": [
                {"name": "sigma", "Delta": 0.518, "spin": 0, "error": 0.001},
                {"name": "epsilon", "Delta": 1.41, "spin": 0, "error": 0.01},
                {"name": "epsilon_prime", "Delta": 3.83, "spin": 0, "error": 0.05},
                {"name": "T", "Delta": 3.0, "spin": 2, "error": 0.001}
            ],
            "ope_coefficients": {
                "sigma_sigma_epsilon": 1.0518,
                "epsilon_epsilon_epsilon": 1.532
            },
            "central_charge": 0.946
        }
    elif theory == "on_n4":
        data = {
            "theory": "O4_model",
            "d": 3,
            "operators": [
                {"name": "phi", "Delta": 0.519, "spin": 0},
                {"name": "s", "Delta": 1.51, "spin": 0},
                {"name": "t", "Delta": 1.24, "spin": 2}
            ],
            "central_charge": 1.1
        }
    else:
        # GenÃ©rico
        data = {
            "theory": theory,
            "d": 3,
            "operators": [
                {"name": "O1", "Delta": 1.0, "spin": 0},
                {"name": "O2", "Delta": 2.0, "spin": 0}
            ]
        }
    
    output_path.write_text(json.dumps(data, indent=2))


def generate_synthetic_lattice_data(output_path: Path):
    """Genera datos sintÃ©ticos tipo lattice QCD."""
    
    T = np.linspace(0.1, 0.5, 50)  # GeV
    Tc = 0.155  # GeV
    
    # EcuaciÃ³n de estado tipo crossover
    t = (T - Tc) / Tc
    
    # PresiÃ³n: transiciÃ³n suave
    p = T**4 * (0.5 * (1 + np.tanh(2 * t)) * 0.9 + 0.1)
    
    # Densidad de energÃ­a
    eps = 3 * p + T * np.gradient(p, T)
    
    # EntropÃ­a
    s = (eps + p) / T
    
    # Î·/s cerca del bound
    eta_s = 0.08 * np.ones_like(T) + 0.1 * np.exp(-((T - Tc) / 0.05)**2)
    
    with h5py.File(output_path, "w") as f:
        f.create_dataset("T", data=T)
        f.create_dataset("pressure", data=p)
        f.create_dataset("energy_density", data=eps)
        f.create_dataset("entropy", data=s)
        f.create_dataset("eta_over_s", data=eta_s)
        f.attrs["Tc"] = Tc


def generate_synthetic_transport_data(material: str, output_path: Path):
    """Genera datos sintÃ©ticos de transporte."""
    
    T = np.linspace(10, 300, 100)  # Kelvin
    
    if "strange" in material.lower():
        # T-lineal
        rho = 10 + 0.5 * T
    else:
        # Fermi liquid: T^2
        rho = 10 + 0.01 * T**2
    
    data = {
        "material": material,
        "T_K": T.tolist(),
        "rho_uOhm_cm": rho.tolist()
    }
    
    output_path.write_text(json.dumps(data, indent=2))


def generate_synthetic_cmb_data(output_path: Path):
    """Genera datos sintÃ©ticos tipo CMB."""
    
    ell = np.arange(2, 2000)
    
    # Espectro simplificado
    ns = 0.965
    As = 2.1e-9
    
    # Dl = l(l+1)Cl / 2Ï€
    Dl = As * (ell / 200.0) ** (ns - 1) * np.exp(-(ell / 1500)**2) * 1e10
    
    data = {
        "ell": ell.tolist(),
        "Cl_TT": (Dl / (ell * (ell + 1)) * 2 * np.pi).tolist(),
        "ns": ns,
        "As": As,
        "H0": 67.4
    }
    
    output_path.write_text(json.dumps(data, indent=2))


# ============================================================
# PIPELINE: PROCESAR DATOS REALES
# ============================================================

def process_real_data(
    adapter: RealDataAdapter,
    source_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Pipeline completo para procesar datos reales.
    """
    # 1. Cargar datos
    adapter.load(source_path)
    
    # 2. Convertir a formato estÃ¡ndar
    boundary_data = adapter.to_boundary_standard()
    
    # 3. Guardar para motor XI
    output_path = output_dir / f"{boundary_data.name}.h5"
    boundary_data.to_hdf5(output_path)
    
    # 4. Obtener predicciones conocidas para validaciÃ³n
    known = adapter.get_known_predictions()
    
    return {
        "name": boundary_data.name,
        "source": boundary_data.source,
        "output_file": str(output_path),
        "n_operators": len(boundary_data.operators),
        "has_horizon": boundary_data.has_horizon,
        "features": boundary_data.features,
        "known_predictions": known
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fase XII: Adaptadores para datos reales"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate-synthetic", "process"],
                        help="Modo de operaciÃ³n")
    parser.add_argument("--source", type=str, default="",
                        help="Archivo o tipo de fuente")
    parser.add_argument("--adapter", type=str, default="bootstrap",
                        choices=["bootstrap", "lattice", "condensed", "cosmology"],
                        help="Tipo de adaptador")
    parser.add_argument("--output-dir", type=str, default="fase12_data",
                        help="Directorio de salida")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "generate-synthetic":
        print("=" * 70)
        print("FASE XII â€” GENERACIÃ“N DE DATOS SINTÃ‰TICOS REALISTAS")
        print("=" * 70)
        
        # Bootstrap: Ising 3D
        generate_synthetic_bootstrap_data("ising3d", output_dir / "ising3d_bootstrap.json")
        print(f"âœ“ Ising 3D bootstrap â†’ {output_dir / 'ising3d_bootstrap.json'}")
        
        # Bootstrap: O(4)
        generate_synthetic_bootstrap_data("on_n4", output_dir / "o4_bootstrap.json")
        print(f"âœ“ O(4) bootstrap â†’ {output_dir / 'o4_bootstrap.json'}")
        
        # Lattice QCD
        generate_synthetic_lattice_data(output_dir / "lattice_qcd.h5")
        print(f"âœ“ Lattice QCD â†’ {output_dir / 'lattice_qcd.h5'}")
        
        # Strange metal
        generate_synthetic_transport_data("strange_metal_cuprate", 
                                         output_dir / "strange_metal.json")
        print(f"âœ“ Strange metal â†’ {output_dir / 'strange_metal.json'}")
        
        # CMB
        generate_synthetic_cmb_data(output_dir / "cmb_planck.json")
        print(f"âœ“ CMB data â†’ {output_dir / 'cmb_planck.json'}")
        
        print("\n" + "=" * 70)
        print("âœ“ Datos sintÃ©ticos generados")
        print("  Siguiente: python fase12_real_data_adapters.py --mode process ...")
        print("=" * 70)
        
    elif args.mode == "process":
        if not args.source:
            print("Error: --source requerido para modo process")
            return 1
        
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: no existe {source_path}")
            return 1
        
        # Seleccionar adaptador
        adapters = {
            "bootstrap": BootstrapAdapter,
            "lattice": LatticeQCDAdapter,
            "condensed": CondensedMatterAdapter,
            "cosmology": CosmologyAdapter
        }
        
        adapter = adapters[args.adapter]()
        
        print("=" * 70)
        print(f"FASE XII â€” PROCESANDO: {source_path}")
        print(f"Adaptador: {args.adapter}")
        print("=" * 70)
        
        result = process_real_data(adapter, source_path, output_dir)
        
        print(f"\n  Nombre:       {result['name']}")
        print(f"  Fuente:       {result['source']}")
        print(f"  Operadores:   {result['n_operators']}")
        print(f"  Horizonte:    {result['has_horizon']}")
        print(f"  Output:       {result['output_file']}")
        
        if result['known_predictions']:
            print("\n  Predicciones conocidas:")
            for k, v in result['known_predictions'].items():
                print(f"    {k}: {v}")
        
        # Guardar manifest
        manifest_path = output_dir / "manifest_fase12.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {"processed": []}
        
        manifest["processed"].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        print(f"\nâœ“ Procesado â†’ {result['output_file']}")
        print(f"  Manifest:  {manifest_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
