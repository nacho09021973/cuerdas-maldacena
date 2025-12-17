#!/usr/bin/env python3
# 02_emergent_geometry_engine.py
# CUERDAS — Bloque A: Geometría emergente (motor de reconstrucción) V 2.2
#
# CAMBIOS EN V 2.2 (respecto a V 2.1):
#   - Añadido modo --mode {train, inference}
#   - En mode='inference': carga checkpoint sandbox, procesa boundary-only, genera .h5
#   - No se accede a bulk_truth en inference (honestidad preservada)
#   - Salida inference: .h5 en geometry_emergent/ con contrato 3.3 del README
#
# CAMBIOS PRINCIPALES EN V 2.1:
#   - Priorización de A(z) y f(z) sobre R(z) y clasificación de familia
#   - Pesos de loss expuestos como constantes configurables al inicio del script
#   - Normalización robusta mejorada (z-score con clipping para R)
#   - Scheduler CosineAnnealing actualizado por EPOCH (no por batch)
#   - Bucle de entrenamiento refactorizado en funciones auxiliares
#   - Métricas R²(A), R²(f) calculadas y mostradas periódicamente en test
#   - Arquitectura refinada: factor residual adaptativo, mejor inicialización
#   - Physics loss documentada con explicación de cada término
#   - Summary JSON extendido con métricas por familia y evolución temporal
#
# OBJETIVO
#   Aprender la geometría de bulk "emergente" a partir de datos de frontera (CFT),
#   sin acceso directo a la métrica real ni al solver de campo.
#   Produce campos A(z), f(z), R(z), etc. para cada universo.
#
# ENTRADAS
#   - runs/sandbox_geometries/boundary/*.h5
#       * Datos CFT por universo: grids, correladores, espectros, etc.
#   - Opcional: configuración de red y entrenamiento:
#       * n_epochs, device, seed, arquitectura, regularización, ...
#
# SALIDAS
#   runs/emergent_geometry/
#     geometry_emergent/
#       <system_name>_emergent.h5
#         - z_grid, A_emergent, f_emergent, R_emergent, ...
#         - metadatos: family_pred, scores, etc.
#     emergent_geometry_summary.json
#       - Métricas de ajuste (train/test), R², errores, familia predicha, ...
#
# RELACIÓN CON OTROS SCRIPTS
#   - Usa como input: boundary/ generados por 01_generate_sandbox_geometries.py
#   - Proporciona geometría emergente a:
#       * 03_discover_bulk_equations.py
#       * 04_geometry_physics_contracts.py
#       * 06_build_bulk_eigenmodes_dataset.py
#       * 08_build_holographic_dictionary.py
#
# HONESTIDAD
#   - No se inyecta bulk_truth en la loss ni en las features de entrenamiento.
#   - Cualquier comparación con bulk_truth sucede aguas abajo, en contratos y análisis.
#   - En modo inference, NO se accede a bulk_truth (CuerdasDataLoader lo bloquea).

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURACIÓN DE PESOS DE LOSS (MODIFICAR AQUÍ)
# ============================================================
# Estos pesos controlan la importancia relativa de cada término.
# V2.1: Prioriza A y f sobre R y familia para datasets pequeños.

LOSS_WEIGHT_A = 2.0          # Warp factor A(z) - PRIORITARIO
LOSS_WEIGHT_F = 2.0          # Blackening factor f(z) - PRIORITARIO
LOSS_WEIGHT_R = 0.001        # Escalar de Ricci R(z) - MUY BAJO (secundario)
LOSS_WEIGHT_ZH = 0.1         # Posición del horizonte z_h - BAJO
LOSS_WEIGHT_FAMILY = 0.05    # Clasificación de familia - MUY BAJO
LOSS_WEIGHT_PHYSICS = 0.05   # Regularización física genérica
LOSS_WEIGHT_PHYSICS_ADS = 0.02  # Regularización específica AdS

# Learning rate y scheduler
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# Frecuencia de evaluación en test (cada N epochs)
EVAL_FREQUENCY = 50


# ============================================================
# UTILIDADES VARIAS
# ============================================================

def set_torch_seed(seed: int = 42):
    """Fija semillas para reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² clásico, con protección para casos degenerados."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 3:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-10:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error con protección."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 1:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


# ============================================================
# EXTRACCIÓN DE FEATURES DEL BOUNDARY
# ============================================================

def extract_correlator_features(G2: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """
    Extrae features básicos del correlador G2(x).
    
    Features extraídos:
    - G2_log_slope: pendiente en escala log-log (relacionado con Δ)
    - G2_log_curvature: curvatura en log-log (desviaciones de power-law)
    - G2_small_x: valor en x pequeño (UV)
    - G2_large_x: valor en x grande (IR)
    """
    features = {}
    
    G2 = np.asarray(G2, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Valores por defecto si no hay datos suficientes
    mask = (G2 > 0) & np.isfinite(G2) & np.isfinite(x) & (x > 0)
    if np.sum(mask) < 5:
        features["G2_log_slope"] = 0.0
        features["G2_log_curvature"] = 0.0
        features["G2_small_x"] = float(G2[0]) if len(G2) > 0 else 0.0
        features["G2_large_x"] = float(G2[-1]) if len(G2) > 0 else 0.0
        return features
    
    x_log = np.log(x[mask] + 1e-8)
    G2_log = np.log(G2[mask] + 1e-8)
    
    try:
        # Ajuste lineal en log-log: G2 ~ x^{-alpha}
        coeffs = np.polyfit(x_log, G2_log, 1)
        slope = coeffs[0]
        features["G2_log_slope"] = float(np.clip(slope, -20, 20))
    except Exception:
        features["G2_log_slope"] = 0.0
    
    try:
        # Curvatura efectiva: ajuste cuadrático
        coeffs2 = np.polyfit(x_log, G2_log, 2)
        curvature = coeffs2[0]
        features["G2_log_curvature"] = float(np.clip(curvature, -10, 10))
    except Exception:
        features["G2_log_curvature"] = 0.0
    
    features["G2_small_x"] = float(np.clip(G2[0], 0, 1e6)) if len(G2) > 0 else 0.0
    features["G2_large_x"] = float(np.clip(G2[-1], 0, 1e6)) if len(G2) > 0 else 0.0
    
    return features


def extract_thermal_features(G2: np.ndarray, x: np.ndarray, T: float) -> Dict[str, float]:
    """
    Features térmicos básicos: presencia de horizonte, escala térmica, etc.
    
    Features extraídos:
    - temperature: temperatura normalizada
    - has_horizon: indicador binario de existencia de horizonte
    - thermal_scale: escala térmica β = 1/T
    - exponential_decay: tasa de decaimiento exponencial a alta T
    """
    features = {}
    
    G2 = np.asarray(G2, dtype=float)
    x = np.asarray(x, dtype=float)
    
    T_arr = np.asarray(T, dtype=float)
    T_scalar = float(T_arr.ravel()[0]) if T_arr.size > 0 else 0.0

    features["temperature"] = float(np.clip(T_scalar, 0, 10))
    features["has_horizon"] = float(T_scalar > 1e-10)

    if T_scalar > 1e-10:
        beta = 1.0 / T_scalar
        features["thermal_scale"] = float(np.clip(beta, 0.1, 100))

        valid = (x > beta) & (G2 > 0) & np.isfinite(G2)
        if np.sum(valid) > 3:
            log_G2 = np.log(G2[valid] + 1e-20)
            try:
                slope_exp = np.polyfit(x[valid], log_G2, 1)[0]
                features["exponential_decay"] = float(np.clip(-slope_exp, -10, 10))
            except Exception:
                features["exponential_decay"] = 0.0
        else:
            features["exponential_decay"] = 0.0
    else:
        features["thermal_scale"] = 0.0
        features["exponential_decay"] = 0.0

    return features


def extract_spectral_features(operators: List[Dict]) -> Dict[str, float]:
    """
    Extrae features del espectro de operadores.
    
    Features extraídos:
    - n_ops: número de operadores
    - Delta_min, Delta_max, Delta_mean: estadísticas de las dimensiones conformes
    """
    features = {}
    
    if not operators:
        features["n_ops"] = 0.0
        features["Delta_min"] = 0.0
        features["Delta_max"] = 0.0
        features["Delta_mean"] = 0.0
        return features
    
    deltas = [op.get("Delta", 0.0) for op in operators]
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        features["n_ops"] = 0.0
        features["Delta_min"] = 0.0
        features["Delta_max"] = 0.0
        features["Delta_mean"] = 0.0
        return features
    
    features["n_ops"] = float(len(deltas))
    features["Delta_min"] = float(np.min(deltas))
    features["Delta_max"] = float(np.max(deltas))
    features["Delta_mean"] = float(np.mean(deltas))
    
    return features


def extract_response_features(
    G_R_real: np.ndarray,
    G_R_imag: np.ndarray,
    omega: np.ndarray,
    k: np.ndarray,
) -> Dict[str, float]:
    """
    Features de la respuesta de Green retardada G_R(omega, k).
    
    Features extraídos:
    - GR_peak_height: altura máxima del pico (relacionado con QNMs)
    - GR_peak_width: ancho del pico (relacionado con lifetime)
    """
    features = {}
    
    G_R_real = np.asarray(G_R_real, dtype=float)
    G_R_imag = np.asarray(G_R_imag, dtype=float)
    
    if G_R_real.ndim != 2 or G_R_imag.ndim != 2:
        return {"GR_peak_height": 0.0, "GR_peak_width": 0.0}
    
    magnitude = np.sqrt(G_R_real**2 + G_R_imag**2)
    mag_flat = magnitude.reshape(-1)
    
    if mag_flat.size == 0:
        return {"GR_peak_height": 0.0, "GR_peak_width": 0.0}
    
    peak_height = float(np.max(mag_flat))
    if peak_height <= 0:
        return {"GR_peak_height": 0.0, "GR_peak_width": 0.0}
    
    thresh = 0.5 * peak_height
    mask = mag_flat >= thresh
    peak_width = float(np.sum(mask) / mag_flat.size)
    
    features["GR_peak_height"] = float(np.clip(peak_height, 0, 1e3))
    features["GR_peak_width"] = float(np.clip(peak_width, 0, 1.0))
    return features


def build_feature_vector(boundary_data: Dict[str, Any], operators: List[Dict]) -> np.ndarray:
    """
    Construye vector de features a partir de boundary_data + operators.
    
    El vector tiene estructura fija para garantizar consistencia entre geometrías.
    """
    all_features: List[float] = []
    
    # 1. Features de correlador G2 vs x (4 features)
    x_grid = boundary_data.get("x_grid", np.linspace(0.1, 10, 100))
    
    # Buscar cualquier correlador G2_* disponible
    G2 = None
    for key in boundary_data:
        if key.startswith("G2_") and isinstance(boundary_data[key], np.ndarray):
            G2 = boundary_data[key]
            break
    
    if G2 is not None:
        corr_feats = extract_correlator_features(G2, x_grid)
        for k in ["G2_log_slope", "G2_log_curvature", "G2_small_x", "G2_large_x"]:
            all_features.append(corr_feats.get(k, 0.0))
    else:
        all_features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 2. Features térmicos (4 features)
    T = boundary_data.get("temperature", boundary_data.get("T", 0.0))
    if isinstance(T, np.ndarray):
        T = float(T.ravel()[0]) if T.size > 0 else 0.0
    
    if G2 is not None:
        thermal_feats = extract_thermal_features(G2, x_grid, T)
        for k in ["temperature", "has_horizon", "thermal_scale", "exponential_decay"]:
            all_features.append(thermal_feats.get(k, 0.0))
    else:
        all_features.extend([float(T), float(T > 1e-10), 0.0, 0.0])
    
    # 3. Features espectrales de operadores (4 features)
    spec_feats = extract_spectral_features(operators)
    for k in ["n_ops", "Delta_min", "Delta_max", "Delta_mean"]:
        all_features.append(spec_feats.get(k, 0.0))
    
    # 4. Features de respuesta G_R (2 features)
    if "G_R_real" in boundary_data and "G_R_imag" in boundary_data:
        G_R_real = boundary_data["G_R_real"]
        G_R_imag = boundary_data["G_R_imag"]
        omega = boundary_data.get("omega_grid", np.linspace(0.1, 10, 50))
        k = boundary_data.get("k_grid", np.linspace(0, 5, 30))
        resp_feats = extract_response_features(G_R_real, G_R_imag, omega, k)
        all_features.append(resp_feats.get("GR_peak_height", 0.0))
        all_features.append(resp_feats.get("GR_peak_width", 0.0))
    else:
        all_features.extend([0.0, 0.0])

    # 5. Observable escalar opcional: "central charge" toy (1 feature)
    c_eff = boundary_data.get("central_charge_eff", None)
    if c_eff is not None:
        c_eff_arr = np.asarray(c_eff, dtype=float)
        c_eff_val = float(c_eff_arr.ravel()[0]) if c_eff_arr.size > 0 else 0.0
    else:
        c_eff_val = 0.0
    all_features.append(c_eff_val)
    
    # 6. Dimensión d como feature explícita (1 feature)
    d = boundary_data.get("d", 4)
    if isinstance(d, np.ndarray):
        d = int(d.ravel()[0]) if d.size > 0 else 4
    all_features.append(float(d))
    
    # Total: 16 features (siempre rellenamos central_charge_eff; 0.0 si falta)
    return np.array(all_features, dtype=np.float32)

# ============================================================
# CARGA SEGURA DE DATOS (BULK vs BOUNDARY)
# ============================================================

class CuerdasDataLoader:
    """
    Encapsula el acceso a los datos de geometría.
    
    - En todos los modos se leen SIEMPRE los datos de *boundary* y la metainformación.
    - El acceso a `bulk_truth` (A_truth, f_truth, R_truth, etc.) SOLO está permitido
      cuando `mode == "train"`.
      
    Esto implementa a nivel de código la separación Sandbox vs. Discovery: cualquier
    script que quiera reutilizar este loader en modo `inference` podrá leer el
    boundary, pero recibirá un error claro si intenta acceder al bulk.
    """
    
    def __init__(self, mode: str = "train"):
        if mode not in ("train", "inference"):
            raise ValueError(f"Modo no reconocido para CuerdasDataLoader: {mode}")
        self.mode = mode
    
    def load_boundary_and_meta(self, f: h5py.File) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Carga datos de frontera y operadores desde un archivo HDF5.
        
        Esta función NO toca `bulk_truth` y es segura en cualquier modo.
        """
        boundary_group = f["boundary"]
        boundary_data: Dict[str, Any] = {}
        
        for key in boundary_group.keys():
            boundary_data[key] = boundary_group[key][:]
        for key in boundary_group.attrs.keys():
            boundary_data[key] = boundary_group.attrs[key]
        
        operators_raw = f.attrs.get("operators", "[]")
        if isinstance(operators_raw, bytes):
            operators_raw = operators_raw.decode("utf-8")
        operators = json.loads(operators_raw)
        return boundary_data, operators
    
    def load_bulk_truth(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, int]:
        """
        Accede al grupo `bulk_truth` solo en modo entrenamiento.
        
        En modo `inference` lanza una excepción explícita para impedir fugas.
        """
        if self.mode != "train":
            raise RuntimeError(
                "Acceso a bulk_truth bloqueado en modo inference/discovery. " 
                "Use solo los datos de boundary en este modo."
            )
        
        if "bulk_truth" not in f:
            raise KeyError("El archivo HDF5 no contiene grupo 'bulk_truth'")
        
        bulk = f["bulk_truth"]
        A_truth = bulk["A_truth"][:]
        f_truth = bulk["f_truth"][:]
        R_truth = bulk["R_truth"][:]
        z_grid = bulk["z_grid"][:]
        z_h = bulk.attrs.get("z_h", 0.0)
        family = bulk.attrs.get("family", "unknown")
        if isinstance(family, bytes):
            family = family.decode("utf-8")
        d_value = int(bulk.attrs.get("d", 4))
        
        return A_truth, f_truth, R_truth, z_grid, float(z_h), str(family), d_value


# ============================================================
# NORMALIZACIÓN ROBUSTA DE TARGETS (V2.1 MEJORADA)
# ============================================================

class TargetNormalizer:
    """
    Normaliza targets de forma robusta para entrenamiento.
    
    V2.1: Usa z-score robusto (mediana + MAD) para A, y percentiles
    con clipping para R que puede tener valores muy extremos.
    """
    
    def __init__(self):
        self.A_mean = None
        self.A_std = None
        self.R_mean = None
        self.R_std = None
        self.f_mean = None
        self.f_std = None
        
    def fit(self, A: np.ndarray, f: np.ndarray, R: np.ndarray):
        """Calcula estadísticas de normalización."""
        # A: z-score robusto (mediana + MAD)
        A_flat = A.flatten()
        A_flat = A_flat[np.isfinite(A_flat)]
        self.A_mean = float(np.median(A_flat))
        mad = np.median(np.abs(A_flat - self.A_mean))
        self.A_std = float(max(mad * 1.4826, 0.1))  # 1.4826 para equivalencia con std normal
        
        # f: normalización simple (ya está en [0,1] típicamente)
        f_flat = f.flatten()
        f_flat = f_flat[np.isfinite(f_flat)]
        self.f_mean = float(np.mean(f_flat))
        self.f_std = float(max(np.std(f_flat), 0.1))
        
        # R: normalización con clipping agresivo (valores pueden ser muy grandes)
        R_flat = R.flatten()
        R_valid = R_flat[np.isfinite(R_flat) & (np.abs(R_flat) < 1e4)]
        if len(R_valid) > 0:
            # Usar percentiles para robustez ante outliers
            self.R_mean = float(np.percentile(R_valid, 50))
            iqr = np.percentile(R_valid, 75) - np.percentile(R_valid, 25)
            self.R_std = float(max(iqr / 1.35, 1.0))  # 1.35 para equivalencia con std normal
        else:
            self.R_mean = -20.0
            self.R_std = 10.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "TargetNormalizer":
        """Reconstruye un TargetNormalizer desde un diccionario (checkpoint)."""
        norm = cls()
        norm.A_mean = d.get("A_mean", 0.0)
        norm.A_std = d.get("A_std", 1.0)
        norm.f_mean = d.get("f_mean", 0.5)
        norm.f_std = d.get("f_std", 0.3)
        norm.R_mean = d.get("R_mean", -20.0)
        norm.R_std = d.get("R_std", 10.0)
        return norm
    
    def normalize_A(self, A: np.ndarray) -> np.ndarray:
        return (A - self.A_mean) / self.A_std
    
    def denormalize_A(self, A_norm: np.ndarray) -> np.ndarray:
        return A_norm * self.A_std + self.A_mean
    
    def normalize_f(self, f: np.ndarray) -> np.ndarray:
        # f ya está típicamente en [0,1], normalización suave
        return (f - self.f_mean) / self.f_std
    
    def denormalize_f(self, f_norm: np.ndarray) -> np.ndarray:
        return f_norm * self.f_std + self.f_mean
    
    def normalize_R(self, R: np.ndarray) -> np.ndarray:
        # Clipping antes de normalizar para evitar explosión
        R_clipped = np.clip(R, self.R_mean - 10 * self.R_std, self.R_mean + 10 * self.R_std)
        return (R_clipped - self.R_mean) / self.R_std
    
    def denormalize_R(self, R_norm: np.ndarray) -> np.ndarray:
        return R_norm * self.R_std + self.R_mean


# ============================================================
# EMERGENT GEOMETRY NETWORK (V2.1 MEJORADA)
# ============================================================

class EmergentGeometryNet(nn.Module):
    """
    Red que mapea features del boundary -> geometría del bulk.
    
    V2.1: 
    - Factor residual adaptativo (0.3 en lugar de 0.1)
    - Mejor inicialización de pesos
    - Dropout calibrado por capa
    - Decoder de f mejorado (sin Sigmoid, usa normalización)
    """
    
    def __init__(
        self,
        n_features: int,
        n_z: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_families: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_z = n_z
        self.n_families = n_families
        
        # Input projection con normalización
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # Bloques residuales tipo "Transformer-lite"
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout * (1 + i * 0.1)),  # Dropout creciente por capa
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Decoders separados para cada salida
        # A(z): warp factor
        self.decoder_A = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_z)
        )
        
        # f(z): blackening factor (sin Sigmoid - normalizado externamente)
        self.decoder_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_z)
        )
        
        # R(z): escalar de Ricci (secundario)
        self.decoder_R = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_z)
        )
        
        # z_h: posición del horizonte
        self.decoder_zh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Clasificación de familia
        self.decoder_family = nn.Linear(hidden_dim, n_families)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización cuidadosa para evitar salidas absurdas."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier con gain moderado
                nn.init.xavier_normal_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Inicialización especial para decoders: salidas cerca de 0
        for decoder in [self.decoder_A, self.decoder_f, self.decoder_R, self.decoder_zh]:
            if isinstance(decoder, nn.Sequential):
                last_layer = decoder[-1]
                if isinstance(last_layer, nn.Linear):
                    nn.init.xavier_normal_(last_layer.weight, gain=0.1)
                    nn.init.zeros_(last_layer.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Normalización de entrada
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = F.gelu(h)
        
        # Bloques residuales con factor adaptativo
        for layer, ln in zip(self.layers, self.layer_norms):
            residual = layer(h)
            h = ln(h + 0.3 * residual)  # Factor residual 0.3 (más expresivo que 0.1)
        
        h = self.final_norm(h)
        
        # Decoders
        A = self.decoder_A(h)
        f_raw = self.decoder_f(h)
        R = self.decoder_R(h)
        z_h = self.decoder_zh(h).squeeze(-1)
        family_logits = self.decoder_family(h)
        
        return {
            "A": A,
            "f": f_raw,  # Sin activación, normalizado externamente
            "R": R,
            "z_h": z_h,
            "family_logits": family_logits,
        }


# ============================================================
# PHYSICS-INFORMED LOSSES (V2.1 DOCUMENTADA)
# ============================================================

def physics_loss_generic(
    A: torch.Tensor, 
    f: torch.Tensor, 
    z: torch.Tensor, 
    d: int
) -> torch.Tensor:
    """
    Regularización física genérica aplicable a TODAS las geometrías.
    
    Términos incluidos:
    1. loss_curvature: penaliza segundas derivadas grandes (suavidad)
    2. loss_smooth: penaliza cambios bruscos en la primera derivada
    3. loss_monotonic_A: A debe decrecer en la región UV (cerca de z=0)
    4. loss_f_bounds: f debe estar aproximadamente en [0, 1]
    
    Estos son priors físicos muy genéricos que NO asumen la forma exacta
    de la solución (no inyectamos A = -log(z/L) ni nada similar).
    """
    n_z = A.shape[1]
    dz = z[1] - z[0]
    
    # Derivadas de A
    dA = (A[:, 2:] - A[:, :-2]) / (2 * dz)
    d2A = (A[:, 2:] - 2 * A[:, 1:-1] + A[:, :-2]) / (dz ** 2)
    
    # Derivadas de f
    d2f = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (dz ** 2)
    
    # 1. Penalizar curvatura excesiva (promueve suavidad)
    loss_curvature = torch.mean(d2A ** 2) + 0.1 * torch.mean(d2f ** 2)
    
    # 2. Penalizar cambios bruscos en pendiente (suavidad de orden superior)
    if dA.shape[1] > 1:
        loss_smooth = torch.mean((dA[:, 1:] - dA[:, :-1]) ** 2)
    else:
        loss_smooth = torch.tensor(0.0, device=A.device)
    
    # 3. A típicamente decrece hacia el interior (dA < 0 en UV)
    # Solo en la primera fracción del grid (región UV)
    n_uv = max(1, int(0.2 * dA.shape[1]))
    loss_monotonic_A = torch.mean(F.relu(dA[:, :n_uv]))  # Penaliza dA > 0 en UV
    
    # 4. f debe estar aproximadamente en [0, 1]
    loss_f_bounds = torch.mean(F.relu(-f) + F.relu(f - 1.0))
    
    # Pesos relativos dentro de esta loss
    total = (
        0.3 * loss_curvature + 
        0.3 * loss_smooth + 
        0.2 * loss_monotonic_A + 
        0.2 * loss_f_bounds
    )
    
    return total


def physics_loss_ads_specific(
    A: torch.Tensor,
    f: torch.Tensor,
    z: torch.Tensor,
    d: int,
    family_mask: torch.Tensor
) -> torch.Tensor:
    """
    Regularización adicional para geometrías clasificadas como AdS-like.
    
    Términos incluidos (solo para muestras con family="ads"):
    1. loss_ads_monotonic: A debe ser monótonamente decreciente en toda la región UV
    2. loss_f_uv: f debe tender a 1 cerca del borde (z → 0)
    
    NOTA: NO imponemos A = -log(z/L) explícitamente. Solo priors cualitativos.
    """
    if torch.sum(family_mask) == 0:
        return torch.tensor(0.0, device=A.device)
    
    A_ads = A[family_mask]
    f_ads = f[family_mask]
    
    n_z = A_ads.shape[1]
    dz = z[1] - z[0]
    
    # Derivada de A para muestras AdS
    dA = (A_ads[:, 2:] - A_ads[:, :-2]) / (2 * dz)
    
    # 1. A monótono decreciente en la mitad UV del grid
    n_uv_half = max(1, int(0.5 * dA.shape[1]))
    loss_ads_monotonic = torch.mean(F.relu(dA[:, :n_uv_half]))
    
    # 2. f → 1 en UV (primeros ~10% del grid)
    n_uv_small = max(1, int(0.1 * f_ads.shape[1]))
    f_uv = f_ads[:, :n_uv_small]
    loss_f_uv = torch.mean((f_uv - 1.0) ** 2)
    
    return 0.5 * loss_ads_monotonic + 0.5 * loss_f_uv


# ============================================================
# FUNCIONES DE ENTRENAMIENTO (V2.1 REFACTORIZADO)
# ============================================================

def train_one_epoch(
    model: nn.Module,
    X: torch.Tensor,
    Y_A: torch.Tensor,
    Y_f: torch.Tensor,
    Y_R: torch.Tensor,
    Y_zh: torch.Tensor,
    Y_family: torch.Tensor,
    z_t: torch.Tensor,
    d_value: int,
    family_map: Dict[str, int],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Entrena una época completa y devuelve las pérdidas medias.
    """
    model.train()
    n_train = X.shape[0]
    
    # Shuffle
    idx = torch.randperm(n_train, device=device)
    X = X[idx]
    Y_A = Y_A[idx]
    Y_f = Y_f[idx]
    Y_R = Y_R[idx]
    Y_zh = Y_zh[idx]
    Y_family = Y_family[idx]
    
    n_batches = int(np.ceil(n_train / batch_size))
    
    # Acumuladores
    losses = {
        "total": 0.0, "A": 0.0, "f": 0.0, "R": 0.0, 
        "zh": 0.0, "family": 0.0, "physics": 0.0, "physics_ads": 0.0
    }
    
    huber = nn.SmoothL1Loss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    
    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n_train)
        
        xb = X[start:end]
        yA = Y_A[start:end]
        yf = Y_f[start:end]
        yR = Y_R[start:end]
        yzh = Y_zh[start:end]
        yfam = Y_family[start:end]
        
        optimizer.zero_grad()
        
        out = model(xb)
        
        # Pérdidas de datos
        loss_A = huber(out["A"], yA)
        loss_f = mse(out["f"], yf)
        loss_R = huber(out["R"], yR)
        loss_zh = huber(out["z_h"], yzh)
        loss_family = ce(out["family_logits"], yfam)
        
        # Pérdidas físicas
        ads_mask = (yfam == family_map["ads"])
        loss_physics = physics_loss_generic(out["A"], out["f"], z_t, d_value)
        loss_physics_ads = physics_loss_ads_specific(out["A"], out["f"], z_t, d_value, ads_mask)
        
        # Pérdida total ponderada
        total = (
            LOSS_WEIGHT_A * loss_A + 
            LOSS_WEIGHT_F * loss_f + 
            LOSS_WEIGHT_R * loss_R +
            LOSS_WEIGHT_ZH * loss_zh + 
            LOSS_WEIGHT_FAMILY * loss_family +
            LOSS_WEIGHT_PHYSICS * loss_physics +
            LOSS_WEIGHT_PHYSICS_ADS * loss_physics_ads
        )
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        
        # Acumular para promedio
        batch_weight = (end - start) / n_train
        losses["total"] += float(total.item()) * batch_weight
        losses["A"] += float(loss_A.item()) * batch_weight
        losses["f"] += float(loss_f.item()) * batch_weight
        losses["R"] += float(loss_R.item()) * batch_weight
        losses["zh"] += float(loss_zh.item()) * batch_weight
        losses["family"] += float(loss_family.item()) * batch_weight
        losses["physics"] += float(loss_physics.item()) * batch_weight
        losses["physics_ads"] += float(loss_physics_ads.item()) * batch_weight
    
    return losses


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    X_test: torch.Tensor,
    Y_A_test: np.ndarray,
    Y_f_test: np.ndarray,
    Y_R_test: np.ndarray,
    Y_zh_test: np.ndarray,
    Y_family_test: np.ndarray,
    normalizer: TargetNormalizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evalúa el modelo en el conjunto de test y devuelve métricas.
    """
    model.eval()
    
    out = model(X_test)
    
    # Desnormalizar predicciones
    A_pred_norm = out["A"].cpu().numpy()
    f_pred_norm = out["f"].cpu().numpy()
    R_pred_norm = out["R"].cpu().numpy()
    zh_pred = out["z_h"].cpu().numpy()
    family_logits = out["family_logits"].cpu().numpy()
    family_pred = np.argmax(family_logits, axis=1)
    
    A_pred = normalizer.denormalize_A(A_pred_norm)
    f_pred = normalizer.denormalize_f(f_pred_norm)
    # Física: f(z) ≥ 0 por causalidad (signatura de la métrica),
    # y f(z) ≤ 1 en geometrías térmicas con horizonte bien definido
    f_pred = np.clip(f_pred, 0.0, 1.0)
    R_pred = normalizer.denormalize_R(R_pred_norm)
    
    metrics = {
        "A_r2": compute_r2(Y_A_test, A_pred),
        "f_r2": compute_r2(Y_f_test, f_pred),
        "R_r2": compute_r2(Y_R_test, R_pred),
        "zh_mae": compute_mae(zh_pred, Y_zh_test),
        "family_accuracy": float(np.mean(family_pred == Y_family_test)),
    }
    
    return metrics, A_pred, f_pred, R_pred, zh_pred, family_pred


# ============================================================
# INFERENCE DE GEOMETRÍA (BOUNDARY-ONLY) - V2.2 NUEVO
# ============================================================

@torch.no_grad()
def run_inference_single(
    model: nn.Module,
    X: np.ndarray,
    normalizer: TargetNormalizer,
    family_map_inv: Dict[int, str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Ejecuta inferencia sobre un único sistema (un vector de features).
    
    Args:
        model: Modelo cargado desde checkpoint
        X: Vector de features [n_features]
        normalizer: TargetNormalizer para desnormalizar salidas
        family_map_inv: Mapeo id -> nombre de familia
        device: Dispositivo torch
    
    Returns:
        Dict con A_pred, f_pred, R_pred, zh_pred, family_pred, family_name
    """
    model.eval()
    
    # Añadir dimensión de batch si es necesario
    if X.ndim == 1:
        X = X[np.newaxis, :]
    
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    out = model(X_t)
    
    # Desnormalizar
    A_pred_norm = out["A"].cpu().numpy()
    f_pred_norm = out["f"].cpu().numpy()
    R_pred_norm = out["R"].cpu().numpy()
    zh_pred = out["z_h"].cpu().numpy()
    family_logits = out["family_logits"].cpu().numpy()
    
    A_pred = normalizer.denormalize_A(A_pred_norm)
    f_pred = normalizer.denormalize_f(f_pred_norm)
    f_pred = np.clip(f_pred, 0.0, 1.0)
    R_pred = normalizer.denormalize_R(R_pred_norm)
    
    family_id = int(np.argmax(family_logits, axis=1)[0])
    family_name = family_map_inv.get(family_id, "unknown")
    
    return {
        "A_pred": A_pred[0],  # Quitar dimensión de batch
        "f_pred": f_pred[0],
        "R_pred": R_pred[0],
        "zh_pred": float(zh_pred[0]),
        "family_pred": family_id,
        "family_name": family_name,
    }


def run_inference_mode(args):
    """
    Modo inference: carga checkpoint sandbox, procesa boundary-only, genera .h5
    
    NO accede a bulk_truth en ningún momento.
    """
    print("=" * 70)
    print("FASE XI V2.2 - MODO INFERENCE (BOUNDARY-ONLY)")
    print("=" * 70)
    
    # Validar argumentos
    if args.checkpoint is None:
        raise ValueError(
            "En mode='inference' debes pasar --checkpoint con un modelo "
            "entrenado en sandbox"
        )
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  Datos:        {data_dir}")
    print(f"  Salida:       {output_dir}")
    print(f"  Dispositivo:  {device}")
    print("=" * 70)
    
    # === CARGAR CHECKPOINT ===
    print("\n>> Cargando checkpoint sandbox...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    
    # Extraer configuración del modelo
    n_features = ckpt["n_features"]
    n_z = ckpt["n_z"]
    hidden_dim = ckpt.get("hidden_dim", 256)
    n_layers = ckpt.get("n_layers", 4)
    family_map = ckpt.get("family_map", {"ads": 0, "lifshitz": 1, "hyperscaling": 2, "deformed": 3, "unknown": 4})
    family_map_inv = {v: k for k, v in family_map.items()}
    
    # Normalización de features (X)
    X_mean = ckpt.get("X_mean", np.zeros(n_features))
    X_std = ckpt.get("X_std", np.ones(n_features))
    if isinstance(X_mean, np.ndarray) and X_mean.ndim > 1:
        X_mean = X_mean.flatten()
    if isinstance(X_std, np.ndarray) and X_std.ndim > 1:
        X_std = X_std.flatten()
    
    # Normalización de targets
    normalizer = TargetNormalizer.from_dict(ckpt.get("normalizer", {}))
    
    # z_grid del checkpoint (CRÍTICO: usar el mismo que en train)
    z_grid = ckpt.get("z_grid", np.linspace(0.01, 5.0, n_z))
    d_value = ckpt.get("d", 4)
    
    print(f"   Modelo: n_features={n_features}, n_z={n_z}, hidden={hidden_dim}, layers={n_layers}")
    print(f"   z_grid: [{z_grid[0]:.3f}, {z_grid[-1]:.3f}], {len(z_grid)} puntos")
    print(f"   d (checkpoint): {d_value}")
    
    # === CREAR MODELO Y CARGAR PESOS ===
    model = EmergentGeometryNet(
        n_features=n_features,
        n_z=n_z,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_families=len(family_map)
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("   Modelo cargado correctamente")
    
    # === PREPARAR DIRECTORIOS DE SALIDA ===
    geom_dir = output_dir / "geometry_emergent"
    geom_dir.mkdir(parents=True, exist_ok=True)

    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    # === CARGAR MANIFEST ===
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No se encontró manifest.json en {data_dir}")
    
    manifest = json.loads(manifest_path.read_text())
    geometries = manifest.get("geometries", [])
    
    print(f"\n>> Procesando {len(geometries)} sistemas...")
    
    # === PROCESAR CADA SISTEMA ===
    loader = CuerdasDataLoader(mode="inference")  # BLOQUEA acceso a bulk_truth
    
    summary_entries = []
    
    for geo_info in geometries:
        name = geo_info["name"]
        h5_path = data_dir / f"{name}.h5"
        
        if not h5_path.exists():
            print(f"   [WARN] No existe: {h5_path}")
            continue
        
        print(f"   Procesando: {name}")
        
        with h5py.File(h5_path, "r") as f:
            # Solo carga boundary (bulk_truth bloqueado)
            boundary_data, operators = loader.load_boundary_and_meta(f)
        
        # Extraer d del boundary o manifest
        d_boundary = geo_info.get("d", boundary_data.get("d", d_value))
        if isinstance(d_boundary, np.ndarray):
            d_boundary = int(d_boundary.ravel()[0])
        boundary_data["d"] = d_boundary
        
        # Construir features
        X = build_feature_vector(boundary_data, operators)
        
        # Normalizar features con estadísticas del checkpoint
        X_norm = (X - X_mean) / X_std
        
        # Inferencia
        preds = run_inference_single(model, X_norm, normalizer, family_map_inv, device)
        
        # === GUARDAR COMO .h5 (IO_CONTRACTS_V1) ===
        out_h5_path = geom_dir / f"{name}_emergent.h5"
        
        with h5py.File(out_h5_path, "w") as f_out:
            # Atributos (IO_CONTRACTS_V1)
            family_pred = preds["family_name"]
            f_out.attrs["system_name"] = name
            # Canónico: 'family' debe existir (family_pred es opcional)
            f_out.attrs["family"] = family_pred
            f_out.attrs["family_pred"] = family_pred
            f_out.attrs["d"] = int(d_boundary)
            f_out.attrs["d_pred"] = int(d_boundary)
            # Canónico: provenance ∈ {"train","inference"}
            f_out.attrs["provenance"] = "inference"
            # Detalle de trazabilidad (no contractual)
            f_out.attrs["provenance_detail"] = "inference_from_boundary_using_sandbox_model"
            f_out.attrs["zh_pred"] = preds["zh_pred"]
            f_out.attrs["checkpoint_source"] = str(checkpoint_path)

            # Datasets (IO_CONTRACTS_V1) — nombres canónicos
            f_out.create_dataset("z_grid", data=z_grid)
            f_out.create_dataset("A_of_z", data=preds["A_pred"])
            f_out.create_dataset("f_of_z", data=preds["f_pred"])
            # Opcional pero útil para contratos/diagnóstico
            f_out.create_dataset("R_of_z", data=preds["R_pred"])

            # Export NPZ (compatibilidad con 03/04)
            A_arr = preds.get('A_pred', preds.get('A_of_z'))
            f_arr = preds.get('f_pred', preds.get('f_of_z'))
            R_arr = preds.get('R_pred', preds.get('R_of_z'))
            if A_arr is None or f_arr is None or R_arr is None:
                raise ValueError(f"[inference] faltan A/f/R para {name}")
            np.savez(
                preds_dir / f"{name}_geometry.npz",
                z=np.asarray(z_grid, dtype=np.float64),
                z_grid=np.asarray(z_grid, dtype=np.float64),
                A_pred=np.asarray(A_arr, dtype=np.float32),
                f_pred=np.asarray(f_arr, dtype=np.float32),
                R_pred=np.asarray(R_arr, dtype=np.float32),
                A_of_z=np.asarray(A_arr, dtype=np.float32),
                f_of_z=np.asarray(f_arr, dtype=np.float32),
                R_of_z=np.asarray(R_arr, dtype=np.float32),
                family_pred=np.array(str(preds.get('family_name','unknown')), dtype=object),
                zh_pred=np.array(float(preds.get('zh_pred', float('nan'))), dtype=np.float32),
                d=np.array(int(d_boundary), dtype=np.int32),
                provenance=np.array('inference', dtype=object),
            )

        summary_entries.append({
            "name": name,
            "h5_input": str(h5_path),
            "h5_output": str(out_h5_path),
            "family_pred": preds["family_name"],
            "zh_pred": preds["zh_pred"],
            "d": d_boundary,
            "provenance": "inference",
            "provenance_detail": "inference_from_boundary_using_sandbox_model",
            "family": preds["family_name"],
        })
        
        print(f"      -> family_pred={preds['family_name']}, zh_pred={preds['zh_pred']:.3f}")
    
    # === ESCRIBIR SUMMARY ===
    summary = {
        "version": "V2.2",
        "mode": "inference",
        "description": (
            "Geometría emergente inferida desde datos boundary-only "
            "usando modelo entrenado en sandbox"
        ),
        "checkpoint": str(checkpoint_path),
        "n_systems": len(summary_entries),
        "z_grid_from_checkpoint": {
            "min": float(z_grid[0]),
            "max": float(z_grid[-1]),
            "n_points": len(z_grid),
        },
        "systems": summary_entries,
        "metrics": None,  # No hay métricas train/test en inference
    }
    
    summary_path = output_dir / "emergent_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # === BANNER FINAL ===
    print("\n" + "=" * 70)
    print("[OK] FASE XI V2.2 - MODO INFERENCE COMPLETADO")
    print(f"  Geometrías:   {geom_dir}")
    print(f"  Summary:      {summary_path}")
    print(f"  Sistemas:     {len(summary_entries)}")
    print("=" * 70)
    print("\nPróximo paso: 03_discover_bulk_equations.py o contratos")


# ============================================================
# MODO TRAIN (CÓDIGO ORIGINAL V2.1)
# ============================================================

def run_train_mode(args):
    """
    Modo train: comportamiento original (sandbox con bulk_truth).
    """
    # Set seed
    set_torch_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    print("=" * 70)
    print("FASE XI V2.2 - EMERGENCIA DE GEOMETRÍA (MODO TRAIN)")
    print("Prioriza A(z) y f(z) sobre R(z) y clasificación")
    print("=" * 70)
    print(f"  Pesos de loss: A={LOSS_WEIGHT_A}, f={LOSS_WEIGHT_F}, R={LOSS_WEIGHT_R}")
    print(f"                 zh={LOSS_WEIGHT_ZH}, family={LOSS_WEIGHT_FAMILY}")
    print(f"                 physics={LOSS_WEIGHT_PHYSICS}, ads={LOSS_WEIGHT_PHYSICS_ADS}")
    print("=" * 70)
    
    # Cargar manifest
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No se encontró manifest.json en {data_dir}")
    manifest = json.loads(manifest_path.read_text())
    
    family_map = {"ads": 0, "lifshitz": 1, "hyperscaling": 2, "deformed": 3, "unknown": 4}
    family_map_inv = {v: k for k, v in family_map.items()}
    
    # Estructuras para datos
    train_data = {
        "X": [], "Y_A": [], "Y_f": [], "Y_R": [], "Y_zh": [], 
        "Y_family": [], "names": [], "families": []
    }
    test_data = {
        "X": [], "Y_A": [], "Y_f": [], "Y_R": [], "Y_zh": [], 
        "Y_family": [], "names": [], "categories": [], "families": []
    }
    
    z_grid = None
    d_value = 4
    
    loader = CuerdasDataLoader(mode="train")
    
    # Cargar datos
    print("\n>> Cargando geometrías...")
    for geo_info in manifest["geometries"]:
        h5_path = data_dir / f"{geo_info['name']}.h5"
        if not h5_path.exists():
            print(f"   [WARN] No existe: {h5_path}")
            continue
            
        category = geo_info["category"]
        
        with h5py.File(h5_path, "r") as f:
            boundary_data, operators = loader.load_boundary_and_meta(f)
            (A_truth, f_truth, R_truth, z_grid_local, z_h, family, d_value_local
            ) = loader.load_bulk_truth(f)
            d_value = d_value_local
        
        if z_grid is None:
            z_grid = z_grid_local
        
        # Añadir d a boundary_data para feature extraction
        boundary_data["d"] = d_value_local
        
        X = build_feature_vector(boundary_data, operators)
        family_id = family_map.get(family, 4)
        
        if category == "known":
            train_data["X"].append(X)
            train_data["Y_A"].append(A_truth)
            train_data["Y_f"].append(f_truth)
            train_data["Y_R"].append(R_truth)
            train_data["Y_zh"].append(z_h if z_h else 0.0)
            train_data["Y_family"].append(family_id)
            train_data["names"].append(geo_info["name"])
            train_data["families"].append(family)
        else:
            test_data["X"].append(X)
            test_data["Y_A"].append(A_truth)
            test_data["Y_f"].append(f_truth)
            test_data["Y_R"].append(R_truth)
            test_data["Y_zh"].append(z_h if z_h else 0.0)
            test_data["Y_family"].append(family_id)
            test_data["names"].append(geo_info["name"])
            test_data["categories"].append(category)
            test_data["families"].append(family)
    
    if len(train_data["X"]) == 0:
        raise ValueError("No hay datos de entrenamiento (category='known')")
    
    # Convertir a arrays
    X_train = np.stack(train_data["X"])
    Y_A_train = np.stack(train_data["Y_A"])
    Y_f_train = np.stack(train_data["Y_f"])
    Y_R_train = np.stack(train_data["Y_R"])
    Y_zh_train = np.array(train_data["Y_zh"])
    Y_family_train = np.array(train_data["Y_family"])
    
    print(f"\n   TRAIN (known):       {len(X_train)} geometrías")
    print(f"   TEST (test/unknown): {len(test_data['X'])} geometrías")
    print(f"   Features:            {X_train.shape[1]}")
    print(f"   n_z (puntos radial): {Y_A_train.shape[1]}")
    
    # Contar por familia en train
    print("\n   Distribución por familia (train):")
    for fam_name, fam_id in family_map.items():
        count = np.sum(Y_family_train == fam_id)
        if count > 0:
            print(f"     {fam_name}: {count}")
    
    # === NORMALIZACIÓN ===
    
    # Features
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    
    # Targets
    normalizer = TargetNormalizer()
    normalizer.fit(Y_A_train, Y_f_train, Y_R_train)
    
    Y_A_train_norm = normalizer.normalize_A(Y_A_train)
    Y_f_train_norm = normalizer.normalize_f(Y_f_train)
    Y_R_train_norm = normalizer.normalize_R(Y_R_train)
    
    print(f"\n   Normalización:")
    print(f"     A: mean={normalizer.A_mean:.3f}, std={normalizer.A_std:.3f}")
    print(f"     f: mean={normalizer.f_mean:.3f}, std={normalizer.f_std:.3f}")
    print(f"     R: mean={normalizer.R_mean:.3f}, std={normalizer.R_std:.3f}")
    
    # Convertir a tensores
    X_train_t = torch.from_numpy(X_train_norm.astype(np.float32)).to(device)
    Y_A_train_t = torch.from_numpy(Y_A_train_norm.astype(np.float32)).to(device)
    Y_f_train_t = torch.from_numpy(Y_f_train_norm.astype(np.float32)).to(device)
    Y_R_train_t = torch.from_numpy(Y_R_train_norm.astype(np.float32)).to(device)
    Y_zh_train_t = torch.from_numpy(Y_zh_train.astype(np.float32)).to(device)
    Y_family_train_t = torch.from_numpy(Y_family_train.astype(np.int64)).to(device)
    z_t = torch.from_numpy(z_grid.astype(np.float32)).to(device)
    
    # Preparar test
    has_test = len(test_data["X"]) > 0
    if has_test:
        X_test = np.stack(test_data["X"])
        X_test_norm = (X_test - X_mean) / X_std
        X_test_t = torch.from_numpy(X_test_norm.astype(np.float32)).to(device)
        Y_A_test = np.stack(test_data["Y_A"])
        Y_f_test = np.stack(test_data["Y_f"])
        Y_R_test = np.stack(test_data["Y_R"])
        Y_zh_test = np.array(test_data["Y_zh"])
        Y_family_test = np.array(test_data["Y_family"])
    
    # === MODELO ===
    
    model = EmergentGeometryNet(
        n_features=X_train.shape[1],
        n_z=Y_A_train.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_families=len(family_map)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n>> Modelo: {n_params:,} parámetros")
    
    # === ENTRENAMIENTO ===
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.n_epochs,
        eta_min=args.lr * 0.01
    )
    
    # Historia de entrenamiento
    history = {
        "train_losses": [],
        "test_metrics": [],
        "epochs": []
    }
    
    best_test_A_r2 = -np.inf
    best_epoch = 0
    
    print("\n>> Iniciando entrenamiento...")
    print("-" * 70)
    
    for epoch in range(1, args.n_epochs + 1):
        # Entrenar una época
        train_losses = train_one_epoch(
            model, X_train_t, Y_A_train_t, Y_f_train_t, Y_R_train_t,
            Y_zh_train_t, Y_family_train_t, z_t, d_value, family_map,
            optimizer, args.batch_size, device
        )
        
        # Actualizar scheduler por ÉPOCA (no por batch)
        scheduler.step()
        
        # Guardar historia
        history["train_losses"].append(train_losses)
        history["epochs"].append(epoch)
        
        # Evaluar en test periódicamente
        should_eval = (epoch % EVAL_FREQUENCY == 0) or (epoch == 1) or (epoch == args.n_epochs)
        
        if has_test and should_eval:
            test_metrics, _, _, _, _, _ = evaluate_on_test(
                model, X_test_t, Y_A_test, Y_f_test, Y_R_test,
                Y_zh_test, Y_family_test, normalizer, device
            )
            history["test_metrics"].append({"epoch": epoch, **test_metrics})
            
            # Tracking del mejor modelo
            if test_metrics["A_r2"] > best_test_A_r2:
                best_test_A_r2 = test_metrics["A_r2"]
                best_epoch = epoch
        
        # Logging
        if args.verbose and (epoch % max(1, args.n_epochs // 20) == 0 or epoch == 1):
            lr_current = scheduler.get_last_lr()[0]
            log_msg = (
                f"[Epoch {epoch:4d}/{args.n_epochs}] "
                f"L_total={train_losses['total']:.4f} | "
                f"L_A={train_losses['A']:.4f} | "
                f"L_f={train_losses['f']:.4f} | "
                f"lr={lr_current:.2e}"
            )
            
            if has_test and should_eval:
                log_msg += f" || Test: A_r2={test_metrics['A_r2']:.3f}, f_r2={test_metrics['f_r2']:.3f}"
            
            print(log_msg)
    
    print("-" * 70)
    
    # === EVALUACIÓN FINAL EN TEST ===
    
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    if has_test:
        print("\n>> Evaluación final en TEST...")
        
        test_metrics_final, A_pred, f_pred, R_pred, zh_pred, family_pred = evaluate_on_test(
            model, X_test_t, Y_A_test, Y_f_test, Y_R_test,
            Y_zh_test, Y_family_test, normalizer, device
        )
        
        print(f"\n   TEST Metrics (final):")
        print(f"   A(z) R²:         {test_metrics_final['A_r2']:.4f}")
        print(f"   f(z) R²:         {test_metrics_final['f_r2']:.4f}")
        print(f"   R(z) R²:         {test_metrics_final['R_r2']:.4f}")
        print(f"   z_h MAE:         {test_metrics_final['zh_mae']:.4f}")
        print(f"   Family accuracy: {test_metrics_final['family_accuracy']:.4f}")
        print(f"\n   Mejor época (por A_r2): {best_epoch} con A_r2={best_test_A_r2:.4f}")
        
        # Métricas por familia
        metrics_by_family = {}
        for fam_name, fam_id in family_map.items():
            mask = Y_family_test == fam_id
            if np.sum(mask) > 0:
                metrics_by_family[fam_name] = {
                    "count": int(np.sum(mask)),
                    "A_r2": compute_r2(Y_A_test[mask], A_pred[mask]),
                    "f_r2": compute_r2(Y_f_test[mask], f_pred[mask]),
                }
        
        print("\n   Métricas por familia:")
        for fam_name, fam_metrics in metrics_by_family.items():
            if fam_metrics["count"] > 0:
                print(f"     {fam_name} (n={fam_metrics['count']}): "
                      f"A_r2={fam_metrics['A_r2']:.3f}, f_r2={fam_metrics['f_r2']:.3f}")
        
        # Guardar predicciones individuales
        for i, name in enumerate(test_data["names"]):
            np.savez(
                preds_dir / f"{name}_geometry.npz",
                z=z_grid,
                A_pred=A_pred[i],
                f_pred=f_pred[i],
                R_pred=R_pred[i],
                A_truth=Y_A_test[i],
                f_truth=Y_f_test[i],
                R_truth=Y_R_test[i],
                zh_pred=zh_pred[i],
                zh_truth=Y_zh_test[i],
                family_pred=family_pred[i],
                family_truth=Y_family_test[i],
                category=test_data["categories"][i]
            )
    else:
        test_metrics_final = {}
        metrics_by_family = {}
    
    # === GUARDAR MODELO ===
    
    model_path = output_dir / "emergent_geometry_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": X_train.shape[1],
        "n_z": Y_A_train.shape[1],
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "family_map": family_map,
        "X_mean": X_mean,
        "X_std": X_std,
        "normalizer": {
            "A_mean": normalizer.A_mean,
            "A_std": normalizer.A_std,
            "f_mean": normalizer.f_mean,
            "f_std": normalizer.f_std,
            "R_mean": normalizer.R_mean,
            "R_std": normalizer.R_std,
        },
        "z_grid": z_grid,
        "d": d_value,
        "loss_weights": {
            "A": LOSS_WEIGHT_A,
            "f": LOSS_WEIGHT_F,
            "R": LOSS_WEIGHT_R,
            "zh": LOSS_WEIGHT_ZH,
            "family": LOSS_WEIGHT_FAMILY,
            "physics": LOSS_WEIGHT_PHYSICS,
            "physics_ads": LOSS_WEIGHT_PHYSICS_ADS,
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_test_A_r2": best_test_A_r2,
    }, model_path)
    
    # === GUARDAR SUMMARY ===
    
    # Extraer última loss de entrenamiento
    final_train_losses = history["train_losses"][-1] if history["train_losses"] else {}
    
    summary = {
        "version": "V2.2",
        "mode": "train",
        "n_train": int(len(X_train)),
        "n_test": int(len(test_data["X"])),
        "n_features": int(X_train.shape[1]),
        "n_z": int(Y_A_train.shape[1]),
        "n_epochs": args.n_epochs,
        "best_epoch": best_epoch,
        "loss_weights": {
            "A": LOSS_WEIGHT_A,
            "f": LOSS_WEIGHT_F,
            "R": LOSS_WEIGHT_R,
            "zh": LOSS_WEIGHT_ZH,
            "family": LOSS_WEIGHT_FAMILY,
            "physics": LOSS_WEIGHT_PHYSICS,
            "physics_ads": LOSS_WEIGHT_PHYSICS_ADS,
        },
        "train_metrics": {
            "final_total_loss": final_train_losses.get("total", 0.0),
            "final_A_loss": final_train_losses.get("A", 0.0),
            "final_f_loss": final_train_losses.get("f", 0.0),
            "final_R_loss": final_train_losses.get("R", 0.0),
            "final_zh_loss": final_train_losses.get("zh", 0.0),
            "final_family_loss": final_train_losses.get("family", 0.0),
            "final_physics_loss": final_train_losses.get("physics", 0.0),
        },
        "test_metrics": test_metrics_final,
        "metrics_by_family": metrics_by_family,
        "normalizer_stats": {
            "A_mean": normalizer.A_mean,
            "A_std": normalizer.A_std,
            "f_mean": normalizer.f_mean,
            "f_std": normalizer.f_std,
            "R_mean": normalizer.R_mean,
            "R_std": normalizer.R_std,
        },
    }
    
    summary_path = output_dir / "emergent_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    
    # === BANNER FINAL ===
    
    print("\n" + "=" * 70)
    print("[OK] FASE XI V2.2 - GEOMETRÍA EMERGENTE COMPLETADA (MODO TRAIN)")
    print(f"  Modelo:       {model_path}")
    print(f"  Predicciones: {preds_dir}")
    print(f"  Summary:      {summary_path}")
    print("=" * 70)
    print("\nPróximo paso: 03_discover_bulk_equations.py")


# ============================================================
# MAIN (V2.2)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CUERDAS - Geometría emergente V2.2 (con modo inference)"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directorio con datos HDF5 de geometrías")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directorio de salida para modelo y predicciones")
    parser.add_argument("--n-epochs", type=int, default=2000,
                        help="Número de épocas de entrenamiento (solo mode=train)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo: 'cpu' o 'cuda'")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Dimensión oculta de la red")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Número de capas residuales")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamaño de batch")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Imprimir progreso detallado")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate inicial")
    
    # === NUEVOS ARGUMENTOS V2.2 ===
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Modo de uso: 'train' (sandbox, con bulk_truth) o "
             "'inference' (datos boundary-only, usando checkpoint pretrained)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al checkpoint del modelo entrenado en sandbox "
             "(solo obligatorio en mode='inference')"
    )
    
    args = parser.parse_args()
    
    # Despachar según modo
    if args.mode == "train":
        run_train_mode(args)
    else:
        run_inference_mode(args)


if __name__ == "__main__":
    main()
