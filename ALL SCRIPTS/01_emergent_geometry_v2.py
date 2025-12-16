#!/usr/bin/env python3
"""
01_emergent_geometry_v2.py - Fase XI: Emergencia de Geometria HONESTA

VERSION CORREGIDA:
    1. Normalizacion robusta de targets (A, f, R)
    2. Proteccion contra valores extremos
    3. Learning rate warmup + cosine decay
    4. Gradient clipping mas agresivo
    5. Logging detallado para diagnostico

MODO DE USO:
    - Entrenamiento: usa geometrias "known" con supervision
    - Inferencia: aplica modelo a "test"/"unknown" solo con boundary data
    - Validacion: compara predicciones con bulk_truth SOLO para metricas finales
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


# ============================================================
# EXTRACCION DE FEATURES DEL BOUNDARY
# ============================================================

def extract_correlator_features(G2: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """Extrae features de un correlador 2-pt."""
    features = {}
    
    valid = (G2 > 0) & (G2 < 1e15) & np.isfinite(G2) & (x > 0)
    if np.sum(valid) < 3:
        return {"slope": -4.0, "curvature": 0.0, "intercept": 0.0, 
                "thermal_deviation": 0.0, "decay_scale": 1.0}
    
    log_x = np.log(x[valid])
    log_G2 = np.log(G2[valid])
    
    try:
        coeffs = np.polyfit(log_x, log_G2, 2)
        slope = coeffs[1]
        curvature = coeffs[0]
        intercept = coeffs[2]
    except:
        slope, curvature, intercept = -4.0, 0.0, 0.0
    
    features["slope"] = float(np.clip(slope, -20, 20))
    features["curvature"] = float(np.clip(curvature, -10, 10))
    features["intercept"] = float(np.clip(intercept, -50, 50))
    
    try:
        pred_power = np.exp(intercept + slope * log_x)
        residuals = log_G2 - np.log(pred_power + 1e-20)
        features["thermal_deviation"] = float(np.clip(np.std(residuals), 0, 10))
    except:
        features["thermal_deviation"] = 0.0
    
    try:
        half_max = 0.5 * np.max(G2[valid])
        idx_half = np.argmin(np.abs(G2[valid] - half_max))
        features["decay_scale"] = float(np.clip(x[valid][idx_half], 0.01, 100))
    except:
        features["decay_scale"] = 1.0
    
    return features


def extract_thermal_features(G2: np.ndarray, x: np.ndarray, T) -> Dict[str, float]:
    """Extrae features relacionados con temperatura finita."""
    features = {}

    # Aseguramos escalar
    T_arr = np.asarray(T, dtype=float)
    T_scalar = float(T_arr.ravel()[0])  # primer valor

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
            except:
                features["exponential_decay"] = 0.0
        else:
            features["exponential_decay"] = 0.0
    else:
        features["thermal_scale"] = 0.0
        features["exponential_decay"] = 0.0

    return features


def extract_spectral_features(operators: List[Dict]) -> Dict[str, float]:
    """Extrae features del espectro de operadores."""
    features = {}
    
    Deltas = [op["Delta"] for op in operators]
    features["n_operators"] = len(operators)
    features["Delta_min"] = float(min(Deltas)) if Deltas else 2.0
    features["Delta_max"] = float(max(Deltas)) if Deltas else 4.0
    features["Delta_mean"] = float(np.mean(Deltas)) if Deltas else 2.0
    features["Delta_gap"] = float(Deltas[1] - Deltas[0]) if len(Deltas) > 1 else 0.0
    
    return features


def extract_response_features(
    G_R_real: np.ndarray,
    G_R_imag: np.ndarray,
    omega: np.ndarray,
    k: np.ndarray
) -> Dict[str, float]:
    """Extrae features de la funcion de Green retardada."""
    features = {}
    
    try:
        G_imag_avg = np.mean(np.abs(G_R_imag), axis=0)
        peak_idx = np.argmax(G_imag_avg)
        features["qnm_omega_real"] = float(np.clip(omega[peak_idx], 0, 20))
        
        half_max = 0.5 * G_imag_avg[peak_idx]
        above_half = G_imag_avg > half_max
        width_idx = np.sum(above_half)
        features["qnm_width"] = float(np.clip(width_idx * (omega[1] - omega[0]), 0, 10))
    except:
        features["qnm_omega_real"] = 1.0
        features["qnm_width"] = 0.1
    
    try:
        G_real_k0 = G_R_real[0, :]
        slope_disp = np.polyfit(omega[:10], G_real_k0[:10], 1)[0]
        features["dispersion_slope"] = float(np.clip(slope_disp, -10, 10))
    except:
        features["dispersion_slope"] = 0.0
    
    return features


def build_feature_vector(boundary_data: Dict, operators: List[Dict]) -> np.ndarray:
    """Construye vector de features completo desde datos del boundary."""
    all_features = []
    
    x_grid = boundary_data.get("x_grid", np.linspace(0.1, 10, 100))
    T = boundary_data.get("temperature", 0.0)
    
    for op in operators:
        name = op["name"]
        G2_key = f"G2_{name}"
        if G2_key in boundary_data:
            G2 = boundary_data[G2_key]
            corr_feats = extract_correlator_features(G2, x_grid)
            for k, v in corr_feats.items():
                all_features.append(v)
    
    if operators and f"G2_{operators[0]['name']}" in boundary_data:
        G2_first = boundary_data[f"G2_{operators[0]['name']}"]
        thermal_feats = extract_thermal_features(G2_first, x_grid, T)
        for k, v in thermal_feats.items():
            all_features.append(v)
    
    spec_feats = extract_spectral_features(operators)
    for k, v in spec_feats.items():
        all_features.append(v)
    
    if "G_R_real" in boundary_data and "G_R_imag" in boundary_data:
        G_R_real = boundary_data["G_R_real"]
        G_R_imag = boundary_data["G_R_imag"]
        omega = boundary_data.get("omega_grid", np.linspace(0.1, 10, 50))
        k = boundary_data.get("k_grid", np.linspace(0, 5, 30))
        resp_feats = extract_response_features(G_R_real, G_R_imag, omega, k)
        for kk, v in resp_feats.items():
            all_features.append(v)
    
    all_features.append(float(boundary_data.get("d", 4)))
    
    return np.array(all_features, dtype=np.float32)


# ============================================================
# NORMALIZACION ROBUSTA DE TARGETS
# ============================================================

class TargetNormalizer:
    """Normaliza targets de forma robusta para entrenamiento."""
    
    def __init__(self):
        self.A_mean = None
        self.A_std = None
        self.R_mean = None
        self.R_std = None
        
    def fit(self, A: np.ndarray, R: np.ndarray):
        """Calcula estadisticas de normalizacion."""
        # Usar mediana y MAD para robustez
        self.A_mean = np.median(A)
        self.A_std = np.maximum(np.median(np.abs(A - self.A_mean)) * 1.4826, 0.1)
        
        # R puede tener valores muy extremos - usar percentiles
        R_valid = R[np.isfinite(R) & (np.abs(R) < 1e6)]
        if len(R_valid) > 0:
            self.R_mean = np.median(R_valid)
            self.R_std = np.maximum(np.percentile(np.abs(R_valid - self.R_mean), 90), 1.0)
        else:
            self.R_mean = -20.0
            self.R_std = 10.0
    
    def normalize_A(self, A: np.ndarray) -> np.ndarray:
        return (A - self.A_mean) / self.A_std
    
    def denormalize_A(self, A_norm: np.ndarray) -> np.ndarray:
        return A_norm * self.A_std + self.A_mean
    
    def normalize_R(self, R: np.ndarray) -> np.ndarray:
        R_clipped = np.clip(R, self.R_mean - 10*self.R_std, self.R_mean + 10*self.R_std)
        return (R_clipped - self.R_mean) / self.R_std
    
    def denormalize_R(self, R_norm: np.ndarray) -> np.ndarray:
        return R_norm * self.R_std + self.R_mean
    
    def state_dict(self) -> Dict:
        return {
            "A_mean": float(self.A_mean), 
            "A_std": float(self.A_std),
            "R_mean": float(self.R_mean), 
            "R_std": float(self.R_std)
        }
    
    def load_state_dict(self, state: Dict):
        self.A_mean = state["A_mean"]
        self.A_std = state["A_std"]
        self.R_mean = state["R_mean"]
        self.R_std = state["R_std"]


# ============================================================
# MODELO: EMERGENT GEOMETRY NETWORK
# ============================================================

class EmergentGeometryNet(nn.Module):
    """Red que mapea features del boundary -> geometria del bulk."""
    
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
        
        # Input projection con normalizacion
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # Trunk: ResNet-style blocks
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Decoders
        self.decoder_A = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_z)
        )
        
        self.decoder_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_z),
            nn.Sigmoid()
        )
        
        self.decoder_zh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.decoder_family = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_families)
        )
        
        self.decoder_R = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_z)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_norm(x)
        h = self.input_proj(h)
        
        for layer in self.layers:
            h = h + 0.1 * layer(h)
        
        h = self.final_norm(h)
        
        A = self.decoder_A(h)
        f = self.decoder_f(h)
        z_h = self.decoder_zh(h).squeeze(-1)
        family_logits = self.decoder_family(h)
        R = self.decoder_R(h)
        
        return {
            "A": A,
            "f": f,
            "z_h": z_h,
            "family_logits": family_logits,
            "R": R
        }


# ============================================================
# PERDIDAS FISICAS
# ============================================================

def physics_loss_generic(
    A_pred: torch.Tensor,
    f_pred: torch.Tensor,
    z: torch.Tensor,
    d: int
) -> torch.Tensor:
    """Perdidas fisicas GENERICAS."""
    loss = torch.tensor(0.0, device=A_pred.device)
    
    # f in [0, 1]
    loss_f_bounds = torch.mean(torch.relu(-f_pred) + torch.relu(f_pred - 1))
    loss = loss + loss_f_bounds
    
    # Regularidad
    loss_A_reg = torch.mean(torch.relu(torch.abs(A_pred) - 10)) * 0.1
    loss = loss + loss_A_reg
    
    # Suavidad
    dz = (z[1] - z[0]).item()
    dA = torch.gradient(A_pred, spacing=(dz,), dim=-1)[0]
    loss_smooth = torch.mean(dA ** 2) * 0.01
    loss = loss + loss_smooth
    
    return loss


def physics_loss_ads_specific(
    A_pred: torch.Tensor,
    f_pred: torch.Tensor,
    z: torch.Tensor,
    d: int,
    family_mask: torch.Tensor
) -> torch.Tensor:
    """Perdidas fisicas especificas para AdS."""
    if family_mask.sum() < 1:
        return torch.tensor(0.0, device=A_pred.device)
    
    A_ads = A_pred[family_mask > 0.5]
    
    loss = torch.tensor(0.0, device=A_pred.device)
    
    # A(z) monotono decreciente para AdS
    dz = (z[1] - z[0]).item()
    dA_ads = torch.gradient(A_ads, spacing=(dz,), dim=-1)[0]
    loss_mono = torch.mean(torch.relu(dA_ads)) * 0.5
    loss = loss + loss_mono
    
    return loss


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_emergent_geometry(
    model: EmergentGeometryNet,
    X_train: torch.Tensor,
    Y_A_train: torch.Tensor,
    Y_f_train: torch.Tensor,
    Y_R_train: torch.Tensor,
    Y_zh_train: torch.Tensor,
    Y_family_train: torch.Tensor,
    z: torch.Tensor,
    d: int,
    n_epochs: int = 5000,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Entrena el modelo SOLO con datos 'known'."""
    
    model = model.to(device)
    X_train = X_train.to(device)
    Y_A_train = Y_A_train.to(device)
    Y_f_train = Y_f_train.to(device)
    Y_R_train = Y_R_train.to(device)
    Y_zh_train = Y_zh_train.to(device)
    Y_family_train = Y_family_train.to(device)
    z = z.to(device)
    
    ads_mask = (Y_family_train == 0).float()
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()
    ce = nn.CrossEntropyLoss()
    
    history = {
        "total": [], "A": [], "f": [], "R": [], "zh": [], 
        "family": [], "generic": [], "ads_specific": []
    }
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(X_train)
        
        loss_A = huber(out["A"], Y_A_train)
        loss_f = mse(out["f"], Y_f_train)
        loss_R = huber(out["R"], Y_R_train)
        loss_zh = huber(out["z_h"], Y_zh_train)
        loss_family = ce(out["family_logits"], Y_family_train)
        
        loss_generic = physics_loss_generic(out["A"], out["f"], z, d)
        loss_ads = physics_loss_ads_specific(out["A"], out["f"], z, d, ads_mask)
        
        total = (
            1.0 * loss_A + 
            1.0 * loss_f + 
            0.5 * loss_R +
            0.5 * loss_zh + 
            0.3 * loss_family +
            0.1 * loss_generic +
            0.1 * loss_ads
        )
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        scheduler.step()
        
        history["total"].append(total.item())
        history["A"].append(loss_A.item())
        history["f"].append(loss_f.item())
        history["R"].append(loss_R.item())
        history["zh"].append(loss_zh.item())
        history["family"].append(loss_family.item())
        history["generic"].append(loss_generic.item())
        history["ads_specific"].append(loss_ads.item())
        
        if verbose and (epoch % max(1, n_epochs // 20) == 0 or epoch == 1):
            acc = (out["family_logits"].argmax(dim=1) == Y_family_train).float().mean()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:5d}: loss={total.item():.4f}, "
                  f"A={loss_A.item():.4f}, f={loss_f.item():.4f}, "
                  f"fam_acc={acc.item():.2f}, lr={current_lr:.2e}")
    
    return history


def inference_on_data(
    model: EmergentGeometryNet,
    X: torch.Tensor,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """Aplica el modelo a datos."""
    model.eval()
    X = X.to(device)
    
    with torch.no_grad():
        out = model(X)
        return {
            "A": out["A"].cpu().numpy(),
            "f": out["f"].cpu().numpy(),
            "R": out["R"].cpu().numpy(),
            "z_h": out["z_h"].cpu().numpy(),
            "family": out["family_logits"].argmax(dim=1).cpu().numpy()
        }


def evaluate_model(
    predictions: Dict[str, np.ndarray],
    Y_A: np.ndarray,
    Y_f: np.ndarray,
    Y_R: np.ndarray,
    Y_zh: np.ndarray,
    Y_family: np.ndarray
) -> Dict[str, Any]:
    """Evalua predicciones contra ground truth."""
    def r2(y_true, y_pred):
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid.sum() < 2:
            return 0.0
        
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else 0.0
        
        return float(np.clip(1 - ss_res / ss_tot, -1, 1))
    
    return {
        "A_r2": r2(Y_A, predictions["A"]),
        "f_r2": r2(Y_f, predictions["f"]),
        "R_r2": r2(Y_R, predictions["R"]),
        "zh_mae": float(np.mean(np.abs(Y_zh - predictions["z_h"]))),
        "family_accuracy": float(np.mean(Y_family == predictions["family"]))
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fase XI v2: Emergencia de geometria (version robusta)"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="fase11_geometry_v2")
    parser.add_argument("--n-epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    print("=" * 70)
    print("FASE XI v2 - EMERGENCIA DE GEOMETRIA (VERSION ROBUSTA)")
    print("=" * 70)
    
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    
    family_map = {"ads": 0, "lifshitz": 1, "hyperscaling": 2, "deformed": 3, "unknown": 4}
    
    train_data = {"X": [], "Y_A": [], "Y_f": [], "Y_R": [], "Y_zh": [], 
                  "Y_family": [], "names": []}
    test_data = {"X": [], "Y_A": [], "Y_f": [], "Y_R": [], "Y_zh": [], 
                 "Y_family": [], "names": [], "categories": []}
    
    z_grid = None
    d_value = 4
    
    for geo_info in manifest["geometries"]:
        h5_path = data_dir / f"{geo_info['name']}.h5"
        category = geo_info["category"]
        
        with h5py.File(h5_path, "r") as f:
            boundary = f["boundary"]
            boundary_data = {}
            
            for key in boundary.keys():
                boundary_data[key] = boundary[key][:]
            for key in boundary.attrs.keys():
                boundary_data[key] = boundary.attrs[key]
            
            bulk = f["bulk_truth"]
            A_truth = bulk["A_truth"][:]
            f_truth = bulk["f_truth"][:]
            R_truth = bulk["R_truth"][:]
            z_grid_local = bulk["z_grid"][:]
            z_h = bulk.attrs.get("z_h", 0.0)
            family = bulk.attrs.get("family", "unknown")
            d_value = int(bulk.attrs.get("d", 4))
            
            operators = json.loads(f.attrs["operators"])
        
        if z_grid is None:
            z_grid = z_grid_local
        
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
        else:
            test_data["X"].append(X)
            test_data["Y_A"].append(A_truth)
            test_data["Y_f"].append(f_truth)
            test_data["Y_R"].append(R_truth)
            test_data["Y_zh"].append(z_h if z_h else 0.0)
            test_data["Y_family"].append(family_id)
            test_data["names"].append(geo_info["name"])
            test_data["categories"].append(category)
    
    X_train = np.stack(train_data["X"])
    Y_A_train = np.stack(train_data["Y_A"])
    Y_f_train = np.stack(train_data["Y_f"])
    Y_R_train = np.stack(train_data["Y_R"])
    Y_zh_train = np.array(train_data["Y_zh"])
    Y_family_train = np.array(train_data["Y_family"])
    
    print(f"\n   TRAIN (known):  {len(X_train)} geometrias")
    print(f"   TEST (test/unknown): {len(test_data['X'])} geometrias")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   n_z:      {Y_A_train.shape[1]}")
    
    # === NORMALIZACION ===
    
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    
    normalizer = TargetNormalizer()
    normalizer.fit(Y_A_train, Y_R_train)
    
    Y_A_train_norm = normalizer.normalize_A(Y_A_train)
    Y_R_train_norm = normalizer.normalize_R(Y_R_train)
    
    print(f"\n   Normalizacion:")
    print(f"     A: mean={normalizer.A_mean:.2f}, std={normalizer.A_std:.2f}")
    print(f"     R: mean={normalizer.R_mean:.2f}, std={normalizer.R_std:.2f}")
    
    X_train_t = torch.from_numpy(X_train_norm.astype(np.float32))
    Y_A_train_t = torch.from_numpy(Y_A_train_norm.astype(np.float32))
    Y_f_train_t = torch.from_numpy(Y_f_train.astype(np.float32))
    Y_R_train_t = torch.from_numpy(Y_R_train_norm.astype(np.float32))
    Y_zh_train_t = torch.from_numpy(Y_zh_train.astype(np.float32))
    Y_family_train_t = torch.from_numpy(Y_family_train.astype(np.int64))
    z_t = torch.from_numpy(z_grid.astype(np.float32))
    
    # === MODELO ===
    
    model = EmergentGeometryNet(
        n_features=X_train.shape[1],
        n_z=Y_A_train.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_families=5
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n>> Modelo: {n_params:,} parametros")
    
    # === ENTRENAMIENTO ===
    
    print("\n>> Entrenando con datos 'known' SOLAMENTE...")
    history = train_emergent_geometry(
        model, X_train_t, Y_A_train_t, Y_f_train_t, Y_R_train_t, 
        Y_zh_train_t, Y_family_train_t, z_t, d_value,
        n_epochs=args.n_epochs, lr=args.lr, device=device, verbose=args.verbose
    )
    
    # === EVALUACION ===
    
    print("\n>> Evaluando en TRAIN (known)...")
    train_preds_norm = inference_on_data(model, X_train_t, device)
    
    train_preds = {
        "A": normalizer.denormalize_A(train_preds_norm["A"]),
        "f": train_preds_norm["f"],
        "R": normalizer.denormalize_R(train_preds_norm["R"]),
        "z_h": train_preds_norm["z_h"],
        "family": train_preds_norm["family"]
    }
    
    train_metrics = evaluate_model(
        train_preds, Y_A_train, Y_f_train, Y_R_train, Y_zh_train, Y_family_train
    )
    
    print(f"   A(z) R2:         {train_metrics['A_r2']:.4f}")
    print(f"   f(z) R2:         {train_metrics['f_r2']:.4f}")
    print(f"   R(z) R2:         {train_metrics['R_r2']:.4f}")
    print(f"   z_h MAE:         {train_metrics['zh_mae']:.4f}")
    print(f"   Family accuracy: {train_metrics['family_accuracy']:.4f}")
    
    # === GUARDAR ===
    
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)
    
    for i, name in enumerate(train_data["names"]):
        np.savez(
            preds_dir / f"{name}_geometry.npz",
            z=z_grid,
            A_pred=train_preds["A"][i],
            f_pred=train_preds["f"][i],
            R_pred=train_preds["R"][i],
            A_truth=Y_A_train[i],
            f_truth=Y_f_train[i],
            R_truth=Y_R_train[i],
            zh_pred=train_preds["z_h"][i],
            zh_truth=Y_zh_train[i],
            family_pred=train_preds["family"][i],
            family_truth=Y_family_train[i],
            category="known"
        )
    
    test_metrics = None
    if len(test_data["X"]) > 0:
        print("\n>> Aplicando a TEST (test/unknown)...")
        
        X_test = np.stack(test_data["X"])
        X_test_norm = (X_test - X_mean) / X_std
        X_test_t = torch.from_numpy(X_test_norm.astype(np.float32))
        
        test_preds_norm = inference_on_data(model, X_test_t, device)
        
        test_preds = {
            "A": normalizer.denormalize_A(test_preds_norm["A"]),
            "f": test_preds_norm["f"],
            "R": normalizer.denormalize_R(test_preds_norm["R"]),
            "z_h": test_preds_norm["z_h"],
            "family": test_preds_norm["family"]
        }
        
        Y_A_test = np.stack(test_data["Y_A"])
        Y_f_test = np.stack(test_data["Y_f"])
        Y_R_test = np.stack(test_data["Y_R"])
        Y_zh_test = np.array(test_data["Y_zh"])
        Y_family_test = np.array(test_data["Y_family"])
        
        test_metrics = evaluate_model(
            test_preds, Y_A_test, Y_f_test, Y_R_test, Y_zh_test, Y_family_test
        )
        
        print(f"\n   TEST Metrics:")
        print(f"   A(z) R2:         {test_metrics['A_r2']:.4f}")
        print(f"   f(z) R2:         {test_metrics['f_r2']:.4f}")
        print(f"   R(z) R2:         {test_metrics['R_r2']:.4f}")
        print(f"   z_h MAE:         {test_metrics['zh_mae']:.4f}")
        print(f"   Family accuracy: {test_metrics['family_accuracy']:.4f}")
        
        for i, name in enumerate(test_data["names"]):
            np.savez(
                preds_dir / f"{name}_geometry.npz",
                z=z_grid,
                A_pred=test_preds["A"][i],
                f_pred=test_preds["f"][i],
                R_pred=test_preds["R"][i],
                A_truth=Y_A_test[i],
                f_truth=Y_f_test[i],
                R_truth=Y_R_test[i],
                zh_pred=test_preds["z_h"][i],
                zh_truth=Y_zh_test[i],
                family_pred=test_preds["family"][i],
                family_truth=Y_family_test[i],
                category=test_data["categories"][i]
            )
    
    model_path = output_dir / "emergent_geometry_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": X_train.shape[1],
        "n_z": Y_A_train.shape[1],
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "X_mean": X_mean,
        "X_std": X_std,
        "normalizer": normalizer.state_dict()
    }, model_path)
    
    summary = {
        "n_train": len(X_train),
        "n_test": len(test_data["X"]),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "final_loss": history["total"][-1] if history["total"] else None,
        "model_path": str(model_path),
        "predictions_dir": str(preds_dir)
    }
    
    summary_path = output_dir / "emergent_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    
    print("\n" + "=" * 70)
    print("[OK] FASE XI v2 - GEOMETRIA EMERGENTE COMPLETADA")
    print(f"  Modelo:       {model_path}")
    print(f"  Predicciones: {preds_dir}")
    print("=" * 70)
    print("\nProximo paso: 02_discover_einstein_v2.py")


if __name__ == "__main__":
    main()
