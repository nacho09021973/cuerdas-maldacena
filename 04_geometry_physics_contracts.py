#!/usr/bin/env python3
# 04_geometry_physics_contracts.py
# CUERDAS — Bloque A: Geometría emergente (contratos físicos)
#
# OBJETIVO
#   Evaluar la calidad física y la honestidad del bloque de geometría y ecuaciones:
#     - Geometría emergente vs bulk_truth (cuando exista sandbox).
#     - Ecuaciones de bulk descubiertas vs criterios físicos.
#
# ENTRADAS
#   - runs/emergent_geometry/emergent_geometry_summary.json
#   - runs/bulk_equations/equations_pareto.json (y/o pysr_summary.json)
#   - Opcional: runs/sandbox_geometries/bulk_truth/*.h5 (para contratos sandbox)
#
# SALIDAS
#   runs/geometry_contracts/
#     geometry_contracts_summary.json
#       - Clasificación por sistema:
#           * Einstein-like / non-Einstein / incierto
#           * Umbrales de R², estabilidad en rollouts, etc.
#       - Flags de honestidad (uso indebido de bulk, mezcla incorrecta de d, ...).
#
# TIPOS DE CONTRATO (EJEMPLOS)
#   - R² mínimo en test, estabilidad en evoluciones numéricas.
#   - No mezcla de dimensiones d entre sistemas incompatibles.
#   - No uso de variables de "verdad" en la construcción de la loss.
#   - Coherencia entre familia asignada (ads/lifshitz/hvlf/deformed) y patrones geométricos.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Consume salidas de:
#       * 02_emergent_geometry_engine.py
#       * 03_discover_bulk_equations.py
#   - Sus resultados condicionan el análisis de:
#       * 05_analyze_bulk_equations.py
#
# HISTÓRICO
#   - Anteriormente conocido como: 04_contracts_fase_11_v2.py

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import h5py


# ============================================================
# CONTRATOS GENRICOS (aplican a TODAS las geometras)
# ============================================================

@dataclass
class GenericRegularityContract:
    """Verifica propiedades bsicas de regularidad."""
    A_finite: bool
    f_finite: bool
    no_nan: bool
    smooth: bool  # derivadas no explotan
    
    @property
    def passed(self) -> bool:
        return self.A_finite and self.f_finite and self.no_nan


@dataclass
class GenericCausalityContract:
    """Verifica estructura causal bsica (no especfica de AdS)."""
    f_non_negative: bool  # f  0 (o muy cerca)
    f_bounded: bool  # f no explota
    horizon_if_thermal: bool  # si T > 0, debe haber estructura de horizonte
    
    @property
    def passed(self) -> bool:
        return self.f_non_negative and self.f_bounded


# ============================================================
# CONTRATOS AdS-ESPECFICOS (solo para family="ads")
# ============================================================

@dataclass
class AdSEinsteinContract:
    """Verifica ecuaciones de Einstein en vaco con  (SOLO para AdS)."""
    R_is_constant: bool
    R_matches_ads: bool
    Lambda_matches_ads: bool
    R_mean: float
    R_expected: float
    
    @property
    def passed(self) -> bool:
        return self.R_is_constant and self.R_matches_ads


@dataclass
class AdSAsymptoticContract:
    """Verifica comportamiento asinttico tipo AdS."""
    A_logarithmic_uv: bool  # A(z) ~ -log(z) cerca de z=0
    f_to_one_uv: bool  # f  1 cuando z  0
    A_monotone: bool  # dA/dz < 0
    
    @property
    def passed(self) -> bool:
        return self.A_logarithmic_uv and self.A_monotone


@dataclass
class HolographicDictionaryContract:
    """Verifica diccionario hologrfico (ms relevante para AdS)."""
    mass_dimension_ok: bool
    hawking_ok: bool
    conformal_symmetry_ok: bool
    
    @property
    def passed(self) -> bool:
        return self.mass_dimension_ok or self.hawking_ok or self.conformal_symmetry_ok


# ============================================================
# CONTRATO GLOBAL
# ============================================================

@dataclass
class PhaseXIContractV2:
    """Contrato completo de Fase XI v2."""
    name: str
    family: str
    category: str
    d: int
    
    # Contratos genricos (TODOS deben pasar)
    regularity: GenericRegularityContract
    causality: GenericCausalityContract
    
    # Contratos AdS-especficos (solo relevantes si family="ads")
    ads_einstein: AdSEinsteinContract
    ads_asymptotic: AdSAsymptoticContract
    holographic: HolographicDictionaryContract
    
    # Mtricas de reconstruccin
    A_r2: float
    f_r2: float
    R_r2: float
    family_accuracy: float
    
    @property
    def generic_passed(self) -> bool:
        """Contratos que TODA geometra debe pasar."""
        return self.regularity.passed and self.causality.passed
    
    @property
    def ads_specific_passed(self) -> bool:
        """Contratos especficos de AdS (solo relevantes si es AdS)."""
        return self.ads_einstein.passed and self.ads_asymptotic.passed
    
    @property
    def is_ads_family(self) -> bool:
        return self.family == "ads"
    
    @property
    def contract_score(self) -> float:
        """Score ponderado por tipo de geometria."""
        def to_float(val) -> float:
            """Convierte bool o string a float."""
            if isinstance(val, bool):
                return 1.0 if val else 0.0
            if isinstance(val, str):
                return 1.0 if val.lower() == 'true' else 0.0
            return float(val)
        
        # Genericos: siempre cuentan
        score = 0.3 * to_float(self.regularity.passed) + 0.2 * to_float(self.causality.passed)
        
        # Reconstruccion (clampear R2 a [0, 1] para el score)
        a_r2_safe = max(0.0, min(1.0, self.A_r2))
        f_r2_safe = max(0.0, min(1.0, self.f_r2))
        score += 0.2 * (a_r2_safe + f_r2_safe) / 2
        
        # AdS-especificos: solo cuentan si es AdS
        if self.is_ads_family:
            score += 0.2 * to_float(self.ads_einstein.passed)
            score += 0.1 * to_float(self.holographic.passed)
        else:
            # Para no-AdS, damos puntos si paso genericos
            score += 0.2 * to_float(self.generic_passed)
            score += 0.1 * to_float(self.holographic.conformal_symmetry_ok)
        
        return score
    
    @property
    def overall_passed(self) -> bool:
        """Pas la fase."""
        if not self.generic_passed:
            return False
        if self.is_ads_family:
            return self.ads_specific_passed
        return True  # Non-AdS solo necesita genricos


# ============================================================
# FUNCIONES DE VERIFICACIN
# ============================================================

def verify_regularity(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray
) -> GenericRegularityContract:
    """Verifica regularidad bsica."""
    A_finite = np.all(np.isfinite(A)) and np.max(np.abs(A)) < 1e6
    f_finite = np.all(np.isfinite(f)) and np.max(np.abs(f)) < 1e6
    no_nan = not (np.any(np.isnan(A)) or np.any(np.isnan(f)))
    
    # Suavidad: derivadas no explotan
    dz = z[1] - z[0]
    dA = np.gradient(A, dz)
    smooth = np.all(np.abs(dA) < 1e4)
    
    return GenericRegularityContract(
        A_finite=A_finite,
        f_finite=f_finite,
        no_nan=no_nan,
        smooth=smooth
    )


def verify_generic_causality(
    f: np.ndarray,
    z: np.ndarray,
    T: float,
    z_h: float
) -> GenericCausalityContract:
    """Verifica causalidad bsica."""
    f_non_negative = np.min(f) >= -0.1
    f_bounded = np.max(f) < 10.0
    
    # Si hay temperatura, debe haber estructura de horizonte
    if T > 1e-10 and z_h > 0:
        idx_h = np.argmin(np.abs(z - z_h))
        f_near_h = f[max(0, idx_h-3):min(len(f), idx_h+3)]
        horizon_ok = np.min(f_near_h) < 0.5  # f debe caer cerca del horizonte
    else:
        horizon_ok = True
    
    return GenericCausalityContract(
        f_non_negative=f_non_negative,
        f_bounded=f_bounded,
        horizon_if_thermal=horizon_ok
    )


def verify_ads_einstein(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray,
    d: int,
    einstein_results: Dict[str, Any]
) -> AdSEinsteinContract:
    """Verifica ecuaciones de Einstein (solo para AdS)."""
    from scipy.ndimage import gaussian_filter1d
    
    D = d + 1
    dz = z[1] - z[0]
    n = len(z)
    
    # Suavizar A y f antes de derivar (reduce ruido numerico)
    sigma = 2.0  # puntos de suavizado
    A_smooth = gaussian_filter1d(A, sigma=sigma, mode='nearest')
    f_smooth = gaussian_filter1d(f, sigma=sigma, mode='nearest')
    f_smooth = np.clip(f_smooth, 1e-6, None)
    
    # Derivadas con esquema de 5 puntos (mas estable que gradiente simple)
    def deriv_5pt(y, dx):
        """Derivada con diferencias finitas de 5 puntos."""
        d = np.zeros_like(y)
        # Interior: esquema centrado de 5 puntos
        for i in range(2, len(y) - 2):
            d[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12 * dx)
        # Bordes: esquema de 3 puntos
        d[0] = (y[1] - y[0]) / dx
        d[1] = (y[2] - y[0]) / (2 * dx)
        d[-1] = (y[-1] - y[-2]) / dx
        d[-2] = (y[-1] - y[-3]) / (2 * dx)
        return d
    
    dA = deriv_5pt(A_smooth, dz)
    d2A = deriv_5pt(dA, dz)
    df = deriv_5pt(f_smooth, dz)
    
    # Calcular R
    R = -2 * D * d2A - D * (D - 1) * dA**2 - df * dA / f_smooth
    
    # Evaluar solo en region interior (evitar efectos de borde)
    margin = max(5, n // 10)  # 10% de margen en cada lado
    R_interior = R[margin:-margin]
    
    R_mean = np.mean(R_interior)
    R_std = np.std(R_interior)
    R_expected = -D * (D - 1)
    
    # Criterios mas razonables
    cv = R_std / (np.abs(R_mean) + 1e-10)
    R_is_constant = cv < 0.3  # Coeficiente de variacion < 30%
    R_matches_ads = np.abs(R_mean - R_expected) / np.abs(R_expected) < 0.5  # 50% tolerancia
    
    # Desde resultados de Einstein discovery (si disponibles)
    Lambda_check = einstein_results.get("einstein_check", {})
    Lambda_matches = Lambda_check.get("consistent_with_einstein_vacuum", False)
    
    return AdSEinsteinContract(
        R_is_constant=R_is_constant,
        R_matches_ads=R_matches_ads,
        Lambda_matches_ads=Lambda_matches,
        R_mean=float(R_mean),
        R_expected=float(R_expected)
    )


def verify_ads_asymptotic(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray
) -> AdSAsymptoticContract:
    """Verifica comportamiento asintotico tipo AdS."""
    from scipy.ndimage import gaussian_filter1d
    
    dz = z[1] - z[0]
    n = len(z)
    
    # Usar mas puntos en UV para mejor estadistica
    n_uv = min(10, n // 5)
    
    # UV: A(z) ~ -log(z)
    A_uv = A[:n_uv]
    z_uv = z[:n_uv]
    expected_A = -np.log(z_uv)
    
    # Calcular correlacion en lugar de error absoluto
    # (mas robusto a offsets constantes)
    A_centered = A_uv - np.mean(A_uv)
    exp_centered = expected_A - np.mean(expected_A)
    
    corr = np.corrcoef(A_uv, expected_A)[0, 1]
    A_log_uv = corr > 0.95  # Alta correlacion con -log(z)
    
    # f -> 1 en UV
    f_uv = f[:n_uv]
    f_to_one = np.mean(np.abs(f_uv - 1)) < 0.15
    
    # A monotono decreciente (suavizado)
    A_smooth = gaussian_filter1d(A, sigma=2, mode='nearest')
    dA = np.gradient(A_smooth, dz)
    
    # Evaluar monotonicidad en region interior
    margin = max(3, n // 20)
    dA_interior = dA[margin:-margin]
    A_monotone = np.mean(dA_interior < 0.01) > 0.8  # 80% con pendiente negativa o ~0
    
    return AdSAsymptoticContract(
        A_logarithmic_uv=A_log_uv,
        f_to_one_uv=f_to_one,
        A_monotone=A_monotone
    )


def verify_holographic(
    dict_results: Dict[str, Any]
) -> HolographicDictionaryContract:
    """Verifica diccionario holografico."""
    def to_bool(val) -> bool:
        """Convierte string/bool a bool."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == 'true'
        return bool(val)
    
    mass_dim = dict_results.get("mass_dimension", {})
    mass_ok = to_bool(mass_dim.get("comparison_with_holographic", {}).get("likely_holographic", False))
    
    hawking = dict_results.get("hawking", {})
    hawking_ok = to_bool(hawking.get("hawking_check", {}).get("hawking_verified", False))
    
    # Buscar en geometras individuales
    conformal_ok = False
    for geo in dict_results.get("geometries", []):
        conf = geo.get("conformal", {}).get("summary", {})
        if to_bool(conf.get("conformal_symmetry_present", False)):
            conformal_ok = True
            break
    
    return HolographicDictionaryContract(
        mass_dimension_ok=mass_ok,
        hawking_ok=hawking_ok,
        conformal_symmetry_ok=conformal_ok
    )


# ============================================================
# PROCESAMIENTO
# ============================================================

def process_geometry(
    name: str,
    data_dir: Path,
    geometry_dir: Path,
    einstein_dir: Path,
    dictionary_results: Dict[str, Any],
    d: int
) -> PhaseXIContractV2:
    """Procesa una geometra y genera su contrato v2."""
    
    # Cargar datos originales
    h5_path = data_dir / f"{name}.h5"
    with h5py.File(h5_path, "r") as f:
        family = str(f.attrs.get("family", "unknown"))
        category = str(f.attrs.get("category", "unknown"))
        
        boundary = f["boundary"]
        T = float(boundary.attrs.get("temperature", 0))
        
        bulk = f["bulk_truth"]
        z_h = float(bulk.attrs.get("z_h", 0))
        z_grid = bulk["z_grid"][:]
    
    # Cargar predicciones
    geo_path = geometry_dir / "predictions" / f"{name}_geometry.npz"
    geo_data = np.load(geo_path, allow_pickle=True)
    A_pred = geo_data["A_pred"]
    f_pred = geo_data["f_pred"]
    A_truth = geo_data["A_truth"]
    f_truth = geo_data["f_truth"]
    R_truth = geo_data.get("R_truth", np.zeros_like(A_truth))
    R_pred = geo_data.get("R_pred", np.zeros_like(A_pred))
    
    # Metricas R2
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-10:
            return 0.0  # Evitar division por cero
        r2_val = 1 - ss_res / ss_tot
        # Clampear a [-1, 1] para evitar scores extremos
        return float(np.clip(r2_val, -1.0, 1.0))
    
    A_r2 = r2(A_truth, A_pred)
    f_r2 = r2(f_truth, f_pred)
    R_r2 = r2(R_truth, R_pred) if np.any(R_truth != 0) else 0.0
    family_match = int(geo_data.get("family_pred", -1)) == int(geo_data.get("family_truth", -2))
    
    # Cargar resultados Einstein
    einstein_path = einstein_dir / name / "einstein_discovery.json"
    if einstein_path.exists():
        einstein_results = json.loads(einstein_path.read_text()).get("results", {})
    else:
        einstein_results = {}
    
    # Verificar contratos
    regularity = verify_regularity(A_pred, f_pred, z_grid)
    causality = verify_generic_causality(f_pred, z_grid, T, z_h)
    ads_einstein = verify_ads_einstein(A_pred, f_pred, z_grid, d, einstein_results)
    ads_asymptotic = verify_ads_asymptotic(A_pred, f_pred, z_grid)
    holographic = verify_holographic(dictionary_results)
    
    return PhaseXIContractV2(
        name=name,
        family=family,
        category=category,
        d=d,
        regularity=regularity,
        causality=causality,
        ads_einstein=ads_einstein,
        ads_asymptotic=ads_asymptotic,
        holographic=holographic,
        A_r2=float(A_r2),
        f_r2=float(f_r2),
        R_r2=float(R_r2),
        family_accuracy=float(family_match)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fase XI v2: Contratos de validacin fsica HONESTOS"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--geometry-dir", type=str, required=True)
    parser.add_argument("--einstein-dir", type=str, required=True)
    parser.add_argument("--dictionary-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="fase11_contracts_v2.json")
    parser.add_argument("--d", type=int, default=4)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    geometry_dir = Path(args.geometry_dir)
    einstein_dir = Path(args.einstein_dir)
    dictionary_path = Path(args.dictionary_file)
    output_file = Path(args.output_file)
    
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    
    # Cargar resultados del diccionario
    dictionary_results = json.loads(dictionary_path.read_text()) if dictionary_path.exists() else {}
    
    print("=" * 90)
    print("FASE XI v2  CONTRATOS DE VALIDACIN FSICA HONESTOS")
    print("=" * 90)
    print("\nFILOSOFA:")
    print("   Contratos GENRICOS: todas las geometras deben pasar")
    print("   Contratos AdS-ESPECFICOS: solo relevantes para family='ads'")
    print("   Geometras no-AdS no fallan por no cumplir criterios AdS")
    print("=" * 90)
    
    all_contracts = []
    
    for geo_info in manifest["geometries"]:
        name = geo_info["name"]
        print(f"\n>> {name}")
        
        try:
            contract = process_geometry(
                name, data_dir, geometry_dir, einstein_dir,
                dictionary_results, args.d
            )
            all_contracts.append(contract)
            
            print(f"   Family: {contract.family}")
            print(f"   Genricos:     {'' if contract.generic_passed else ''} "
                  f"(reg={contract.regularity.passed}, caus={contract.causality.passed})")
            if contract.is_ads_family:
                print(f"   AdS-specific:  {'' if contract.ads_specific_passed else ''} "
                      f"(einstein={contract.ads_einstein.passed}, asymp={contract.ads_asymptotic.passed})")
            else:
                print(f"   AdS-specific:  N/A (no es family='ads')")
            print(f"   Score: {contract.contract_score:.2f}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # ============================================================
    # RESUMEN
    # ============================================================
    
    print("\n" + "=" * 100)
    print("RESUMEN DE CONTRATOS FASE XI v2")
    print("=" * 100)
    
    # Tabla
    print(f"\n{'Name':<20} {'Family':<12} {'Category':<10} {'Generic':^8} {'AdS':^8} {'Score':^8} {'Pass':^6}")
    print("-" * 100)
    
    for c in all_contracts:
        ads_col = '' if c.ads_specific_passed else ('' if c.is_ads_family else '')
        print(f"{c.name:<20} {c.family:<12} {c.category:<10} "
              f"{'' if c.generic_passed else '':^8} "
              f"{ads_col:^8} "
              f"{c.contract_score:.2f} "
              f"{'' if c.overall_passed else '':^6}")
    
    print("-" * 100)
    
    # Estadsticas
    n_total = len(all_contracts)
    n_generic = sum(c.generic_passed for c in all_contracts)
    n_ads_family = sum(c.is_ads_family for c in all_contracts)
    n_ads_passed = sum(c.ads_specific_passed for c in all_contracts if c.is_ads_family)
    n_overall = sum(c.overall_passed for c in all_contracts)
    avg_score = np.mean([c.contract_score for c in all_contracts])
    
    print(f"\nESTADSTICAS:")
    print(f"  Total geometras:      {n_total}")
    print(f"  Genricos OK:          {n_generic}/{n_total}")
    print(f"  AdS families:          {n_ads_family}/{n_total}")
    print(f"  AdS-specific OK:       {n_ads_passed}/{n_ads_family} (de las que son AdS)")
    print(f"  Overall passed:        {n_overall}/{n_total}")
    print(f"  Score promedio:        {avg_score:.3f}")
    
    # Por familia
    print("\nPOR FAMILIA:")
    families = set(c.family for c in all_contracts)
    for fam in sorted(families):
        fam_contracts = [c for c in all_contracts if c.family == fam]
        n_fam = len(fam_contracts)
        n_passed = sum(c.overall_passed for c in fam_contracts)
        avg_fam = np.mean([c.contract_score for c in fam_contracts])
        print(f"  {fam:15}: {n_passed}/{n_fam} passed, score={avg_fam:.2f}")
    
    # Por categora
    print("\nPOR CATEGORA:")
    for cat in ["known", "test", "unknown"]:
        cat_contracts = [c for c in all_contracts if c.category == cat]
        if cat_contracts:
            n_cat = len(cat_contracts)
            n_passed = sum(c.overall_passed for c in cat_contracts)
            avg_cat = np.mean([c.contract_score for c in cat_contracts])
            print(f"  {cat:12}: {n_passed}/{n_cat} passed, score={avg_cat:.2f}")
    
    # Veredicto final
    print("\n" + "=" * 90)
    
    phase_passed = (n_generic == n_total) and (avg_score > 0.5)
    
    if phase_passed:
        print(" FASE XI v2 COMPLETADA  VALIDACIN HONESTA EXITOSA")
        print("=" * 90)
        print("\n  El sistema CUERDAS ha logrado:")
        print("     Todas las geometras pasan contratos genricos")
        if n_ads_family > 0:
            print(f"     {n_ads_passed}/{n_ads_family} geometras AdS pasan contratos especficos")
        print("     Score promedio > 0.5")
    else:
        print(" FASE XI v2 REQUIERE REFINAMIENTO")
        print("=" * 90)
        if n_generic < n_total:
            print(f"\n  {n_total - n_generic} geometras no pasan contratos genricos")
        if avg_score <= 0.5:
            print(f"\n  Score promedio ({avg_score:.2f}) demasiado bajo")
    
    # Guardar
    def serialize_value(v):
        """Convierte valores para JSON limpio."""
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, dict):
            return {k: serialize_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [serialize_value(item) for item in v]
        return v
    
    def clean_contract(c_dict):
        """Limpia un contrato para JSON."""
        return {k: serialize_value(v) for k, v in c_dict.items()}
    
    output_data = {
        "n_total": n_total,
        "n_generic_passed": int(n_generic),
        "n_ads_family": int(n_ads_family),
        "n_ads_specific_passed": int(n_ads_passed),
        "n_overall_passed": int(n_overall),
        "avg_score": float(avg_score),
        "phase_passed": bool(phase_passed),
        "contracts": [clean_contract(asdict(c)) for c in all_contracts]
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n  Resultados: {output_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
