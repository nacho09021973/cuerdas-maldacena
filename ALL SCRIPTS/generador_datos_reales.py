#!/usr/bin/env python3
# generador_datos_reales.py
# CUERDAS — Utilidad interna: generador de datos físicos (reales o sintéticos) para Fase XII
# Versión 1.0 (diciembre 2025)
#
# Este script NO forma parte numerada del pipeline (sin "faseXX_").
# Se usa solo como herramienta auxiliar para:
#   - Construir descriptores limpios de teorías concretas (ej. Ising 3D bootstrap)
#   - Generar datos sintéticos "realistas" (bootstrap / lattice / condensed / cosmología)
#
# Filosofía de honestidad:
#   - Todos los números "reales" (Δ_sigma, Δ_epsilon, c, etc.) vienen explícitos del código,
#     inspirados en literatura estándar, pero este script está pensado como SANDBOX / TESTING.
#   - Nada de esto se usa para entrenar el diccionario emergente con la fórmula de Maldacena
#     codificada; son solo entradas de datos físicos para Fase XII.

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import h5py


# =============================================================================
# 1. Descriptor explícito para Ising 3D (bootstrap)
#    (destilado de fase12_adapter_ising3d_bootstrap.py)
# =============================================================================

def build_ising3d_descriptor() -> Dict[str, Any]:
    """
    Construye un descriptor compacto para el modelo Ising 3D,
    pensado para conectar con datos de bootstrap.

    Estructura pensada como input "limpio" para adaptadores de Fase XII.
    """
    descriptor = {
        "system_name": "ising3d_bootstrap",
        "source": "bootstrap",
        "theory": "Ising 3D",
        "d": 3,
        "operators": [
            {
                "name": "sigma",
                "Delta": 0.518,
                "spin": 0,
                "role": "order_parameter",
                "error": 0.001
            },
            {
                "name": "epsilon",
                "Delta": 1.41,
                "spin": 0,
                "role": "energy_density",
                "error": 0.01
            },
            {
                "name": "epsilon_prime",
                "Delta": 3.83,
                "spin": 0,
                "role": "next_scalar",
                "error": 0.05
            },
            {
                "name": "T",
                "Delta": 3.0,
                "spin": 2,
                "role": "stress_tensor",
                "error": 0.001
            }
        ],
        "metadata": {
            "central_charge_normalized": 0.946,
            "reference": "Bootstrap Ising 3D (literatura estándar, valores aproximados)",
            "notes": [
                "Este descriptor es un stub honesto para tests de Fase XII.",
                "No codifica ninguna relación holográfica Δ ↔ m²L².",
                "Uso previsto: servir como 'real_data' para adaptadores bootstrap."
            ]
        }
    }
    return descriptor


def save_ising3d_descriptor(output_dir: Path) -> Path:
    """
    Guarda el descriptor de Ising 3D en JSON.

    Devuelve la ruta del archivo generado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ising3d_descriptor.json"
    desc = build_ising3d_descriptor()
    path.write_text(json.dumps(desc, indent=2))
    return path


# =============================================================================
# 2. Generadores sintéticos "realistas" (destilado de fase12_real_data_adapters.py)
# =============================================================================

def generate_synthetic_bootstrap_data(theory: str, output_path: Path):
    """Genera datos sintéticos tipo bootstrap para testing (Ising 3D, O(4), genérico)."""
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
        # Genérico mínimo
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
    """Genera datos sintéticos tipo lattice QCD (ecuación de estado + eta/s)."""
    T = np.linspace(0.1, 0.5, 50)  # GeV
    Tc = 0.155  # GeV

    # Ecuación de estado tipo crossover
    t = (T - Tc) / Tc
    # Presión: transición suave
    p = T**4 * (0.5 * (1 + np.tanh(2 * t)) * 0.9 + 0.1)
    # Densidad de energía
    eps = 3 * p + T * np.gradient(p, T)
    # Entropía
    s = (eps + p) / T
    # η/s cerca del bound
    eta_s = 0.08 * np.ones_like(T) + 0.1 * np.exp(-((T - Tc) / 0.05)**2)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("T", data=T)
        f.create_dataset("pressure", data=p)
        f.create_dataset("energy_density", data=eps)
        f.create_dataset("entropy", data=s)
        f.create_dataset("eta_over_s", data=eta_s)
        f.attrs["Tc"] = Tc


def generate_synthetic_transport_data(material: str, output_path: Path):
    """Genera datos sintéticos de transporte (strange metal vs Fermi liquid)."""
    T = np.linspace(10, 300, 100)  # Kelvin

    if "strange" in material.lower():
        # Resistividad lineal en T
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
    """Genera datos sintéticos tipo CMB (espectro TT simplificado)."""
    ell = np.arange(2, 2000)

    # Espectro simplificado
    ns = 0.965
    As = 2.1e-9

    # Dl = l(l+1)Cl / 2π
    Dl = As * (ell / 200.0) ** (ns - 1) * np.exp(-(ell / 1500)**2) * 1e10

    data = {
        "ell": ell.tolist(),
        "Cl_TT": (Dl / (ell * (ell + 1)) * 2 * np.pi).tolist(),
        "ns": ns,
        "As": As,
        "H0": 67.4
    }

    output_path.write_text(json.dumps(data, indent=2))


def generate_all_synthetic(output_dir: Path):
    """
    Replica (y simplifica) el modo "generate-synthetic" de fase12_real_data_adapters.py:
      - Bootstrap: Ising 3D, O(4)
      - Lattice QCD (HDF5)
      - Strange metal (JSON)
      - CMB (JSON)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap: Ising 3D
    generate_synthetic_bootstrap_data("ising3d", output_dir / "ising3d_bootstrap.json")
    print(f"✓ Ising 3D bootstrap → {output_dir / 'ising3d_bootstrap.json'}")

    # Bootstrap: O(4)
    generate_synthetic_bootstrap_data("on_n4", output_dir / "o4_bootstrap.json")
    print(f"✓ O(4) bootstrap    → {output_dir / 'o4_bootstrap.json'}")

    # Lattice QCD
    generate_synthetic_lattice_data(output_dir / "lattice_qcd.h5")
    print(f"✓ Lattice QCD       → {output_dir / 'lattice_qcd.h5'}")

    # Strange metal
    generate_synthetic_transport_data(
        "strange_metal_cuprate",
        output_dir / "strange_metal.json"
    )
    print(f"✓ Strange metal     → {output_dir / 'strange_metal.json'}")

    # CMB
    generate_synthetic_cmb_data(output_dir / "cmb_planck.json")
    print(f"✓ CMB (Planck-like) → {output_dir / 'cmb_planck.json'}")

    print("\n======================================================================")
    print("✓ Datos sintéticos generados (bootstrap / lattice / condensed / CMB)")
    print("  Siguiente típico:")
    print("    - Para tests de XI: usar estos archivos como entrada de adaptadores reales.")
    print("    - Para XII: construir posteriormente boundary.h5 + manifest_fase12.json.")
    print("======================================================================")


# =============================================================================
# 3. CLI principal
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CUERDAS — Generador interno de datos reales/sintéticos para Fase XII"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="ising3d_descriptor",
        choices=["ising3d_descriptor", "synthetic_all"],
        help=(
            "Qué generar:\n"
            "  - ising3d_descriptor: solo descriptor JSON compacto de Ising 3D.\n"
            "  - synthetic_all: paquete completo de datos sintéticos (bootstrap, lattice, etc.)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="real_data_sandbox",
        help="Directorio de salida (se creará si no existe).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("CUERDAS — GENERADOR INTERNO DE DATOS REALES/SINTÉTICOS")
    print("=" * 70)
    print(f"Preset:      {args.preset}")
    print(f"Output dir:  {output_dir}")
    print("=" * 70)

    if args.preset == "ising3d_descriptor":
        path = save_ising3d_descriptor(output_dir)
        print(f"✓ Descriptor Ising 3D guardado en: {path}")
        print("\nUso típico:")
        print("  - Como input 'limpio' para un BootstrapAdapter especializado.")
        print("  - O como bloque 'real_data' dentro de un fase12_report.json stub.")
    elif args.preset == "synthetic_all":
        generate_all_synthetic(output_dir)
        print("\nUso típico:")
        print("  - Paso previo a ejecutar adaptadores reales (bootstrap/lattice/condensed/cosmology).")
        print("  - Buena batería de pruebas para Fase XII completa.")
    else:
        print(f"Preset no soportado: {args.preset}")
        return 1

    print("\n[OK] Generación completada sin errores aparentes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
