#!/usr/bin/env python3
"""
analyze_lambda_sl_scatter.py

Pequeño análisis exploratorio para el dataset emergente XI→XII.c (v2_lambda_sl):

- Construye un DataFrame con (Delta, d, lambda_sl_emergent).
- Calcula lambda_sl_theory = Delta * (Delta - d).
- Calcula un R² "naive" para la fórmula teórica.
- Genera dos scatter plots:
    1) lambda_sl_emergent vs Delta
    2) lambda_sl_emergent vs lambda_sl_theory

Uso sugerido:

  (.venv39) python analyze_lambda_sl_scatter.py \\
    --input-json scripts/smoke_fase12c_v2/fase11_bulk_for_fase12c_lambda_sl.json \\
    --output-dir scripts/smoke_fase12c_v2/analysis

"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza λ_SL vs Δ para el dataset emergente XI→XII.c (v2_lambda_sl)."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Ruta al JSON generado por make_fase11_bulk_for_fase12c_v2.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio donde guardar los gráficos (por defecto: <carpeta_input>/analysis)",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Filtra por familia (ads, lifshitz, hvlf, etc.). Si se omite, usa todas.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=None,
        help="Filtra por dimensión d. Si se omite, usa todas.",
    )
    return parser.parse_args()


def load_v2_lambda_sl(path: Path) -> pd.DataFrame:
    """Carga el JSON v2_lambda_sl y construye un DataFrame plano."""
    with path.open("r") as f:
        data: Dict[str, Any] = json.load(f)

    records: List[Dict[str, Any]] = []
    for system in data.get("systems", []):
        sys_name = system.get("geometry_name", "unknown")
        family = system.get("family", "unknown")
        d = system.get("d", 4)
        source = system.get("lambda_source", "bulk_eigenmode")

        lambda_list = system.get("lambda_sl_bulk", [])
        Delta_list = system.get("Delta_bulk_uv", [])

        if len(lambda_list) != len(Delta_list):
            print(
                f"[WARN] Sistema {sys_name}: "
                f"len(Delta)={len(Delta_list)} != len(lambda_sl)={len(lambda_list)}. "
                f"Se ignora este sistema."
            )
            continue

        for i, (Delta, lam) in enumerate(zip(Delta_list, lambda_list)):
            records.append(
                {
                    "system": sys_name,
                    "family": family,
                    "d": int(d),
                    "source": source,
                    "mode_index": i,
                    "Delta": float(Delta),
                    "lambda_sl_emergent": float(lam),
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No se han construido registros; revisa el JSON de entrada.")

    return df


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula R² a mano para evitar dependencias adicionales."""
    if y_true.size == 0:
        return np.nan
    mu = np.mean(y_true)
    ss_tot = np.sum((y_true - mu) ** 2)
    if ss_tot == 0.0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1.0 - ss_res / ss_tot


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el JSON de entrada: {input_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ANÁLISIS λ_SL vs Δ — DATASET XI→XII.c (v2_lambda_sl)")
    print("=" * 80)
    print(f"Input JSON:  {input_path}")
    print(f"Output dir:  {output_dir}")
    if args.family:
        print(f"Filtro family = {args.family}")
    if args.d is not None:
        print(f"Filtro d      = {args.d}")
    print("=" * 80)

    df = load_v2_lambda_sl(input_path)

    # Filtros opcionales
    if args.family is not None:
        df = df[df["family"] == args.family]
    if args.d is not None:
        df = df[df["d"] == args.d]

    if df.empty:
        print("[WARN] Tras aplicar filtros, el DataFrame está vacío.")
        return 0

    # Construir columna teórica
    df["lambda_sl_theory"] = df["Delta"] * (df["Delta"] - df["d"])

    y_true = df["lambda_sl_emergent"].values
    y_theory = df["lambda_sl_theory"].values

    r2_theory = r2_score(y_true, y_theory)

    print("\nRESUMEN NUMÉRICO (naive):")
    print(f"  n puntos        : {len(df)}")
    print(f"  R²(λ_theory=Δ(Δ-d)) vs λ_emergent : {r2_theory: .4f}")

    # Scatter 1: lambda_sl_emergent vs Delta
    plt.figure()
    plt.scatter(df["Delta"].values, y_true, s=10)
    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$\lambda_{\mathrm{SL, emergent}}$")
    plt.title("Emergent λ_SL vs Δ")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    out1 = output_dir / "scatter_lambda_vs_delta.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  Guardado scatter λ_SL vs Δ en: {out1}")

    # Scatter 2: lambda_sl_emergent vs lambda_sl_theory
    plt.figure()
    plt.scatter(y_theory, y_true, s=10)
    plt.xlabel(r"$\lambda_{\mathrm{SL, theory}} = \Delta(\Delta - d)$")
    plt.ylabel(r"$\lambda_{\mathrm{SL, emergent}}$")
    plt.title("Emergent λ_SL vs λ_SL, theory (Δ(Δ-d))")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    out2 = output_dir / "scatter_lambda_vs_lambda_theory.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Guardado scatter λ_emergent vs λ_theory en: {out2}")

    print("\nHecho.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
