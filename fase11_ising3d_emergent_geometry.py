#!/usr/bin/env python3
# fase11_ising3d_emergent_geometry.py
#
# Wrapper para ejecutar Fase XI (motor de geometría emergente V2.1)
# sobre el dataset de contorno real/sintético de Ising 3D.
#
# IDEA:
#   En lugar de usar los sandbox estándar de Fase XI, apuntamos
#   directamente al directorio de datos de Fase XII:
#
#     fase12_data_boundary/
#       ├─ manifest.json          (compatible con 02_emergent_geometry_engine.py)
#       └─ ising_3d.h5            (boundary + metadatos de Ising 3D)
#
#   y escribimos las geometrías emergentes en:
#
#     runs/fase12_ising_real/fase11_geometry_from_real/
#       ├─ geometry_emergent/
#       │    └─ ising_3d_emergent.h5   (z, A(z), f(z), R(z), familia, etc.)
#       └─ emergent_geometry_summary.json
#
# USO TÍPICO:
#   python fase11_ising3d_emergent_geometry.py \
#       --data-dir fase12_data_boundary \
#       --output-dir runs/fase12_ising_real/fase11_geometry_from_real \
#       --n-epochs 2000 \
#       --device cpu
#
# NOTA IMPORTANTE:
#   Este script NO toca el código de 02_emergent_geometry_engine.py.
#   Simplemente construye una llamada robusta vía subprocess con los
#   argumentos adecuados, para que XI trate a Ising igual que a los
#   sandbox (mismo contrato de datos).

import argparse
import subprocess
import sys
from pathlib import Path


def run_emergent_geometry(
    data_dir: Path,
    output_dir: Path,
    n_epochs: int = 2000,
    device: str = "cpu",
    hidden_dim: int = 256,
    n_layers: int = 4,
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True,
    mode: str = "inference",
    checkpoint: Path | None = None,
) -> int:
    """
    Lanza 02_emergent_geometry_engine.py con una configuración razonable
    para el caso Ising 3D.

    Devuelve el código de retorno del proceso hijo.
    """

    script_path = Path(__file__).with_name("02_emergent_geometry_engine.py")
    if not script_path.exists():
        raise FileNotFoundError(
            f"No se encontró 02_emergent_geometry_engine.py junto a este script: {script_path}"
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de datos de entrada: {data_dir}"
        )

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No se encontró manifest.json en {data_dir}. "
            f"Asegúrate de haber ejecutado antes el adaptador real "
            f"(fase12_real_data_adapters.py o equivalente)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--hidden-dim",
        str(hidden_dim),
        "--n-layers",
        str(n_layers),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--mode",
        mode,
    ]

    if mode == "train":
        cmd.extend(["--n-epochs", str(n_epochs)])

    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])

    if verbose:
        cmd.append("--verbose")

    print("=" * 70)
    print("FASE XI (wrapper Ising 3D) — Geometría emergente desde datos reales")
    print("=" * 70)
    print(f"  Script 02   : {script_path}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  n_epochs    : {n_epochs}")
    print(f"  device      : {device}")
    print(f"  hidden_dim  : {hidden_dim}")
    print(f"  n_layers    : {n_layers}")
    print(f"  batch_size  : {batch_size}")
    print(f"  seed        : {seed}")
    print("=" * 70)
    print("Comando lanzado:\n  ", " ".join(cmd))
    print("=" * 70)

    result = subprocess.run(cmd)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper Fase XI para geometría emergente de Ising 3D (real/sintético)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="fase12_data_boundary",
        help="Directorio con manifest.json y ising_3d.h5 generados por el adaptador real",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/fase12_ising_real/fase11_geometry_from_real",
        help="Directorio de salida para la geometría emergente de Ising",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=2000,
        help="Número de épocas de entrenamiento (se pasa a 02_emergent_geometry_engine.py)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo para PyTorch: 'cpu' o 'cuda'",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Dimensión oculta de la red (se pasa a 02)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Número de capas residuales (se pasa a 02)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch (se pasa a 02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (se pasa a 02)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Si se pasa, desactiva el flag --verbose del motor XI",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    verbose = not args.no_verbose

    checkpoint = Path("runs/fase11_sandbox/emergent_geometry_model.pt")  # ajusta ruta

    retcode = run_emergent_geometry(
        data_dir=data_dir,
        output_dir=output_dir,
        n_epochs=args.n_epochs,  # solo se usará si mode="train"
        device=args.device,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=verbose,
        mode="inference",
        checkpoint=checkpoint,
    )

    if retcode != 0:
        print(f"\n[ERROR] 02_emergent_geometry_engine.py terminó con código {retcode}")
        sys.exit(retcode)

    print("\n[OK] Geometría emergente de Ising 3D generada correctamente.")
    print(f"     Revisa: {output_dir}/geometry_emergent/ y emergent_geometry_summary.json")


if __name__ == "__main__":
    main()
