#!/usr/bin/env python3
"""
run_fase_11_v2.py â€” Ejecuta el pipeline completo de Fase XI v2 (HONESTO)

DIFERENCIAS vs v1:
    1. Train/test split honesto por category (known vs test/unknown)
    2. Physics losses AdS-especÃ­ficas solo para family=ads
    3. RelaciÃ³n mÂ²LÂ² = Î”(Î”-d) descubierta genuinamente, no construida
    4. Contratos separados: genÃ©ricos vs AdS-especÃ­ficos
    5. PySR limpio (sin parÃ¡metros obsoletos)

USO:
    python run_fase_11_v2.py --output-dir fase11_output_v2

    O con argumentos especÃ­ficos:
    python run_fase_11_v2.py --output-dir fase11_output_v2 --n-epochs 1000 --niterations 50
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


# ======================================================================
# UTILIDADES
# ======================================================================

def run_command(cmd, description: str) -> bool:
    """Ejecuta un comando de subprocess y muestra salida amigable."""
    print("\n" + "=" * 75)
    print(description)
    print("=" * 75)
    print(">> Comando:", " ".join(map(str, cmd)))
    print("-" * 75)
    
    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
        )
    except KeyboardInterrupt:
        print("\nâŒ Ejecutado interrumpido por el usuario.")
        return False
    
    if result.returncode != 0:
        print(f"\nâŒ El comando terminÃ³ con cÃ³digo {result.returncode}")
        return False
    
    print("\nâœ… Paso completado correctamente.")
    return True


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


# ======================================================================
# PIPELINE PRINCIPAL
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fase XI v2: Pipeline completo HONESTO"
    )
    parser.add_argument("--output-dir", type=str, default="fase11_output_v2",
                        help="Directorio base de salida")
    parser.add_argument("--n-epochs", type=int, default=3000,
                        help="Ã‰pocas de entrenamiento para geometrÃ­a")
    parser.add_argument("--niterations", type=int, default=100,
                        help="Iteraciones de PySR")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo (cpu/cuda)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Saltar generaciÃ³n de datos y usar datos existentes")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verboso para algunos pasos")
    
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Directorios
    data_dir = base_dir / "data"
    geometry_dir = base_dir / "geometry"
    einstein_dir = base_dir / "einstein"
    dictionary_dir = base_dir / "dictionary"
    contracts_file = base_dir / "contracts_v2.json"
    
    print("=" * 70)
    print("FASE XI v2 â€” PIPELINE COMPLETO HONESTO")
    print("=" * 70)
    print(f"\nOutput base:     {base_dir}")
    print(f"Data:            {data_dir}")
    print(f"Geometry:        {geometry_dir}")
    print(f"Einstein:        {einstein_dir}")
    print(f"Dictionary:      {dictionary_dir}")
    print(f"Contracts:       {contracts_file}")
    print("-" * 70)
    print(f"Semilla:         {args.seed}")
    print(f"Device:          {args.device}")
    print(f"N epochs geom.:  {args.n_epochs}")
    print(f"PySR iterations: {args.niterations}")
    print("=" * 70)
    
    # ==================================================================
    # PASO 0: GeneraciÃ³n de datos (usa generador v3)
    # ==================================================================
    
    if not args.skip_generate:
        # Usar el generador v3 (00_generate_fase_11_v3.py)
        # que ya separa correctamente boundary vs bulk_truth
        cmd = [
            sys.executable, "00_generate_fase_11_v3.py",
            "--output-dir", str(data_dir),
            "--seed", str(args.seed),
            "--n-operators", "3",
        ]
        
        if not run_command(cmd, "Paso 0: GeneraciÃ³n de datos CFT"):
            print("\nâš  Intentando con datos existentes...")
            if not data_dir.exists():
                print("âŒ No hay datos y generaciÃ³n fallÃ³")
                return 1
    else:
        print("\n>> Usando datos existentes (--skip-generate)")
        if not data_dir.exists():
            print("âŒ No existe el directorio de datos, no se puede continuar")
            return 1
    
    # Comprobamos que exista manifest.json
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"\nâŒ No existe manifest.json en {data_dir}")
        return 1
    
    manifest = load_json(manifest_path)
    print("\nðŸ“„ MANIFEST DE DATOS:")
    print(f"  GeometrÃ­as totales: {len(manifest.get('geometries', []))}")
    print(f"  Claves: {list(manifest.keys())}")
    
    # ==================================================================
    # PASO 1: Emergencia de geometrÃ­a (v2 - honesta)
    # ==================================================================
    
    geometry_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "01_emergent_geometry_v2.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(geometry_dir),
        "--n-epochs", str(args.n_epochs),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.verbose:
        cmd.append("--verbose")
    
    if not run_command(cmd, "Paso 1: Emergencia de geometrÃ­a (HONESTA)"):
        print("âŒ FallÃ³ emergencia de geometrÃ­a")
        return 1
    
    # ==================================================================
    # PASO 2: Descubrimiento de Einstein (v2 - sin asumir)
    # ==================================================================
    
    einstein_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "02_discover_einstein_v2.py",
        "--geometry-dir", str(geometry_dir),
        "--output-dir", str(einstein_dir),
        "--niterations", str(args.niterations),
        "--seed", str(args.seed),
    ]
    
    if not run_command(cmd, "Paso 2: Descubrimiento de Einstein (SIN ASUMIR)"):
        print("âŒ FallÃ³ descubrimiento de Einstein")
        return 1
    
    # ==================================================================
    # PASO 3: Diccionario hologrÃ¡fico (v3 - genuino)
    # ==================================================================
    
    dictionary_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "03_holographic_dictionary_v3.py",
        "--data-dir", str(data_dir),
        "--geometry-dir", str(geometry_dir),
        "--output-dir", str(dictionary_dir),
        "--seed", str(args.seed),
    ]
    
    if not run_command(cmd, "Paso 3: Diccionario hologrÃ¡fico (genuino)"):
        print("âŒ FallÃ³ construcciÃ³n del diccionario hologrÃ¡fico")
        return 1
    
    # ==================================================================
    # PASO 4: Contratos de Fase XI v2
    # ==================================================================
    
    dictionary_summary = dictionary_dir / "holographic_dictionary_v3_summary.json"
    
    cmd = [
        sys.executable, "04_contracts_fase_11_v2.py",
        "--data-dir", str(data_dir),
        "--geometry-dir", str(geometry_dir),
        "--einstein-dir", str(einstein_dir),
        "--dictionary-file", str(dictionary_summary),
        "--output-file", str(contracts_file),
    ]
    
    if not run_command(cmd, "Paso 4: Contratos de Fase XI v2"):
        print("âŒ FallÃ³ validaciÃ³n de contratos")
        return 1
    
    # ==================================================================
    # RESUMEN FINAL
    # ==================================================================
    
    if not contracts_file.exists():
        print("\nâŒ No se encontrÃ³ el fichero de contratos final")
        return 1
    
    contracts = load_json(contracts_file)
    
    print("\n" + "=" * 70)
    print("RESUMEN FASE XI v2")
    print("=" * 70)
    
    phase_passed = contracts.get("phase_passed", False)
    print(f"\n  Â¿Fase XI v2 pasada?: {'âœ… SÃ' if phase_passed else 'âŒ NO'}")
    
    if phase_passed:
        print("\n  Detalles de contratos clave:")
        for key, val in contracts.items():
            if key == "phase_passed":
                continue
            print(f"    - {key}: {val}")
        print("\n" + "=" * 70)
        print("âœ“ FASE XI v2 COMPLETADA CON Ã‰XITO")
        print("    âœ“ GeometrÃ­a emergente entrenada")
        print("    âœ“ Ecuaciones tipo Einstein descubiertas sin asumir forma")
        print("    âœ“ Diccionario hologrÃ¡fico obtenido de manera genuina")
        print("    âœ“ Contratos genÃ©ricos vs AdS-especÃ­ficos separados")
    else:
        print("\n" + "=" * 70)
        print("âš  FASE XI v2 REQUIERE REFINAMIENTO")
        print("=" * 70)
    
    print(f"\n  Resultados completos: {base_dir}")
    print("=" * 70)
    
    return 0 if contracts['phase_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
