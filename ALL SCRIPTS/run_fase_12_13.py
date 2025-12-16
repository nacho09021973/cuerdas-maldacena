#!/usr/bin/env python3
"""
run_fase_12_13.py â€” Runner unificado para Fases XII y XIII

FASE XII: CUERDAS en el mundo real
    - Adapta datos fÃ­sicos reales al formato del motor XI
    - Valida coherencia con fÃ­sica conocida
    - Genera predicciones verificables

FASE XIII: Explorador universal de teorÃ­as
    - Escanea el espacio de teorÃ­as
    - Identifica outliers y regiones interesantes
    - Construye atlas de teorÃ­as

USO:
    # Fase XII completa
    python run_fase_12_13.py --phase 12 --output-dir cuerdas_real_world
    
    # Fase XIII completa
    python run_fase_12_13.py --phase 13 --output-dir cuerdas_explorer --n-theories 200
    
    # Ambas fases
    python run_fase_12_13.py --phase both --output-dir cuerdas_complete
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json


def run_command(cmd: list, description: str) -> bool:
    """Ejecuta un comando y reporta resultado."""
    print(f"\n{'='*70}")
    print(f">> {description}")
    print(f"{'='*70}")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\nâœ— ERROR en: {description}")
        return False
    
    print(f"\nâœ“ Completado: {description}")
    return True


def run_fase12(output_dir: Path, fase11_dir: Path) -> bool:
    """Ejecuta Fase XII completa."""
    
    print("\n" + "=" * 70)
    print("FASE XII â€” CUERDAS EN EL MUNDO REAL")
    print("=" * 70)
    
    data_dir = output_dir / "real_data"
    predictions_dir = output_dir / "predictions"
    
    # Paso 1: Generar datos sintÃ©ticos realistas
    cmd = [
        sys.executable, "fase12_real_data_adapters.py",
        "--mode", "generate-synthetic",
        "--output-dir", str(data_dir)
    ]
    
    if not run_command(cmd, "Fase XII.1: Generar datos sintÃ©ticos realistas"):
        return False
    
    # Paso 2: Procesar cada tipo de datos
    adapters = [
        ("bootstrap", "ising3d_bootstrap.json"),
        ("bootstrap", "o4_bootstrap.json"),
        ("lattice", "lattice_qcd.h5"),
        ("condensed", "strange_metal.json"),
        ("cosmology", "cmb_planck.json")
    ]
    
    for adapter, filename in adapters:
        source = data_dir / filename
        if not source.exists():
            print(f"   âš  No existe {source}, saltando...")
            continue
        
        cmd = [
            sys.executable, "fase12_real_data_adapters.py",
            "--mode", "process",
            "--adapter", adapter,
            "--source", str(source),
            "--output-dir", str(data_dir)
        ]
        
        if not run_command(cmd, f"Fase XII.2: Procesar {filename} con {adapter}"):
            print(f"   âš  FallÃ³ procesamiento de {filename}")
            continue
    
    # Paso 3: Motor de predicciones
    # Nota: requiere outputs de Fase XI
    if fase11_dir.exists():
        cmd = [
            sys.executable, "fase12_prediction_engine.py",
            "--data-dir", str(data_dir),
            "--fase11-dir", str(fase11_dir),
            "--output-dir", str(predictions_dir)
        ]
        
        if not run_command(cmd, "Fase XII.3: Motor de predicciones"):
            print("   âš  Motor de predicciones requiere outputs de Fase XI")
    else:
        print(f"\n   âš  No existe {fase11_dir}")
        print("     Para predicciones completas, primero ejecutar Fase XI")
        print("     Los datos adaptados estÃ¡n listos en:", data_dir)
    
    print("\n" + "=" * 70)
    print("FASE XII â€” RESUMEN")
    print("=" * 70)
    
    # Mostrar archivos generados
    manifest_path = data_dir / "manifest_fase12.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"\n  Sistemas procesados: {len(manifest.get('processed', []))}")
        for item in manifest.get("processed", []):
            print(f"    - {item['name']} ({item['source']})")
    
    return True


def run_fase13(output_dir: Path, n_theories: int, seed: int) -> bool:
    """Ejecuta Fase XIII completa."""
    
    print("\n" + "=" * 70)
    print("FASE XIII â€” EXPLORADOR UNIVERSAL DE TEORÃAS")
    print("=" * 70)
    
    # Paso 1: Generar teorÃ­as
    cmd = [
        sys.executable, "fase13_theory_explorer.py",
        "--mode", "generate",
        "--n-theories", str(n_theories),
        "--output-dir", str(output_dir),
        "--seed", str(seed)
    ]
    
    if not run_command(cmd, "Fase XIII.1: Generar teorÃ­as sintÃ©ticas"):
        return False
    
    # Paso 2: Explorar (ejecutar pipeline)
    cmd = [
        sys.executable, "fase13_theory_explorer.py",
        "--mode", "explore",
        "--output-dir", str(output_dir),
        "--seed", str(seed)
    ]
    
    if not run_command(cmd, "Fase XIII.2: Explorar espacio de teorÃ­as"):
        return False
    
    # Paso 3: Analizar
    cmd = [
        sys.executable, "fase13_theory_explorer.py",
        "--mode", "analyze",
        "--output-dir", str(output_dir)
    ]
    
    if not run_command(cmd, "Fase XIII.3: Analizar atlas de teorÃ­as"):
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Runner unificado para Fases XII y XIII"
    )
    parser.add_argument("--phase", type=str, required=True,
                        choices=["12", "13", "both"],
                        help="Fase a ejecutar")
    parser.add_argument("--output-dir", type=str, default="cuerdas_output",
                        help="Directorio base de salida")
    parser.add_argument("--fase11-dir", type=str, default="fase11_output_v2",
                        help="Directorio con outputs de Fase XI")
    parser.add_argument("--n-theories", type=int, default=100,
                        help="NÃºmero de teorÃ­as para Fase XIII")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fase11_dir = Path(args.fase11_dir)
    
    print("=" * 70)
    print("CUERDAS â€” FASES XII/XIII")
    print("=" * 70)
    print(f"\n  Fase:       {args.phase}")
    print(f"  Output:     {output_dir}")
    print(f"  Fase XI:    {fase11_dir}")
    print(f"  Seed:       {args.seed}")
    
    success = True
    
    if args.phase in ["12", "both"]:
        fase12_dir = output_dir / "fase12"
        success = success and run_fase12(fase12_dir, fase11_dir)
    
    if args.phase in ["13", "both"]:
        fase13_dir = output_dir / "fase13"
        success = success and run_fase13(fase13_dir, args.n_theories, args.seed)
    
    print("\n" + "=" * 70)
    if success:
        print("âœ“ EJECUCIÃ“N COMPLETADA")
    else:
        print("âš  EJECUCIÃ“N CON WARNINGS")
    print("=" * 70)
    print(f"\n  Resultados en: {output_dir}")
    
    if args.phase in ["12", "both"]:
        print(f"\n  FASE XII:")
        print(f"    - Datos adaptados: {output_dir}/fase12/real_data/")
        print(f"    - Predicciones:    {output_dir}/fase12/predictions/")
    
    if args.phase in ["13", "both"]:
        print(f"\n  FASE XIII:")
        print(f"    - Atlas:      {output_dir}/fase13/theory_atlas.json")
        print(f"    - AnÃ¡lisis:   {output_dir}/fase13/fase13_analysis.json")
    
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
