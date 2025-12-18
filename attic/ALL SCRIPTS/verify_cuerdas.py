#!/usr/bin/env python3
"""
verify_cuerdas.py — Verifica la integridad del proyecto CUERDAS

Comprueba:
1. Estructura de carpetas
2. Scripts requeridos
3. Configuración válida
4. Dependencias

USO:
    python verify_cuerdas.py [--fix]
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Estructura requerida
REQUIRED_DIRS = [
    "runs/fase11",
    "runs/fase12", 
    "runs/fase13",
    "data_real/bootstrap",
    "data_real/lattice",
    "data_real/condensed_matter",
    "data_real/cosmology",
    "data_processed",
    "scripts",
    "config"
]

# Scripts canónicos requeridos
REQUIRED_SCRIPTS = [
    "scripts/run_cuerdas.py",
    "scripts/00_generate_fase_11_v3.py",
    "scripts/01_emergent_geometry_v2.py",
    "scripts/02_discover_einstein_v2.py",
    "scripts/03_holographic_dictionary_v3.py",
    "scripts/04_contracts_fase_11_v2.py",
    "scripts/fase12_real_data_adapters.py",
    "scripts/fase12_prediction_engine.py",
    "scripts/fase12c_emergent_dictionary_real.py",
    "scripts/fase13_theory_explorer.py",
    "scripts/make_fase11_for_fase12c_v3.py",
    "config/cuerdas_config.py",
    "config/__init__.py"
]

# READMEs requeridos
REQUIRED_DOCS = [
    "README.md",
    "runs/fase11/README.md",
    "runs/fase12/README.md",
    "runs/fase13/README.md"
]

# Dependencias Python
REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "h5py",
    "torch",
    "pandas",
    "sklearn"
]

OPTIONAL_PACKAGES = [
    "pysr",
    "matplotlib",
    "seaborn"
]


def check_structure(base_dir: Path, fix: bool = False) -> List[str]:
    """Verifica estructura de carpetas."""
    errors = []
    
    for dir_path in REQUIRED_DIRS:
        full_path = base_dir / dir_path
        if not full_path.exists():
            if fix:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Creado: {dir_path}")
            else:
                errors.append(f"Falta directorio: {dir_path}")
    
    return errors


def check_scripts(base_dir: Path) -> List[str]:
    """Verifica scripts requeridos."""
    errors = []
    
    for script in REQUIRED_SCRIPTS:
        if not (base_dir / script).exists():
            errors.append(f"Falta script: {script}")
    
    return errors


def check_docs(base_dir: Path) -> List[str]:
    """Verifica documentación."""
    errors = []
    
    for doc in REQUIRED_DOCS:
        if not (base_dir / doc).exists():
            errors.append(f"Falta documentación: {doc}")
    
    return errors


def check_dependencies() -> Tuple[List[str], List[str]]:
    """Verifica dependencias Python."""
    missing_required = []
    missing_optional = []
    
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(pkg)
    
    for pkg in OPTIONAL_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing_optional.append(pkg)
    
    return missing_required, missing_optional


def check_config(base_dir: Path) -> List[str]:
    """Verifica que la configuración sea importable."""
    errors = []
    
    config_path = base_dir / "config"
    if config_path.exists():
        sys.path.insert(0, str(base_dir))
        try:
            from config import CUERDAS_VERSION, FAMILIES, THRESHOLDS
            print(f"  ✓ Config OK (v{CUERDAS_VERSION})")
            print(f"    Familias: {list(FAMILIES.keys())}")
        except ImportError as e:
            errors.append(f"Error importando config: {e}")
        finally:
            sys.path.pop(0)
    else:
        errors.append("No existe directorio config/")
    
    return errors


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verifica proyecto CUERDAS")
    parser.add_argument("--fix", action="store_true", help="Crear directorios faltantes")
    parser.add_argument("--base-dir", type=str, default=".", help="Directorio base")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    print("="*60)
    print("CUERDAS — Verificación de Proyecto")
    print("="*60)
    print(f"Base: {base_dir.absolute()}")
    print()
    
    all_errors = []
    
    # 1. Estructura
    print("1. Verificando estructura de carpetas...")
    errors = check_structure(base_dir, fix=args.fix)
    all_errors.extend(errors)
    if not errors:
        print("  ✓ Estructura OK")
    else:
        for e in errors:
            print(f"  ✗ {e}")
    
    # 2. Scripts
    print("\n2. Verificando scripts...")
    errors = check_scripts(base_dir)
    all_errors.extend(errors)
    if not errors:
        print("  ✓ Scripts OK")
    else:
        for e in errors:
            print(f"  ✗ {e}")
    
    # 3. Documentación
    print("\n3. Verificando documentación...")
    errors = check_docs(base_dir)
    all_errors.extend(errors)
    if not errors:
        print("  ✓ Documentación OK")
    else:
        for e in errors:
            print(f"  ✗ {e}")
    
    # 4. Dependencias
    print("\n4. Verificando dependencias...")
    missing_req, missing_opt = check_dependencies()
    if missing_req:
        for pkg in missing_req:
            all_errors.append(f"Falta dependencia requerida: {pkg}")
            print(f"  ✗ Falta (requerido): {pkg}")
    else:
        print("  ✓ Dependencias requeridas OK")
    
    if missing_opt:
        for pkg in missing_opt:
            print(f"  ⚠ Falta (opcional): {pkg}")
    
    # 5. Configuración
    print("\n5. Verificando configuración...")
    errors = check_config(base_dir)
    all_errors.extend(errors)
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
    
    # Resumen
    print("\n" + "="*60)
    if all_errors:
        print(f"✗ VERIFICACIÓN FALLIDA ({len(all_errors)} errores)")
        print("\nPara crear directorios faltantes: python verify_cuerdas.py --fix")
        return 1
    else:
        print("✓ VERIFICACIÓN EXITOSA")
        print("\nEl proyecto está correctamente sincronizado.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
