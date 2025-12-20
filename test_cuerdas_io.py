#!/usr/bin/env python3
"""
test_cuerdas_io.py — Tests para el módulo cuerdas_io

Ejecutar:
    python test_cuerdas_io.py

Este test verifica:
1. Escritura y lectura de manifest
2. Resolución de rutas con manifest
3. Resolución de rutas con fallback legacy
4. Clase RunContext
5. Compatibilidad con argparse
"""

import json
import shutil
import sys
import tempfile
import pytest
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

# ... tu clase Result ...

# evita que pytest intente coleccionarla como clase de tests

@pytest.fixture
def results() -> "Result":
    return Result()

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from cuerdas_io import (
    MANIFEST_FILENAME,
    SUBDIR_PREDICTIONS,
    SUBDIR_GEOMETRY_EMERGENT,
    SUBDIR_BULK_EQUATIONS,
    write_run_manifest,
    load_run_manifest,
    update_run_manifest,
    resolve_predictions_dir,
    resolve_geometry_emergent_dir,
    resolve_bulk_equations_dir,
    resolve_data_dir,
    RunContext,
    add_run_dir_argument,
)


class Result:
    """Acumula resultados de tests."""
    
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
    
    def ok(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")
    
    def fail(self, name: str, reason: str):
        self.failed.append((name, reason))
        print(f"  ✗ {name}: {reason}")
    
    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"RESULTADO: {len(self.passed)}/{total} tests pasados")
        if self.failed:
            print("Tests fallidos:")
            for name, reason in self.failed:
                print(f"  - {name}: {reason}")
        print(f"{'='*60}")
        return len(self.failed) == 0


def test_manifest_write_read(results: Result, tmp_dir: Path):
    """Test: escribir y leer manifest."""
    name = "manifest_write_read"
    
    try:
        artifacts = {
            "data_dir": "../sandbox",
            "predictions_dir": "predictions",
            "geometry_emergent_dir": "geometry_emergent",
        }
        
        manifest_path = write_run_manifest(tmp_dir, artifacts, {"test": True})
        
        assert manifest_path.exists(), "Manifest no fue creado"
        
        loaded = load_run_manifest(tmp_dir)
        assert loaded is not None, "Manifest no se pudo cargar"
        assert loaded["artifacts"]["predictions_dir"] == "predictions"
        assert loaded["metadata"]["test"] is True
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_manifest_update(results: Result, tmp_dir: Path):
    """Test: actualizar manifest existente."""
    name = "manifest_update"
    
    try:
        # Crear manifest inicial
        write_run_manifest(tmp_dir, {"initial": "value"})
        
        # Actualizar
        update_run_manifest(tmp_dir, {"added": "new_value"})
        
        loaded = load_run_manifest(tmp_dir)
        assert loaded["artifacts"]["initial"] == "value"
        assert loaded["artifacts"]["added"] == "new_value"
        assert "updated_at" in loaded
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_resolve_predictions_from_manifest(results: Result, tmp_dir: Path):
    """Test: resolver predictions desde manifest."""
    name = "resolve_predictions_manifest"
    
    try:
        # Crear estructura
        preds_dir = tmp_dir / SUBDIR_PREDICTIONS
        preds_dir.mkdir()
        
        write_run_manifest(tmp_dir, {"predictions_dir": SUBDIR_PREDICTIONS})
        
        resolved = resolve_predictions_dir(run_dir=tmp_dir)
        assert resolved is not None
        assert resolved == preds_dir
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_resolve_predictions_legacy(results: Result, tmp_dir: Path):
    """Test: resolver predictions sin manifest (fallback legacy)."""
    name = "resolve_predictions_legacy"
    
    try:
        # Crear estructura legacy
        legacy_geom = tmp_dir / "legacy_geometry"
        legacy_preds = legacy_geom / "predictions"
        legacy_preds.mkdir(parents=True)
        
        resolved = resolve_predictions_dir(geometry_dir=legacy_geom)
        assert resolved is not None
        assert resolved == legacy_preds
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_resolve_geometry_emergent(results: Result, tmp_dir: Path):
    """Test: resolver geometry_emergent."""
    name = "resolve_geometry_emergent"
    
    try:
        geom_dir = tmp_dir / SUBDIR_GEOMETRY_EMERGENT
        geom_dir.mkdir()
        
        write_run_manifest(tmp_dir, {"geometry_emergent_dir": SUBDIR_GEOMETRY_EMERGENT})
        
        resolved = resolve_geometry_emergent_dir(run_dir=tmp_dir)
        assert resolved is not None
        assert resolved == geom_dir
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_resolve_bulk_equations(results: Result, tmp_dir: Path):
    """Test: resolver bulk_equations."""
    name = "resolve_bulk_equations"
    
    try:
        eq_dir = tmp_dir / SUBDIR_BULK_EQUATIONS
        eq_dir.mkdir()
        
        write_run_manifest(tmp_dir, {"bulk_equations_dir": SUBDIR_BULK_EQUATIONS})
        
        resolved = resolve_bulk_equations_dir(run_dir=tmp_dir)
        assert resolved is not None
        assert resolved == eq_dir
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_run_context_with_manifest(results: Result, tmp_dir: Path):
    """Test: RunContext con manifest."""
    name = "run_context_manifest"
    
    try:
        # Crear estructura completa
        (tmp_dir / SUBDIR_PREDICTIONS).mkdir()
        (tmp_dir / SUBDIR_GEOMETRY_EMERGENT).mkdir()
        
        write_run_manifest(tmp_dir, {
            "predictions_dir": SUBDIR_PREDICTIONS,
            "geometry_emergent_dir": SUBDIR_GEOMETRY_EMERGENT,
        })
        
        ctx = RunContext(run_dir=tmp_dir)
        
        assert ctx.has_manifest
        assert ctx.predictions_dir is not None
        assert ctx.geometry_emergent_dir is not None
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_run_context_legacy(results: Result, tmp_dir: Path):
    """Test: RunContext sin manifest (modo legacy)."""
    name = "run_context_legacy"
    
    try:
        # Crear estructura legacy
        legacy_dir = tmp_dir / "legacy"
        (legacy_dir / "predictions").mkdir(parents=True)
        (legacy_dir / "geometry_emergent").mkdir(parents=True)
        
        ctx = RunContext(geometry_dir=legacy_dir)
        
        assert not ctx.has_manifest
        assert ctx.predictions_dir is not None
        assert ctx.geometry_emergent_dir is not None
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_run_context_from_args(results: Result, tmp_dir: Path):
    """Test: RunContext.from_args()."""
    name = "run_context_from_args"
    
    try:
        # Simular args de argparse
        args = Namespace(
            run_dir=str(tmp_dir),
            data_dir=None,
            geometry_dir=None,
            einstein_dir=None,
            dictionary_file=None,
            output_dir=str(tmp_dir),
        )
        
        (tmp_dir / SUBDIR_PREDICTIONS).mkdir(exist_ok=True)
        write_run_manifest(tmp_dir, {"predictions_dir": SUBDIR_PREDICTIONS})
        
        ctx = RunContext.from_args(args)
        
        assert ctx.run_dir == tmp_dir
        assert ctx.predictions_dir is not None
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_run_context_add_artifact(results: Result, tmp_dir: Path):
    """Test: añadir artefactos y escribir manifest."""
    name = "run_context_add_artifact"
    
    try:
        ctx = RunContext(run_dir=tmp_dir, output_dir=tmp_dir)
        
        ctx.add_artifact("predictions_dir", "predictions")
        ctx.add_artifact("custom_key", "custom_value")
        ctx.add_metadata("script", "test")
        
        manifest_path = ctx.write_manifest()
        assert manifest_path is not None
        assert manifest_path.exists()
        
        loaded = load_run_manifest(tmp_dir)
        assert loaded["artifacts"]["predictions_dir"] == "predictions"
        assert loaded["artifacts"]["custom_key"] == "custom_value"
        assert loaded["metadata"]["script"] == "test"
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_run_context_ensure_subdirs(results: Result, tmp_dir: Path):
    """Test: crear subdirectorios estándar."""
    name = "run_context_ensure_subdirs"
    
    try:
        ctx = RunContext(output_dir=tmp_dir)
        ctx.ensure_subdirs()
        
        assert (tmp_dir / SUBDIR_PREDICTIONS).exists()
        assert (tmp_dir / SUBDIR_GEOMETRY_EMERGENT).exists()
        assert (tmp_dir / SUBDIR_BULK_EQUATIONS).exists()
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def test_priority_manifest_over_legacy(results: Result, tmp_dir: Path):
    """Test: manifest tiene prioridad sobre rutas legacy."""
    name = "priority_manifest_over_legacy"
    
    try:
        # Crear dos directorios de predictions
        manifest_preds = tmp_dir / "manifest_predictions"
        legacy_preds = tmp_dir / "legacy" / "predictions"
        manifest_preds.mkdir()
        legacy_preds.mkdir(parents=True)
        
        # El manifest apunta a manifest_predictions
        write_run_manifest(tmp_dir, {"predictions_dir": "manifest_predictions"})
        
        # Resolver con ambos: manifest debe ganar
        resolved = resolve_predictions_dir(
            run_dir=tmp_dir,
            geometry_dir=tmp_dir / "legacy"
        )
        
        assert resolved == manifest_preds, f"Expected {manifest_preds}, got {resolved}"
        
        results.ok(name)
    except Exception as e:
        results.fail(name, str(e))


def main():
    print("=" * 60)
    print("TEST: cuerdas_io.py")
    print("=" * 60 + "\n")
    
    results = Result()
    
    # Crear directorio temporal para cada test
    tests = [
        test_manifest_write_read,
        test_manifest_update,
        test_resolve_predictions_from_manifest,
        test_resolve_predictions_legacy,
        test_resolve_geometry_emergent,
        test_resolve_bulk_equations,
        test_run_context_with_manifest,
        test_run_context_legacy,
        test_run_context_from_args,
        test_run_context_add_artifact,
        test_run_context_ensure_subdirs,
        test_priority_manifest_over_legacy,
    ]
    
    for test_fn in tests:
        tmp_dir = Path(tempfile.mkdtemp(prefix="cuerdas_io_test_"))
        try:
            test_fn(results, tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
