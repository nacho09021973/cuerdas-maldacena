#!/usr/bin/env python3
"""
cuerdas_io.py — Protocolo IO Determinista para CUERDAS-Maldacena

Este módulo proporciona un protocolo IO unificado basado en:
- Un `run_manifest.json` por ejecución que captura todas las rutas
- Funciones de resolución que soportan tanto el nuevo layout como legacy
- Un único argumento `--run-dir` que reemplaza múltiples flags de ruta

VERSIÓN: 2.0
FECHA: 2025-12
CONTRATO: IO_CONTRACTS_V1.md + IO_LAYOUT_V2.md

USO:
    # En scripts:
    from cuerdas_io import RunContext
    
    ctx = RunContext.from_args(args)  # Lee manifest si existe
    predictions_dir = ctx.predictions_dir
    geometry_emergent_dir = ctx.geometry_emergent_dir
    
    # Al final del script 02:
    ctx.write_manifest()

LAYOUT NUEVO (v2):
    runs/<run_id>/
        run_manifest.json           # Manifest con todas las rutas
        geometry_emergent/          # Salida de 02: *.h5
        predictions/                # Salida de 02: *.npz
        bulk_equations/             # Salida de 03
        geometry_contracts/         # Salida de 04
        bulk_eigenmodes/            # Salida de 06
        emergent_dictionary/        # Salida de 07
        holographic_dictionary/     # Salida de 08
        ...
        
COMPATIBILIDAD LEGACY:
    - Si no existe `run_manifest.json`, las funciones `resolve_*` aplican fallbacks
    - Los flags antiguos (--geometry-dir, --data-dir, etc.) siguen funcionando
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ============================================================
# CONSTANTES
# ============================================================

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_VERSION = "2.0"

# Subdirectorios estándar
SUBDIR_GEOMETRY_EMERGENT = "geometry_emergent"
SUBDIR_PREDICTIONS = "predictions"
SUBDIR_BULK_EQUATIONS = "bulk_equations"
SUBDIR_GEOMETRY_CONTRACTS = "geometry_contracts"
SUBDIR_BULK_EIGENMODES = "bulk_eigenmodes"
SUBDIR_EMERGENT_DICTIONARY = "emergent_dictionary"
SUBDIR_HOLOGRAPHIC_DICTIONARY = "holographic_dictionary"
SUBDIR_FASE12 = "fase12"

# Archivos de summary estándar
FILE_EMERGENT_SUMMARY = "emergent_geometry_summary.json"
FILE_EINSTEIN_SUMMARY = "einstein_discovery_summary.json"
FILE_CONTRACTS_SUMMARY = "geometry_contracts_summary.json"


# ============================================================
# FUNCIONES DE MANIFEST
# ============================================================

def write_run_manifest(
    run_dir: Union[str, Path],
    artifacts: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Escribe un run_manifest.json en el directorio de ejecución.
    
    Args:
        run_dir: Directorio raíz de la ejecución
        artifacts: Diccionario con rutas de artefactos (relativas a run_dir)
        metadata: Metadatos opcionales (timestamp, versión, etc.)
    
    Returns:
        Path al manifest escrito
    
    Ejemplo de artifacts:
        {
            "data_dir": "../sandbox_geometries",
            "checkpoint": "checkpoint.pt",
            "geometry_emergent_dir": "geometry_emergent",
            "predictions_dir": "predictions",
            "summary_json": "emergent_geometry_summary.json",
            "systems": [
                {
                    "name": "ads_d3_Tfinite_known_000",
                    "h5_output": "geometry_emergent/ads_d3_Tfinite_known_000_emergent.h5",
                    "npz_output": "predictions/ads_d3_Tfinite_known_000_geometry.npz"
                }
            ]
        }
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir.resolve()),
        "artifacts": artifacts,
        "metadata": metadata or {}
    }
    
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    
    return manifest_path


def load_run_manifest(run_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Carga el run_manifest.json de un directorio de ejecución.
    
    Args:
        run_dir: Directorio raíz de la ejecución
    
    Returns:
        Diccionario con el manifest, o None si no existe
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    
    if not manifest_path.exists():
        return None
    
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, IOError) as e:
        print(f"[WARN] Error leyendo manifest {manifest_path}: {e}")
        return None


def update_run_manifest(
    run_dir: Union[str, Path],
    updates: Dict[str, Any],
    section: str = "artifacts"
) -> Path:
    """
    Actualiza un manifest existente añadiendo nuevos artefactos.
    
    Args:
        run_dir: Directorio raíz de la ejecución
        updates: Diccionario con nuevas entradas
        section: Sección a actualizar ("artifacts" o "metadata")
    
    Returns:
        Path al manifest actualizado
    """
    run_dir = Path(run_dir)
    manifest = load_run_manifest(run_dir) or {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir.resolve()),
        "artifacts": {},
        "metadata": {}
    }
    
    manifest["updated_at"] = datetime.now().isoformat()
    
    if section not in manifest:
        manifest[section] = {}
    
    manifest[section].update(updates)
    
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    
    return manifest_path


# ============================================================
# FUNCIONES DE RESOLUCIÓN DE RUTAS
# ============================================================

def resolve_predictions_dir(
    run_dir: Optional[Union[str, Path]] = None,
    geometry_dir: Optional[Union[str, Path]] = None,
    manifest: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Resuelve el directorio de predictions (*.npz de geometría emergente).
    
    Prioridad:
        1. manifest["artifacts"]["predictions_dir"] (si existe manifest)
        2. run_dir / "predictions" (si existe)
        3. geometry_dir / "predictions" (fallback legacy)
        4. None
    
    Args:
        run_dir: Directorio raíz de ejecución (nuevo layout)
        geometry_dir: Directorio de geometría (legacy)
        manifest: Manifest ya cargado (opcional, evita re-lectura)
    
    Returns:
        Path al directorio de predictions, o None si no se puede resolver
    """
    # Cargar manifest si no se proporciona
    if manifest is None and run_dir is not None:
        manifest = load_run_manifest(run_dir)
    
    # Opción 1: desde manifest
    if manifest is not None:
        artifacts = manifest.get("artifacts", {})
        if "predictions_dir" in artifacts:
            base = Path(manifest.get("run_dir", run_dir or "."))
            return base / artifacts["predictions_dir"]
    
    # Opción 2: run_dir / predictions
    if run_dir is not None:
        candidate = Path(run_dir) / SUBDIR_PREDICTIONS
        if candidate.exists():
            return candidate
    
    # Opción 3: geometry_dir / predictions (legacy)
    if geometry_dir is not None:
        candidate = Path(geometry_dir) / SUBDIR_PREDICTIONS
        if candidate.exists():
            return candidate
    
    return None


def resolve_geometry_emergent_dir(
    run_dir: Optional[Union[str, Path]] = None,
    geometry_dir: Optional[Union[str, Path]] = None,
    manifest: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Resuelve el directorio de geometry_emergent (*.h5 de geometría emergente).
    
    Prioridad:
        1. manifest["artifacts"]["geometry_emergent_dir"]
        2. run_dir / "geometry_emergent"
        3. geometry_dir / "geometry_emergent" (legacy)
        4. geometry_dir (si contiene *.h5 directamente)
        5. None
    """
    if manifest is None and run_dir is not None:
        manifest = load_run_manifest(run_dir)
    
    # Opción 1: desde manifest
    if manifest is not None:
        artifacts = manifest.get("artifacts", {})
        if "geometry_emergent_dir" in artifacts:
            base = Path(manifest.get("run_dir", run_dir or "."))
            return base / artifacts["geometry_emergent_dir"]
    
    # Opción 2: run_dir / geometry_emergent
    if run_dir is not None:
        candidate = Path(run_dir) / SUBDIR_GEOMETRY_EMERGENT
        if candidate.exists():
            return candidate
    
    # Opción 3: geometry_dir / geometry_emergent
    if geometry_dir is not None:
        candidate = Path(geometry_dir) / SUBDIR_GEOMETRY_EMERGENT
        if candidate.exists():
            return candidate
        # Opción 4: geometry_dir contiene *.h5 directamente
        if list(Path(geometry_dir).glob("*.h5")):
            return Path(geometry_dir)
    
    return None


def resolve_bulk_equations_dir(
    run_dir: Optional[Union[str, Path]] = None,
    einstein_dir: Optional[Union[str, Path]] = None,
    manifest: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Resuelve el directorio de bulk_equations (salida de 03).
    
    Prioridad:
        1. manifest["artifacts"]["bulk_equations_dir"]
        2. run_dir / "bulk_equations"
        3. einstein_dir (legacy)
        4. None
    """
    if manifest is None and run_dir is not None:
        manifest = load_run_manifest(run_dir)
    
    # Opción 1: desde manifest
    if manifest is not None:
        artifacts = manifest.get("artifacts", {})
        if "bulk_equations_dir" in artifacts:
            base = Path(manifest.get("run_dir", run_dir or "."))
            return base / artifacts["bulk_equations_dir"]
    
    # Opción 2: run_dir / bulk_equations
    if run_dir is not None:
        candidate = Path(run_dir) / SUBDIR_BULK_EQUATIONS
        if candidate.exists():
            return candidate
    
    # Opción 3: einstein_dir (legacy)
    if einstein_dir is not None:
        return Path(einstein_dir)
    
    return None


def resolve_data_dir(
    run_dir: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    manifest: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Resuelve el directorio de datos de entrada (sandbox_geometries).
    
    Prioridad:
        1. manifest["artifacts"]["data_dir"]
        2. data_dir (argumento explícito)
        3. None
    """
    if manifest is None and run_dir is not None:
        manifest = load_run_manifest(run_dir)
    
    # Opción 1: desde manifest
    if manifest is not None:
        artifacts = manifest.get("artifacts", {})
        if "data_dir" in artifacts:
            base = Path(manifest.get("run_dir", run_dir or "."))
            data_path = base / artifacts["data_dir"]
            # Si es ruta relativa, resolverla desde run_dir
            if data_path.exists():
                return data_path.resolve()
    
    # Opción 2: data_dir explícito
    if data_dir is not None:
        return Path(data_dir).resolve()
    
    return None


def resolve_dictionary_file(
    run_dir: Optional[Union[str, Path]] = None,
    dictionary_file: Optional[Union[str, Path]] = None,
    manifest: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Resuelve el archivo de diccionario holográfico.
    
    Prioridad:
        1. manifest["artifacts"]["dictionary_file"]
        2. run_dir / "holographic_dictionary/holographic_dictionary_summary.json"
        3. dictionary_file (argumento explícito)
        4. None
    """
    if manifest is None and run_dir is not None:
        manifest = load_run_manifest(run_dir)
    
    # Opción 1: desde manifest
    if manifest is not None:
        artifacts = manifest.get("artifacts", {})
        if "dictionary_file" in artifacts:
            base = Path(manifest.get("run_dir", run_dir or "."))
            return base / artifacts["dictionary_file"]
    
    # Opción 2: run_dir / holographic_dictionary/...
    if run_dir is not None:
        candidate = Path(run_dir) / SUBDIR_HOLOGRAPHIC_DICTIONARY / "holographic_dictionary_summary.json"
        if candidate.exists():
            return candidate
    
    # Opción 3: dictionary_file explícito
    if dictionary_file is not None:
        return Path(dictionary_file)
    
    return None


# ============================================================
# CLASE RunContext
# ============================================================

class RunContext:
    """
    Contexto de ejecución que encapsula la resolución de rutas.
    
    Uso:
        ctx = RunContext(run_dir="runs/my_run")
        
        # O desde argumentos de argparse
        ctx = RunContext.from_args(args)
        
        # Acceso a rutas
        preds = ctx.predictions_dir
        geom = ctx.geometry_emergent_dir
        
        # Actualizar manifest al final
        ctx.add_artifact("bulk_equations_dir", "bulk_equations")
        ctx.write_manifest()
    """
    
    def __init__(
        self,
        run_dir: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        geometry_dir: Optional[Union[str, Path]] = None,
        einstein_dir: Optional[Union[str, Path]] = None,
        dictionary_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Inicializa el contexto de ejecución.
        
        Args:
            run_dir: Directorio raíz de ejecución (nuevo layout)
            data_dir: Directorio de datos de entrada (legacy)
            geometry_dir: Directorio de geometría emergente (legacy)
            einstein_dir: Directorio de ecuaciones bulk (legacy)
            dictionary_file: Archivo de diccionario (legacy)
            output_dir: Directorio de salida (puede ser igual a run_dir)
        """
        self._run_dir = Path(run_dir) if run_dir else None
        self._data_dir = Path(data_dir) if data_dir else None
        self._geometry_dir = Path(geometry_dir) if geometry_dir else None
        self._einstein_dir = Path(einstein_dir) if einstein_dir else None
        self._dictionary_file = Path(dictionary_file) if dictionary_file else None
        self._output_dir = Path(output_dir) if output_dir else self._run_dir
        
        # Cargar manifest si existe
        self._manifest = None
        if self._run_dir:
            self._manifest = load_run_manifest(self._run_dir)
        
        # Artefactos pendientes de escribir
        self._pending_artifacts: Dict[str, Any] = {}
        self._pending_metadata: Dict[str, Any] = {}
    
    @classmethod
    def from_args(cls, args) -> "RunContext":
        """
        Crea un RunContext desde argumentos de argparse.
        
        Soporta tanto el nuevo `--run-dir` como los flags legacy.
        """
        return cls(
            run_dir=getattr(args, "run_dir", None),
            data_dir=getattr(args, "data_dir", None),
            geometry_dir=getattr(args, "geometry_dir", None),
            einstein_dir=getattr(args, "einstein_dir", None),
            dictionary_file=getattr(args, "dictionary_file", None),
            output_dir=getattr(args, "output_dir", None)
        )
    
    @property
    def run_dir(self) -> Optional[Path]:
        """Directorio raíz de ejecución."""
        return self._run_dir
    
    @property
    def output_dir(self) -> Optional[Path]:
        """Directorio de salida (puede ser igual a run_dir)."""
        return self._output_dir or self._run_dir
    
    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        """Manifest cargado (si existe)."""
        return self._manifest
    
    @property
    def has_manifest(self) -> bool:
        """Indica si existe un manifest válido."""
        return self._manifest is not None
    
    @property
    def predictions_dir(self) -> Optional[Path]:
        """Resuelve el directorio de predictions."""
        return resolve_predictions_dir(
            run_dir=self._run_dir,
            geometry_dir=self._geometry_dir,
            manifest=self._manifest
        )
    
    @property
    def geometry_emergent_dir(self) -> Optional[Path]:
        """Resuelve el directorio de geometry_emergent."""
        return resolve_geometry_emergent_dir(
            run_dir=self._run_dir,
            geometry_dir=self._geometry_dir,
            manifest=self._manifest
        )
    
    @property
    def bulk_equations_dir(self) -> Optional[Path]:
        """Resuelve el directorio de bulk_equations."""
        return resolve_bulk_equations_dir(
            run_dir=self._run_dir,
            einstein_dir=self._einstein_dir,
            manifest=self._manifest
        )
    
    @property
    def data_dir(self) -> Optional[Path]:
        """Resuelve el directorio de datos de entrada."""
        return resolve_data_dir(
            run_dir=self._run_dir,
            data_dir=self._data_dir,
            manifest=self._manifest
        )
    
    @property
    def dictionary_file(self) -> Optional[Path]:
        """Resuelve el archivo de diccionario."""
        return resolve_dictionary_file(
            run_dir=self._run_dir,
            dictionary_file=self._dictionary_file,
            manifest=self._manifest
        )
    
    def add_artifact(self, key: str, value: Any) -> None:
        """
        Añade un artefacto al manifest pendiente.
        
        Args:
            key: Clave del artefacto (ej: "predictions_dir")
            value: Valor (ej: "predictions" o Path)
        """
        if isinstance(value, Path):
            value = str(value)
        self._pending_artifacts[key] = value
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Añade metadatos al manifest pendiente.
        """
        self._pending_metadata[key] = value
    
    def write_manifest(self) -> Optional[Path]:
        """
        Escribe el manifest con los artefactos acumulados.
        
        Returns:
            Path al manifest escrito, o None si no hay run_dir
        """
        if self._output_dir is None:
            return None
        
        # Combinar artefactos existentes con pendientes
        artifacts = {}
        if self._manifest:
            artifacts = self._manifest.get("artifacts", {}).copy()
        artifacts.update(self._pending_artifacts)
        
        # Combinar metadatos
        metadata = {}
        if self._manifest:
            metadata = self._manifest.get("metadata", {}).copy()
        metadata.update(self._pending_metadata)
        
        return write_run_manifest(self._output_dir, artifacts, metadata)
    
    def ensure_subdirs(self) -> None:
        """Crea los subdirectorios estándar si no existen."""
        if self._output_dir is None:
            return
        
        for subdir in [
            SUBDIR_GEOMETRY_EMERGENT,
            SUBDIR_PREDICTIONS,
            SUBDIR_BULK_EQUATIONS,
            SUBDIR_GEOMETRY_CONTRACTS,
            SUBDIR_BULK_EIGENMODES,
            SUBDIR_EMERGENT_DICTIONARY,
            SUBDIR_HOLOGRAPHIC_DICTIONARY
        ]:
            (self._output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def summary(self) -> Dict[str, Any]:
        """Retorna un resumen del contexto para debug."""
        return {
            "run_dir": str(self._run_dir) if self._run_dir else None,
            "output_dir": str(self._output_dir) if self._output_dir else None,
            "has_manifest": self.has_manifest,
            "resolved": {
                "data_dir": str(self.data_dir) if self.data_dir else None,
                "predictions_dir": str(self.predictions_dir) if self.predictions_dir else None,
                "geometry_emergent_dir": str(self.geometry_emergent_dir) if self.geometry_emergent_dir else None,
                "bulk_equations_dir": str(self.bulk_equations_dir) if self.bulk_equations_dir else None,
                "dictionary_file": str(self.dictionary_file) if self.dictionary_file else None,
            }
        }


# ============================================================
# UTILIDADES DE ARGPARSE
# ============================================================

def add_run_dir_argument(parser, required: bool = False) -> None:
    """
    Añade el argumento --run-dir a un parser de argparse.
    
    Args:
        parser: ArgumentParser
        required: Si es True, hace obligatorio el argumento
    """
    parser.add_argument(
        "--run-dir",
        type=str,
        required=required,
        default=None,
        help="Directorio raíz de ejecución con run_manifest.json. "
             "Si se proporciona, resuelve automáticamente las rutas de "
             "predictions, geometry_emergent, bulk_equations, etc. "
             "Compatible con flags legacy (--data-dir, --geometry-dir, etc.)"
    )


def validate_paths_or_run_dir(args, required_paths: List[str]) -> None:
    """
    Valida que existan las rutas necesarias via --run-dir o flags legacy.
    
    Args:
        args: Argumentos parseados
        required_paths: Lista de paths requeridos (ej: ["predictions_dir", "data_dir"])
    
    Raises:
        ValueError si faltan rutas requeridas
    """
    ctx = RunContext.from_args(args)
    
    missing = []
    for path_name in required_paths:
        resolved = getattr(ctx, path_name, None)
        if resolved is None or (isinstance(resolved, Path) and not resolved.exists()):
            missing.append(path_name)
    
    if missing:
        raise ValueError(
            f"Rutas no encontradas: {missing}. "
            f"Proporciona --run-dir con manifest válido o usa flags legacy."
        )


# ============================================================
# MAIN (para tests rápidos)
# ============================================================

if __name__ == "__main__":
    import sys
    
    print("=== cuerdas_io.py - Test rápido ===\n")
    
    # Test 1: Crear manifest
    test_dir = Path("/tmp/cuerdas_io_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        "data_dir": "../sandbox_geometries",
        "predictions_dir": "predictions",
        "geometry_emergent_dir": "geometry_emergent",
        "systems": [
            {"name": "test_system", "npz_output": "predictions/test_system_geometry.npz"}
        ]
    }
    
    manifest_path = write_run_manifest(test_dir, artifacts, {"script": "test"})
    print(f"✓ Manifest escrito: {manifest_path}")
    
    # Test 2: Cargar manifest
    loaded = load_run_manifest(test_dir)
    assert loaded is not None
    assert loaded["artifacts"]["predictions_dir"] == "predictions"
    print(f"✓ Manifest cargado correctamente")
    
    # Test 3: RunContext
    ctx = RunContext(run_dir=test_dir)
    assert ctx.has_manifest
    print(f"✓ RunContext creado con manifest")
    print(f"  Resumen: {json.dumps(ctx.summary(), indent=2)}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n✓ Test completado. Directorio temporal eliminado.")
