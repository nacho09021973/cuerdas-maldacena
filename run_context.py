#!/usr/bin/env python3
"""
run_context.py - Sistema Determinístico de Rutas para CUERDAS-MALDACENA V3

Este módulo elimina el caos de rutas implementando un único patrón:
    runs/<experiment>/<stage_id>_<stage_slug>/

Uso típico:
    ctx = RunContext.from_args(args)  # args.experiment requerido
    output_dir = ctx.stage_dir()       # automático basado en script
    ctx.register_outputs({"predictions": "*.h5"})
    ctx.save_manifest()

Compatibilidad:
    - Si se pasa --experiment, usa el nuevo layout
    - Si solo se pasan flags legacy, funciona pero emite WARNING
    - run_manifest.json es la fuente de verdad

Autor: Refactor para CUERDAS-MALDACENA
Versión: 3.0 (experiment-centric)
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

__version__ = "3.0.0"
__all__ = [
    "RunContext",
    "get_stage_info",
    "resolve_experiment_dir",
    "STAGE_REGISTRY",
    "ALIAS_MAPPING",
    "add_experiment_args",
    "validate_experiment_args",
]

logger = logging.getLogger(__name__)

# =============================================================================
# REGISTRO DE STAGES (fuente de verdad para mappings)
# =============================================================================

STAGE_REGISTRY = {
    "00": {
        "slug": "validate_io_contracts",
        "outputs": ["validation_report.json"],
        "aliases": [],
    },
    "01": {
        "slug": "generate_sandbox_geometries",
        "outputs": ["boundary/*.h5", "bulk_truth/*.h5", "manifest.json"],
        "aliases": ["sandbox_data"],
    },
    "02": {
        "slug": "emergent_geometry_engine",
        "outputs": [
            "predictions/*.h5",
            "geometry_emergent/*.h5",
            "emergent_geometry_summary.json",
            "model_checkpoint.pt"
        ],
        "aliases": ["predictions", "geometry_emergent"],
    },
    "03": {
        "slug": "discover_bulk_equations",
        "outputs": ["equations_pareto.json", "einstein_discovery_summary.json"],
        "aliases": ["bulk_equations"],
    },
    "04": {
        "slug": "geometry_physics_contracts",
        "outputs": ["geometry_contracts_summary.json"],
        "aliases": ["geometry_contracts"],
    },
    "04b": {
        "slug": "negative_control_contracts",
        "outputs": ["negative_control_summary.json"],
        "aliases": ["negative_controls"],
    },
    "05": {
        "slug": "analyze_bulk_equations",
        "outputs": ["bulk_equations_report.json", "bulk_equations_analysis.txt"],
        "aliases": ["bulk_equations_analysis"],
    },
    "06": {
        "slug": "build_bulk_eigenmodes_dataset",
        "outputs": ["bulk_modes_dataset.csv", "bulk_modes_meta.json"],
        "aliases": ["bulk_eigenmodes"],
    },
    "07": {
        "slug": "emergent_lambda_sl_dictionary",
        "outputs": ["lambda_sl_dictionary_report.json"],
        "aliases": ["emergent_dictionary"],
    },
    "07b": {
        "slug": "discover_lambda_delta_relation",
        "outputs": ["lambda_delta_relation_report.json"],
        "aliases": ["lambda_delta_relation"],
    },
    "08": {
        "slug": "build_holographic_dictionary",
        "outputs": ["holographic_dictionary_v3_summary.json"],
        "aliases": ["holographic_dictionary"],
    },
    "09": {
        "slug": "real_data_and_dictionary_contracts",
        "outputs": ["contracts_12_13.json", "real_data_contracts_summary.json"],
        "aliases": ["contracts"],
    },
}

# Mapping inverso: alias -> stage_id
ALIAS_MAPPING = {}
for stage_id, info in STAGE_REGISTRY.items():
    for alias in info["aliases"]:
        ALIAS_MAPPING[alias] = stage_id


def get_stage_info(script_name: str) -> Dict[str, Any]:
    """
    Extrae stage_id y slug del nombre del script.
    
    Args:
        script_name: Nombre del script (e.g., "02_emergent_geometry_engine.py")
    
    Returns:
        Dict con stage_id, slug, outputs, aliases
    """
    basename = os.path.basename(script_name)
    
    # Extraer stage_id del patrón NN_ o NNx_
    if basename.startswith(tuple(f"{i:02d}" for i in range(10))):
        parts = basename.split("_", 1)
        stage_id = parts[0]
        
        # Manejar sufijos como 04b, 07b
        if stage_id not in STAGE_REGISTRY:
            for reg_id in STAGE_REGISTRY:
                if basename.startswith(reg_id + "_"):
                    stage_id = reg_id
                    break
    else:
        raise ValueError(f"Script '{basename}' no sigue el patrón NN_*.py")
    
    if stage_id not in STAGE_REGISTRY:
        raise ValueError(f"Stage '{stage_id}' no está en STAGE_REGISTRY")
    
    info = STAGE_REGISTRY[stage_id].copy()
    info["stage_id"] = stage_id
    return info


def resolve_experiment_dir(experiment: str, base_dir: Optional[Path] = None) -> Path:
    """Resuelve el directorio raíz de un experimento."""
    if base_dir is None:
        base_dir = Path.cwd() / "runs"
    return (base_dir / experiment).resolve()


class RunContext:
    """
    Contexto de ejecución para un stage del pipeline CUERDAS.
    
    Encapsula:
    - Resolución automática de rutas
    - Creación de directorios
    - Gestión de run_manifest.json
    - Creación de alias (symlinks)
    """
    
    MANIFEST_VERSION = "3.0"
    MANIFEST_FILENAME = "run_manifest.json"
    
    def __init__(
        self,
        experiment: str,
        script_name: str,
        base_dir: Optional[Path] = None,
        legacy_mode: bool = False,
    ):
        self.experiment = experiment
        self.script_name = script_name
        self.legacy_mode = legacy_mode
        
        try:
            self._stage_info = get_stage_info(script_name)
        except ValueError as e:
            if legacy_mode:
                self._stage_info = {
                    "stage_id": "XX",
                    "slug": "unknown",
                    "outputs": [],
                    "aliases": [],
                }
            else:
                raise
        
        self._base_dir = Path(base_dir) if base_dir else Path.cwd() / "runs"
        self._run_dir = (self._base_dir / experiment).resolve()
        self._stage_dir = self._run_dir / f"{self.stage_id}_{self.stage_slug}"
        
        self._manifest: Optional[Dict[str, Any]] = None
        self._outputs_registered: Dict[str, str] = {}
        self._created_at = datetime.now().isoformat()
        self._cmdline = " ".join(sys.argv)
    
    @property
    def stage_id(self) -> str:
        return self._stage_info["stage_id"]
    
    @property
    def stage_slug(self) -> str:
        return self._stage_info["slug"]
    
    @property
    def run_dir(self) -> Path:
        return self._run_dir
    
    @property
    def manifest_path(self) -> Path:
        return self._run_dir / self.MANIFEST_FILENAME
    
    def stage_dir(self, create: bool = True) -> Path:
        """Retorna el directorio de salida de este stage."""
        if create and not self._stage_dir.exists():
            self._stage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creado directorio de stage: {self._stage_dir}")
        return self._stage_dir
    
    def ensure_dir(self, subpath: str = "") -> Path:
        """Asegura que existe un subdirectorio dentro del stage."""
        target = self._stage_dir / subpath if subpath else self._stage_dir
        target.mkdir(parents=True, exist_ok=True)
        return target
    
    def output_path(self, filename: str) -> Path:
        """Construye path completo para un archivo de salida."""
        return self.stage_dir() / filename
    
    def resolve_input_from_stage(
        self,
        stage_id: str,
        relative_path: str = "",
    ) -> Path:
        """Resuelve la ruta a un output de un stage anterior."""
        if stage_id not in STAGE_REGISTRY:
            raise ValueError(f"Stage '{stage_id}' desconocido")
        
        source_slug = STAGE_REGISTRY[stage_id]["slug"]
        source_dir = self._run_dir / f"{stage_id}_{source_slug}"
        
        if not source_dir.exists():
            # Intentar buscar por alias
            for alias in STAGE_REGISTRY[stage_id]["aliases"]:
                alias_path = self._run_dir / alias
                if alias_path.exists():
                    source_dir = alias_path.resolve()
                    break
            else:
                raise FileNotFoundError(
                    f"Stage {stage_id} no encontrado en {self._run_dir}"
                )
        
        target = source_dir / relative_path if relative_path else source_dir
        return target
    
    def resolve_alias(self, alias: str) -> Path:
        """Resuelve un alias canónico a su path real."""
        alias_path = self._run_dir / alias
        if alias_path.exists():
            return alias_path.resolve()
        
        if alias in ALIAS_MAPPING:
            stage_id = ALIAS_MAPPING[alias]
            return self.resolve_input_from_stage(stage_id)
        
        raise FileNotFoundError(f"Alias '{alias}' no encontrado")
    
    def create_aliases(self) -> List[Path]:
        """Crea symlinks canónicos para los outputs de este stage."""
        created = []
        
        for alias in self._stage_info.get("aliases", []):
            alias_path = self._run_dir / alias
            
            if alias in ("predictions", "geometry_emergent"):
                target = self._stage_dir / alias
            else:
                target = self._stage_dir
            
            if alias in ("predictions", "geometry_emergent"):
                target.mkdir(parents=True, exist_ok=True)
            
            try:
                if alias_path.is_symlink():
                    alias_path.unlink()
                elif alias_path.exists():
                    continue
                
                rel_target = os.path.relpath(target, self._run_dir)
                alias_path.symlink_to(rel_target)
                created.append(alias_path)
                logger.info(f"Alias creado: {alias} -> {rel_target}")
            except OSError as e:
                logger.warning(f"No se pudo crear alias '{alias}': {e}")
        
        return created
    
    def link_alias(self, alias_name: str, target_subpath: str) -> Optional[Path]:
        """Crea un symlink específico."""
        alias_path = self._run_dir / alias_name
        target = self._stage_dir / target_subpath
        
        try:
            if alias_path.is_symlink():
                alias_path.unlink()
            
            target.mkdir(parents=True, exist_ok=True)
            rel_target = os.path.relpath(target, self._run_dir)
            alias_path.symlink_to(rel_target)
            return alias_path
        except OSError as e:
            logger.warning(f"No se pudo crear alias '{alias_name}': {e}")
            return None
    
    def load_manifest(self) -> Dict[str, Any]:
        """Carga o crea el run_manifest.json."""
        if self._manifest is not None:
            return self._manifest
        
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {
                "manifest_version": self.MANIFEST_VERSION,
                "created_at": self._created_at,
                "experiment": self.experiment,
                "run_dir": str(self._run_dir),
                "stages": {},
                "aliases": {},
            }
        return self._manifest
    
    def register_outputs(self, outputs: Dict[str, str]) -> None:
        """Registra outputs producidos por este stage."""
        self._outputs_registered.update(outputs)
    
    def save_manifest(self) -> Path:
        """Guarda el run_manifest.json actualizado."""
        manifest = self.load_manifest()
        
        manifest["stages"][self.stage_id] = {
            "slug": self.stage_slug,
            "script": self.script_name,
            "cmdline": self._cmdline,
            "executed_at": self._created_at,
            "stage_dir": str(self._stage_dir.relative_to(self._run_dir)),
            "outputs": self._outputs_registered,
        }
        
        for alias in self._stage_info.get("aliases", []):
            alias_path = self._run_dir / alias
            if alias_path.exists():
                manifest["aliases"][alias] = {
                    "stage_id": self.stage_id,
                    "path": str(alias_path.relative_to(self._run_dir)),
                    "is_symlink": alias_path.is_symlink(),
                }
        
        manifest["updated_at"] = datetime.now().isoformat()
        
        self._run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return self.manifest_path
    
    @classmethod
    def from_args(cls, args, script_name: Optional[str] = None) -> "RunContext":
        """Crea RunContext desde argparse Namespace."""
        if script_name is None:
            script_name = os.path.basename(sys.argv[0])
        
        experiment = getattr(args, "experiment", None)
        
        if experiment:
            return cls(
                experiment=experiment,
                script_name=script_name,
                legacy_mode=False,
            )
        
        # Fallback a modo legacy
        legacy_output = getattr(args, "output_dir", None) or \
                       getattr(args, "output", None) or \
                       getattr(args, "run_dir", None)
        
        if legacy_output:
            warnings.warn(
                "DEPRECATED: Usa --experiment <nombre> en lugar de --output-dir. "
                "El modo legacy será eliminado en futuras versiones.",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy_path = Path(legacy_output)
            experiment = legacy_path.name
            base_dir = legacy_path.parent
            
            return cls(
                experiment=experiment,
                script_name=script_name,
                base_dir=base_dir,
                legacy_mode=True,
            )
        
        raise ValueError(
            "Se requiere --experiment <nombre>. "
            "Ejemplo: --experiment debug_Tfinite"
        )
    
    @classmethod
    def from_manifest(cls, manifest_path: Union[str, Path], script_name: Optional[str] = None) -> "RunContext":
        """Crea RunContext desde un run_manifest.json existente."""
        manifest_path = Path(manifest_path)
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        experiment = manifest.get("experiment", manifest_path.parent.name)
        base_dir = manifest_path.parent.parent
        
        if script_name is None:
            script_name = os.path.basename(sys.argv[0])
        
        ctx = cls(experiment=experiment, script_name=script_name, base_dir=base_dir)
        ctx._manifest = manifest
        return ctx
    
    def __repr__(self) -> str:
        return f"RunContext(experiment='{self.experiment}', stage='{self.stage_id}_{self.stage_slug}')"
    
    def summary(self) -> str:
        return f"""=== RunContext ===
Experiment: {self.experiment}
Stage: {self.stage_id}_{self.stage_slug}
Run Dir: {self._run_dir}
Stage Dir: {self._stage_dir}
Manifest: {self.manifest_path}
Legacy Mode: {self.legacy_mode}"""


# =============================================================================
# FUNCIONES DE COMPATIBILIDAD
# =============================================================================

def write_run_manifest(run_dir: Path, artifacts: Dict[str, str], metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Escribe un run_manifest.json (compatibilidad con V2)."""
    manifest = {
        "manifest_version": RunContext.MANIFEST_VERSION,
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir.resolve()),
        "artifacts": artifacts,
        "metadata": metadata or {},
    }
    manifest_path = run_dir / RunContext.MANIFEST_FILENAME
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_run_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Carga un run_manifest.json existente."""
    manifest_path = run_dir / RunContext.MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def update_run_manifest(run_dir: Path, updates: Dict[str, Any]) -> Path:
    """Actualiza un run_manifest.json existente."""
    manifest = load_run_manifest(run_dir) or {
        "manifest_version": RunContext.MANIFEST_VERSION,
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir.resolve()),
    }
    manifest.update(updates)
    manifest["updated_at"] = datetime.now().isoformat()
    manifest_path = run_dir / RunContext.MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# =============================================================================
# HELPERS CLI
# =============================================================================

def add_experiment_args(parser) -> None:
    """Añade argumento --experiment a un ArgumentParser."""
    # Solo añade --experiment, los scripts pueden tener sus propios --output-dir/--run-dir
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        help="Nombre del experimento. Outputs en runs/<experiment>/.",
    )


def validate_experiment_args(args) -> str:
    """Valida y retorna el nombre del experimento."""
    if hasattr(args, "experiment") and args.experiment:
        return args.experiment
    
    legacy_path = getattr(args, "output_dir", None) or getattr(args, "run_dir", None)
    if legacy_path:
        warnings.warn("DEPRECATED: Usa --experiment <nombre>.", DeprecationWarning)
        return Path(legacy_path).name
    
    print("ERROR: Se requiere --experiment <nombre>", file=sys.stderr)
    print("Ejemplo: python script.py --experiment mi_experimento", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test del módulo run_context")
    add_experiment_args(parser)
    parser.add_argument("--script", default="02_emergent_geometry_engine.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    experiment = validate_experiment_args(args)
    ctx = RunContext(experiment=experiment, script_name=args.script)
    
    print(ctx.summary())
    print(f"\nStage directory: {ctx.stage_dir()}")
    ctx.create_aliases()
    ctx.register_outputs({"test_output": "test.json"})
    manifest_path = ctx.save_manifest()
    print(f"\nManifest: {manifest_path}")
