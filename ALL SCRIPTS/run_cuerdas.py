#!/usr/bin/env python3
"""
run_cuerdas.py — Runner Unificado del Pipeline CUERDAS

Orquesta las fases XI → XII → XII.c → XIII con handoff explícito.
Cada fase puede ejecutarse por separado o en secuencia.

USO:
    # Pipeline completo
    python run_cuerdas.py --phases all --run-id experiment_001
    
    # Solo Fase XI
    python run_cuerdas.py --phases 11 --run-id test_xi
    
    # Fases XII y XIII (requiere XI previo)
    python run_cuerdas.py --phases 12,13 --run-id real_world --fase11-run prev_run
    
    # Modo debug (parámetros reducidos)
    python run_cuerdas.py --phases all --run-id debug_001 --debug

ESTRUCTURA DE SALIDA:
    runs/
    ├── fase11/{run_id}/
    │   ├── data/
    │   ├── geometry/
    │   ├── einstein/
    │   ├── dictionary/
    │   ├── contracts_v2.json
    │   └── manifest.json
    ├── fase12/{run_id}/
    │   ├── real_data/
    │   ├── predictions/
    │   └── manifest.json
    └── fase13/{run_id}/
        ├── theory_atlas.json
        ├── analysis/
        └── manifest.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Importar configuración central
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config.cuerdas_config import (
        CUERDAS_VERSION,
        PIPELINE_VERSION,
        THRESHOLDS,
        EXPLORER_CONFIG,
        create_manifest,
        get_standard_paths
    )
except ImportError:
    # Fallback si no está el módulo
    CUERDAS_VERSION = "1.0.0"
    PIPELINE_VERSION = {"fase11": "v2", "fase12": "v1", "fase13": "v1"}
    
    def create_manifest(phase, version, seed, config, outputs):
        return {
            "phase": phase, "version": version, "seed": seed,
            "config": config, "outputs": outputs,
            "timestamp": datetime.now().isoformat()
        }


class CuerdasRunner:
    """Orquestador del pipeline CUERDAS."""
    
    def __init__(
        self,
        base_dir: Path,
        run_id: str,
        seed: int = 42,
        debug: bool = False,
        verbose: bool = False
    ):
        self.base_dir = base_dir
        self.run_id = run_id
        self.seed = seed
        self.debug = debug
        self.verbose = verbose
        
        # Parámetros según modo
        if debug:
            self.n_epochs = 100
            self.niterations = 20
            self.n_theories = 50
            self.n_known = 5
            self.n_test = 3
            self.n_unknown = 2
        else:
            self.n_epochs = 3000
            self.niterations = 200
            self.n_theories = 1000
            self.n_known = 20
            self.n_test = 10
            self.n_unknown = 5
        
        # Estado del pipeline
        self.results: Dict[str, Any] = {}
        
    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Ejecuta un comando y reporta resultado."""
        print(f"\n{'='*70}")
        print(f">> {description}")
        print(f"{'='*70}")
        print(f"   Comando: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=not self.verbose
            )
            
            if result.returncode != 0:
                print(f"\n✗ ERROR: {description}")
                if result.stderr:
                    print(f"   stderr: {result.stderr.decode()[:500]}")
                return False
                
            print(f"\n✓ Completado: {description}")
            return True
            
        except Exception as e:
            print(f"\n✗ EXCEPCIÓN: {e}")
            return False
    
    def _save_manifest(self, phase: str, phase_dir: Path, config: Dict, outputs: List[str]):
        """Guarda manifest para una fase."""
        manifest = create_manifest(
            phase=phase,
            version=PIPELINE_VERSION.get(phase, "v1"),
            seed=self.seed,
            config=config,
            outputs=outputs
        )
        manifest["run_id"] = self.run_id
        manifest["debug_mode"] = self.debug
        
        manifest_path = phase_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"   Manifest guardado: {manifest_path}")
    
    # ================================================================
    # FASE XI
    # ================================================================
    
    def run_fase11(self, output_dir: Optional[Path] = None) -> bool:
        """Ejecuta Fase XI completa."""
        
        print("\n" + "="*70)
        print("FASE XI — EMERGENCIA DE GEOMETRÍA")
        print("="*70)
        
        fase11_dir = output_dir or (self.base_dir / "runs" / "fase11" / self.run_id)
        fase11_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir = fase11_dir / "data"
        geometry_dir = fase11_dir / "geometry"
        einstein_dir = fase11_dir / "einstein"
        dictionary_dir = fase11_dir / "dictionary"
        contracts_file = fase11_dir / "contracts_v2.json"
        
        # Paso 0: Generación de datos
        cmd = [
            sys.executable, "00_generate_fase_11_v3.py",
            "--output-dir", str(data_dir),
            "--seed", str(self.seed),
            "--n-known", str(self.n_known),
            "--n-test", str(self.n_test),
            "--n-unknown", str(self.n_unknown),
        ]
        
        if not self._run_command(cmd, "Fase XI.0: Generación de datos CFT"):
            return False
        
        # Paso 1: Emergencia de geometría
        cmd = [
            sys.executable, "01_emergent_geometry_v2.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(geometry_dir),
            "--n-epochs", str(self.n_epochs),
            "--seed", str(self.seed),
        ]
        
        if not self._run_command(cmd, "Fase XI.1: Emergencia de geometría"):
            return False
        
        # Paso 2: Descubrimiento de Einstein
        cmd = [
            sys.executable, "02_discover_einstein_v2.py",
            "--geometry-dir", str(geometry_dir),
            "--output-dir", str(einstein_dir),
            "--niterations", str(self.niterations),
            "--seed", str(self.seed),
        ]
        
        if not self._run_command(cmd, "Fase XI.2: Descubrimiento de Einstein"):
            return False
        
        # Paso 3: Diccionario holográfico
        cmd = [
            sys.executable, "03_holographic_dictionary_v3.py",
            "--data-dir", str(data_dir),
            "--geometry-dir", str(geometry_dir),
            "--output-dir", str(dictionary_dir),
            "--seed", str(self.seed),
        ]
        
        if not self._run_command(cmd, "Fase XI.3: Diccionario holográfico"):
            return False
        
        # Paso 4: Contratos
        dictionary_summary = dictionary_dir / "holographic_dictionary_v3_summary.json"
        cmd = [
            sys.executable, "04_contracts_fase_11_v2.py",
            "--data-dir", str(data_dir),
            "--geometry-dir", str(geometry_dir),
            "--einstein-dir", str(einstein_dir),
            "--dictionary-file", str(dictionary_summary),
            "--output-file", str(contracts_file),
        ]
        
        if not self._run_command(cmd, "Fase XI.4: Validación de contratos"):
            return False
        
        # Guardar manifest
        self._save_manifest(
            phase="fase11",
            phase_dir=fase11_dir,
            config={
                "n_epochs": self.n_epochs,
                "niterations": self.niterations,
                "n_known": self.n_known,
                "n_test": self.n_test,
                "n_unknown": self.n_unknown
            },
            outputs=[
                str(data_dir),
                str(geometry_dir),
                str(einstein_dir),
                str(dictionary_dir),
                str(contracts_file)
            ]
        )
        
        # Leer resultado de contratos
        if contracts_file.exists():
            contracts = json.loads(contracts_file.read_text())
            self.results["fase11"] = {
                "passed": contracts.get("phase_passed", False),
                "contracts": contracts,
                "output_dir": str(fase11_dir)
            }
            
            if contracts.get("phase_passed"):
                print("\n✓ FASE XI COMPLETADA CON ÉXITO")
            else:
                print("\n⚠ FASE XI COMPLETADA CON WARNINGS")
        
        return True
    
    # ================================================================
    # FASE XII
    # ================================================================
    
    def run_fase12(
        self,
        fase11_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> bool:
        """Ejecuta Fase XII (datos reales)."""
        
        print("\n" + "="*70)
        print("FASE XII — CUERDAS EN EL MUNDO REAL")
        print("="*70)
        
        fase12_dir = output_dir or (self.base_dir / "runs" / "fase12" / self.run_id)
        fase12_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir = fase12_dir / "real_data"
        predictions_dir = fase12_dir / "predictions"
        
        # Paso 1: Generar datos sintéticos realistas
        cmd = [
            sys.executable, "fase12_real_data_adapters.py",
            "--mode", "generate-synthetic",
            "--output-dir", str(data_dir)
        ]
        
        if not self._run_command(cmd, "Fase XII.1: Generar datos sintéticos"):
            return False
        
        # Paso 2: Procesar adaptadores
        adapters = [
            ("bootstrap", "ising3d_bootstrap.json"),
            ("bootstrap", "o4_bootstrap.json"),
        ]
        
        if not self.debug:
            adapters.extend([
                ("lattice", "lattice_qcd.h5"),
                ("condensed", "strange_metal.json"),
                ("cosmology", "cmb_planck.json")
            ])
        
        for adapter, filename in adapters:
            source = data_dir / filename
            if not source.exists():
                print(f"   ⚠ No existe {source}, saltando...")
                continue
            
            cmd = [
                sys.executable, "fase12_real_data_adapters.py",
                "--mode", "process",
                "--adapter", adapter,
                "--source", str(source),
                "--output-dir", str(data_dir)
            ]
            
            self._run_command(cmd, f"Fase XII.2: Procesar {filename}")
        
        # Paso 3: Motor de predicciones (si hay XI disponible)
        if fase11_dir and fase11_dir.exists():
            cmd = [
                sys.executable, "fase12_prediction_engine.py",
                "--data-dir", str(data_dir),
                "--fase11-dir", str(fase11_dir),
                "--output-dir", str(predictions_dir)
            ]
            
            self._run_command(cmd, "Fase XII.3: Motor de predicciones")
        else:
            print("\n   ⚠ Sin Fase XI, saltando predicciones")
        
        # Guardar manifest
        self._save_manifest(
            phase="fase12",
            phase_dir=fase12_dir,
            config={
                "adapters": [a[0] for a in adapters],
                "fase11_dir": str(fase11_dir) if fase11_dir else None
            },
            outputs=[str(data_dir), str(predictions_dir)]
        )
        
        self.results["fase12"] = {
            "passed": True,
            "output_dir": str(fase12_dir)
        }
        
        print("\n✓ FASE XII COMPLETADA")
        return True
    
    # ================================================================
    # FASE XII.c
    # ================================================================
    
    def run_fase12c(
        self,
        input_file: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> bool:
        """Ejecuta Fase XII.c (diccionario emergente)."""
        
        print("\n" + "="*70)
        print("FASE XII.c — DICCIONARIO HOLOGRÁFICO EMERGENTE")
        print("="*70)
        
        fase12c_dir = output_dir or (self.base_dir / "runs" / "fase12" / self.run_id / "dictionary_emergent")
        fase12c_dir.mkdir(parents=True, exist_ok=True)
        
        if input_file is None:
            # Buscar en fase11
            fase11_dict = self.base_dir / "runs" / "fase11" / self.run_id / "dictionary" / "holographic_dictionary_v3_summary.json"
            if fase11_dict.exists():
                input_file = fase11_dict
            else:
                print("   ✗ No se encontró input para XII.c")
                return False
        
        cmd = [
            sys.executable, "fase12c_emergent_dictionary_real.py",
            "--input-file", str(input_file),
            "--output-dir", str(fase12c_dir),
            "--seed", str(self.seed),
        ]
        
        if self.debug:
            cmd.append("--ops-minimal")
        
        if not self._run_command(cmd, "Fase XII.c: Diccionario emergente"):
            return False
        
        self._save_manifest(
            phase="fase12c",
            phase_dir=fase12c_dir,
            config={"input_file": str(input_file), "ops_minimal": self.debug},
            outputs=[str(fase12c_dir)]
        )
        
        self.results["fase12c"] = {
            "passed": True,
            "output_dir": str(fase12c_dir)
        }
        
        print("\n✓ FASE XII.c COMPLETADA")
        return True
    
    # ================================================================
    # FASE XIII
    # ================================================================
    
    def run_fase13(self, output_dir: Optional[Path] = None) -> bool:
        """Ejecuta Fase XIII (explorador de teorías)."""
        
        print("\n" + "="*70)
        print("FASE XIII — EXPLORADOR UNIVERSAL DE TEORÍAS")
        print("="*70)
        
        fase13_dir = output_dir or (self.base_dir / "runs" / "fase13" / self.run_id)
        fase13_dir.mkdir(parents=True, exist_ok=True)
        
        # Paso 1: Generar teorías
        cmd = [
            sys.executable, "fase13_theory_explorer.py",
            "--mode", "generate",
            "--n-theories", str(self.n_theories),
            "--output-dir", str(fase13_dir),
            "--seed", str(self.seed)
        ]
        
        if not self._run_command(cmd, "Fase XIII.1: Generar teorías"):
            return False
        
        # Paso 2: Explorar
        cmd = [
            sys.executable, "fase13_theory_explorer.py",
            "--mode", "explore",
            "--output-dir", str(fase13_dir),
            "--seed", str(self.seed)
        ]
        
        if not self._run_command(cmd, "Fase XIII.2: Explorar espacio"):
            return False
        
        # Paso 3: Analizar
        cmd = [
            sys.executable, "fase13_theory_explorer.py",
            "--mode", "analyze",
            "--output-dir", str(fase13_dir)
        ]
        
        if not self._run_command(cmd, "Fase XIII.3: Analizar atlas"):
            return False
        
        self._save_manifest(
            phase="fase13",
            phase_dir=fase13_dir,
            config={"n_theories": self.n_theories},
            outputs=[str(fase13_dir / "theory_atlas.json")]
        )
        
        self.results["fase13"] = {
            "passed": True,
            "output_dir": str(fase13_dir)
        }
        
        print("\n✓ FASE XIII COMPLETADA")
        return True
    
    # ================================================================
    # PIPELINE COMPLETO
    # ================================================================
    
    def run_all(self) -> bool:
        """Ejecuta pipeline completo XI → XII → XII.c → XIII."""
        
        print("\n" + "="*70)
        print(f"CUERDAS v{CUERDAS_VERSION} — PIPELINE COMPLETO")
        print("="*70)
        print(f"  Run ID:     {self.run_id}")
        print(f"  Seed:       {self.seed}")
        print(f"  Debug mode: {self.debug}")
        print("="*70)
        
        success = True
        
        # Fase XI
        if not self.run_fase11():
            print("\n⚠ Fase XI falló, continuando...")
            success = False
        
        fase11_dir = self.base_dir / "runs" / "fase11" / self.run_id
        
        # Fase XII
        if not self.run_fase12(fase11_dir=fase11_dir):
            print("\n⚠ Fase XII falló, continuando...")
            success = False
        
        # Fase XII.c
        if not self.run_fase12c():
            print("\n⚠ Fase XII.c falló, continuando...")
            success = False
        
        # Fase XIII
        if not self.run_fase13():
            print("\n⚠ Fase XIII falló")
            success = False
        
        # Resumen final
        self._print_summary()
        
        return success
    
    def _print_summary(self):
        """Imprime resumen del pipeline."""
        print("\n" + "="*70)
        print("RESUMEN CUERDAS")
        print("="*70)
        
        for phase, result in self.results.items():
            status = "✓" if result.get("passed") else "✗"
            print(f"  {status} {phase}: {result.get('output_dir', 'N/A')}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="CUERDAS — Runner Unificado del Pipeline"
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="all",
        help="Fases a ejecutar: 'all', '11', '12', '12c', '13', o combinación '11,12'"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador del run (default: timestamp)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Directorio base del proyecto"
    )
    parser.add_argument(
        "--fase11-run",
        type=str,
        default=None,
        help="Run ID de Fase XI previa (para fases que la requieren)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria global"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug con parámetros reducidos"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar output completo de subprocesos"
    )
    
    args = parser.parse_args()
    
    # Generar run_id si no se especifica
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path(args.base_dir)
    
    runner = CuerdasRunner(
        base_dir=base_dir,
        run_id=args.run_id,
        seed=args.seed,
        debug=args.debug,
        verbose=args.verbose
    )
    
    # Parsear fases
    phases = args.phases.lower()
    
    if phases == "all":
        success = runner.run_all()
    else:
        phase_list = [p.strip() for p in phases.split(",")]
        success = True
        
        fase11_dir = None
        if args.fase11_run:
            fase11_dir = base_dir / "runs" / "fase11" / args.fase11_run
        
        for phase in phase_list:
            if phase == "11":
                success = runner.run_fase11() and success
                fase11_dir = base_dir / "runs" / "fase11" / args.run_id
            elif phase == "12":
                success = runner.run_fase12(fase11_dir=fase11_dir) and success
            elif phase == "12c":
                success = runner.run_fase12c() and success
            elif phase == "13":
                success = runner.run_fase13() and success
            else:
                print(f"⚠ Fase desconocida: {phase}")
        
        runner._print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
