#!/usr/bin/env python3
"""
Script maestro para ejecutar el pipeline completo o por bloques.

Ejemplos:
    python run_pipeline.py --bloque A --family AdS --quick-test
    python run_pipeline.py --bloque B --input runs/mi_experimento
    python run_pipeline.py --completo --experiment ising3d_full
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Añade el directorio actual al path para importar tus módulos
sys.path.insert(0, str(Path(__file__).parent))

from config import PATHS, save_experiment_config

class PipelineRunner:
    """Orquesta la ejecución de los scripts del pipeline"""
    
    BLOQUE_SCRIPTS = {
        'A': ['01_generate_sandbox_geometries.py',
              '02_emergent_geometry_engine.py',
              '03_discover_bulk_equations.py',
              '04_geometry_physics_contracts.py',
              '05_analyze_bulk_equations.py'],
        'B': ['bulk_scalar_solver.py',
              '06_build_bulk_eigenmodes_dataset.py'],
        'C': ['07_emergent_lambda_sl_dictionary.py',
              '08_build_holographic_dictionary.py',
              '09_real_data_and_dictionary_contracts.py']
    }
    
    def __init__(self, experiment_name=None):
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.results = {}
        
    def run_script(self, script_name, args=None):
        """Ejecuta un script individual con manejo de errores"""
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)
        
        print(f"\n{'='*60}")
        print(f"🚀 Ejecutando: {script_name}")
        print(f"📝 Comando: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {script_name} completado exitosamente")
            if result.stdout.strip():
                print(f"Salida:\n{result.stdout[-500:]}")  # Últimas 500 líneas
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error en {script_name}:")
            print(f"Código salida: {e.returncode}")
            print(f"Stderr:\n{e.stderr[-1000:]}")
            return False
    
    def run_bloque(self, bloque, quick_test=False):
        """Ejecuta todos los scripts de un bloque"""
        print(f"\n{'#'*60}")
        print(f"INICIANDO BLOQUE {bloque}")
        print(f"{'#'*60}")
        
        scripts = self.BLOQUE_SCRIPTS.get(bloque, [])
        if not scripts:
            print(f"⚠️  Bloque desconocido: {bloque}")
            return False
        
        for script in scripts:
            if not Path(script).exists():
                print(f"⚠️  Script no encontrado: {script}")
                continue
                
            args = []
            if quick_test:
                args.append("--quick-test")
            if self.experiment_name:
                args.extend(["--experiment", self.experiment_name])
            
            success = self.run_script(script, args)
            if not success:
                print(f"❌ Bloque {bloque} interrumpido por error en {script}")
                return False
        
        print(f"\n✅ BLOQUE {bloque} COMPLETADO")
        return True
    
    def run_completo(self, quick_test=False):
        """Ejecuta el pipeline completo A->B->C"""
        print(f"\n{'#'*60}")
        print(f"PIPELINE COMPLETO: Experimento {self.experiment_name}")
        print(f"{'#'*60}")
        
        # Crear configuración del experimento
        config = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "quick_test": quick_test
        }
        save_experiment_config(self.experiment_name, config)
        
        # Ejecutar bloques en secuencia
        for bloque in ['A', 'B', 'C']:
            if not self.run_bloque(bloque, quick_test):
                print(f"❌ Pipeline abortado en bloque {bloque}")
                return False
        
        print(f"\n{'🎉'*20}")
        print(f"PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"Resultados en: {PATHS.RUNS / self.experiment_name}")
        print(f"{'🎉'*20}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Orquestador del pipeline CUERDAS-Maldacena')
    parser.add_argument('--bloque', choices=['A', 'B', 'C'], help='Ejecutar solo un bloque')
    parser.add_argument('--completo', action='store_true', help='Ejecutar pipeline completo A->B->C')
    parser.add_argument('--experiment', help='Nombre del experimento (por defecto: fecha+hora)')
    parser.add_argument('--quick-test', action='store_true', help='Ejecutar en modo rápido (datos reducidos)')
    
    args = parser.parse_args()
    
    # Asegurar estructura de carpetas
    PATHS.ensure_structure()
    
    runner = PipelineRunner(args.experiment)
    
    if args.completo:
        success = runner.run_completo(args.quick_test)
    elif args.bloque:
        success = runner.run_bloque(args.bloque, args.quick_test)
    else:
        print("❌ Especifica --bloque o --completo")
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()