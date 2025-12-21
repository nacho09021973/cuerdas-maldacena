# config.py
"""
Configuración centralizada de CUERDAS-Maldacena.

NO hardcodear rutas en los scripts. Usar:
    from config import PATHS, PARAMS
"""

from pathlib import Path
import json

# ---------------------------------------------------------------------
# RUTAS ABSOLUTAS (se ajustan automáticamente)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

class PATHS:
    """Estructura de carpetas del proyecto"""
    # Directorios principales
    RUNS = BASE_DIR / "runs"
    BOUNDARY = BASE_DIR / "boundary"
    NOTEBOOKS = BASE_DIR / "notebooks"
    
    # Subdirectorios dentro de runs
    SANDBOX = RUNS / "sandbox_geometries"
    SANDBOX_BOUNDARY = SANDBOX / "boundary"
    SANDBOX_BULK_TRUTH = SANDBOX / "bulk_truth"
    
    @classmethod
    def ensure_structure(cls):
        """Crea toda la estructura de carpetas si no existe"""
        for attr in dir(cls):
            if not attr.startswith('_') and isinstance(getattr(cls, attr), Path):
                path = getattr(cls, attr)
                if not path.exists():
                    print(f"📁 Creando carpeta: {path}")
                    path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# PARÁMETROS POR BLOQUE
# ---------------------------------------------------------------------
class PARAMS:
    """Parámetros ajustables del pipeline"""
    
    # Bloque A: Geometría emergente
    class GEOMETRY:
        BATCH_SIZE = 32
        EPOCHS = 1000
        LEARNING_RATE = 1e-3
        HIDDEN_DIMS = [64, 128, 128, 64]  # Arquitectura MLP
        
    # Bloque B: Solver escalar
    class SOLVER:
        R_POINTS = 1000
        BOUNDARY_TOL = 1e-6
        MAX_ITER = 10000
        
    # Bloque C: Diccionario (PySR)
    class PYSR:
        POPULATION_SIZE = 100
        NITERATIONS = 50
        BINARY_OPERATORS = ["+", "*", "-", "/"]
        UNARY_OPERATORS = ["exp", "log", "abs", "sqrt"]
        COMPLEXITY_EPSILON = 1e-10
        
# ---------------------------------------------------------------------
# FUNCIONES DE UTILIDAD
# ---------------------------------------------------------------------
def load_json_config(path):
    """Carga configuración desde JSON, con valores por defecto"""
    default = {
        "geometry": PARAMS.GEOMETRY.__dict__,
        "solver": PARAMS.SOLVER.__dict__,
        "pysr": PARAMS.PYSR.__dict__
    }
    
    if Path(path).exists():
        with open(path, 'r') as f:
            user_config = json.load(f)
        # Fusiona con defaults
        for key in default:
            if key in user_config:
                default[key].update(user_config[key])
    return default

def save_experiment_config(experiment_name, config_dict):
    """Guarda configuración de un experimento específico"""
    exp_dir = PATHS.RUNS / experiment_name
    exp_dir.mkdir(exist_ok=True)
    
    config_path = exp_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return config_path