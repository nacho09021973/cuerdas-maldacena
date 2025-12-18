# IO Layout V2 - Protocolo Determinístico de Rutas

## Motivación

El pipeline CUERDAS tiene múltiples scripts (02-09) que deben pasarse rutas entre sí. Antes de esta mejora, cada script tenía sus propios argumentos `--data-dir`, `--geometry-dir`, `--output-dir`, etc., lo que llevaba a:

1. **Caos de rutas**: Errores "0/0 summaries" cuando las rutas no coincidían
2. **Difícil reproducibilidad**: Recordar qué directorio pasó a cuál script
3. **Encadenamiento manual**: Cada script requería múltiples flags

## Solución: `run_manifest.json` + `--run-dir`

Ahora cada run produce un `run_manifest.json` que registra dónde están los artefactos. Los scripts posteriores solo necesitan `--run-dir` para encontrar todo automáticamente.

## Estructura de Directorios (V2)

```
runs/my_experiment/
├── run_manifest.json          # ← Fuente de verdad
├── predictions/               # Salida de 02 (modo inference)
│   ├── system_A.h5
│   └── system_B.h5
├── geometry_emergent/         # Salida de 02 (modo train)
│   ├── system_A.h5
│   └── system_B.h5
├── bulk_equations/            # Salida de 03
│   ├── pareto_equations.json
│   └── einstein_discovery_summary.json
├── geometry_contracts/        # Salida de 04
│   └── geometry_contracts_summary.json
├── bulk_equations_analysis/   # Salida de 05
│   └── bulk_equations_report.json
├── bulk_eigenmodes/           # Salida de 06
│   ├── bulk_modes_dataset.csv
│   └── bulk_modes_meta.json
├── emergent_dictionary/       # Salida de 07
│   └── lambda_sl_dictionary_report.json
├── holographic_dictionary/    # Salida de 08
│   └── holographic_dictionary_v3_summary.json
└── contracts/                 # Salida de 09
    └── contracts_12_13.json
```

## Schema de `run_manifest.json`

```json
{
  "manifest_version": "2.0",
  "created_at": "2025-12-18T12:00:00",
  "run_dir": "/absolute/path/to/run",
  "artifacts": {
    "data_dir": "../sandbox_geometries",
    "predictions_dir": "predictions",
    "geometry_emergent_dir": "geometry_emergent",
    "bulk_equations_dir": "bulk_equations",
    "geometry_contracts_dir": "geometry_contracts",
    "bulk_equations_analysis_dir": "bulk_equations_analysis",
    "bulk_eigenmodes_dir": "bulk_eigenmodes",
    "emergent_dictionary_dir": "emergent_dictionary",
    "holographic_dictionary_dir": "holographic_dictionary",
    "contracts_dir": "contracts",
    "systems": [
      {"name": "system_A", "h5_output": "predictions/system_A.h5"},
      {"name": "system_B", "h5_output": "predictions/system_B.h5"}
    ]
  },
  "metadata": {
    "script": "02_emergent_geometry_engine.py",
    "mode": "inference",
    "version": "2.0"
  }
}
```

## API de `cuerdas_io.py`

### Funciones de Resolución

```python
from cuerdas_io import (
    resolve_predictions_dir,
    resolve_geometry_emergent_dir,
    resolve_bulk_equations_dir,
    resolve_data_dir,
    resolve_dictionary_file,
    load_run_manifest,
    update_run_manifest,
    write_run_manifest,
)

# Resolver directorio de predicciones
predictions = resolve_predictions_dir(run_dir=Path("runs/my_run"))
# → Path("runs/my_run/predictions") si existe en manifest

# Resolver geometría emergente
geom = resolve_geometry_emergent_dir(run_dir=Path("runs/my_run"))
# → Path("runs/my_run/geometry_emergent")
```

### Clase RunContext

```python
from cuerdas_io import RunContext

ctx = RunContext.from_run_dir(Path("runs/my_run"))
print(ctx.predictions_dir)        # Path resuelto
print(ctx.geometry_emergent_dir)  # Path resuelto
print(ctx.bulk_equations_dir)     # Path resuelto

# Añadir artefacto
ctx.add_artifact("my_output", "my_dir/output.json")
ctx.save()
```

## Uso por Script

### 02_emergent_geometry_engine.py

```bash
# Modo inference - escribe run_manifest.json
python 02_emergent_geometry_engine.py \
  --data-dir runs/sandbox \
  --output-dir runs/my_run \
  --checkpoint model.pt \
  --inference-only
```

### 03_discover_bulk_equations.py

```bash
# NUEVO: Solo necesita --run-dir
python 03_discover_bulk_equations.py --run-dir runs/my_run

# LEGACY: Sigue funcionando
python 03_discover_bulk_equations.py --geometry-dir runs/my_run/predictions
```

### 04_geometry_physics_contracts.py

```bash
# NUEVO: Solo necesita --run-dir
python 04_geometry_physics_contracts.py --run-dir runs/my_run

# LEGACY: Sigue funcionando con 4 argumentos
python 04_geometry_physics_contracts.py \
  --data-dir A --geometry-dir B --einstein-dir C --dictionary-file D
```

### 05_analyze_bulk_equations.py

```bash
# NUEVO
python 05_analyze_bulk_equations.py --run-dir runs/my_run

# LEGACY
python 05_analyze_bulk_equations.py --input file.json --output analysis.txt
```

### 06_build_bulk_eigenmodes_dataset.py

```bash
# NUEVO
python 06_build_bulk_eigenmodes_dataset.py --run-dir runs/my_run

# LEGACY
python 06_build_bulk_eigenmodes_dataset.py --geometry-dir runs/geometry_emergent
```

### 07_emergent_lambda_sl_dictionary.py

```bash
# NUEVO
python 07_emergent_lambda_sl_dictionary.py --run-dir runs/my_run

# LEGACY
python 07_emergent_lambda_sl_dictionary.py --input-file data.json --output-dir output/
```

### 08_build_holographic_dictionary.py

```bash
# NUEVO
python 08_build_holographic_dictionary.py --run-dir runs/my_run

# LEGACY
python 08_build_holographic_dictionary.py --data-dir runs/geometry
```

### 09_real_data_and_dictionary_contracts.py

```bash
# NUEVO
python 09_real_data_and_dictionary_contracts.py --phase both --run-dir runs/my_run

# LEGACY
python 09_real_data_and_dictionary_contracts.py \
  --phase both \
  --fase12-report report.json \
  --fase13-analysis analysis.json
```

## Pipeline Simplificado

```bash
# Generar geometrías sandbox
python 01_generate_sandbox_geometries.py --output-dir runs/sandbox

# Entrenar modelo de geometría emergente
python 02_emergent_geometry_engine.py \
  --data-dir runs/sandbox \
  --output-dir runs/my_run \
  --epochs 100

# ¡El resto del pipeline solo necesita --run-dir!
python 03_discover_bulk_equations.py --run-dir runs/my_run
python 04_geometry_physics_contracts.py --run-dir runs/my_run
python 05_analyze_bulk_equations.py --run-dir runs/my_run
python 06_build_bulk_eigenmodes_dataset.py --run-dir runs/my_run
python 07_emergent_lambda_sl_dictionary.py --run-dir runs/my_run
python 08_build_holographic_dictionary.py --run-dir runs/my_run
python 09_real_data_and_dictionary_contracts.py --phase both --run-dir runs/my_run
```

## Compatibilidad

### Garantías

1. **Todos los argumentos legacy siguen funcionando**
2. **`--run-dir` es aditivo, no obligatorio**
3. **Mezclar `--run-dir` con args legacy permite override**
4. **Runs existentes sin manifest funcionan normalmente**
5. **Sin cambios breaking a IO_CONTRACTS_V1.md**

### Prioridad de Resolución

1. Argumento explícito (ej. `--geometry-dir`)
2. Manifest en `--run-dir`
3. Default del script

## Beneficios

| Antes | Después |
|-------|---------|
| 4+ argumentos por script | 1 argumento (`--run-dir`) |
| Errores de ruta frecuentes | Resolución automática |
| Difícil reproducibilidad | Todo en manifest |
| Encadenamiento manual | Pipeline determinístico |

## Migración

### Para runs existentes

No se requiere ninguna acción. Los scripts detectan si hay `run_manifest.json` y lo usan; si no existe, usan los argumentos legacy.

### Para nuevos runs

El script 02 crea automáticamente `run_manifest.json`. Los scripts posteriores lo actualizan con sus artefactos.

### Crear manifest manualmente

```python
from cuerdas_io import write_run_manifest

write_run_manifest(
    run_dir=Path("runs/legacy_run"),
    artifacts={
        "predictions_dir": "predictions",
        "data_dir": "../sandbox",
    },
    metadata={"note": "migrado manualmente"}
)
```

## Tests

Ver `test_cuerdas_io.py` para 12 tests que verifican:
- Lectura/escritura de manifest
- Resolución desde manifest
- Resolución legacy
- RunContext
- Prioridad manifest sobre legacy
