# PR: Protocolo IO Determinístico V2 (Completo)

## Resumen

Este PR implementa un protocolo determinístico de IO basado en `run_manifest.json` para eliminar el caos de rutas entre scripts del pipeline. Todos los scripts (02-09) ahora soportan `--run-dir` como argumento simplificado.

## Cambios

### Nuevo módulo: `cuerdas_io.py`

Funciones para gestionar `run_manifest.json`:
- `write_run_manifest()` - Crear manifest
- `load_run_manifest()` - Leer manifest
- `update_run_manifest()` - Actualizar manifest
- `resolve_*_dir()` - Resolver rutas desde manifest
- `RunContext` - Clase para encapsular resolución

### Scripts modificados

| Script | Cambios |
|--------|---------|
| `02_emergent_geometry_engine.py` | Escribe `run_manifest.json` al final |
| `03_discover_bulk_equations.py` | +`--run-dir`, `--geometry-dir` ahora opcional |
| `04_geometry_physics_contracts.py` | +`--run-dir`, todos los paths ahora opcionales |
| `05_analyze_bulk_equations.py` | +`--run-dir`, `--input` ahora opcional |
| `06_build_bulk_eigenmodes_dataset.py` | +`--run-dir`, `--geometry-dir` ahora opcional |
| `07_emergent_lambda_sl_dictionary.py` | +`--run-dir`, `--input-file` y `--output-dir` ahora opcionales |
| `08_build_holographic_dictionary.py` | +`--run-dir`, `--data-dir` ahora opcional |
| `09_real_data_and_dictionary_contracts.py` | +`--run-dir`, inputs ahora opcionales |

### Nueva documentación

- `docs/IO_LAYOUT_V2.md` - Especificación completa del protocolo

### Nuevos tests

- `test_cuerdas_io.py` - 12 tests para el módulo cuerdas_io

## Uso simplificado

```bash
# Antes (4+ argumentos por script)
python 04_geometry_physics_contracts.py \
  --data-dir A --geometry-dir B --einstein-dir C --dictionary-file D

# Después (1 argumento)
python 04_geometry_physics_contracts.py --run-dir runs/my_run
```

## Pipeline completo

```bash
python 01_generate_sandbox_geometries.py --output-dir runs/sandbox
python 02_emergent_geometry_engine.py --data-dir runs/sandbox --output-dir runs/my_run
python 03_discover_bulk_equations.py --run-dir runs/my_run
python 04_geometry_physics_contracts.py --run-dir runs/my_run
python 05_analyze_bulk_equations.py --run-dir runs/my_run
python 06_build_bulk_eigenmodes_dataset.py --run-dir runs/my_run
python 07_emergent_lambda_sl_dictionary.py --run-dir runs/my_run
python 08_build_holographic_dictionary.py --run-dir runs/my_run
python 09_real_data_and_dictionary_contracts.py --phase both --run-dir runs/my_run
```

## Compatibilidad

- ✅ Todos los argumentos legacy siguen funcionando
- ✅ `--run-dir` es aditivo, no obligatorio
- ✅ Runs existentes sin manifest funcionan
- ✅ Sin cambios breaking a IO_CONTRACTS_V1.md

## Archivos incluidos

```
io_v2_pr_complete/
├── cuerdas_io.py                           # Módulo de IO
├── test_cuerdas_io.py                      # Tests
├── IO_LAYOUT_V2.md                         # Documentación
├── 02_emergent_geometry_engine.py          # Modificado
├── 03_discover_bulk_equations.py           # Modificado
├── 04_geometry_physics_contracts.py        # Modificado
├── 05_analyze_bulk_equations.py            # Modificado
├── 06_build_bulk_eigenmodes_dataset.py     # Modificado
├── 07_emergent_lambda_sl_dictionary.py     # Modificado
├── 08_build_holographic_dictionary.py      # Modificado
└── 09_real_data_and_dictionary_contracts.py # Modificado
```

## Verificación

Todos los scripts compilan correctamente:
```
✓ 02 OK
✓ 03 OK
✓ 04 OK
✓ 05 OK
✓ 06 OK
✓ 07 OK
✓ 08 OK
✓ 09 OK
```

## Integración

Para integrar en el repositorio:

```bash
# Desde la raíz del repo
cp io_v2_pr_complete/cuerdas_io.py .
cp io_v2_pr_complete/test_cuerdas_io.py .
cp io_v2_pr_complete/02_*.py .
cp io_v2_pr_complete/03_*.py .
cp io_v2_pr_complete/04_*.py .
cp io_v2_pr_complete/05_*.py .
cp io_v2_pr_complete/06_*.py .
cp io_v2_pr_complete/07_*.py .
cp io_v2_pr_complete/08_*.py .
cp io_v2_pr_complete/09_*.py .
mkdir -p docs && cp io_v2_pr_complete/IO_LAYOUT_V2.md docs/

# Verificar
python -m pytest test_cuerdas_io.py -v
```
