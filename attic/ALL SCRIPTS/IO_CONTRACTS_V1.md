# IO_CONTRACTS_V1 — Contrato de formatos y metadatos (CUERDAS‑Maldacena)

**Versión:** v1.0  
**Fecha:** 2025-12-12  

## 1. Alcance

Este contrato define requisitos **MUST/SHOULD** para los artefactos que conectan el pipeline:

- **Bloque A:** `01_generate_sandbox_geometries.py` → `02_emergent_geometry_engine.py` → `03_discover_bulk_equations.py` / `04_geometry_physics_contracts.py` / `05_analyze_bulk_equations.py`
- **Bloque B:** `02_emergent_geometry_engine.py` → `bulk_scalar_solver.py`/`06_build_bulk_eigenmodes_dataset.py`
- **Bloque C:** `06_build_bulk_eigenmodes_dataset.py` → `07_emergent_lambda_sl_dictionary.py` / `08_build_holographic_dictionary.py` → `09_real_data_and_dictionary_contracts.py`

El objetivo es eliminar fallos downstream causados por:

- metadatos inconsistentes (p.ej. `d`),
- nombres de datasets ambiguos,
- layouts HDF5 no unificados.

Para validación automática, use `00_validate_io_contracts.py`.

## 2. Reglas generales (MUST)

1. **No defaults silenciosos en metadatos críticos.** Si falta `d` o `family`, el consumidor debe fallar (o el validador debe marcar FAIL). No se permite “asumir `d=4`”.
2. **Monotonía y consistencia de grid radial.** Todo `z_grid`/`bulk_truth/z_grid` MUST ser 1D, estrictamente creciente, finito.
3. **Coherencia de shapes.** Para cada geometría, `A*`, `f*`, `R*` MUST tener la misma longitud que `z_grid`.
4. **Coherencia `d` ↔ nombre.** Si el nombre del fichero contiene `_d<k>_`, entonces el atributo `d` MUST ser `k`.
5. **Metadatos como attrs.** `d`, `family`, `system_name/name` MUST existir como atributos raíz del HDF5.

Notas:
- Este contrato no impone contenido físico (por honestidad); impone *interfaces*.


## 3. Formato HDF5 — Sandbox (salida de 01)

### 3.1. Layout canónico (archivo único por sistema)

Ruta típica: `runs/sandbox_geometries/<system>.h5`

**Root attrs (MUST):**
- `name` o `system_name` (string)
- `family` (string)
- `d` (int)
- `z_dyn` (float; usar 1.0 si no aplica)
- `theta` (float; usar 0.0 si no aplica)
- `category` (string: `known|test|unknown`)

**Grupo `boundary/` (MUST):**
- Debe existir el grupo `boundary`.
- `boundary.attrs['d']` MUST existir y coincidir con `root.attrs['d']`.
- `boundary.attrs['family']` MUST existir y coincidir con `root.attrs['family']`.
- El resto de datasets dentro de `boundary/` puede variar (correladores, grids, etc.), pero MUST ser reproducibles y auto-consistentes.

**Grupo `bulk_truth/` (MUST en sandbox; MUST NOT usarse como input de training/inference):**
- Debe existir el grupo `bulk_truth` en sandbox.
- Datasets mínimos (MUST):
  - `bulk_truth/z_grid` (1D)
  - `bulk_truth/A_truth` (1D)
  - `bulk_truth/f_truth` (1D)
- Datasets recomendados (SHOULD):
  - `bulk_truth/R_truth` (1D)
  - `bulk_truth/z_h` como attr o dataset escalar

### 3.2. Layout legacy aceptado (split boundary/bulk_truth)

El README histórico menciona rutas tipo `runs/sandbox_geometries/boundary/*.h5` y `runs/sandbox_geometries/bulk_truth/*.h5`.
El validador acepta este dialecto **solo si**:

- boundary file contiene `root attrs` mínimos y datasets boundary en root, **o** en `boundary/`.
- bulk_truth file contiene `root attrs` mínimos y `z_grid`, `A_truth`, `f_truth`.
- ambos ficheros comparten el mismo `system_name` y `d`.


## 4. Formato HDF5 — Geometría emergente (salida de 02)

Ruta típica: `runs/**/geometry_emergent/<system>_emergent.h5`

**Root attrs (MUST):**
- `system_name` (string)
- `family_pred` (string)
- `d` (int)
- `provenance` (string)

**Root attrs (SHOULD):**
- `d_pred` (int)
- `checkpoint_source` (string)
- `zh_pred` (float)

**Datasets (MUST):**
- `z_grid` (1D)
- `A_emergent` (1D)
- `f_emergent` (1D)
- `R_emergent` (1D)

**Aliases permitidos (SHOULD, no requeridos):**
- `A_of_z` ≡ `A_emergent`
- `f_of_z` ≡ `f_emergent`
- `R_of_z` ≡ `R_emergent`

Regla: los consumidores MUST leer primero los nombres canónicos actuales (`*_emergent`) y, si no existen, buscar aliases `*_of_z`.

**Compatibilidad de metadatos:**
- Si existe `family`, debe ser coherente con `family_pred`.
- Si el nombre codifica `_d<k>_`, entonces `d` MUST ser `k`.


## 5. Dataset de modos escalares (salida de 06)

Archivo canónico:
- `runs/bulk_eigenmodes/bulk_modes_dataset.csv`

**Columnas mínimas (MUST):**
- `system_name` (str)
- `family` (str)
- `d` (int)
- `mode_id` (int)
- `lambda_sl` (float)
- `Delta_UV` (float o vacío)

**Columnas recomendadas (SHOULD):**
- `z_dyn` (float)
- `theta` (float)
- `quality_flag` (str)
- `is_ground_state` (bool)

Reglas:
- `lambda_sl` es λ_SL (no se interpreta como masa por defecto).
- Si existe `m2L2_legacy`, debe ser alias/compatibilidad, no columna principal.

## 6. Report de diccionario emergente (salida de 07)

Archivo canónico:
- `runs/emergent_dictionary/lambda_sl_dictionary_report.json`

**Campos mínimos (MUST):**
- `config` (objeto)
- `discovery_results.best_equation` (string)
- `discovery_results.test_metrics` (objeto con, al menos, `r2` o `mae`)
- `data_stats` (objeto)
- `nomenclature_version` = `v2_lambda_sl`

Reglas:
- Comparaciones con fórmulas teóricas (p.ej. Δ(Δ−d)) MUST ser explícitamente post‑hoc.

## 7. Validador automático

Use `00_validate_io_contracts.py` para escanear directorios y producir un reporte PASS/WARN/FAIL por archivo.

- MUST ser read‑only por defecto (no modifica datos).
- MUST fallar (exit != 0) si hay cualquier FAIL.

