# Contrato IO v1 — CUERDAS-Maldacena

> **Versión:** 1.0  
> **Fecha:** 2025-12  
> **Estado:** ACTIVO  

Este documento define el **contrato de entrada/salida** entre los bloques del pipeline CUERDAS-Maldacena.
Es complementario al README.md y tiene carácter **normativo**: cualquier archivo que no cumpla estos requisitos debe ser rechazado por el validador `00_validate_io_contracts.py`.

---

## 0. Principios generales

1. **Fail-fast**: Si falta un campo obligatorio, el validador debe **FAIL**, nunca poner defaults silenciosos.
2. **Canónico sobre legacy**: Los nombres canónicos (`*_of_z`) son obligatorios para escritura. Los legacy (`*_emergent`) solo se aceptan en lectura para compatibilidad.
3. **Consistencia de metadatos**: El atributo `d` debe ser coherente con el nombre del archivo si este codifica `_d<k>_`.
4. **Monotonicidad**: `z_grid` debe ser estrictamente creciente.
5. **Longitudes consistentes**: Todos los arrays de un mismo archivo deben tener la misma longitud que `z_grid`.

---

## 1. Artefacto: Sandbox HDF5

**Productor:** `01_generate_sandbox_geometries.py`  
**Consumidor:** `02_emergent_geometry_engine.py` (modo train)  
**Ubicación:** `runs/sandbox_geometries/*.h5` (archivo combinado) o `runs/sandbox_geometries/boundary/*.h5` + `runs/sandbox_geometries/bulk_truth/*.h5` (separados)

### 1.1 Atributos raíz (obligatorios)

| Atributo      | Tipo    | Descripción                                      | Valores válidos                                |
|---------------|---------|--------------------------------------------------|------------------------------------------------|
| `name`        | str     | Identificador único (= filename sin `.h5`)       | Cualquier string válido                        |
| `system_name` | str     | Alias de `name` (puede ser igual)                | Cualquier string válido                        |
| `family`      | str     | Familia de geometría                             | `ads`, `lifshitz`, `hvlf`, `deformed`, `unknown` |
| `d`           | int     | Dimensión del boundary                           | 2, 3, 4, 5, ...                                |
| `category`    | str     | Categoría de split                               | `known`, `test`, `unknown`                     |

### 1.2 Atributos raíz (opcionales pero recomendados)

| Atributo   | Tipo  | Descripción                          | Default si ausente |
|------------|-------|--------------------------------------|--------------------|
| `z_dyn`    | float | Exponente dinámico (Lifshitz)        | 1.0                |
| `theta`    | float | Exponente de hyperscaling violation  | 0.0                |
| `z_h`      | float | Posición del horizonte               | —                  |
| `operators`| str   | JSON string con lista de operadores  | `"[]"`             |

### 1.3 Grupo `boundary/` (obligatorio)

Contiene los datos de frontera que serán input de `02` en modo train/inference.

**Atributos del grupo `boundary/`:**
- `d` (int) — debe coincidir con el atributo raíz
- `family` (str) — debe coincidir con el atributo raíz

**Datasets típicos:**
- `x_grid`, `omega_grid`, `k_grid`, `temperature`
- `G_R_real`, `G_R_imag` (correladores)

### 1.4 Grupo `bulk_truth/` (obligatorio en sandbox)

Contiene la geometría "verdadera" para validación en contratos.

**Atributos del grupo:**
- `z_h` (float, si aplica)

**Datasets obligatorios:**

| Dataset       | Tipo  | Descripción                       |
|---------------|-------|-----------------------------------|
| `z_grid`      | 1D    | Grid radial, estrictamente creciente |
| `A_truth`     | 1D    | Warp factor verdadero             |
| `f_truth`     | 1D    | Blackening factor verdadero       |

**Datasets opcionales:**
- `R_truth` (Ricci scalar verdadero)

### 1.5 Reglas de integridad

1. `len(A_truth) == len(f_truth) == len(z_grid)`
2. `z_grid` debe ser monótono creciente: `z_grid[i+1] > z_grid[i]` para todo `i`
3. Si el nombre contiene `_d<k>_`, entonces `attrs["d"]` debe ser igual a `k`

---

## 2. Artefacto: Geometría Emergente HDF5

**Productor:** `02_emergent_geometry_engine.py`  
**Consumidores:** `03`, `04`, `06`, `08`  
**Ubicación:** `runs/emergent_geometry/geometry_emergent/*.h5`

### 2.1 Atributos raíz (obligatorios)

| Atributo      | Tipo | Descripción                              |
|---------------|------|------------------------------------------|
| `system_name` | str  | Identificador del sistema                |
| `family`      | str  | Familia de geometría (verdad o inferida) |
| `d`           | int  | Dimensión del boundary                   |
| `provenance`  | str  | Origen del archivo (`train`, `inference`)|

### 2.2 Atributos raíz (opcionales)

| Atributo           | Tipo | Descripción                                    |
|--------------------|------|------------------------------------------------|
| `family_pred`      | str  | Predicción de familia por la red (si difiere)  |
| `d_pred`           | int  | Predicción de dimensión por la red             |
| `checkpoint_source`| str  | Path al checkpoint usado                       |

### 2.3 Datasets (obligatorios) — NOMBRES CANÓNICOS

| Dataset   | Tipo | Descripción                            |
|-----------|------|----------------------------------------|
| `z_grid`  | 1D   | Grid radial, estrictamente creciente   |
| `A_of_z`  | 1D   | Warp factor emergente                  |
| `f_of_z`  | 1D   | Blackening factor emergente            |

### 2.4 Datasets (opcionales)

| Dataset   | Tipo | Descripción          |
|-----------|------|----------------------|
| `R_of_z`  | 1D   | Ricci scalar emergente |
| `phi_of_z`| 1D   | Dilaton (si aplica)  |

### 2.5 Aliases legacy (SOLO LECTURA)

Los siguientes nombres son **solo para lectura** de archivos antiguos. **Nunca escribir archivos nuevos con estos nombres.**

| Legacy         | Canónico  |
|----------------|-----------|
| `A_emergent`   | `A_of_z`  |
| `f_emergent`   | `f_of_z`  |
| `R_emergent`   | `R_of_z`  |

### 2.6 Reglas de integridad

1. `len(A_of_z) == len(f_of_z) == len(z_grid)`
2. `z_grid` monótono creciente
3. `family` debe existir (no se acepta solo `family_pred`)
4. Si `provenance == "inference"`, no debe existir grupo `bulk_truth/`

---

## 3. Artefacto: Dataset de Modos Bulk

**Productor:** `06_build_bulk_eigenmodes_dataset.py`  
**Consumidor:** `07_emergent_lambda_sl_dictionary.py`  
**Ubicación:** `runs/bulk_eigenmodes/bulk_modes_dataset.csv` + `runs/bulk_eigenmodes/bulk_modes_meta.json`

### 3.1 Columnas CSV (obligatorias)

| Columna       | Tipo  | Descripción                              |
|---------------|-------|------------------------------------------|
| `system_name` | str   | Identificador del sistema                |
| `family`      | str   | Familia de geometría                     |
| `d`           | int   | Dimensión del boundary                   |
| `mode_id`     | int   | Índice del modo (0 = ground state)       |
| `lambda_sl`   | float | Autovalor Sturm-Liouville                |
| `Delta_UV`    | float | Dimensión conforme extraída              |

### 3.2 Columnas CSV (opcionales)

| Columna          | Tipo  | Descripción                      |
|------------------|-------|----------------------------------|
| `z_dyn`          | float | Exponente dinámico               |
| `theta`          | float | Hyperscaling violation           |
| `is_ground_state`| bool  | `True` si `mode_id == 0`         |
| `quality_flag`   | str   | `OK`, `UV_UNRELIABLE`, `NEGATIVE_LAMBDA` |
| `norm`           | float | Norma del modo propio            |

### 3.3 Regla de nomenclatura

**IMPORTANTE:** La columna se llama `lambda_sl`, **NO** `m2L2`.  
Si existe código legacy que use `m2L2`, debe ser un alias de lectura, nunca la columna principal.

### 3.4 Metadatos JSON

El archivo `bulk_modes_meta.json` debe contener:

```json
{
  "created_at": "ISO timestamp",
  "n_systems": int,
  "n_modes_total": int,
  "source_geometry_dir": "path",
  "nomenclature_version": "v2_lambda_sl"
}
```

---

## 4. Artefacto: Reporte de Diccionario λ_SL

**Productor:** `07_emergent_lambda_sl_dictionary.py`  
**Consumidor:** `09_real_data_and_dictionary_contracts.py`, `make_fase12_report_from_emergent.py`  
**Ubicación:** `runs/emergent_dictionary/lambda_sl_dictionary_report.json`

### 4.1 Campos obligatorios

```json
{
  "config": {
    "seed": int,
    "operators": [...],
    "iterations": int
  },
  "discovery_results": {
    "best_equation": "string con la ecuación",
    "test_metrics": {
      "r2": float,
      "mae": float,
      "pearson": float
    }
  },
  "data_stats": {
    "Delta_range": [min, max],
    "lambda_sl_range": [min, max],
    "n_points_train": int,
    "n_points_test": int
  },
  "nomenclature_version": "v2_lambda_sl"
}
```

### 4.2 Campos opcionales (recomendados)

```json
{
  "theory_comparison": {
    "theory_r2": float,
    "compatible_with_standard": bool,
    "notes": "string"
  }
}
```

---

## 5. Artefacto: Atlas Holográfico

**Productor:** `08_build_holographic_dictionary.py`  
**Consumidores:** `04`, `09`  
**Ubicación:** `runs/holographic_dictionary/holographic_dictionary_summary.json`

### 5.1 Estructura obligatoria

```json
{
  "by_system": {
    "<system_name>": {
      "family": str,
      "d": int,
      "n_points": int,
      "Delta": [float, ...],
      "lambda_sl": [float, ...],
      "geometries_included": [str, ...],
      "source": "hdf5" | "correlator" | "manual"
    }
  },
  "metadata": {
    "created_at": "ISO timestamp",
    "source_dir": "path"
  }
}
```

---

## 6. Artefacto: Reporte Fase XII

**Productores:** `make_fase12_report_from_emergent.py` (emergente) o manual (stub)  
**Consumidor:** `09_real_data_and_dictionary_contracts.py`  
**Ubicación:** `runs/fase12_*/fase12/predictions/fase12_report*.json`

### 6.1 Estructura obligatoria

```json
{
  "phase": 12,
  "description": str,
  "systems": [
    {
      "name": str,
      "source": "bootstrap" | "lattice" | "exact" | "synthetic",
      "d": int,
      "dictionary_source": "manual" | "emergent_lambda_sl_v2" | ...,
      "geometry": {
        "predicted_family": str,
        "operators_predicted": [
          {"name": str, "Delta": float, ...}
        ]
      },
      "dictionary": {
        "provenance": "manual" | "emergent",
        "operators_predicted": [...]
      }
    }
  ]
}
```

### 6.2 Semántica de `dictionary_source`

| Valor                    | Significado                                           |
|--------------------------|-------------------------------------------------------|
| `"manual"`               | Diccionario impuesto a mano. PASS = solo tubería OK.  |
| `"emergent_lambda_sl_v2"`| Diccionario inferido. PASS = evidencia de física.     |

---

## 7. Matriz Productor → Consumidor

| Productor | Artefacto                        | Consumidores       |
|-----------|----------------------------------|--------------------|
| 01        | `sandbox/*.h5`                   | 02                 |
| 02        | `geometry_emergent/*.h5`         | 03, 04, 06, 08     |
| 02        | `emergent_geometry_summary.json` | 04                 |
| 03        | `equations_pareto.json`          | 04, 05             |
| 06        | `bulk_modes_dataset.csv`         | 07                 |
| 07        | `lambda_sl_dictionary_report.json`| 09, make_fase12   |
| 08        | `holographic_dictionary_summary.json` | 04, 07, 09    |
| make_fase12| `fase12_report*.json`           | 09                 |

---

## 8. Validador: Checklist por archivo

El script `00_validate_io_contracts.py` debe verificar:

### Para sandbox HDF5:
- [ ] Existen atributos `name`, `family`, `d`, `category`
- [ ] `d` es consistente con el nombre si contiene `_d<k>_`
- [ ] Existe grupo `boundary/` con atributos `d`, `family`
- [ ] Existe grupo `bulk_truth/` con datasets `z_grid`, `A_truth`, `f_truth`
- [ ] Longitudes consistentes
- [ ] `z_grid` monótono creciente

### Para geometría emergente HDF5:
- [ ] Existen atributos `system_name`, `family`, `d`, `provenance`
- [ ] Existen datasets canónicos `z_grid`, `A_of_z`, `f_of_z`
- [ ] Si faltan canónicos pero existen legacy → WARN (solo lectura)
- [ ] Si faltan ambos → FAIL
- [ ] Longitudes consistentes
- [ ] `z_grid` monótono creciente
- [ ] Si `provenance == "inference"` → no debe existir `bulk_truth/`

### Para CSV de modos:
- [ ] Columnas obligatorias presentes: `system_name`, `family`, `d`, `mode_id`, `lambda_sl`, `Delta_UV`
- [ ] No hay columna `m2L2` como principal (solo alias)
- [ ] Valores numéricos válidos (no NaN, no inf)

### Para JSONs:
- [ ] Campos obligatorios según sección correspondiente
- [ ] `nomenclature_version` presente donde se requiera

---

## 9. Política de cambios

Cualquier modificación a este contrato requiere:

1. **Aprobación explícita** del responsable del proyecto
2. **Plan de migración** para archivos existentes
3. **Actualización del validador** antes de cualquier cambio en scripts productores
4. **Bump de versión** del contrato (v1 → v2, etc.)

---

## 10. Historial de versiones

| Versión | Fecha    | Cambios                                      |
|---------|----------|----------------------------------------------|
| v1.0    | 2025-12  | Versión inicial. Congelación post-EXP-001.   |
