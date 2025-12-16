# EXP-0001 — Solver Sturm–Liouville: estabilidad y dataset de modos (baseline)
**Fecha:** 2025-12-12  
**Fase(s):** XI → Bloque B (espectro escalar)  
**Estado:** 🔄 en curso | ✅ interesante | ⚪ neutro | ❌ descartado | 🐞 bug

## 1. Configuración experimental

- **Scripts implicados:**
  - `bulk_scalar_solver.py` (cálculo de autovalores λ_SL y estimación de Δ_UV)
  - `06_build_bulk_eigenmodes_dataset.py` (agregación en dataset “bulk_modes_summary.json”)

- **Datos de entrada:**
  - Directorio de geometrías: `runs/fase11_sandbox/fase11_output_v2/data`
  - Ejemplos: `.../<geom_1>.h5`, `.../<geom_2>.h5`
  - Datasets HDF5 usados (si aplica): `bulk_truth/z_grid`, `bulk_truth/A_truth`, `bulk_truth/f_truth`

- **Parámetros clave:**
  - `n_eigs = 6`
  - `discard_negative = true`
  - Estimación UV: ajuste de potencia en el primer `20%` del grid radial (frac_uv=0.2)

## 2. Objetivo del experimento

Validar la capa más baja del pipeline del diccionario:

1) Que el solver produce espectros **numéricamente razonables**:
   - λ_SL positivos (tras descartar negativos numéricos).
   - orden creciente de autovalores.
   - Δ_UV estimable (no-None) en una fracción significativa de modos.

2) Que el constructor de dataset:
   - exporta `lambda_sl_bulk` + `Delta_bulk_uv`
   - y reporta “nomenclature_version=v2_lambda_sl” sin depender de claves legacy.

Tipo de hipótesis:
- [x] Técnica (pipeline / IO / consistencia)
- [x] Numérica (estabilidad básica)
- [ ] Física (diccionario λ↔Δ aún NO)

## 3. Resultados principales (rellenar tras correr)

- **Resumen solver (por geometría):**
  - Geometría: <geom_1>
    - λ_SL: [...]
    - Δ_UV: [...]
    - Observaciones: (p.ej. “modo 0 sin Δ_UV fiable”)
  - Geometría: <geom_2>
    - λ_SL: [...]
    - Δ_UV: [...]
    - Observaciones: ...

- **Métricas agregadas (dataset):**
  - n_geometries_processed: ...
  - n_geometries_solver_failed: ...
  - fraction_modes_with_finite_Delta_uv: ...
  - compat_used_keys: [...]
  - all_v2_clean: true/false

- **Archivos generados relevantes:**
  - `runs/experiments/EXP-0001/solver_outputs/solver_<geom>.json`
  - `runs/experiments/EXP-0001/bulk_modes_summary.json`

## 4. Interpretación (breve y honesta)

- Si λ_SL sale estable y el dataset se construye limpio:
  - “El Bloque B es funcional y trazable. La máquina produce observables internos (λ_SL, Δ_UV) con consistencia mínima.”
- Si Δ_UV falla mucho (muchos None):
  - “La estimación UV es frágil con el layout actual (BC Dirichlet / grid / normalización). No es un fallo físico; es un punto a robustecer.”

## 5. Siguientes pasos derivados

- [ ] EXP-0002: sensibilidad a resolución (submuestreo del grid radial y comparación de λ_SL).
- [ ] EXP-0003: mismo test pero sobre geometría emergente desde Ising (modo inference) cuando el Bloque B esté estable.
- [ ] EXP-0004: solo entonces, empezar diccionario λ_SL↔Δ con más puntos y evaluación honesta (no overfitting).

## 6. Notas adicionales

- Nota de honestidad operativa:
  - λ_SL son autovalores Sturm–Liouville; no se interpretan como masas holográficas automáticamente.
