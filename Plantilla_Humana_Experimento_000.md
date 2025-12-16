# EXP-0001 — [título corto]
**Fecha:** 2025-12-12  
**Fase(s):** XI / XII / Diccionario λ_SL↔Δ  
**Estado:** 🔄 en curso | ✅ interesante | ⚪ neutro | ❌ descartado | 🐞 bug

## 1. Configuración experimental

- **Scripts implicados:**  
  - `02_emergent_geometry_engine.py` (modo: `train` / `inference`)  
  - `06_build_bulk_eigenmodes_dataset.py`  
  - `07_emergent_lambda_sl_dictionary.py`  
- **Checkpoint geometría:**  
  - Ruta: `runs/fase11_sandbox/emergent_geometry_model.pt`  
- **Datos de entrada:**
  - Sandbox / Real: `sandbox_ads+lif` / `Ising3D_stub` / etc.
  - Ficheros: `runs/.../sandbox_geometries.h5`, `runs/.../ising_stub.json`, ...
- **Semilla(s) aleatorias:** `seed=1234` (PyTorch, numpy, etc.)
- **Parámetros clave:**
  - `n_geometries_train=60`, `n_geometries_test=30`
  - `pysr_niterations=...`, `population_size=...`
  - `max_degree`, `max_depth`, etc.

## 2. Objetivo del experimento

- ¿Qué queríamos comprobar?  
  - Ejemplo: “Ver si con pocos puntos (φ libre + Ising3D σ, ε′, σ′) PySR encuentra *alguna* relación razonable λ_SL↔Δ que pase Fase XII (stub).”
  - Tipo de hipótesis:
    - [ ] Técnica (pipeline / bugs)
    - [ ] Numérica (estabilidad / ruido)
    - [x] Física (estructura de diccionario, dependencia en d, etc.)

## 3. Resultados principales

- **Fórmula(s) descubiertas (resumen humano):**
  - `Δ ≈ 1/2 + 1.50 √λ + (400/9) * λ² / (λ - π²/(200 d))` (fórmula emergente candidata)
- **Métricas relevantes:**
  - RMSE diccionario: `≈ 2e-6` sobre {φ libre, σ, ε′, σ′ (Ising3D)}
  - Contratos Fase XII: `PASS (1/1)`  
- **Archivos generados relevantes:**
  - `runs/fase12_ising_real/fase12/predictions/fase12_report.json`
  - `runs/fase12_ising_real/dictionary/pysr_run_0001.json`

## 4. Interpretación (breve y honesta)

- **Lectura rápida:**
  - “La fórmula clava los 4 puntos casi a precisión máquina, pero tiene 3 parámetros + estructura → riesgo fuerte de overfitting.”
  - “La estructura √λ + polo sugiere conexión natural con m²L²=Δ(Δ−d) deformado.”
- **Limitaciones:**
  - Número de puntos muy pequeño.
  - Solo d=3 (más un punto trivial φ libre).
  - No test serio en d=2 ni en otros modelos (O(N), etc.).
- **Clasificación del experimento:**
  - [x] “Funcionamiento del pipeline” validado.
  - [x] “Señal interesante, pero no concluyente.”
  - [ ] “Descubrimiento físico consolidado.”

## 5. Siguientes pasos derivados de este experimento

- [ ] Correr experimento análogo con Ising 2D (cuando tengamos λ_SL emergentes).
- [ ] Probar ansatz general en d con (d−2)/2 + √λ + Padé y más puntos.
- [ ] Definir contrato post-hoc WF para comparar con expansiones ε→0.

## 6. Notas adicionales / anécdotas

- Profundidad personal / contexto:  
  > “Este fue el primer experimento donde la máquina devolvió una ley emergente λ_SL↔Δ para datos reales (Ising 3D stub). Se considera un hito técnico de CUERDAS 2025, aunque aún no es una ley física confirmada.”
