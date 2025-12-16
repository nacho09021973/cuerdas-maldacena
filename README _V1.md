# CUERDAS-Maldacena  
Motor holográfico de **geometría emergente** y **diccionario λ_SL ↔ Δ** para QFT reales

Este README está pensado para que **cualquier IA** (o humano) que reciba:

- estos **10 scripts “núcleo”**:

  - `01_generate_sandbox_geometries.py`
  - `02_emergent_geometry_engine.py`
  - `03_discover_bulk_equations.py`
  - `04_geometry_physics_contracts.py`
  - `05_analyze_bulk_equations.py`
  - `bulk_scalar_solver.py`
  - `06_build_bulk_eigenmodes_dataset.py`
  - `07_emergent_lambda_sl_dictionary.py`
  - `08_build_holographic_dictionary.py`
  - `09_real_data_and_dictionary_contracts.py`

y **este único README**, pueda:

1. Entender el pipeline completo de CUERDAS-Maldacena.
2. Saber en qué entorno debe trabajar.
3. Respetar los formatos de datos críticos.
4. No romper la **honestidad científica** del proyecto.
5. Saber **qué tipo de ayuda se espera** de ella.
6. Ejecutar un **run mínimo reproducible** para comprobar compatibilidad.
7. Entender las **extensiones Fase XII/XIII** para datos reales (Ising 3D, etc.).

Adicionalmente, el repositorio puede incluir algunos scripts auxiliares:

- `fase11_ising3d_emergent_geometry.py`  
- `make_fase12_report_from_emergent.py`  
- `extended_physics_contracts_fase12_13.py`  
- (opcional) `04b_negative_control_contracts.py`  

Estos extienden el pipeline núcleo hacia **datos reales** y **contratos avanzados**, sin cambiar el corazón del flujo A–B–C.

Todo lo que antes se llamaba “Fase 8/9/11/12” se considera ahora **fase cero histórica**.  
Lo que importa es este pipeline actual en tres bloques:

- **Bloque A — Geometría emergente y ecuaciones de campo.**
- **Bloque B — Espectro escalar y dataset λ_SL.**
- **Bloque C — Diccionario holográfico y contratos con datos reales.**

Más un módulo común:

- `bulk_scalar_solver.py` — Solver escalar estilo Sturm–Liouville (λ_SL, Δ_UV).


---

## 0. Cómo usar este README (humano o IA)

Si estás leyendo esto como **IA**:

- **Lee al menos las secciones 1, 2, 3, 4 y 6 completas** antes de proponer cambios.
- **Nunca cambies** nombres de campos, estructuras de datos o semántica de etiquetas sin dejar claro el impacto.
- Cuando propongas modificaciones:
  - Indica qué scripts tocas.
  - Indica qué ficheros en `runs/` cambian de forma.
  - Mantén siempre la compatibilidad con el **run mínimo** descrito en la sección 6.

Si eres **humano**:

- Usa este documento como contrato técnico con las IAs.
- Si una IA propone algo que viola explícitamente este README, se descarta.


---

## 1. Visión global del pipeline

### 1.1. Bloques

```text
┌───────────────────────────────┐
│   BLOQUE A: GEOMETRÍA        │
│   (emergente + ecuaciones)   │
└─────────────┬────────────────┘
              │
              v
┌───────────────────────────────┐
│   BLOQUE B: ESPECTRO ESCALAR │
│   (λ_SL y dataset limpio)    │
└─────────────┬────────────────┘
              │
              v
┌───────────────────────────────┐
│   BLOQUE C: DICCIONARIO      │
│   HOLOGRÁFICO + CONTRATOS    │
│   (incluye datos reales)     │
└───────────────────────────────┘
````

### 1.2. Scripts principales (núcleo)

#### Bloque A — Geometría y ecuaciones

* `01_generate_sandbox_geometries.py`
  Genera “universos sandbox” con datos de `boundary/` + `bulk_truth/` a partir de familias tipo AdS, Lifshitz, hyperscaling, deformed, etc.

* `02_emergent_geometry_engine.py` (V2.2)
  Aprende la **geometría emergente** a partir de los datos de frontera.
  Soporta modos:

  * `--mode train`: entrena en sandbox (boundary + bulk_truth).
  * `--mode inference`: usa un checkpoint entrenado para procesar **solo boundary** (real o stub).

* `03_discover_bulk_equations.py`
  Aplica regresión simbólica (PySR) sobre la geometría emergente para descubrir ecuaciones de campo (para A, f, R, etc.), **sin imponer Einstein a priori**.

* `04_geometry_physics_contracts.py`
  Aplica contratos físicos a geometría + ecuaciones:

  * regularidad y causalidad genéricas;
  * contratos específicos de AdS (Einstein-like, asintótica AdS, diccionario compatible);
  * score global por geometría.

* `05_analyze_bulk_equations.py`
  Analiza la familia de ecuaciones descubiertas:

  * patrones universales vs específicos de familia;
  * dependencia en d, z_dyn, θ;
  * relación complejidad vs error vs clasificación física.

#### Bloque B — Espectro escalar

* `bulk_scalar_solver.py`
  Módulo de solver escalar tipo Sturm–Liouville:

  * calcula autovalores λ_SL y modos propios;
  * extrae Δ_UV cuando corresponde;
  * actúa como backend para 06.

* `06_build_bulk_eigenmodes_dataset.py`
  Recorre las geometrías emergentes, llama al solver escalar y construye un dataset limpio `(Δ_UV, λ_SL)` por sistema.

#### Bloque C — Diccionario + datos reales

* `07_emergent_lambda_sl_dictionary.py`
  Aprende un **diccionario emergente** λ_SL ↔ Δ a partir del dataset de modos (y/o diccionario v3):

  * usa PySR 1.5.9 con librería de operadores sobria;
  * detecta y filtra métodos de extracción “sospechosos” (e.g. usos analíticos de Δ(Δ−d));
  * produce un reporte con la mejor ecuación, métricas de test y comparación **post-hoc** con Δ(Δ−d).

* `08_build_holographic_dictionary.py`
  Construye un **atlas holográfico interno** (`holographic_dictionary_summary.json`) a partir de:

  * geometrías de 02;
  * operadores extraídos de correladores o de atributos HDF5 (modo control/emergent);
  * opcionalmente, redescubre la relación masa-dimensión (m²L² vs Δ) usando PySR como diagnóstico.

* `09_real_data_and_dictionary_contracts.py`
  Punto de encuentro entre:

  * diccionario emergente λ_SL ↔ Δ;
  * atlas holográfico interno;
  * sistemas físicos reales (Ising 3D, O(N), QCD-like, CMB, ...).

  Aplica contratos de Fase XII/XIII:

  * coherencia Δ_predicho vs Δ_bootstrap/lattice;
  * interpretación honesta según `dictionary_source` / `provenance`;
  * contratos extendidos importados de `extended_physics_contracts_fase12_13.py`.

### 1.3. Scripts auxiliares (extensiones Fase XI/XII/XIII)

Estos scripts **no forman parte del “núcleo mínimo”**, pero extienden el pipeline a datos reales:

* `fase11_ising3d_emergent_geometry.py`
  Wrapper para ejecutar 02 en modo `inference` sobre **Ising 3D real/sintético**:

  * Entrada: `fase12_data_boundary/manifest.json` + `ising_3d.h5`.
  * Salida: `runs/fase12_ising_real/fase11_geometry_from_real/geometry_emergent/*.h5`.

* `make_fase12_report_from_emergent.py`
  Construye un reporte tipo Fase XII para Ising 3D a partir de:

  * `bulk_modes_dataset_ising.json` (salida de 06 para Ising);
  * `lambda_sl_dictionary_report.json` (salida de 07);
  * descriptor opcional `ising3d_descriptor.json`.

  Salida: un `fase12_report.json` con `dictionary_source != "manual"` y `operators_predicted` rellenos usando el diccionario emergente.

* `extended_physics_contracts_fase12_13.py`
  Implementa **contratos extendidos** de Fase XII (datos reales) y Fase XIII (exploración de teorías):

  * `ExtendedContractsFase12`:

    * contratos de exponentes críticos (ν, η, α, etc.) con tolerancia 3σ y relaciones de hyperscaling;
    * torre de operadores (identidad + primario + descendientes) con bounds de unitariedad adaptados a Ising 3D;
    * contratos de gap espectral para teorías críticas vs gappeadas.

  * `ExtendedContractsFase13`:

    * contratos de suavidad, continuidad y coherencia entre puntos de una “nube” de teorías en el espacio de parámetros.

  Este módulo es consumido por `09_real_data_and_dictionary_contracts.py`.

* (Opcional) `04b_negative_control_contracts.py`
  Script para **controles negativos** sobre geometría y ecuaciones (por ejemplo, datasets mezclados o corruptos) para verificar que los contratos de 04 detectan situaciones “malas” como deben.
  Si este archivo está presente, se considera complementario a 04 pero no forma parte del run mínimo.

---

## 2. Entorno y dependencias (ENVIRONMENT_AND_DEPENDENCIES)

Esta sección NO es un “pin” rígido, sino la **fotografía del entorno actual** en el que se ha probado CUERDAS-Maldacena.
Cualquier IA que proponga cambios debe mantener la compatibilidad con este entorno o proponer un plan claro de migración.

### 2.1. Cómo obtener las versiones actuales

Antes de tocar nada, el procedimiento correcto es:

```bash
python - << 'EOF'
import importlib

pkgs = ["numpy", "scipy", "pandas", "h5py", "torch", "pysr"]
for name in pkgs:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{name} == {ver}")
    except ImportError:
        print(f"{name} NOT INSTALLED")
EOF
```

Y, si hace falta, mirar también `juliacall`:

```bash
python - << 'EOF'
try:
    import juliacall
    print("juliacall == ", juliacall.__version__)
except ImportError:
    print("juliacall NOT INSTALLED")
EOF
```

### 2.2. Estado actual de referencia

A día de hoy, el entorno de referencia de CUERDAS-Maldacena es (según la instalación actual):

* Python: 3.11.x
* numpy: (ver comando anterior)
* scipy: (ver comando anterior)
* pandas: (ver comando anterior)
* h5py: (ver comando anterior)
* torch: 2.9.1+cu128 en la máquina con GPU (en otras máquinas puede no estar instalado o estar en versión CPU)
* **pysr: 1.5.9**
* **juliacall: 0.9.26**

> NOTA PARA IAs:
> Con `pysr == 1.5.9` se asume la **API moderna**, es decir:
>
> * `from pysr import PySRRegressor`
> * parámetros como `elementwise_loss`, `nested_constraints`, `parallelism="serial"`,
>   `deterministic=True`, `random_state=seed`, `batching`, `batch_size`, etc.
>   No uséis ejemplos basados en APIs obsoletas.

### 2.3. GPU vs CPU

* **Regla dura**: el pipeline debe poder ejecutarse **íntegramente en CPU**.

  * Ningún script puede “asumir” que hay GPU.
  * Cualquier uso de GPU debe ser opcional y controlado:

    * bandera `--device cpu|cuda` o lógica equivalente.

* Uso de GPU permitido (pero no obligatorio) en:

  * entrenamiento de redes en `02_emergent_geometry_engine.py`,
  * entrenamiento/búsqueda en `07_emergent_lambda_sl_dictionary.py`.

En consecuencia:

* IAs pueden sugerir optimizaciones GPU, pero:

  * Deben mantener un **camino claro CPU-only**.
  * Deben documentar cómo seleccionar `--device cpu` sin sorpresas.

### 2.4. Reglas para cambios de versión

Para preservar estabilidad y reproducibilidad:

1. **No cambiar versiones mayores** de Python, PySR, JuliaCall o PyTorch sin:

   * Explicar el beneficio concreto (bug crítico, feature imprescindible).
   * Diseñar un **test canario** (típicamente el “run mínimo” de la sección 6).

2. Si se propone un cambio de versión de PySR:

   * Comprobar que:

     * `PySRRegressor` sigue existiendo.
     * Los parámetros usados en los scripts siguen siendo válidos.

3. Si se introduce una librería nueva:

   * Debe ser:

     * Justificada (no “por comodidad”).
     * Opcional si es pesada.

   * No se aceptan frameworks gigantes si solo aportan azúcar sintáctico.

Prioridades:

1. El pipeline corre en CPU con el entorno actual.
2. Los formatos de datos descritos en la sección 3 siguen siendo válidos.
3. El run mínimo (sección 6) sigue funcionando sin cambios en la línea de comandos.

---

## 3. Formatos de datos (DATA_FORMATS)

**Contrato estricto recomendado:** ver `IO_CONTRACTS_V1.md`.
Para validación automática (read-only), use `00_validate_io_contracts.py` antes de ejecutar el resto del pipeline.

**IMPORTANTE PARA IAs:**
No cambiar nombres de campos ni tipos básicos sin señalar explícitamente el impacto y cómo mantener compatibilidad.

A continuación se describen los formatos **lógicos** de los ficheros clave.
No es un contrato rígido de todos los campos, pero sí de los campos mínimos esperados.

### 3.1. `runs/sandbox_geometries/boundary/*.h5`

Fichero HDF5 por sistema “sandbox”.

**Atributos raíz mínimos**:

* `system_name: str`
* `family: str` (`"ads"`, `"lifshitz"`, `"hvlf"`, `"deformed"`, …)
* `d: int`
* `z_dyn: float` (si aplica)
* `theta: float` (si aplica)

**Datasets típicos**:

* `x_grid`: 1D array.
* `temperature`: 1D array o escalar repetido.
* `omega_grid`: 1D array (frecuencias).
* `k_grid`: 1D array (momentos).
* `G_R_real`, `G_R_imag`: correlador retarded (parte real/imag).
* Otros correladores `G2_<O>` según diseño.

Uso: **entrada principal** para `02_emergent_geometry_engine.py`.

### 3.2. `runs/sandbox_geometries/bulk_truth/*.h5`

Geometría “de verdad” para sandbox.

**Atributos raíz mínimos**:

* `system_name`, `family`, `d`, `z_dyn`, `theta`, `z_h`, …

**Datasets típicos**:

* `z_grid`: 1D array.
* `A_truth(z)`, `f_truth(z)`, `R_truth(z)`.

Uso: **solo** en contratos 04 y análisis (no entra en la loss).

### 3.3. `runs/emergent_geometry/geometry_emergent/*.h5`

Geometría reconstruida por 02.

**Atributos mínimos**:

* `system_name`
* `family_pred`
* `d_pred` (si existe)
* otros metadatos de entrenamiento.

**Datasets típicos**:

* `z_grid`
* `A_emergent`
* `f_emergent`
* `R_emergent`
* opcionalmente un grupo `boundary/` con datos resumidos.

Uso:

* Input para 03 (ecuaciones).
* Input para 06 (espectro escalar).
* Input para 08 (atlas holográfico).

### 3.4. `runs/emergent_geometry/emergent_geometry_summary.json`

Resumen global de 02.

Campos mínimos:

* `n_train`, `n_test`

* `train_metrics` y `test_metrics` con:

  * `A_r2`, `f_r2`, `R_r2`, etc.

* Por sistema:

  * `system_name`, `family_pred`, métricas individuales.

Uso: contratos 04.

### 3.5. `runs/bulk_equations/equations_pareto.json`

Resumen de ecuaciones halladas por 03.

Esquema lógico:

```json
{
  "geometries": [
    {
      "name": "ads_d3_000",
      "family": "ads",
      "equations": [
        {
          "target": "R_scalar",
          "expression": "(-20.0 * square(x2)) - (10.0 * x3)",
          "complexity": 7,
          "loss": 1.23e-3,
          "r2": 0.998
        }
      ]
    }
  ],
  "pysr_config": { ... }
}
```

Uso: 04 (contratos) y 05 (análisis).

### 3.6. `runs/geometry_contracts/geometry_contracts_summary.json`

Salida de 04.

Campos típicos:

* Por sistema (`PhaseXIContractV2`):

  * `name`, `family`, `category`, `d`
  * sub-bloques de contratos (regularidad, causalidad, AdS, holográfico)
  * métricas de reconstrucción (`A_r2`, `f_r2`, `R_r2`, `family_accuracy`)
  * `contract_score`, `overall_passed`.

Uso: diagnóstico global y consumo por 05.

### 3.7. `runs/bulk_eigenmodes/bulk_modes_dataset.csv`

Dataset central para 06/07.

Columnas **mínimas**:

* `system_name: str`
* `family: str`
* `d: int`
* `z_dyn: float` (si aplica)
* `theta: float` (si aplica)
* `mode_id: int`
* `lambda_sl: float`
* `Delta_UV: float`

Columnas extra permitidas:

* `is_ground_state: bool`
* `norm: float`
* banderas de calidad, etc.

Uso: input directo de 07.

Además, para Ising 3D, 06 puede escribir también un JSON tipo:

* `bulk_modes_dataset_ising.json`

con estructura compatible con los loaders de 07 y de `make_fase12_report_from_emergent.py`.

### 3.8. `runs/emergent_dictionary/lambda_sl_dictionary_report.json`

Salida de 07.

Esquema lógico:

```json
{
  "config": { ... },
  "discovery_results": {
    "best_equation": "expression string",
    "test_metrics": {
      "r2": 0.97,
      "mae": 0.02,
      "pearson": 0.99
    }
  },
  "theory_comparison": {
    "theory_r2": 0.95,
    "compatible_with_maldacena": false
  },
  "data_stats": {
    "Delta_range": [0.5, 4.0],
    "lambda_sl_range": [...]
  }
}
```

Uso principal: 09 y `make_fase12_report_from_emergent.py`.

### 3.9. `runs/holographic_dictionary/holographic_dictionary_summary.json`

Atlas interno generado por 08.

Estructura típica:

```json
{
  "by_system": {
    "ads_d3": {
      "family": "ads",
      "d": 3,
      "n_points": 12,
      "Delta": [0.52, 1.41, ...],
      "m2L2_emergent": [...],
      "geometries_included": ["ads_d3_000", "ads_d3_001"],
      "source": "hdf5"
    }
  },
  "discoveries": { ... }
}
```

Uso: 07 (en modo `dictionary_v3`) y 09.

### 3.10. Ejemplo de reporte Fase XII (sistema real)

Para Ising 3D ya adaptado:

```json
{
  "phase": 12,
  "description": "...",
  "systems": [
    {
      "name": "ising3d_bootstrap",
      "source": "bootstrap",
      "d": 3,
      "T": 0.0,
      "geometry": {
        "predicted_family": "ads_d3_bootstrap_stub",
        "operators_predicted": [
          { "name": "sigma",   "Delta": 0.518, "..." },
          { "name": "epsilon", "Delta": 1.41,  "..." }
        ]
      },
      "dictionary": {
        "provenance": "manual",
        "operators_predicted": [
          { "name": "sigma",   "Delta": 0.518, "..." },
          { "name": "epsilon", "Delta": 1.41,  "..." }
        ]
      },
      "dictionary_source": "manual"
    }
  ]
}
```

Puntos clave:

* `dictionary.provenance` y `dictionary_source` controlan la interpretación (manual vs emergente).
* Mientras sean `"manual"`, los contratos de 09 solo validan la **tubería técnica**, no descubrimientos.

Cuando se usa `make_fase12_report_from_emergent.py`, el mismo formato se rellena con:

* `dictionary_source`: algo tipo `"emergent_lambda_sl_v2"`.
* `operators_predicted`: generados a partir de λ_SL observados + ecuación emergente de 07.

### 3.11. `runs/real_data_contracts/real_data_and_dictionary_contracts_summary.json`

Salida de 09.

Esquema lógico:

* Por sistema real (ej. Ising 3D):

  * `system_name`
  * `source` (bootstrap, lattice, …)
  * `contracts`:

    * p.ej. `ising3d_consistency`, `critical_exponents`, `operator_tower`, `spectral_gap`, etc.
    * `status`: `"PASS"` / `"FAIL"`
    * detalles numéricos (diferencias de Δ, nσ, gaps, bounds de unitariedad).

Uso: evaluación final del pipeline contra datos del mundo real.

---

## 4. Honestidad y contratos (HONESTY_AND_CONTRACTS)

Esta sección es la “constitución” del proyecto.
Cualquier IA debe respetar estos puntos.

### 4.1. Principio 1 — No inyectar teoría en la loss

Fórmulas teóricas como:

[
m^2 L^2 = \Delta(\Delta - d)
]

u otras relaciones “de libro”:

* **NO** pueden ser utilizadas:

  * como parte de la **función de pérdida**;
  * como **features** de entrada;
  * para “regularizar” modelos en `02`, `03`, `06`, `07`.

Solo pueden usarse:

* en scripts de:

  * análisis,
  * diagnóstico,
  * contratos (`04`, `08`, `09`, `extended_physics_contracts_fase12_13.py`),
* siempre marcadas explícitamente como **checks post-hoc**.

### 4.2. Principio 2 — Separación sandbox vs real

* **Sandbox**:

  * Existe `bulk_truth`.
  * Se puede usar en contratos (04) para comprobar:

    * R²,
    * derivadas,
    * Einstein-like vs non-Einstein.

* **Real**:

  * No hay `bulk_truth`.
  * Solo hay datos de referencia (bootstrap, lattice, observaciones).
  * Los contratos en 09/extended:

    * comparan diccionario emergente ↔ datos de referencia;
    * nunca contra una métrica exacta que el modelo haya visto.

### 4.3. Principio 3 — Contratos como juez, no como entrenador

Scripts de contratos:

* `04_geometry_physics_contracts.py`
* `09_real_data_and_dictionary_contracts.py`
* `extended_physics_contracts_fase12_13.py`
* (opcional) `04b_negative_control_contracts.py`

Su papel:

* Leer outputs ya entrenados.
* Evaluar si cumplen ciertas condiciones físicas y de honestidad.
* Reportar PASS/FAIL, scores, etc.

**Nunca** deben:

* reentrenar modelos;
* ajustar pesos;
* modificar datasets en función del resultado de contratos.

### 4.4. Etiquetas de `provenance` y `dictionary_source`

Valores típicos:

* `"manual"`:

  * Diccionario impuesto a mano (p.ej. σ, ε del bootstrap en Ising 3D).
  * Cualquier PASS implica “solo tubería técnica OK”.

* `"emergent_lambda_sl_v2"` o similar:

  * Diccionario inferido a partir de:

    * geometría emergente (bloque A),
    * espectro escalar (bloque B),
    * ecuaciones descubiertas (PySR),
    * contratos de honestidad.

Contratos en 09 + extendidos deben:

* interpretar los resultados en función de estas etiquetas;
* no vender como “descubrimiento físico” algo que venga de `"manual"`.

### 4.5. Cosas explícitamente prohibidas para IAs

* Introducir `Δ(Δ−d)` (u otras fórmulas de diccionario conocidas) en:

  * `02_emergent_geometry_engine.py`
  * `03_discover_bulk_equations.py`
  * `06_build_bulk_eigenmodes_dataset.py`
  * `07_emergent_lambda_sl_dictionary.py`

* Cambiar silenciosamente:

  * nombres de campos críticos (`Delta_UV`, `lambda_sl`, `system_name`, `family`, `d`, …);
  * estructura de ficheros en `runs/` sin plan de migración.

* Mezclar:

  * datos de sandbox con datos reales dentro del mismo dataset sin etiquetas claras.

* Forzar resultados “bonitos”:

  * ajustar contratos para que siempre pasen;
  * “tunear” umbrales sólo para hacer pasar un caso concreto sin justificación física.

---

## 5. Guía para IAs colaboradoras (AI_COLLAB_GUIDE)

### 5.1. Rol esperado de las IAs

Las IAs que colaboran en CUERDAS-Maldacena NO están para “descubrir física mágicamente”.
Su aportación es:

* **Ingeniería**:

  * refactorizar código sin romper IO;
  * mejorar legibilidad y estructura;
  * añadir tests / validaciones.

* **Metodología**:

  * diseñar contratos más completos y honestos;
  * proponer experimentos adicionales (nuevos universos sandbox, nuevos tipos de ruido).

* **Asistencia científica**:

  * ayudar a buscar literatura relevante;
  * sugerir nuevas validaciones en bloque C (p.ej. nuevos exponentes o gaps para Ising, O(N), QCD).

### 5.2. Tareas que SÍ se desean

Ejemplos:

* **Código / ingeniería**:

  * Limpiar funciones largas en 02, 03, 06, 07.
  * Añadir logs más claros (pero no excesivos).
  * Proponer tests unitarios (e.g. para `bulk_scalar_solver.solve`).

* **Validación**:

  * Nuevos contratos en 04/09/extended:

    * condiciones físicas adicionales;
    * checks de estabilidad y sensibilidad a ruido.

* **Análisis**:

  * Visualizaciones o resúmenes de:

    * ecuaciones en 05;
    * diccionario emergente en 07;
    * atlas en 08;
    * contratos reales en 09.

* **Literatura**:

  * Localizar valores estándar para nuevos sistemas de referencia:

    * O(N) 3D,
    * teorías QCD-like,
    * strange metals,
    * CMB, etc.

### 5.3. Tareas que NO deben hacer sin aprobación explícita

* Cambiar **formatos de fichero**:

  * renombrar columnas/keys existentes;
  * cambiar `.csv → .parquet` o `.json → .yaml` sin capa de compatibilidad.

* Cambiar **semántica** de:

  * `provenance`,
  * `dictionary_source`,
  * etiquetas de contratos.

* Introducir:

  * nuevas dependencias pesadas;
  * cambios de backend numérico (por ejemplo, pasar todo a JAX) sin plan.

Cualquier cambio de este tipo debe venir acompañado de:

1. explicación explícita,
2. plan de migración incremental,
3. modo “legacy” que siga funcionando.

### 5.4. Terreno donde se quiere creatividad

* Extender el solver escalar a:

  * campos con spin > 0;
  * otros tipos de potencial.

* Diseñar adaptadores para:

  * modelos O(N),
  * teorías QCD-like,
  * sistemas de materia condensada (strange metals),
  * CMB (Planck, etc.).

* Proponer nuevos **contratos Fase XIII** para explorar nubes de teorías y encontrar outliers “raros”.

---

## 6. Run mínimo reproducible (MINIMAL_RUN)

Esta sección define un flujo **mínimo** que debe seguir funcionando después de cualquier cambio importante.

### 6.1. Run mínimo sandbox (A → B → C técnico)

1. **Generar sandbox**

   ```bash
   python 01_generate_sandbox_geometries.py \
     --output-dir runs/sandbox_geometries \
     --n-known 3 --n-test 2 --n-unknown 1
   ```

2. **Geometría emergente (entrenamiento)**

   ```bash
   python 02_emergent_geometry_engine.py \
     --data-dir runs/sandbox_geometries \
     --output-dir runs/emergent_geometry \
     --n-epochs 200 \
     --device cpu \
     --mode train \
     --seed 42
   ```

3. **Descubrir ecuaciones de bulk**

   Se asume que 02 escribe predicciones en algo tipo `runs/emergent_geometry/predictions/*.npz`.

   ```bash
   python 03_discover_bulk_equations.py \
     --geometry-dir runs/emergent_geometry \
     --output-dir runs/bulk_equations \
     --d 4 \
     --niterations 50 \
     --maxsize 12
   ```

4. **Contratos de geometría (Fase XI honesta)**

   Requiere:

   * `runs/sandbox_geometries/manifest.json`
   * `runs/emergent_geometry/emergent_geometry_summary.json`
   * `runs/bulk_equations/einstein_discovery_summary.json` (o equivalente)
   * diccionario v3 de 08 (para campos de masa, si aplica).

   Ejemplo:

   ```bash
   python 04_geometry_physics_contracts.py \
     --data-dir runs/sandbox_geometries \
     --geometry-dir runs/emergent_geometry \
     --einstein-dir runs/bulk_equations \
     --dictionary-file runs/holographic_dictionary/holographic_dictionary_summary.json \
     --output-file runs/geometry_contracts/geometry_contracts_summary.json \
     --d 4
   ```

5. **Análisis de ecuaciones (opcional, pero recomendado)**

   ```bash
   python 05_analyze_bulk_equations.py \
     --input runs/bulk_equations/einstein_discovery_summary.json \
     --output runs/bulk_equations_analysis/bulk_equations_report.json
   ```

6. **Dataset de modos de bulk**

   ```bash
   python 06_build_bulk_eigenmodes_dataset.py \
     --geometry-dir runs/emergent_geometry/geometry_emergent \
     --output-csv runs/bulk_eigenmodes/bulk_modes_dataset.csv
   ```

7. **Diccionario λ_SL ↔ Δ (emergente)**

   Primero, asegurar que los datos están en formato compatible (`bulk_modes_dataset.csv` o un JSON intermedio).
   Ejemplo usando un JSON derivado:

   ```bash
   # (si se dispone de un .json con nomenclatura v2_lambda_sl o dictionary_v3)
   python 07_emergent_lambda_sl_dictionary.py \
     --input-file runs/bulk_eigenmodes/bulk_modes_dataset_v2.json \
     --output-dir runs/emergent_dictionary \
     --iterations 200 \
     --seed 42 \
     --ops-minimal
   ```

8. **Atlas holográfico interno**

   ```bash
   python 08_build_holographic_dictionary.py \
     --data-dir runs/emergent_geometry/geometry_emergent \
     --output-summary runs/holographic_dictionary/holographic_dictionary_summary.json \
     --mass-source hdf5 \
     --compute-m2-from-delta
   ```

9. **Contratos con datos reales (stub manual)**

   Como run mínimo, se puede usar un reporte Fase XII stub con `dictionary_source="manual"`.

   ```bash
   python 09_real_data_and_dictionary_contracts.py \
     --phase 12 \
     --fase12-report runs/fase12_ising_real/fase12/predictions/fase12_report_stub.json \
     --output-file runs/real_data_contracts/contracts_fase12_stub.json
   ```

   El objetivo aquí es solo comprobar que:

   * 09 ejecuta sin error;
   * los contratos etiquetan correctamente el caso “manual” como **validación técnica**, no física.

### 6.2. Extensión recomendada: pipeline Ising 3D emergente (A_real → B → C_real)

Este run no es obligatorio para todas las IAs, pero es la **meta natural** del proyecto:

1. Preparar `fase12_data_boundary/` (adaptador real fuera de este README).

2. Ejecutar:

   ```bash
   python fase11_ising3d_emergent_geometry.py \
     --data-dir fase12_data_boundary \
     --output-dir runs/fase12_ising_real/fase11_geometry_from_real \
     --n-epochs 2000 \
     --device cpu
   ```

3. Construir modos de bulk para Ising:

   ```bash
   python 06_build_bulk_eigenmodes_dataset.py \
     --geometry-dir runs/fase12_ising_real/fase11_geometry_from_real/geometry_emergent \
     --output-csv runs/fase12_ising_real/bulk_modes_dataset_ising.csv \
     --output-json runs/fase12_ising_real/bulk_modes_dataset_ising.json
   ```

4. Aprender diccionario emergente con 07 (usando sandbox + Ising si se desea).

5. Construir reporte Fase XII emergente:

   ```bash
   python make_fase12_report_from_emergent.py \
     --bulk-modes runs/fase12_ising_real/bulk_modes_dataset_ising.json \
     --dictionary-summary runs/emergent_dictionary/lambda_sl_dictionary_report.json \
     --output runs/fase12_ising_real/fase12/predictions/fase12_report_emergent.json \
     --descriptor fase12_data_boundary/ising3d_descriptor.json
   ```

6. Ejecutar contratos reales con diccionario emergente:

   ```bash
   python 09_real_data_and_dictionary_contracts.py \
     --phase 12 \
     --fase12-report runs/fase12_ising_real/fase12/predictions/fase12_report_emergent.json \
     --output-file runs/fase12_ising_real/contracts_fase12_emergent.json
   ```

En este escenario:

* un PASS ya no es solo “tubería técnica correcta”;
* empieza a ser evidencia de que el diccionario emergente λ_SL ↔ Δ está alineado con física real (dentro de los umbrales de contratos extendidos).

---

## 7. Resumen

* Este README define el **contrato técnico y ético** del proyecto CUERDAS-Maldacena.
* El núcleo son los 10 scripts listados al inicio; los demás son extensiones honestas hacia datos reales.
* Cualquier cambio debe respetar:

  1. ejecución en CPU,
  2. formatos de datos,
  3. run mínimo sandbox,
  4. reglas de honestidad (no inyectar teoría en la loss).

Si una IA respeta todo lo anterior, es bienvenida a ayudar a:

* limpiar código,
* endurecer contratos,
* extender el atlas holográfico,
* empujar el diccionario emergente λ_SL ↔ Δ hacia QFT reales más allá de Ising 3D.

