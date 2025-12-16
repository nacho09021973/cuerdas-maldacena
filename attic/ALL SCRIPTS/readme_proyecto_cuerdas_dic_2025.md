# Proyecto CUERDAS — Resumen para ingeniería

## 1. Dónde estábamos

### 1.1. Fases I–X: entrenamiento en sandbox

- Trabajábamos con **toy models holográficos** (AdS4/CFT3, etc.).
- Pipeline principal:
  - **KAN** para aproximar funciones.
  - **PySR** para destilar ecuaciones simbólicas.
- Objetivo: comprobar que el stack ML es capaz de **redescubrir leyes conocidas** antes de buscar física nueva.

### 1.2. Fase XI: motor de geometría emergente

- A partir de datos “de frontera” (CFT), el sistema aprende:
  - Una **geometría de bulk** emergente.
  - Ecuaciones de movimiento efectivas.
- Había un módulo de **diccionario holográfico** que conectaba:
  - Parámetros del boundary (Δ, d, etc.).
  - Con una cantidad de bulk que llamábamos **m2L2**.

### 1.3. El problema (antes del giro)

- El solver numérico de modos escalares resolvía un **problema de valores propios** (operador Sturm–Liouville).
- Nosotros llamábamos directamente a esos autovalores **m2L2**, como si ya fueran masas holográficas.
- El diccionario intentaba aprender la relación Δ, d → m2L2, pero:
  - El nombre y algunos supuestos internos ya sugerían la fórmula de Maldacena:
    \[
    m^2 L^2 = \Delta(\Delta - d)
    \]
- Riesgo: **auto-confirmación**.
  - El sistema podía parecer que “descubre” algo que en realidad le estamos insinuando vía nomenclatura y diseño.

---

## 2. El giro que hemos pegado

### 2.1. Cambio de contrato: de “masas” a autovalores honestos

Antes:
- El solver devolvía algo etiquetado como `m2L2`.
- El diccionario apuntaba a “reconstruir” ese m2L2.

Ahora:
- El solver se redefine como **módulo numérico puro**, ajeno a la interpretación holográfica.
- Los autovalores se llaman:

  - `lambda_sl` (λ_SL) → **lo que realmente calcula el solver**.
  - `m2L2_legacy` → alias solo para compatibilidad con datos antiguos.

**Idea clave**:  
El solver solo calcula **eigenvalues**.  
Interpretar esos eigenvalues como “masas m²L² de Maldacena” es trabajo de otra capa, y solo después del ajuste.

### 2.2. Nuevos scripts y separación de responsabilidades

Hemos introducido tres piezas limpias:

1. **`bulk_scalar_solver_v2.py`**
   - Input: geometría de bulk (rejillas z, A(z), f(z)).
   - Output:
     - Espectro `lambda_sl`.
     - `m2L2_legacy` solo como alias de compatibilidad.
     - Exponentes UV.
   - No “sabe” nada de Maldacena ni de m²L² como concepto físico.

2. **`make_fase11_bulk_for_fase12c_v2.py`**
   - Puente entre **Fase XI** y **Fase XII.c**.
   - Funciones:
     - Lee geometrías generadas en XI (formato nuevo o legacy).
     - Llama a `bulk_scalar_solver_v2`.
     - Construye un dataset emergente con:
       - `lambda_sl_bulk`.
       - `lambda_source` (fuente: solver v2, legacy, etc.).
     - Marca el dataset con `nomenclature_version = "v2_lambda_sl"` para trazabilidad.

3. **`fase12c_emergent_dictionary_v2.py`**
   - Toma como **target de PySR**:
     - `lambda_sl_emergent` (no `m2L2_emergent`).
   - Objetivo del ajuste:
     \[
     \lambda_\text{SL} = f(\Delta, d, \ldots)
     \]
   - La comparación con Maldacena:
     \[
     \lambda_\text{theory} = \Delta(\Delta - d)
     \]
     se hace:
     - **A posteriori**, como check.
     - Nunca como input del entrenamiento.

En términos de arquitectura:

- Antes:
  - Solver + diccionario + teoría estaban **acoplados** (nombres y supuestos físicos mezclados con el código numérico).
- Ahora:
  - Hay una separación clara:
    - Capa numérica → λ_SL.
    - Capa de datos → construcción del dataset XI→XII.c.
    - Capa ML → ajuste simbólico con PySR sobre λ_SL.
    - Capa física → validación comparando con Maldacena y otros contratos.

---

## 3. A dónde vamos ahora

### 3.1. Estabilizar el pipeline honesto XI → XII.c

Objetivo técnico inmediato (smoke test):

1. Generar un mini-dataset de Fase XI (sandbox).
2. Pasarlo por la cadena:
   - `bulk_scalar_solver_v2.py`.
   - `make_fase11_bulk_for_fase12c_v2.py`.
   - `fase12c_emergent_dictionary_v2.py` (PySR con pocos operadores e iteraciones).
3. Verificar que:
   - El target en todos los ficheros nuevos es `lambda_sl_*`.
   - `m2L2` solo aparece como `*_legacy` o en comparaciones a posteriori.
   - El diccionario aprende una relación razonable λ_SL ↔ (Δ, d) **sin** tener la fórmula de Maldacena incrustada.

Resultado esperado:
- Un pipeline XI→XII.c **sólido, trazable y honesto**, listo para usarse con datos que sí importan.

### 3.2. Fase XII real: datos del mundo físico

Una vez esté estable el eje λ_SL, el siguiente paso de CUERDAS es usarlo con **sistemas reales**, por ejemplo:

- Espectros del **conformal bootstrap** (Ising 3D, O(N)…).
- **Lattice QCD**.
- Sistemas de **materia condensada** (strange metals, etc.).
- Datos cosmológicos (p.ej. **CMB**).

Para cada sistema:

1. Un **adaptador** traduce el dataset real al formato que entiende la Fase XI (boundary).
2. Fase XI propone:
   - Geometría de bulk.
   - Ecuaciones efectivas.
3. El solver v2 calcula λ_SL para esa geometría.
4. El diccionario v2 busca relaciones entre λ_SL y los parámetros del sistema (Δ, d, escalas, etc.).
5. Los contratos XII/XIII evalúan:
   - Compatibilidad con AdS estándar (Maldacena).
   - O aparición de geometrías más exóticas (Lifshitz, HV, deformadas, “unknown”…).

En lenguaje de ingeniería:

- Fases I–X han sido **tests de integración** en un entorno controlado (sandbox).
- El giro λ_SL:
  - Limpia la **API interna**.
  - Aísla responsabilidades.
  - Elimina fugas de “conocimiento teórico” dentro del entrenamiento.
- Lo que viene ahora:
  - Usar el sistema como un **motor de descubrimiento** sobre datos reales.
  - Donde:
    - El solver numérico solo resuelve problemas.
    - El ML solo ajusta patrones a los datos.
    - La física se aplica en la fase de **validación y lectura**, no en el core del fit.

