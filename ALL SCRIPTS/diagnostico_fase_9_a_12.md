# DIAGNÓSTICO FINAL – FASE XI → XII.c (AdS / Diccionario de Masas)

## 0. Objetivo del documento

Este documento resume el diagnóstico final sobre el **mismatch** entre:

- La **ecuación emergente** que CUERDAS (Fase XII.c) descubre entre las cantidades bulk producidas en Fase XI, y
- La **fórmula teórica estándar de Maldacena** para escalares en AdS\(_{d+1}\):

\[
m^2 L^2 = \Delta(\Delta - d).
\]

El propósito es dejar por escrito:

1. Qué hace realmente el pipeline XI → XII.c.
2. Qué relación emergente observamos en los datos tipo AdS.
3. Por qué **no** tiene sentido forzar estos datos a obedecer la fórmula de Maldacena.
4. Qué implicaciones tiene esto para las fases XII–XIII del proyecto CUERDAS.

---

## 1. Resumen del flujo XI → XII.c

### 1.1. Fase XI – Solver escalar y extracción de datos bulk

En Fase XI se trabaja con ficheros `.h5` que contienen geometrías y resultados del solver escalar para distintas familias (`ads`, `lifshitz`, `hyperscaling`, `deformed`, `unknown`) y dimensiones de contorno \(d\in\{3,4,5\}\).

Puntos clave del solver escalar (`bulk_scalar_solver.py`):

- El operador se implementa como un problema **Sturm–Liouville** en la radial \(z\):

  \[
  L \phi = -\frac{1}{\sqrt{-g}}\partial_z\left( \sqrt{-g} g^{zz} \partial_z \phi \right),
  \]

  con condiciones de contorno tipo Dirichlet en \(z_\text{min}, z_\text{max}\).

- Se resuelve el problema de autovalores:

  \[
  L \phi_n = \lambda_n \phi_n,
  \]

  donde, por teoría de Sturm–Liouville, los autovalores \(\lambda_n\) son **reales y positivos**.

- Para cada modo se extraen dos cantidades numéricas:

  1. Un **exponente UV emergente**, \(\Delta_\text{UV}\), a partir del comportamiento
     \[
     |\phi(z)| \sim z^{\Delta_\text{UV}},
     \]
     ajustado en la parte más UV del grid.
  2. Un **autovalor emergente** que se guarda con nombre tipo `m2L2_bulk` y luego `m2L2_emergent`, pero que en realidad es \(\lambda_n\) del problema Sturm–Liouville.

En resumen: XI produce pares \((\Delta_\text{UV}, \lambda_\text{SL})\) para cada “operador” emergente, etiquetados por sistema, familia y dimensión \(d\).

### 1.2. Preparación de datos para XII.c

Dos scripts hacen de puente:

- `make_fase11_bulk_for_fase12c.py`  
  Lee los `.h5` de Fase XI y construye un JSON con todos los operadores emergentes:

  - `Delta_bulk_uv` → luego `Delta`.
  - `m2L2_bulk` → luego `m2L2_emergent`.
  - Metadatos: `family`, `d`, `name`, etc.

- `make_fase11_bulk_legacy_for_fase12c.py`  
  Aplica filtros de calidad y construye un JSON “legacy” para XII.c.

  Resultado reportado:

  - Operadores totales (antes de filtro): **270**
  - Operadores usados (después de filtro): **239**
  - Sistemas finales: **90**
  - Fichero: `data_processed/fase11_bulk_legacy_for_fase12c.json`

A partir de este JSON se construye una versión restringida a sistemas AdS:

- `data_processed/fase11_bulk_legacy_ads_only.json`  
  Contiene sólo `systems` con `family = "ads"`.

### 1.3. Fase XII.c – Motor de diccionario emergente (PySR)

`fase12c_emergent_dictionary_real.py`:

- Toma como entrada un JSON tipo legacy (`*_for_fase12c.json`).
- Construye un dataset con columnas:
  - `Delta` (x0),
  - `d` (x1),
  - `m2L2_emergent` (target y),
  - y otros metadatos (familia, sistema, etc.).
- Ejecuta **Symbolic Regression (PySR)** para buscar una ecuación:

  \[
  y = f(x_0, x_1) = f(\Delta, d)
  \]

  que explique \(m^2L^2_\text{emergent}\).
- Compara esa ecuación emergente con la fórmula teórica de Maldacena:

  \[
  m^2_\text{teo} L^2 = \Delta(\Delta - d).
  \]

La comparación se hace con métricas en test (R², MAE, correlación) y se marca explícitamente si la fórmula teórica es “compatible” o no con los datos emergentes.

---

## 2. Resultados globales (todas las familias mezcladas)

### 2.1. Ecuación emergente global

Ejecutando XII.c sobre **todas las familias** (`ads`, `lifshitz`, `hyperscaling`, `deformed`, `unknown`):

- **Mejor ecuación global**:

  \[
  m^2L^2 \approx \left(\exp(\Delta) - \frac{d}{0.26366538}\right)^2.
  \]

- Métricas en test:

  - R² ≈ **0.5814**
  - MAE ≈ **30.53**
  - Pearson ≈ **0.80**

- Comparación con la fórmula teórica:

  - Fórmula: \(m^2L^2 = \Delta(\Delta - d)\).
  - R² teórico ≈ **-1.7082**.
  - Marcado como **“Compatible: False”**.

Interpretación:

- El mejor fit simbólico global **no** se parece en absoluto a la parábola \(\Delta(\Delta-d)\).
- La R² teórica negativa indica que, a nivel global, la fórmula de Maldacena explica peor los datos que una constante.

Esto es el primer indicio fuerte de que:

> Los \(m^2L^2_\text{emergent}\) producidos por XI en el “mix” global de geometrías no siguen el diccionario estándar.

---

## 3. Análisis por familia y dimensión d

Sobre `fase11_bulk_legacy_for_fase12c.json` se computó, por bloque `(family, d)`, el R² de:

\[
m^2L^2_\text{teo} = \Delta(\Delta - d)
\]

frente a los datos emergentes.

Resumen de R² teóricos por bloque:

- `ads`:
  - d=3: R² ≈ **-1.8862**
  - d=4: R² ≈ **-2.0415**
  - d=5: R² ≈ **-1.5514**
- `lifshitz`:
  - d=3: R² ≈ **-1.9002**
  - d=4: R² ≈ **-2.1356**
  - d=5: R² ≈ **-1.6712**
- `hyperscaling`:
  - d=3: R² ≈ **-1.4580**
  - d=4: R² ≈ **-1.7664**
  - d=5: R² ≈ **-2.4004**
- `deformed`:
  - d=3: R² ≈ **-2.1816**
  - d=5: R² ≈ **-2.0276**
- `unknown`:
  - d=3: R² ≈ **-2.2551**
  - d=4: R² ≈ **-1.7181**
  - d=5: R² ≈ **-1.9018**

Conclusión:

- En **ninguna** familia ni dimensión el diccionario teórico \(m^2L^2 = \Delta(\Delta-d)\) se ajusta bien a los datos emergentes: todos los R² están alrededor de -1.5 a -2.

---

## 4. Análisis AdS-only

Dado que la comparación con Maldacena tiene sentido, en primer lugar, en fondos AdS, se restringió el análisis a `family = "ads"`.

### 4.1. Datos AdS-only

A partir de `fase11_bulk_legacy_ads_only.json` se obtiene:

- Nº total de operadores AdS: **57**.
- Distribución de \(d\):

  - d=3: **28** operadores.
  - d=4: **9** operadores.
  - d=5: **20** operadores.

- Estadísticos globales (Ads-only):

  - `Delta`:
    - min ≈ **-0.009**
    - max ≈ **2.767**
    - media ≈ **1.568**
  - `m2L2`:
    - min ≈ **1.665**
    - max ≈ **258.85**
    - media ≈ **87.56**

Es importante destacar que:

- Para d=3: \(\Delta \in [-0.009, 1.831]\), media ≈ 1.118.
- Para d=5: \(\Delta \in [1.024, 2.767]\), media ≈ 2.222.

En AdS/CFT estándar, con \(\Delta < d\), la fórmula \(m^2L^2 = \Delta(\Delta-d)\) daría valores **negativos** de m²L². Aquí, en cambio, los datos emergentes de XI dan **m²L² positivos y grandes**.

### 4.2. Ecuación emergente AdS-only (XII.c)

Ejecutando XII.c sólo sobre AdS-only:

- **Mejor ecuación**:

  \[
  m^2 L^2 \approx (x_1 \cdot (x_1 \cdot 11.574053)) + (35.533646 - (x_0^2 \cdot 42.214573)),
  \]
  es decir, aproximadamente:
  \[
  m^2 L^2 \approx 11.57\,d^2 + 35.53 - 42.21\,\Delta^2.
  \]

- Métricas en test:

  - R² ≈ **0.9827**
  - MAE ≈ **7.99**
  - Pearson ≈ **0.9926**

- Comparación con teoría en este subset:

  - Fórmula teórica: \(m^2L^2 = \Delta(\Delta-d)\).
  - R² teórico ≈ **-1.1357**.
  - Marcado como **“Compatible: False”**.

Es decir:

- En AdS-only, XII.c encuentra una ley **simple, cuadrática y muy bien ajustada** entre \(\Delta\) y \(d\).
- Pero esa ley no se parece en absoluto a la parábola \(\Delta(\Delta-d)\).

### 4.3. Ajustes por dimensión d (fits internos)

Sobre AdS-only se realizaron ajustes separados por dimensión \(d\):

Para cada d se ajustó:

\[
m^2 L^2 \approx a_d\,\Delta^2 + b_d.
\]

Resultados:

- d = 3 (n=28):

  - Δ: min ≈ -0.009, max ≈ 1.831, media ≈ 1.118.
  - m²L²: min ≈ 3.872, max ≈ 142.038, media ≈ 71.893.
  - Fit:
    - \(a_3 \approx -44.836\)
    - \(b_3 \approx 144.606\)
    - R² ≈ **0.9757**

- d = 4 (n=9):

  - Δ: min ≈ 0.536, max ≈ 2.158, media ≈ 1.513.
  - m²L²: min ≈ 17.190, max ≈ 200.470, media ≈ 101.991.
  - Fit:
    - \(a_4 \approx -41.955\)
    - \(b_4 \approx 215.872\)
    - R² ≈ **0.9875**

- d = 5 (n=20):

  - Δ: min ≈ 1.024, max ≈ 2.767, media ≈ 2.222.
  - m²L²: min ≈ 1.665, max ≈ 258.847, media ≈ 102.999.
  - Fit:
    - \(a_5 \approx -40.980\)
    - \(b_5 \approx 317.720\)
    - R² ≈ **0.9478**

Conclusiones:

- El coeficiente de \(\Delta^2\), \(a_d\), es **negativo y casi constante**:
  - En torno a \(-42\) para d=3,4,5.
- El offset \(b_d\) **crece con d**, de forma compatible con un término cuadrático en d (como `11.57 d² + const` de la ecuación de XII.c).

Esto confirma una ley interna del tipo:

\[
m^2_\text{emergent} L^2 \approx -C\,\Delta_\text{UV}^2 + B(d),
\quad C \approx 42,
\]

con muy buen ajuste (R² ≈ 0.95–0.99 por cada d).

### 4.4. Test de “diccionario” con variantes de Δ y signo

Para comprobar si había algún truco trivial (root equivocado, signo de m²), se evaluaron las siguientes variantes en AdS-only, por cada d:

1. \(m^2_\text{teo} = \Delta(\Delta-d)\)
2. \(m^2_\text{teo} = \Delta'(\Delta'-d)\), con \(\Delta' = d-\Delta\) (el otro root de la ecuación cuadrática).
3. Comparaciones con m2 y -m2: \(m2\) vs m²_teo, \(-m2\) vs m²_teo.

Resultados:

- d = 3:

  - R²( m2,  Δ(Δ−d) ) ≈ **-1.8862**
  - R²( m2,  Δ'(Δ'−d) ) ≈ **-1.8862**
  - R²(−m2, Δ(Δ−d) ) ≈ **-1.7575**
  - R²(−m2, Δ'(Δ'−d)) ≈ **-1.7575**

- d = 4:

  - R²( m2,  Δ(Δ−d) ) ≈ **-2.0415**
  - R²( m2,  Δ'(Δ'−d) ) ≈ **-2.0415**
  - R²(−m2, Δ(Δ−d) ) ≈ **-1.8340**
  - R²(−m2, Δ'(Δ'−d)) ≈ **-1.8340**

- d = 5:

  - R²( m2,  Δ(Δ−d) ) ≈ **-1.5514**
  - R²( m2,  Δ'(Δ'−d) ) ≈ **-1.5514**
  - R²(−m2, Δ(Δ−d) ) ≈ **-1.2600**
  - R²(−m2, Δ'(Δ'−d)) ≈ **-1.2600**

Observaciones:

- Δ(Δ−d) y Δ'(Δ'−d) son algebraicamente **idénticas** (de ahí los mismos R²).
- Ninguna combinación de:
  - root (Δ vs d−Δ),
  - ni flip de signo m2 → −m2,
  mejora los R² a un valor aceptable: siguen siendo muy negativos.

Esto demuestra que **no hay una reparametrización simple** que convierta el espectro emergente de XI en el de la fórmula de Maldacena.

---

## 5. Test de calibración AdS (Sturm–Liouville vs Maldacena)

Un test adicional, sobre geometrías AdS puras sintéticas, comparó:

- Autovalores numéricos \(\lambda_\text{num}\) del problema elíptico:

  \[
  L\phi_n = \lambda_n \phi_n,
  \]

- Con “m² teórico” definido como \(\Delta(\Delta-d)\) usando \(\Delta_\text{UV}\) extraído numéricamente.

Resultados típicos para d=3,4,5:

- Modo 0:
  - \(\lambda_0 \approx 1.0\)
  - Δ_UV no definido (modo trivial / gauge)
- Modos 1–9:
  - \(\lambda_n > 0\) siempre (por construcción Sturm–Liouville).
  - \(m^2L^2_\text{teo} = \Delta(\Delta-d) < 0\) para Δ < d.
  - Los “ratios” \(\lambda_\text{num} / m^2L^2_\text{teo}\) son negativos, con valor medio ≈ **-2.84 ± 2.03**.

Conclusión del test:

- Se están comparando **objetos diferentes**:
  - \(\lambda_\text{num}\): autovalor positivo del operador radial,
  - \(m^2L^2_\text{teo}\): parámetro de masa (que puede ser negativo) en la ecuación de Klein–Gordon usada en el análisis de Maldacena.

No hay ninguna razón matemática para que \(\lambda_n\) coincida con \(\Delta(\Delta-d)\) en este setup: no se está resolviendo la KG con m² fijado y analizando el comportamiento asintótico, sino un espectro de Sturm–Liouville con condiciones de contorno globales.

---

## 6. Diagnóstico final

Con todos estos datos, el diagnóstico final es:

1. **El pipeline XI → XII.c es honesto y consistente consigo mismo**:
   - No hay inyección de la fórmula de Maldacena en ningún punto.
   - PySR encuentra una relación emergente con R² muy alto (≈0.98) en AdS-only,
   - Y detecta correctamente que la fórmula teórica \(m^2L^2 = \Delta(\Delta-d)\) **no** describe los datos (R² negativos).

2. **Los “m2L2_emergent” de XI no son “m² holográficos de Maldacena”**:
   - Son autovalores positivos de un problema de Sturm–Liouville con condiciones de contorno finitas.
   - La variable \(\Delta_\text{UV}\) es un exponente numérico ajustado en una región UV del grid, no necesariamente la Δ del análisis asintótico estándar.
   - En este contexto, no hay motivo para que \(\lambda\) satisfaga \(\lambda = \Delta(\Delta-d)\); y de hecho, los datos muestran que no ocurre.

3. **El diccionario emergente interno del sandbox AdS** es:

   - De forma empírica:
     \[
     \lambda_\text{SL} \equiv m^2L^2_\text{emergent} \approx 11.6 d^2 + 36 - 42\,\Delta_\text{UV}^2
     \]
     con R² ≈ 0.98 en AdS-only.
   - Y por dimensión fija:
     \[
     \lambda_\text{SL} \approx a_d \Delta_\text{UV}^2 + b_d, \quad
     a_d \approx -42, \quad b_d \text{ creciente con } d.
     \]

   Esta es la **ley efectiva** descubierta por CUERDAS en el playground XI, y es coherente internamente.

4. **No existe una reparametrización simple (Δ→d−Δ, m²→−m²)** que reconcilie el espectro emergente con la fórmula de Maldacena:
   - Todos los R² evaluados en estas variantes siguen siendo ~ -1.2…-2.0.

5. **Conclusión conceptual**:

   > Fase XI implementa un playground AdS-like en el que el espectro de autovalores y exponente UV están relacionados por un diccionario emergente distinto al de la teoría continua. Fase XII.c ha redescubierto ese diccionario de forma autónoma y ha demostrado, cuantitativamente, que no coincide con \(m^2L^2 = \Delta(\Delta-d)\). El mismatch no es un bug, sino una diferencia de objeto matemático entre:
   >
   > - “masa holográfica m² de Maldacena”, y
   > - “autovalor de Sturm–Liouville en un fondo discreto con condiciones de contorno”.

---

## 7. Implicaciones para CUERDAS (Fases XII–XIII)

De cara al proyecto CUERDAS:

1. **Uso de XI/XII.c como motor emergente**:
   - Se puede seguir utilizando XI/XII.c como motor de descubrimiento de diccionarios emergentes para datos reales (Ising 3D, lattice QCD, etc.).
   - Lo único que hay que hacer es **nombrar con precisión** las variables:
     - Llamar, por ejemplo, `lambda_sl` en lugar de `m2L2` cuando se trate del autovalor numérico del solver.
     - Dejar claro en la documentación qué se entiende por “masa” en cada contexto.

2. **Comparación con Maldacena**:
   - Debe tratarse como un **módulo de análisis separado**:
     - Para comparar resultados de un solver “teórico” específico con la fórmula estándar.
     - No como una expectativa que deba cumplirse automáticamente en el sandbox emergente.

3. **Posible módulo de control positivo “AdS puro”** (para futuro):
   - Si se desea un control positivo que reproduzca literalmente \(m^2L^2 = \Delta(\Delta-d)\), habría que implementar un solver distinto:
     - Resolver la KG con m² fijado en una métrica AdS exacta,
     - Ajustar Δ del comportamiento asintótico,
     - Comparar con la fórmula analítica.
   - Este módulo coexistiría con XI/XII.c, pero no lo sustituye: serían dos piezas complementarias.

En resumen:

> XI y XII.c ya están “bien” para lo que hacen: describir y explotar la física emergente de un playground numérico, no replicar exactamente la AdS/CFT continua. El trabajo de diagnóstico ha servido para demostrar esa honestidad y delimitar el dominio de validez del diccionario emergente.

---
