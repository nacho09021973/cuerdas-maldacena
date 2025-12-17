# CHECKLIST MAI — Modo de Análisis Interno (v1.0)

**Proyecto:** CUERDAS–Maldacena  
**Propósito:** evitar autoengaño, detectar tensiones temprano y no bloquear el descubrimiento con teoría mal aplicada.  
**Ámbito:** uso interno (no publicación).

---

## 0. Principio rector
> **Ningún contrato CONDICIONAL o CONJETURAL entra en entrenamiento.**  
Todo contrato no estricto se usa **solo post-hoc** (análisis, etiquetado, comparación).

---

## 1. Etiquetado obligatorio por contrato (antes de analizar resultados)
Para **cada resultado** (geometría, espectro, correlador, diccionario), asignar **una etiqueta**:

- **WL (Wilson loops)** → `EXTENDIDO`
- **m–Δ (masa–dimensión)** → `CONDICIONAL`
- **CORR (2/3-point)** → `CONDICIONAL / EJEMPLAR`
- **GBB (geom. bulk–boundary)** → `CONJETURAL`

**Regla:** si el análisis depende de un contrato `CONDICIONAL` o `CONJETURAL`, queda marcado automáticamente como **POST-HOC**.

---

## 2. Regla de entrenamiento (NO negociable)
**PROHIBIDO en loss / regularizadores / filtros previos:**
- Relación m–Δ
- Correladores esperados
- Forma AdS de la métrica
- Criterios de confinamiento “esperados”

**PERMITIDO (y recomendado):**
- Métricas libres
- Clustering no supervisado
- Comparación **a posteriori** con contratos

---

## 3. Checklist rápido (30 segundos) por resultado
Contestar **Sí / No**:

1. ¿Estoy forzando una forma AdS? → **STOP** (GBB es conjetural)
2. ¿Uso m–Δ como verdad dura? → **STOP** (solo post-hoc)
3. ¿Interpreto correladores fuera del régimen de supergravedad? → **STOP**
4. ¿El resultado existe sin teoría previa? → **OK**
5. ¿Puedo describirlo sin nombrar Maldacena? → **MEJOR**

> Si fallan 1–3: **no descartar**; **re-etiquetar** como exploratorio.

---

## 4. Uso correcto de la herramienta M99 (interno)
**Único objetivo:** delimitar **herencia vs invención**.

**Preguntas correctas:**
- “¿M99 afirma X explícitamente?”
- “¿Esto es ejemplo o ley general en M99?”

**Preguntas incorrectas:**
- “¿Debería salir AdS aquí?”

---

## 5. Clasificación final de resultados (decide acciones)
Cada resultado cae en **una sola** categoría:

- **HEREDADO** — Soportado explícitamente por M99
- **EXTENDIDO** — Compatible, no explícito
- **EXPLORATORIO** — No en M99, pero consistente
- **OUTLIER** — No encaja con contratos conocidos

> **EXPLORATORIOS** y **OUTLIERS** son prioritarios para descubrimiento.

---

## 6. Señales de alarma temprana
Si aparece cualquiera de estas frases internas:
- “Esto debería salir así por AdS/CFT…”
- “Seguro que m–Δ corrige esto…”

→ Ejecutar M99 solo para **verificar alcance**, no para forzar corrección.

---

## 7. Qué NO hacer (lista corta)
- No usar “según Maldacena” como argumento interno
- No descartar resultados por “no parecer AdS”
- No mezclar contratos con fases del pipeline

---

## 8. Registro mínimo recomendado (opcional)
Para cada experimento, anotar:
- Contratos tocados y etiquetas
- Dónde se aplicaron (post-hoc)
- Clasificación final (HEREDADO / EXTENDIDO / EXPLORATORIO / OUTLIER)

---

**Estado:** activo  
**Última revisión:** hoy  
**Propietario:** equipo CUERDAS–Maldacena

