
**`README_BIBLIOTECA_INDEXADA.md`**

## 0. Qué es este documento

> Este documento describe el uso, extensión y principios de una **biblioteca indexada de artículos fundacionales en holografía, CFT e Ising**, diseñada para ser utilizada tanto por humanos como por agentes de IA.
>
> No es un resumen divulgativo.
> No es un paper.
> Es un **manual de trabajo honesto**.

---

## 1. Objetivo de la biblioteca

* Convertir artículos largos y densos (ej. Maldacena 1999) en **objetos consultables, citables y verificables**.
* Permitir preguntas del tipo:

  * “¿Dónde se define X?”
  * “¿Qué se afirma exactamente sobre Y?”
  * “¿Hay contradicciones internas o con nuestros contratos?”
* Servir como **base documental** para:

  * validación post-hoc,
  * contratos físicos,
  * auditoría de resultados del pipeline.

**La biblioteca NO genera teoría.
La biblioteca NO interpreta por nosotros.
La biblioteca permite *localizar y contrastar*.**

---

## 2. Principios fundamentales (muy importante)

### 2.1 Honestidad epistemológica

* **Nunca** se inyecta teoría conocida en:

  * funciones de pérdida,
  * features,
  * regularizadores,
  * entrenamiento.
* Los artículos solo se usan:

  * como **referencia externa**,
  * en **checks post-hoc**,
  * o como **auditoría de coherencia**.

### 2.2 Trazabilidad total

Toda afirmación debe poder rastrearse a:

* artículo,
* página,
* párrafo,
* bounding box.

No hay “según Maldacena…”, solo:

> *M99:p0099:para002 dice exactamente…*

---

## 3. Qué contiene la biblioteca (estado actual)

### 3.1 Artículo M99 — Maldacena et al. (1999)

* **Referencia:** hep-th/9905111 v3
* **Contenido indexado:**

  * páginas (`pages.jsonl`)
  * bloques (`blocks.jsonl`)
  * párrafos (`paragraphs.jsonl`)
  * secciones jerárquicas (`toc.json`)
* **Herramienta asociada:**

  * `tools/m99_index.py`
  * agente de consulta GPT-5.2 (citation-only)

### 3.2 Artículo Ising 2D (en preparación)

* **Tema:** Ising bidimensional, correladores, estructura conforme.
* **Estado:**

  * documento identificado
  * pendiente de indexación completa
* **Objetivo:**

  * servir como contrapunto **no holográfico** a Maldacena,
  * validar contratos desde un CFT exactamente soluble.

*(Aquí se irán añadiendo más artículos: Ising 3D bootstrap, QCD efectiva, etc.)*

---

## 4. Estructura de carpetas (contrato)

Ejemplo:

```
data/
  corpus/
    m99/
      manifest.json
      pages.jsonl
      blocks.jsonl
      paragraphs.jsonl
      toc.json
    ising2d/
      manifest.json
      pages.jsonl
      paragraphs.jsonl
tools/
  m99_index.py
  ising_index.py
agents/
  gpt52_m99_agent.md
  gpt52_ising_agent.md
```

**Regla:**
Cada artículo vive en **su propia carpeta autocontenida**.

---

## 5. Qué se puede hacer con la biblioteca

### 5.1 Búsqueda citada

Ejemplo:

```bash
python tools/m99_index.py query --text "Wilson loop" --max 5 --jsonl
```

Resultado:

* citas exactas,
* sin interpretación,
* sin mezcla de fuentes.

### 5.2 Auditoría de contratos

Ejemplos:

* ¿El pipeline reproduce algo que **contradice** explícitamente al texto?
* ¿Hay afirmaciones fuertes en el paper que **no aparecen nunca** en los datos?

### 5.3 Uso por agentes de IA

Los agentes:

* **no pueden resumir libremente**,
* **no pueden extrapolar teoría**,
* **solo pueden responder con citas** del corpus.

---

## 6. Qué NO es esta biblioteca

* ❌ No es un LLM entrenado con papers
* ❌ No es un motor de “resúmenes inteligentes”
* ❌ No es un sustituto de leer los artículos

Es una **prótesis cognitiva** para trabajo riguroso.

---

## 7. Cómo añadir un nuevo artículo (workflow)

1. Añadir PDF original.
2. Indexar páginas y párrafos.
3. Generar `manifest.json`.
4. Validar:

   * conteo de páginas,
   * ausencia de ruido (números sueltos, headers).
5. (Opcional) Crear agente de consulta.

**No se añade nada al pipeline antes de esto.**

---

## 8. Relación con el pipeline CUERDAS-MALDACENA

* La biblioteca es **externa al entrenamiento**.
* Se conecta solo en:

  * análisis,
  * contratos,
  * informes (`fase12`, `fase13`).
* Nunca se usa para “guiar” al modelo.

---

## 9. Estado del documento

> Este README es **abierto y vivo**.
>
> Se actualizará a medida que:
>
> * entren nuevos artículos,
> * aparezcan nuevos contratos,
> * se detecten errores o ambigüedades.

---

## 10. Nota final

> Si solo un resultado sobrevive a todos los contratos,
> **ese** será el que importe.
>
> La biblioteca existe para que ese resultado pueda defenderse
> **sin trucos y sin atajos**.

---

