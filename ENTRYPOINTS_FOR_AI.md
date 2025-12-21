# ENTRYPOINTS_FOR_AI.md  
## CUERDAS-MALDACENA — Acceso mínimo, honesto y verificable para IAs

Este documento define **el único interfaz autorizado** para que una IA interactúe con el proyecto **CUERDAS-MALDACENA**.

No es un README general.  
No describe el pipeline completo.  
No concede acceso implícito al repositorio GitHub.

Su objetivo es **evitar alucinaciones, malentendidos y uso indebido del conocimiento**.

---

## 1. Principio fundamental (obligatorio)

⚠️ **La IA NO tiene acceso al repositorio GitHub como tal.**

Para la IA, GitHub **no existe como repo**, ramas o commits.  
Solo existen **URLs explícitas de archivos públicos (raw)** o **archivos proporcionados directamente**.

Si un archivo no aparece listado aquí o no se proporciona explícitamente:
> **Debe considerarse inexistente.**

---

## 2. Fuente de verdad humana (informativa)

La fuente de verdad del proyecto es el repositorio GitHub:

- Repo: https://github.com/nacho09021973/cuerdas-maldacena  
- Rama principal: `main`

⚠️ **Esto es solo informativo.**  
La IA **no puede** navegar ni asumir el contenido del repo sin URLs explícitas.

---

## 3. Entry points autorizados para IAs (consumibles)

La IA **solo puede usar** los siguientes tipos de recursos:

### 3.1 Corpus indexados (lectura estricta, citation-only)

Ejemplo (Maldacena 1999):

- `manifest.json`
- `toc.json`
- `pages.jsonl`
- `blocks.jsonl`
- `paragraphs.jsonl`
- `paragraphs_sections.jsonl`

Publicados vía `raw.githubusercontent.com`, por ejemplo:

https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/<COMMIT>/data/corpus/m99/manifest.json
https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/<COMMIT>/data/corpus/m99/paragraphs_sections.jsonl

yaml
Copiar código

📌 **Regla**:  
La IA solo puede responder sobre el contenido **explícitamente presente** en esos ficheros.

---

### 3.2 Archivos proporcionados directamente en la conversación

Ejemplos:
- logs de error
- diffs o `.patch`
- fragmentos de código
- contratos (`.md`)
- reportes (`.json`)

📌 **Regla**:  
La IA **no puede inferir contexto externo** más allá del archivo recibido.

---

### 3.3 Manifests y contratos

Archivos como:
- `run_manifest.json`
- contratos físicos (`docs/contratos/*.md`)
- protocolos IO (`IO_LAYOUT_V2.md`)

Se consideran **documentos normativos**.

📌 **Regla**:  
Si un contrato prohíbe algo, la IA **debe respetarlo**, incluso si “conoce” la teoría por entrenamiento previo.

---

## 4. Prohibiciones explícitas (muy importante)

La IA **NO DEBE**:

- ❌ asumir fórmulas físicas conocidas si no aparecen en los datos
- ❌ “rellenar” lagunas con conocimiento externo sin etiquetarlo
- ❌ usar teoría conocida en entrenamiento, losses o features
- ❌ decir “según Maldacena…” sin cita localizable
- ❌ inventar estructura del repo
- ❌ asumir que ha leído el README general del proyecto

Cualquier violación invalida la respuesta.

---

## 5. Modo de respuesta esperado (formato)

Cuando se consulte un corpus indexado:

1. Buscar coincidencias textuales.
2. Devolver **hits citables**, incluyendo:
   - `section_id`
   - `page_id`
   - `para_id`
   - `bbox` (si existe)
3. Redactar una síntesis **mínima**, derivada solo de esos hits.
4. Si no hay evidencia:
   > “No hay soporte explícito en el corpus para esta afirmación.”

---

## 6. Filosofía del proyecto (contexto para la IA)

CUERDAS-MALDACENA no busca:
- confirmar teoría conocida,
- ajustar curvas,
- ni producir narrativa elegante.

Busca:
- falsación,
- contratos duros,
- supervivencia de resultados bajo auditoría extrema.

> Si solo 1 resultado sobrevive entre 500.000,
> **ese** es el que importa.

La IA es una **herramienta de apoyo**, no una fuente de autoridad.

---

## 7. Regla final

Si tienes dudas sobre:
- si puedes usar un archivo,
- si una inferencia es legítima,
- si una afirmación está soportada,

👉 **di explícitamente que no tienes evidencia suficiente**.

Eso es comportamiento correcto.

