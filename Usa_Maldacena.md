Perfecto: objetivo cumplido. Con esos `HTTP/2 200` ya has demostrado que **los artefactos del corpus M99 están publicados en `main` y accesibles vía `raw.githubusercontent.com`**, es decir, consumibles por cualquier IA con acceso web (incluyéndome a mí). ([GitHub][1])

Además, el `manifest.json` ya está en el repo y expone metadatos clave (fuente `hep-th/9905111`, `pdf_sha256`, timestamp y lista de ficheros), lo cual deja el corpus **autodescriptivo y trazable**. ([GitHub][2])

## Qué debe consumir una IA (lo mínimo indispensable)

Tu “entrypoint” real para consultas es:

* `data/corpus/m99/paragraphs_sections.jsonl` (texto por párrafo + `section_id` + `page_id` + `para_id` + `bbox`) ([GitHub][1])
  y, para estructura:
* `data/corpus/m99/toc.json` ([GitHub][3])
  y, para trazabilidad:
* `data/corpus/m99/manifest.json` ([GitHub][2])

En concreto, el JSONL contiene entradas con campos como `doc_id`, `page_id`, `para_id`, `bbox`, `text` y `section_id`. ([GitHub][1])

## Enlaces “raw” (para pegarlos en prompts o agentes)

(Te los pongo en bloque de código para que puedas copiarlos tal cual.)

```text
https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/manifest.json
https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/toc.json
https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/paragraphs_sections.jsonl
https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/paragraphs.jsonl
https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/blocks.jsonl
https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/pages.jsonl
```

## Prompt corto “universal” para cualquier IA (incluyéndome)

Úsalo tal cual; obliga a respuestas “citables” desde el corpus, sin inventar:

> Tienes acceso web a un corpus indexado del paper M99 (hep-th/9905111) en GitHub, en formato JSONL. Usa como fuente primaria `paragraphs_sections.jsonl` y como estructura `toc.json`.
> Procedimiento: (1) busca por string los términos relevantes en `paragraphs_sections.jsonl`; (2) devuelve 3–10 hits con `section_id`, `page_id`, `para_id`, `bbox` y un snippet corto; (3) redacta una síntesis mínima estrictamente derivada de esos hits; (4) si no hay hits, dilo explícitamente y sugiere 3 términos alternativos.
> Prohibido: extrapolar sin evidencia en el corpus, o “rellenar” lagunas con conocimiento externo sin etiquetarlo como externo.

## Recomendación final para máxima reproducibilidad

Como quieres una única rama (bien), pero también quieres estabilidad para agentes:

* Cuando uses el corpus en experimentos o reportes, **pínchalo por commit** en vez de `main`:

  1. obtiene el hash: `git rev-parse HEAD`
  2. usa URLs del estilo:

     ```text
     https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/<COMMIT>/data/corpus/m99/paragraphs_sections.jsonl
     ```

Así garantizas que una IA hoy y otra dentro de 6 meses consultan exactamente el mismo contenido.

Si me dices “quiero comprobar X del paper (ej. Wilson loops, diccionario field↔operator, condiciones de contorno…)”, ya puedo hacerlo consultando directamente este corpus publicado en GitHub y devolviéndote referencias `page_id/para_id/bbox` de forma verificable.

[1]: https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/paragraphs_sections.jsonl "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/manifest.json "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/nacho09021973/cuerdas-maldacena/main/data/corpus/m99/toc.json "raw.githubusercontent.com"
