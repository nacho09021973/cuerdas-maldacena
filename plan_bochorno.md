
0) Principios no negociables

P0.1 Una ejecución es un objeto auditable

Si no puedes reconstruir qué comando, qué inputs, qué versión, y dónde quedaron los outputs, entonces esa ejecución no existe a efectos del proyecto.

P0.2 Separación absoluta de capas

Infra/IO: rutas, manifests, schemas, orquestación, CI.

Ciencia: modelos, losses, contratos físicos, métricas físicas.

Datos: adapters, truth, datasets, splits.


No se mezclan en el mismo PR salvo casos quirúrgicos.

P0.3 El pipeline no “busca”; resuelve

Se prohíbe find/heurísticas dentro de etapas salvo fallback explícito, registrado y con warning. La resolución de rutas se hace por:

1. CLI


2. run_manifest.json


3. legacy resolver (siempre registrando resolved_from_legacy=true)



P0.4 Contrato de interfaz (Artifact API) primero, ciencia después

Ningún contrato físico vale si el artefacto ni siquiera está bien definido y validado.


---

1) Reglas estrictas de ejecución y layout

1.1 Run Root único

Todo output vive bajo:

runs/<experiment>/

Prohibido escribir fuera (salvo caches explícitas).

1.2 Stage Dir canónico

Cada etapa escribe en:

runs/<experiment>/<NN>_<stage_slug>/

Ejemplo:

runs/debug_Tfinite/02_emergent_geometry_engine/

runs/debug_Tfinite/06_build_bulk_eigenmodes_dataset/


1.3 Alias canónicos (para consumo downstream)

En runs/<experiment>/ existen alias (symlink o carpeta real) con nombres estables:

predictions/

geometry_emergent/

bulk_equations/

geometry_contracts/

bulk_eigenmodes/

dictionary/ (o los que defináis)


Estos alias apuntan a los outputs reales de la etapa productora. Downstream consume alias, no rutas de stage.

1.4 Fuente de verdad: run_manifest.json

Cada stage debe escribir:

inputs consumidos

outputs producidos (rutas relativas)

comando, git SHA, entorno (mínimo)



---

2) Contratos operativos (lo que os faltaba)

2.1 Contrato de estado y exit codes (obligatorio)

Todas las etapas devuelven:

0 OK

1 WARNING (parcial, pero usable)

2 INCOMPLETE (faltan inputs / artefactos requeridos; no se debe continuar)

3 ERROR (bug o fallo no recuperable)


Además generan siempre un stage_summary.json machine-readable.

stage_summary.json (schema mínimo)

{
  "stage": "06_build_bulk_eigenmodes_dataset",
  "experiment": "test_v3",
  "status": "OK|WARNING|INCOMPLETE|ERROR",
  "exit_code": 0,
  "counts": {
    "inputs_total": 40,
    "inputs_ok": 38,
    "inputs_failed": 2,
    "outputs_rows": 1200
  },
  "errors": [],
  "warnings": [],
  "artifacts": {
    "dataset_csv": "bulk_eigenmodes/bulk_modes_dataset.csv"
  },
  "meta": {
    "git_sha": "...",
    "created_at": "..."
  }
}

Regla: si no existe stage_summary.json, el stage se considera fallido.

2.2 Contrato de artefactos (Artifact API) versionado

Cada tipo de artefacto se define con:

schema_name

schema_version

campos mínimos obligatorios

validador automático


Ejemplos de tipos para vuestro pipeline:

1. GeometryPrediction (salida de 02)

Formato: NPZ (canónico)

Obligatorio: z, A, f, meta



2. BulkEquations (salida de 03)

JSON versionado



3. GeometryContracts (salida de 04)

JSON + contracts_summary.json para CI



4. EigenmodesDataset (salida de 06)

CSV + meta JSON (schema version)



5. Dictionary (07–08)

JSON/NPZ versionado




Ejemplo: GeometryPrediction NPZ (schema v0.1)

Claves obligatorias:

z (1D)

A (1D)

f (1D)

meta_json (string JSON) o meta.json al lado


meta mínimo:

{
  "schema_name": "GeometryPrediction",
  "schema_version": "0.1",
  "system_id": "ads_d3_Tfinite_test_000",
  "family": "ads|lifshitz|unknown",
  "d": 3,
  "z_h": 0.1,
  "source_stage": "02_emergent_geometry_engine",
  "git_sha": "..."
}

2.3 Contrato de naming (system_id)

Se prohíben heurísticas distintas por script. Debe existir un system_id estable:

derivado del dataset + índice o de un hash determinista del meta

usado para unir outputs entre etapas



---

3) Enforcements automáticos (para que no vuelva a pasar)

3.1 tools/assert_artifacts.py (gate duro)

Un script que:

lee run_manifest.json

valida existencia y schema de artefactos requeridos por la siguiente etapa

si falta algo: exit 2 con mensaje exacto missing_artifact: ...


Se ejecuta:

en el orquestador entre etapas

en CI antes de etapas costosas


3.2 Gate “no writes outside run”

En CI (y opcionalmente local) se falla si se detectan outputs fuera de runs/<experiment>/.
Esto elimina de raíz outputs/, results/, fase11_* generándose accidentalmente.

3.3 Validadores obligatorios por tipo

En cada etapa, al inicio:

validate_inputs()
al final:

validate_outputs()


Con mensajes útiles y exit 2 si falta lo mínimo.


---

4) Orquestación: un solo punto de entrada

4.1 run_pipeline.py

Debe existir un orquestador oficial:

python run_pipeline.py --experiment X --from 01 --to 09 --config configs/X.json

Responsabilidades:

crea runs/<experiment>/

llama etapas en orden

escribe run_manifest.json global

ejecuta assert_artifacts entre etapas

consolida summaries


Regla: un “run serio” se lanza siempre desde aquí.


---

5) Observabilidad y trazabilidad

Por etapa, obligatorio:

logs/stdout.log

logs/stderr.log

cmd.txt

meta_env.json (python version + pip freeze hash + GPU/CPU info mínimo)

stage_summary.json

actualización de run_manifest.json


Sin esto, no hay debugging serio.


---

6) Control de configuración y reproducibilidad

6.1 Config único por experimento

Cada run guarda una copia inmutable del config:

runs/<experiment>/config.resolved.json

Regla: los flags CLI solo sobreescriben; el “source of truth” del run es el config guardado.

6.2 Determinismo

Ordenar siempre el input set (sorted(glob))

--seed obligatorio o registrado siempre

Registrar git_sha + versiones (mínimo)



---

7) Plan de saneamiento del legado (sin destruir historia)

7.1 Congelar carpetas legacy

outputs/, results/, fase11_*, smoke_* se declaran legacy:

no se borran

no se escriben más

se consumen solo vía “importer”


7.2 Importer/migrador controlado

tools/import_legacy_run.py:

toma una carpeta legacy

crea un runs/<experiment>/ nuevo

symlinks a los artefactos legacy detectados

escribe run_manifest.json con resolved_from_legacy=true


Así recuperas valor sin perpetuar el caos.


---

8) CI serio (dos carriles)

8.1 Carril rápido (en cada PR)

python -m py_compile (entrypoints 00–09)

unit tests de loaders/validators

toy E2E (01→06) con dataset minúsculo

gate “no writes outside run”


8.2 Carril nightly

smoke más pesado (01→09 cuando esté)

controles negativos (04c)

regression metrics (comparar contra golden run)



---

9) “Definition of Done” para una etapa

Una etapa no está “hecha” hasta que cumple:

1. Acepta --experiment


2. Lee inputs desde alias/manifest


3. Valida inputs (schema)


4. Produce outputs en layout canónico


5. Valida outputs (schema)


6. Actualiza manifest


7. Escribe summary + logs


8. Pasa toy smoke en CI



Si no, es “en desarrollo”.


---

10) Hoja de ruta extensiva (orden recomendado)

Workstream A — Infra/IO (cerrar lo que ya empezasteis)

1. Consolidar V3 en main (PR + merge)


2. Estabilizar run_context.py como API única


3. Añadir stage_summary.json a 00–09


4. Añadir assert_artifacts.py y gates



Workstream B — Artifact API v0.1

5. Escribir docs/ARTIFACT_API.md con tablas 00–09:

inputs/outputs + schema_version



6. Implementar validadores por tipo:

NPZ geometry

CSV eigenmodes

JSON equations/contracts/dictionary




Workstream C — Orquestación

7. Implementar run_pipeline.py (01→06 primero)


8. Integrar 07–09 gradualmente con assert_artifacts



Workstream D — Contratos de interfaz etapa a etapa

9. Formalizar y cerrar 06→07, 07→08, 08→09


10. Cada cierre incluye:



schema + validator

gate en CI

ejemplo de run reproducible


Workstream E — Regresión y “golden runs”

11. Definir 2–3 runs “golden” pequeños:



ads control

Tfinite

ising3d mini


12. CI nightly compara outputs agregados (hash + métricas)



Workstream F — Deuda técnica (ya con infra sólida)

13. Propagación de metadata (family, d, z_h)


14. Diagnóstico numérico (Delta_UV negativos)


15. Robustez de loaders (H5 attrs/datasets, etc.)




---

11) Reglas de disciplina (para que el equipo no recaiga)

Prohibido añadir un nuevo output sin registrarlo en manifest.

Prohibido leer un input sin validarlo.

Prohibido introducir un formato nuevo sin schema_version + validador.

Prohibido “arreglar un bug” mezclando ciencia + infra en el mismo PR.

PR que rompe toy smoke no entra.



---

Siguiente acción concreta (para convertir el plan en ejecución)

Si me confirmas que actualmente tenéis en main:

V3 (00–05)

contrato 02→06 (06 consume NPZ)


entonces el siguiente PR “serio” que haría es único y mecánico:

PR: “stage_summary + assert_artifacts + gate no-writes”

Porque a partir de ahí:

cualquier incompatibilidad de formatos deja de ser sorpresa,

