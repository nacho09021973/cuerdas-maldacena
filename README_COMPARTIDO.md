# README_COMPARTIDO.md

Este directorio compartido **no es** la fuente de verdad del proyecto.

## Fuente de verdad (GitHub)

- Repo: **https://github.com/nacho09021973/cuerdas-maldacena**
- Rama principal: **main**
- Commit puntero (rama de limpieza / PR): **4f8078a6**
- Fecha: **2025-12-16**
- Responsable de actualizar este puntero: **ignac**

## Propósito de esta carpeta compartida

Usar esta carpeta solo como **buzón temporal** para:
- Logs de errores para depuración (salidas de consola, stack traces).
- Parches (`.patch`) o diffs cuando haga falta revisar cambios.
- Notas temporales para coordinar trabajo entre personas/IA.
- Archivos pequeños auxiliares que no deban entrar aún en Git.

**Regla de oro:** Todo lo que ya esté en GitHub se borra de aquí.

## Qué NO debe vivir aquí

- Copias “paralelas” del repositorio.
- Versiones duplicadas de scripts.
- Datos grandes (`.h5`, `.pt`, outputs pesados en `runs/`) salvo un caso de depuración explícito y temporal.

## Flujo de trabajo recomendado

1. Trabajo real en el repo Git (GitHub) en una rama.
2. Si algo falla: copia aquí **solo** el log/trace mínimo necesario.
3. Se corrige en Git.
4. Se sube PR.
5. Se borra el material temporal de esta carpeta.

Cuando fijes un puntero “estable”, actualiza arriba el hash y la fecha.

## Run mínimo (recordatorio)

Referencia: el run mínimo reproducible y los contratos IO deben estar descritos en el README del repo.
Este fichero solo apunta al commit/rama relevante para el trabajo.

---

### Registro breve de sesiones (opcional)

- 2025-12-16: limpieza repo (canon 00–09, attic/, docs, ignore logs/) | commit: 4f8078a6
- <YYYY-MM-DD>: <qué se hizo> | commit: <hash>
