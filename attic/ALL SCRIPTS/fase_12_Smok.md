# Fase XII — Ising 3D (Bootstrap) — Smoke Real v0

Este directorio recoge el **primer smoke real** de la Fase XII con un sistema físico de verdad:  
el **modelo de Ising 3D** visto desde el conformal bootstrap.

El objetivo de este experimento NO es descubrir nada nuevo, sino:

- Verificar que la **arquitectura de Fase XII** funciona con un sistema real.
- Probar que los **contratos de Fase XII** pueden validar ese sistema.
- Dejar claro qué partes son todavía *stub/manual* y cuáles son *arquitectura estable*.

---

## 1. Estructura de archivos

```text
runs/
  fase12_ising_real/
    fase12/
      predictions/
        fase12_report.json      ← Reporte Fase XII para Ising 3D (stub)
    contracts_fase12.json       ← Resultado de contracts_fase_12_13.py (fase12)
scripts/
  data/
    ising3d_bootstrap.json      ← Adapter v0: descriptor de Ising 3D (bootstrap)
  fase12_adapter_ising3d_bootstrap.py   ← Script que genera el descriptor
  fase12_ising_real_stub.py             ← Stub que construye fase12_report.json
