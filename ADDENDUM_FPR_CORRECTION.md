# ADDENDUM: Corrección a FPR (False Positive Rate)

**Versión:** 1.1  
**Fecha:** 2025-12-21  
**Cambio:** Métrica corregida de `pass_rate` a `false_positive_rate`

---

## Problema con el diseño v2

El diseño original usaba:

```python
pass_rate = n_passed / n_total  # ← INCORRECTO
```

Esto mezclaba:
- Contratos que **deben fallar** (ej. `ising3d_consistency` para datos no-CFT)
- Señales de **falso positivo holográfico** (el pipeline cree que es AdS cuando no lo es)

**Resultado:** `pass_rate` quedaba artificialmente bajo por construcción, no por mérito.

---

## Corrección: False Positive Rate (FPR)

### Nueva métrica

```
FPR = (señales holográficas disparadas) / (señales evaluables)
```

Donde:
- **Señal disparada** = el check indica holografía (True, score ≥ umbral)
- **Señal evaluable** = hay artefactos suficientes para computarla

### Qué mide

> "¿El pipeline se autoengaña creyendo que hay holografía cuando NO debería?"

### Umbrales (sin cambios)

| FPR | Status | Interpretación |
|-----|--------|----------------|
| < 0.2 | SUCCESS | Pipeline honesto |
| 0.2-0.5 | WARNING | Investigar señales disparadas |
| ≥ 0.5 | ALERT | Falso positivo sistemático |

---

## Separación conceptual

### A) HolographicSignals (entran en FPR)

Checks binarios de "¿parece holográfico?":

| Señal | Dispara si... |
|-------|---------------|
| `family_ads_like` | Familia clasificada contiene "AdS" |
| `einstein_score_high` | Score ≥ 0.5 |
| `dictionary_converged` | Convergencia = True |
| `deltas_in_physical_range` | >50% de Δ en rango 0.3-4.0 |
| `bulk_equations_clean` | n_equations > 0 con score > 0.3 |
| `delta_sigma_match` | Algún Δ ≈ 0.518 (±0.1) |
| `delta_epsilon_match` | Algún Δ ≈ 1.41 (±0.15) |

### B) ExpectedFailContracts (NO entran en FPR)

Contratos que deben fallar por diseño:

| Contrato | Por qué debe fallar |
|----------|---------------------|
| `ising3d_consistency` | Datos son campo masivo, no CFT |

Estos son **informativos**. Si pasan, es información adicional, pero no inflan/deflatan el FPR.

### C) Coverage

```
coverage = n_evaluable / n_total_signals
```

Un SUCCESS con coverage bajo es sospechoso (falso confort).

---

## Nueva estructura de salida

```json
"negative_control": {
  "status": "SUCCESS",
  "false_positive_rate": 0.14,
  "coverage": 0.86,
  "n_signals_triggered": 1,
  "n_signals_evaluable": 7,
  "n_signals_total": 7,
  "fpr_threshold": 0.2,
  "signals": [
    {
      "name": "family_ads_like",
      "status": "not_triggered",
      "value": "flat",
      "triggered": false,
      "evaluable": true
    },
    {
      "name": "einstein_score_high",
      "status": "triggered",
      "value": 0.62,
      "threshold": 0.5,
      "triggered": true,
      "evaluable": true
    }
  ],
  "expected_fail_contracts": [
    {"name": "ising3d_consistency", "passed": false, "note": "expected-fail"}
  ],
  "rationale": "FPR=14.3% < 20%. 1/7 señales disparadas. Pipeline honesto."
}
```

---

## Cambios en CLI

| v2 | v3 |
|----|-----|
| `--negative-control-max-pass-rate` | `--negative-control-fpr-threshold` |

---

## Regla de oro

En control negativo:

> **No se mide "qué pasa".**  
> Se mide **"si el pipeline se autoengaña y cree que pasa".**

Eso es lo que cuantifica el FPR.

---

## Archivos actualizados

- `09_real_data_and_dictionary_contracts_v3.py` — Implementación corregida
- Este addendum — Documentación del cambio

---

*Cambio propuesto por revisión epistemológica - Diciembre 2025*
