# DISEÑO TÉCNICO: Integración Control Negativo Anti-Holografía en 09

**Versión:** 1.0  
**Fecha:** 2025-12-21  
**Estado:** LISTO PARA REVISIÓN

---

## 1. Resumen Ejecutivo

Este documento describe la integración del control negativo (generado por `04c_negative_controls.py`) en el script agregador de contratos `09_real_data_and_dictionary_contracts.py`.

**Objetivo:** Detectar falsos positivos holográficos verificando que datos explícitamente anti-holográficos **NO** pasen los contratos del pipeline.

**Resultado:** Un nuevo bloque `"negative_control"` en el JSON de salida que documenta el status de la verificación.

---

## 2. Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────┐
│  04c_negative_controls.py                                       │
│  ───────────────────────                                        │
│  Input: parámetros (mass, lattice_size, seed)                   │
│  Output: runs/negative_control_{id}/negative_control_{id}.h5    │
│    attrs: IS_NEGATIVE_CONTROL=1, EXPECTED_HOLOGRAPHIC=0         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline normal (02 → 03 → ... → 08)                           │
│  ─────────────────────────────────────                          │
│  Procesa el HDF5 como datos normales                            │
│  Output: runs/negative_control_{id}/                            │
│    ├── geometry_emergent/                                       │
│    ├── bulk_equations/                                          │
│    ├── emergent_dictionary/                                     │
│    └── holographic_dictionary/                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  09_real_data_and_dictionary_contracts.py                       │
│  ─────────────────────────────────────────                      │
│  Flags nuevos:                                                  │
│    --negative-control-run-dir runs/negative_control_{id}/       │
│    --negative-control-h5 (opcional, autodetecta)                │
│    --require-negative-control                                   │
│    --negative-control-max-pass-rate 0.2                         │
│                                                                 │
│  Pasos:                                                         │
│    1. Verificar HDF5 (IS_NEGATIVE_CONTROL=1)                    │
│    2. Cargar artefactos (geometry, einstein, dictionary)        │
│    3. Ejecutar contratos sobre artefactos                       │
│    4. Calcular pass_rate                                        │
│    5. Determinar status (SUCCESS/WARNING/ALERT)                 │
│    6. Añadir bloque "negative_control" al JSON                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Nuevos Argumentos CLI

| Argumento | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--negative-control-run-dir` | Path | None | Directorio del run sobre datos anti-holográficos |
| `--negative-control-h5` | Path | None | HDF5 original (autodetecta si no se da) |
| `--require-negative-control` | Flag | False | Si ALERT → exit 1 |
| `--negative-control-max-pass-rate` | Float | 0.2 | Umbral para SUCCESS |

**Compatibilidad:** Todos los argumentos son opcionales. Sin ellos, el script funciona exactamente igual que antes.

---

## 4. Nuevo Contrato: `contract_anti_holography`

```python
def contract_anti_holography(
    self,
    predicted_family: str,
    predicted_Deltas: List[float],
    einstein_score: Optional[float] = None,
    dictionary_converged: Optional[bool] = None
) -> Dict[str, Any]:
```

**Checks que realiza:**

| Check | Señal holográfica | Señal anti-holográfica |
|-------|-------------------|------------------------|
| Familia geométrica | AdS-like | Otra (flat, dS, unknown) |
| Einstein score | ≥ 0.5 | < 0.5 |
| Diccionario convergió | Sí | No |
| Δ en rango físico | > 50% | ≤ 50% |

**Evaluación:**
- Si mayoría de señales son anti-holográficas → `passed = False` (BIEN para control negativo)
- Si mayoría son holográficas → `passed = True` (MAL para control negativo)

---

## 5. Estructura del Bloque `negative_control`

```json
{
  "fase12": { ... },
  "fase13": { ... },
  "negative_control": {
    "status": "SUCCESS",
    "pass_rate": 0.15,
    "n_contracts": 5,
    "n_passed": 1,
    "n_failed": 4,
    "max_pass_rate_threshold": 0.2,
    "h5_path": "runs/negative_control_20251221/negative_control_20251221.h5",
    "h5_verified": true,
    "IS_NEGATIVE_CONTROL": 1,
    "EXPECTED_HOLOGRAPHIC": 0,
    "h5_metadata": {
      "type": "massive_scalar_flat_space",
      "mass": 1.0,
      "lattice_size": 100,
      "dimension": 2,
      "conformal": false
    },
    "artifacts_summary": {
      "geometry_found": true,
      "einstein_found": true,
      "dictionary_found": true
    },
    "contracts_passed": [
      {"name": "has_predicted_Deltas", "passed": true}
    ],
    "contracts_failed": [
      {"name": "family_is_ads_like", "passed": false, "got": "flat"},
      {"name": "Delta_sigma_match", "passed": false, "reason": "..."}
    ],
    "rationale": "El pipeline detectó correctamente que los datos anti-holográficos no producen holografía válida."
  }
}
```

---

## 6. Lógica de Status

```python
def evaluate_status(pass_rate, max_threshold=0.2):
    if pass_rate < max_threshold:
        return "SUCCESS"    # Pipeline honesto
    elif pass_rate < 0.5:
        return "WARNING"    # Investigar
    else:
        return "ALERT"      # Posible falso positivo
```

| pass_rate | Status | Interpretación |
|-----------|--------|----------------|
| < 20% | SUCCESS | ✓ El pipeline distingue holografía real de ruido |
| 20-50% | WARNING | ⚠ Algunos contratos pasan sin razón física |
| ≥ 50% | ALERT | 🚨 El pipeline produce falsos positivos |

---

## 7. Justificación de Honestidad Epistemológica

### 7.1 No hay inyección de teoría

- Los datos anti-holográficos pasan por el pipeline **sin modificaciones**.
- Los contratos se aplican **post-hoc** sobre outputs ya generados.
- No se modifica entrenamiento, losses, features ni regularizadores.

### 7.2 Los contratos son observables empíricos

Los checks comparan:
- Familia predicha (observable del modelo de geometría)
- Δ predichos (observables del diccionario)
- Einstein score (observable de symbolic regression)

Con valores de referencia (bootstrap Ising 3D). Esto es **comparación post-hoc**, no guía.

### 7.3 El control es falsable

Si el pipeline pasa contratos sobre datos anti-holográficos:
- Eso es **evidencia de problema** (falso positivo)
- El sistema dispara ALERT
- Se documenta para auditoría

El diseño está construido para **detectar fallas**, no ocultarlas.

### 7.4 Transparencia total

El bloque `negative_control` documenta:
- Qué contratos pasaron (falsos positivos menores)
- Qué contratos fallaron (comportamiento esperado)
- Metadata completa del HDF5 de origen
- Rationale de la conclusión

---

## 8. Protección Contra Falsos Positivos Sistemáticos

### 8.1 Escenario de riesgo (sin control negativo)

1. El pipeline entrena sobre datos variados
2. Produce geometría "AdS-like" para casi todo
3. Los contratos pasan porque el diccionario está calibrado
4. **Conclusión falsa:** "El pipeline descubre holografía real"

### 8.2 Con control negativo

1. Se genera sistema **explícitamente no-holográfico**
2. El pipeline procesa estos datos
3. Si los contratos pasan → ALERT
4. El investigador sabe que hay problema sistemático

### 8.3 Tipos de problemas que detecta

| Problema | Cómo se manifiesta |
|----------|-------------------|
| Overfitting del clasificador de familias | `family_is_ads_like` pasa para flat space |
| Diccionario siempre converge | `dictionary_converged` es True siempre |
| Δ predichos en rango por casualidad | `Delta_sigma_match` pasa |
| Einstein score inflado | `einstein_score_high` para ecuaciones triviales |

---

## 9. Uso Típico

### 9.1 Generar control negativo

```bash
python 04c_negative_controls.py \
  --output_dir runs/negative_control_test \
  --mass 1.0 \
  --lattice_size 100 \
  --seed 42 \
  --generate_only
```

### 9.2 Ejecutar pipeline sobre control negativo

```bash
python 02_emergent_geometry_engine.py \
  --data-dir runs/negative_control_test \
  --output-dir runs/negative_control_test

python 03_discover_bulk_equations.py --run-dir runs/negative_control_test
# ... resto del pipeline
```

### 9.3 Verificar contratos incluyendo control negativo

```bash
python 09_real_data_and_dictionary_contracts.py \
  --phase both \
  --run-dir runs/main_experiment \
  --negative-control-run-dir runs/negative_control_test \
  --require-negative-control
```

### 9.4 Interpretar resultado

```
>> Ejecutando control negativo desde runs/negative_control_test

   Status: SUCCESS
   Pass rate: 15.0%

RESUMEN FINAL
======================================================================
  fase12: OK (3/4)
  fase13: OK (4/5)
  negative_control: SUCCESS (pass_rate=15.0%)

  Output: runs/main_experiment/contracts/contracts_12_13.json
======================================================================
```

---

## 10. Cambios Respecto al Script Original

### Funciones añadidas:

1. `verify_negative_control_h5()` - Valida HDF5 con atributos correctos
2. `find_negative_control_h5()` - Busca HDF5 en directorio
3. `load_negative_control_artifacts()` - Carga geometría, einstein, diccionario
4. `run_negative_control_check()` - Orquesta la verificación completa
5. `ContractsFase12.contract_anti_holography()` - Nuevo contrato

### Argumentos CLI añadidos:

- `--negative-control-run-dir`
- `--negative-control-h5`
- `--require-negative-control`
- `--negative-control-max-pass-rate`

### Modificaciones al main():

- Bloque de ejecución de control negativo
- Variable `negative_control_alert` para exit code
- Inclusión de `negative_control` en JSON de salida

### Compatibilidad:

- **100% backward compatible** - Sin los nuevos argumentos, funciona igual
- **Sin cambios a formatos existentes** - Solo añade bloque opcional
- **Sin dependencias nuevas** - h5py ya era opcional

---

## 11. Tests Sugeridos

```python
def test_negative_control_h5_verification():
    """Verifica que HDF5 sin atributos falle."""
    
def test_negative_control_pass_rate_calculation():
    """Verifica cálculo correcto de pass_rate."""
    
def test_negative_control_status_thresholds():
    """Verifica SUCCESS/WARNING/ALERT según pass_rate."""
    
def test_require_negative_control_exit_code():
    """Verifica exit 1 cuando ALERT + --require."""
    
def test_backward_compatibility():
    """Verifica que sin argumentos nuevos funciona igual."""
```

---

## 12. Próximos Pasos Recomendados

1. **Revisar este diseño** - Validar que cumple expectativas
2. **Integrar en repo** - Copiar `09_...v2.py` y testear
3. **Documentar en README** - Añadir sección sobre control negativo
4. **Añadir a CI** - `--require-negative-control` en pipeline de tests
5. **Ejecutar primer run completo** - Validar con datos reales

---

## 13. Notas de Implementación

### 13.1 Manejo de artefactos faltantes

Si no se encuentran artefactos del pipeline:
- Se documenta en `errors`
- Status queda como `INCOMPLETE`
- No se fuerza fallo (puede ser que el pipeline no se ejecutó)

### 13.2 Reutilización de contratos

Se reutiliza `ContractsFase12` para ejecutar los mismos contratos sobre datos del control negativo. Esto garantiza consistencia.

### 13.3 Threshold configurable

El umbral de 0.2 es configurable vía `--negative-control-max-pass-rate`. Esto permite ajustar sensibilidad según el proyecto.

---

*Documento preparado para revisión por Nacho.*
*Proyecto CUERDAS-Maldacena - Diciembre 2025*
