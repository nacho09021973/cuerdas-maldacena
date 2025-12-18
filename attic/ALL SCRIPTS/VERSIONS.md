# Scripts: Versiones Canónicas vs Legacy

## ✅ USAR (versiones canónicas)

### Fase XI
| Script | Versión | Descripción |
|--------|---------|-------------|
| `00_generate_fase_11_v3.py` | v3 | Generador de datos CFT |
| `01_emergent_geometry_v2.py` | v2 | Learner de geometría |
| `02_discover_einstein_v2.py` | v2 | Descubridor de Einstein |
| `03_holographic_dictionary_v3.py` | v3 | Diccionario holográfico |
| `04_contracts_fase_11_v2.py` | v2 | Validador de contratos |
| `run_fase_11_v2.py` | v2 | Runner de Fase XI |

### Fase XII
| Script | Versión | Descripción |
|--------|---------|-------------|
| `fase12_real_data_adapters.py` | v1 | Adaptadores de datos reales |
| `fase12_prediction_engine.py` | v1 | Motor de predicciones |
| `fase12c_emergent_dictionary_real.py` | v1 | Diccionario emergente (CANÓNICO) |

### Fase XIII
| Script | Versión | Descripción |
|--------|---------|-------------|
| `fase13_theory_explorer.py` | v1 | Explorador de teorías |

### Utilidades
| Script | Versión | Descripción |
|--------|---------|-------------|
| `make_fase11_for_fase12c_v3.py` | v3 | Puente XI → XII.c |
| `contracts_fase_12_13.py` | v1 | Contratos para XII/XIII |
| `ecuaciones_emd.py` | v1 | Solver EMD |
| `analyze_discovered_equations.py` | v1 | Análisis de ecuaciones |

### Runners
| Script | Versión | Descripción |
|--------|---------|-------------|
| `run_cuerdas.py` | v1 | **Runner unificado (NUEVO)** |
| `run_fase_12_13.py` | v1 | Runner legacy XII/XIII |

---

## ⚠️ LEGACY (no usar directamente)

| Script | Reemplazado por |
|--------|-----------------|
| `03_holographic_dictionary_v2.py` | `03_holographic_dictionary_v3.py` |
| `make_fase11_for_fase12c.py` | `make_fase11_for_fase12c_v3.py` |
| `fase12c_emergent_dictionary.py` | `fase12c_emergent_dictionary_real.py` |
| `fase12c_emergent_dictionary_real_BACKUP.py` | Para referencia solamente |
| `fase12c_emergent_dictionary_real_ORIGINAL.py` | Para referencia solamente |

---

## 🗑️ CANDIDATOS A ELIMINAR

Estos archivos pueden eliminarse una vez confirmada la estabilidad:

```bash
# Una vez que v1.0 esté estable:
rm 03_holographic_dictionary_v2.py
rm make_fase11_for_fase12c.py
rm fase12c_emergent_dictionary.py
rm fase12c_emergent_dictionary_real_BACKUP.py
rm fase12c_emergent_dictionary_real_ORIGINAL.py
rm from_google_import_genai.py  # No se usa
```

---

## 📝 Notas

1. **`run_cuerdas.py`** es el nuevo runner unificado y debería ser el punto de entrada principal.

2. Los archivos `_BACKUP` y `_ORIGINAL` se mantienen temporalmente para referencia en caso de regresiones.

3. `from_google_import_genai.py` parece ser un stub no utilizado, confirmar antes de eliminar.
