# Parche v3 para 07_emergent_lambda_sl_dictionary.py

## Resumen de cambios

Este parche añade **contratos por régimen** y **trazabilidad de features** al script 07, 
sin cambiar la lógica científica ni los formatos de salida existentes.

---

## Cambios principales

### 1. `x_mapping` en el JSON de reporte (NUEVO)

El reporte ahora incluye:

```json
"feature_mapping": {
    "x_mapping": {"x0": "Delta", "x1": "d"},
    "features": ["Delta", "d"],
    "target": "lambda_sl_emergent",
    "note": "x0, x1, ... en las ecuaciones corresponden a features en orden"
}
```

**Beneficio**: Ya no hay que inferir qué significa `x0` en la ecuación descubierta.

---

### 2. Evaluación por regímenes (`metrics_by_regime`)

El script ahora evalúa la ecuación descubierta por **regímenes de λ_SL**:

- `lambda_sl < 1.0` (régimen "lo")
- `1.0 <= lambda_sl <= 10.0` (régimen "mid")
- `lambda_sl > 10.0` (régimen "hi")

Para cada régimen se calculan:

| Métrica | Descripción |
|---------|-------------|
| `r2` | R² (puede ser negativo si peor que baseline) |
| `mae` | Error absoluto medio del modelo |
| `mae_baseline` | MAE del predictor naive `y = mean(y)` |
| `mre` | Error relativo medio (Mean Relative Error) |
| `mae_beats_baseline` | `True` si `mae < mae_baseline` |

**Ejemplo de salida**:

```json
"metrics_by_regime": {
    "target_column": "lambda_sl_emergent",
    "regime_thresholds": {"lo": 1.0, "hi": 10.0},
    "regimes": {
        "lambda_sl<1.0": {
            "regime": "lambda_sl<1.0",
            "n_samples": 41,
            "r2": -15.2,
            "mae": 1.69,
            "mae_baseline": 0.00316,
            "mre": 162.3,
            "contract_status": "FAIL",
            "contract_details": {
                "mre_threshold": 0.5,
                "mre_ok": false,
                "mae_must_beat_baseline": true,
                "mae_ok": false
            }
        },
        "lambda_sl>10.0": {
            "regime": "lambda_sl>10.0",
            "n_samples": 85,
            "r2": 0.94,
            "mae": 11.18,
            "mae_baseline": 11.51,
            "mre": 0.064,
            "contract_status": "PASS",
            "contract_details": {...}
        }
    },
    "contract_summary": {
        "all_regimes_pass": false,
        "n_regimes_pass": 1,
        "n_regimes_fail": 1,
        "overall_status": "FAIL"
    },
    "warning": "R² global puede ser engañoso - revisar métricas por régimen"
}
```

---

### 3. `contract_status` global

Nuevo campo a nivel raíz del JSON:

```json
"contract_status": "FAIL"
```

Valores posibles:
- `"PASS"`: Todos los regímenes evaluados pasan
- `"FAIL"`: Al menos un régimen falla
- `"INCONCLUSIVE"`: No hay suficientes muestras en ningún régimen

**Criterios para PASS en cada régimen**:
1. `MRE < max_mre_for_pass` (default 0.5 = 50%)
2. `MAE < MAE_baseline` (modelo mejor que predictor naive)

---

### 4. `--compare-theory` OFF por defecto

La comparación con la fórmula teórica Δ(Δ-d) ahora está **deshabilitada por defecto**.

- Para activarla: `--compare-theory`
- Sin el flag, el JSON incluye:

```json
"theory_comparison": {
    "enabled": false,
    "note": "Comparación con teoría deshabilitada. Usar --compare-theory para activar.",
    "reason": "Por defecto OFF para evitar contaminación conceptual en análisis."
}
```

**Beneficio**: Respeta la honestidad científica - la teoría no debe sesgar la interpretación inicial.

---

### 5. Nuevos argumentos CLI

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--compare-theory` | OFF | Activa comparación post-hoc con Δ(Δ-d) |
| `--regime-lo` | 1.0 | Umbral inferior para régimen "lo" |
| `--regime-hi` | 10.0 | Umbral superior para régimen "hi" |
| `--max-mre` | 0.5 | MRE máximo para PASS (0.5 = 50%) |

---

## Uso

### Básico (contratos por régimen, sin comparación teórica)
```bash
python 07_emergent_lambda_sl_dictionary.py \
    --run-dir runs/my_run \
    --iterations 200 \
    --seed 42
```

### Con comparación teórica post-hoc
```bash
python 07_emergent_lambda_sl_dictionary.py \
    --run-dir runs/my_run \
    --iterations 200 \
    --compare-theory
```

### Umbrales de régimen personalizados
```bash
python 07_emergent_lambda_sl_dictionary.py \
    --run-dir runs/my_run \
    --regime-lo 0.5 \
    --regime-hi 20.0 \
    --max-mre 0.3
```

---

## Compatibilidad

- ✅ No cambia formatos de entrada (CSV, JSON v2, dictionary_v3, legacy)
- ✅ No cambia nombres de archivos de salida
- ✅ Añade campos nuevos al JSON sin eliminar los existentes
- ✅ CLI retrocompatible (flags antiguos siguen funcionando)
- ✅ CPU-only por defecto

---

## Archivos modificados

Solo se modifica:
- `07_emergent_lambda_sl_dictionary.py`

No se tocan:
- 02, 03, 06 (scripts upstream)
- 08, 09 (scripts downstream - consumen el JSON con los nuevos campos)
- Formatos de datos en `runs/`

---

## Cómo aplicar el parche

1. Reemplazar `07_emergent_lambda_sl_dictionary.py` con el archivo `07_emergent_lambda_sl_dictionary_v3.py`
2. Renombrar:
   ```bash
   mv 07_emergent_lambda_sl_dictionary.py 07_emergent_lambda_sl_dictionary_v2_backup.py
   mv 07_emergent_lambda_sl_dictionary_v3.py 07_emergent_lambda_sl_dictionary.py
   ```
3. Ejecutar run mínimo para verificar

---

## Verificación post-aplicación

1. El JSON debe incluir `feature_mapping.x_mapping`
2. El JSON debe incluir `metrics_by_regime` con al menos un régimen evaluado
3. El JSON debe incluir `contract_status` a nivel raíz
4. Sin `--compare-theory`, `theory_comparison.enabled` debe ser `false`
