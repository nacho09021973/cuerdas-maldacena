# Diseño Técnico: Control Negativo para CUERDAS-Maldacena

**Fecha:** 2025-12-21  
**Estado:** DRAFT - Requiere revisión

---

## 1. Resumen del Diseño

El script `04c_negative_controls.py` implementa un control negativo que:
1. Genera datos sintéticos de un campo escalar masivo en espacio plano
2. Los procesa a través del pipeline
3. Verifica que los contratos fallen (ausencia de holografía)
4. Documenta el "fallo esperado" como evidencia de honestidad científica

---

## 2. Decisiones de Diseño

### 2.1 Elección del sistema anti-holográfico

**Campo escalar masivo en flat space** porque:
- El término m²φ² rompe explícitamente la simetría conforme
- Los correladores decaen exponencialmente: G(r) ~ exp(-mr)/r^α
- No hay analogía AdS/CFT para este sistema
- Fácil de generar y verificar analíticamente

**Alternativas consideradas:**
- Gas ideal clásico: más simple pero menos cercano a física de campos
- Teoría de Yang-Mills en fase confinada: demasiado complejo
- CFT no unitaria: técnicamente conforme pero patológica

### 2.2 Formato de datos

He asumido compatibilidad con el formato HDF5 existente:
```
/negative_control/
  field            # Configuración del campo φ(x)
  correlator_2pt   # G(r) = <φ(0)φ(r)>
  /pseudo_boundary/
    G2_phi         # Pseudo-correladores (formato pipeline)
    distances
  [attrs]
    IS_NEGATIVE_CONTROL = 1
    EXPECTED_HOLOGRAPHIC = 0
```

### 2.3 Criterio de éxito

| Pass rate | Interpretación |
|-----------|----------------|
| < 20% | ✓ Sistema detecta no-holografía |
| 20-50% | ⚠ Advertencia, investigar |
| > 50% | 🚨 Posible falso positivo |

---

## 3. Puntos que Requieren Confirmación

### 3.1 Integración con pipeline existente

**Pregunta:** ¿Cómo se invoca el pipeline sobre datos nuevos?

He dejado `run_pipeline_on_negative_control()` como placeholder. Necesito saber:

1. ¿Los scripts 02-06 se invocan secuencialmente vía CLI?
   ```bash
   python 02_emergent_geometry_engine.py --input data.h5 --output runs/...
   python 04_geometry_physics_contracts.py --geometry_dir runs/...
   ```

2. ¿O hay un orquestador/runner?

3. ¿Qué formato exacto espera `02_emergent_geometry_engine.py` como input?

### 3.2 Contratos existentes

**Pregunta:** ¿Cuáles son los contratos específicos en `04_geometry_physics_contracts.py`?

Necesito la lista para mapear cuáles *deberían* fallar:

| Contrato | Debería fallar? | Razón |
|----------|-----------------|-------|
| Causalidad | ✓ | No hay horizonte en flat space |
| Regularidad | ? | Depende de la definición |
| Gap espectral | ✓ | No hay espectro conforme |
| Unitaridad | ✓ | Dimensiones fake violan |

### 3.3 Formato de pseudo-boundary data

**Pregunta:** ¿El pipeline espera correladores en formato específico?

He creado `pseudo_boundary_data` con:
```python
{
  'G2': {'phi': array, 'phi_squared': array},
  'fake_dimensions': {'phi': 0.1, 'phi_squared': -0.5},
  'distances': array
}
```

¿Esto es compatible con lo que lee `06_discover_symbolic_equations.py`?

### 3.4 Ubicación en el repo

**Pregunta:** ¿Dónde debería vivir este script?

Opciones:
- `04c_negative_controls.py` (junto a otros contratos)
- `tests/negative_controls.py` (separado como test)
- `tools/negative_control_generator.py` (como utilidad)

### 3.5 Semilla por defecto

**Pregunta:** ¿Deberíamos fijar una semilla canónica para reproducibilidad?

Propuesta: `--seed 42` como default documentado para runs de referencia.

---

## 4. Extensiones Futuras

### 4.1 Controles negativos adicionales

| Sistema | Por qué anti-holográfico |
|---------|-------------------------|
| Ruido blanco | Sin correlaciones |
| CFT trivial | Δ = 0 para todo |
| Teoría topológica | Sin grados locales |

### 4.2 Controles positivos (para comparar)

| Sistema | Por qué debería funcionar |
|---------|--------------------------|
| Ising 2D exacto | CFT soluble conocida |
| AdS puro sintético | Diccionario exacto |
| N=4 SYM en límites | Correspondencia probada |

---

## 5. Próximos Pasos

1. **Nacho confirma** puntos de la Sección 3
2. **Implementar integración** con pipeline real
3. **Ejecutar primera corrida** y verificar pass rate
4. **Documentar resultados** en hardening_plan

---

## 6. Código Pendiente

```python
# TODO: Implementar en run_pipeline_on_negative_control()

def run_pipeline_on_negative_control(h5_path, pipeline_dir):
    """
    REQUIERE: 
    - Saber cómo invocar 02_emergent_geometry_engine.py
    - Saber cómo recoger resultados de 04_geometry_physics_contracts.py
    - Parsear output de contratos para contar pass/fail
    """
    # Ejemplo de lo que podría ser:
    import subprocess
    
    # 1. Ejecutar geometría emergente
    result = subprocess.run([
        'python', pipeline_dir / '02_emergent_geometry_engine.py',
        '--input', str(h5_path),
        '--output_dir', str(h5_path.parent / 'geometry')
    ], capture_output=True)
    
    # 2. Ejecutar contratos
    result = subprocess.run([
        'python', pipeline_dir / '04_geometry_physics_contracts.py',
        '--geometry_dir', str(h5_path.parent / 'geometry'),
        '--output_json', str(h5_path.parent / 'contracts.json')
    ], capture_output=True)
    
    # 3. Parsear resultados
    with open(h5_path.parent / 'contracts.json') as f:
        contracts = json.load(f)
    
    return {
        'contracts_passed': [c for c in contracts if c['passed']],
        'contracts_failed': [c for c in contracts if not c['passed']]
    }
```

---

*Este documento es un draft para discusión. No commitear sin revisión.*
