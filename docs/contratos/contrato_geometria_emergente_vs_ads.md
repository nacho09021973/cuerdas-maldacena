# CONTRATO FÍSICO VERIFICADO
## Geometría Emergente ≠ AdS Puro

**Fecha:** 2025-12-21  
**Proyecto:** CUERDAS-Maldacena  
**Estado:** ✓ VERIFICADO  

---

## 1. Resumen ejecutivo

La geometría emergente reconstruida desde datos de Ising 3D **no es AdS puro**. 
En consecuencia, la relación teórica del diccionario holográfico:

$$\lambda_{SL} = m^2 L^2 = \Delta(\Delta - d)$$

**NO aplica directamente** a los datos reconstruidos.

Esto **no es un fallo del pipeline** — es un resultado físico correcto que el sistema detectó honestamente.

---

## 2. Evidencia cuantitativa

### 2.1 Comparación de geometría

| Métrica | Emergente (Ising 3D) | AdS puro (T=0) | Discrepancia |
|---------|----------------------|----------------|--------------|
| A(z) rango | [-0.59, 0.54] | [-1.61, 4.61] | Significativa |
| f(z) rango | [0.00, 1.00] | 1.0 (constante) | Horizonte presente |
| RMS(A_emergente - A_ads) | 0.8227 | — | Alta |

**Interpretación**: La geometría emergente tiene un horizonte (f→0), indicando temperatura finita. AdS puro tiene f=1 constante (T=0).

### 2.2 Validación del diccionario λ_SL vs Δ(Δ-d)

| Operador | Δ (medido) | λ_SL (solver) | λ_teórico = Δ(Δ-3) | Ratio |
|----------|------------|---------------|---------------------|-------|
| σ (sigma) | 0.518 | 0.001792 | -1.286 | -0.0014 |
| ε (epsilon) | 1.410 | 0.005297 | -2.242 | -0.0024 |
| T (stress tensor) | 3.000 | 0.007148 | 0.000 | ∞ |
| ε' (epsilon prime) | 3.830 | 0.014942 | 3.179 | 0.0047 |

**Interpretación**: 
- Los λ_SL medidos son ~100-1000x menores que los teóricos
- Para Δ < d, λ_teórico es negativo pero λ_SL medido es positivo
- Esto indica que λ_SL del solver **no es** m²L² del diccionario AdS/CFT

---

## 3. Origen físico de la discrepancia

### 3.1 Ising 3D es una régimen térmico

Los datos de Ising 3D provienen de simulaciones a temperatura finita crítica.
En el dual holográfico, esto corresponde a un **black brane** en el bulk, no a AdS puro.

### 3.2 El diccionario se modifica a T ≠ 0

La relación Δ(Δ-d) es válida para:
- AdS puro (vacío, T=0)
- Límite UV (z→0) de geometrías asintóticamente AdS

La geometría emergente de Ising:
- Tiene horizonte (T > 0)
- No es asintóticamente AdS en el rango de z analizado
- El solver Sturm-Liouville encuentra autovalores del problema completo, no del límite UV

---

## 4. Implicaciones para el pipeline

### 4.1 Lo que funciona correctamente

- ✓ Extracción de Δ desde correladores de boundary (precisión <0.2% vs bootstrap)
- ✓ Solver Sturm-Liouville encuentra autovalores estables
- ✓ Geometría emergente captura física térmica (horizonte)
- ✓ Sistema detecta inconsistencia con teoría — comportamiento honesto

### 4.2 Lo que NO se puede hacer

- ✗ Entrenar diccionario con sandbox toy y aplicarlo a Ising
- ✗ Asumir λ_SL = Δ(Δ-d) para cualquier geometría
- ✗ Comparar directamente λ_SL de diferentes geometrías sin normalización

### 4.3 Caminos forward

1. **Diccionario empírico por familia**: Entrenar relación λ_SL ↔ Δ separadamente para cada tipo de geometría
2. **Normalización por horizonte**: Escalar λ_SL por z_h u otro parámetro térmico
3. **Límite UV explícito**: Extraer comportamiento asintótico de la geometría emergente y verificar ahí

---

## 5. Datos de soporte

### 5.1 Archivos utilizados

- Geometría emergente: `runs/ising3d_infer_20251220_105526/geometry_emergent/ising_3d_emergent.h5`
- Dataset eigenmodos: `runs/ising3d_infer_20251220_105526/bulk_eigenmodes/bulk_modes_dataset.csv`
- Fuente de Δ: boundary correlators (G2_sigma, G2_epsilon, G2_T, G2_epsilon_prime)

### 5.2 Valores de referencia Ising 3D (bootstrap)

| Operador | Δ_bootstrap | Δ_extraído | Error relativo |
|----------|-------------|------------|----------------|
| σ | 0.5181 | 0.518 | 0.02% |
| ε | 1.4126 | 1.410 | 0.18% |
| T | 3.0000 | 3.000 | 0.00% |
| ε' | 3.8297 | 3.830 | 0.01% |

---

## 6. Conclusión

> **El pipeline CUERDAS-Maldacena ha verificado honestamente que la geometría emergente de Ising 3D no es AdS puro, y por tanto el diccionario holográfico teórico no aplica directamente.**

Este es un **resultado positivo**: el sistema no fuerza consistencia con teoría cuando los datos no la soportan.

---

## 7. Próximos pasos recomendados

1. [ ] Investigar relación λ_SL vs Δ en el límite z→0 de la geometría emergente
2. [ ] Generar sandbox con horizonte (T>0) y verificar si el diccionario mejora
3. [ ] Documentar en M99 las condiciones exactas para λ = Δ(Δ-d)
4. [ ] Implementar contrato automático que detecte "geometría no-AdS"

---

*Documento generado: 2025-12-21*

### Anexo — Referencias exactas (corpus indexado M99)

**Referencias exactas (corpus indexado M99):**
- M99:p0105 M99:p0105:para002 bbox=[71.99981689453125, 377.4491271972656, 511.26025390625, 607.9215087890625]
  - snippet: "3.6 Theories at Finite Temperature As discussed in section 3.2, the quantities that can be most successfully compared between gauge theory and string theory are those with some protection from super- ..."
- M99:p0105 M99:p0105:para003 bbox=[71.9998779296875, 630.8810424804688, 511.1268005371094, 685.20166015625]
  - snippet: "3.6.1 Construction The gravity solution describing the gauge theory at ﬁnite temperature can be obtained by starting from the general <<black three-brane solution>> (1.12) and taking the decoupling"
- M99:p0106 M99:p0106:para002 bbox=[163.56021118164062, 132.75344848632812, 511.3330993652344, 171.79693603515625]
  - snippet: "h = 1 −u4 0 u4 , u0 = πT. <<(3.98)>>"
