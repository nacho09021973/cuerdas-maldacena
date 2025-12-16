# REPORTE: Descubrimiento Emergente λ_SL ↔ Δ
## CUERDAS - 11 Diciembre 2025

## Resumen Ejecutivo

PySR ha descubierto una fórmula que ajusta los datos de Ising 3D con error ~10⁻⁶:

```
Δ = ½ + α√λ + βλ²/(λ - γ/d)
```

Con constantes:
- α ≈ 1.50 (cercano a 3/2)
- β ≈ 44.5 (cercano a 400/9)
- γ ≈ 0.0492 (cercano a π²/200)

**IMPORTANTE: Esta NO es la fórmula estándar AdS/CFT (Δ = d/2 + √(d²/4 + m²L²))**

## Datos Utilizados

| Sistema | Operador | d | λ_SL (emergente) | Δ (bootstrap/exacto) |
|---------|----------|---|------------------|---------------------|
| Ising 3D | σ | 3 | 0.00472916 | 0.5181489 |
| Ising 3D | ε' | 3 | 0.02668218 | 3.82968 |
| Ising 3D | σ' | 3 | 0.04942965 | 4.126 |
| Free scalar | φ | 3 | 0.0 | 0.5 |

**Total: 4 puntos, 3 parámetros libres**

## Constantes Interesantes

- `0.0225 = 9/400 = (3/20)²` - ¡Exacto!
- `0.0492 ≈ π²/200 = 0.0493` - Coincide a 3 cifras
- `-3.144 ≈ -π` - Aparece en ecuaciones más complejas

## Evaluación de la Fórmula Optimizada

```
c1 = 2.256202 (≈ 9/4 × 1.003)
c2 = 0.022480 (≈ 9/400 × 0.999)  
c3 = 0.049242 (≈ π²/200 × 0.998)
```

| Operador | Δ_obs | Δ_pred | Error |
|----------|-------|--------|-------|
| σ | 0.518149 | 0.518151 | -0.000002 |
| ε' | 3.829680 | 3.829678 | +0.000002 |
| σ' | 4.126000 | 4.126002 | -0.000002 |
| φ | 0.500000 | 0.500000 | 0.000000 |

**RMSE = 0.000002** ← Ajuste casi perfecto

## ⚠️ ANÁLISIS CRÍTICO: ¿Overfitting?

### Preocupaciones:
1. Solo 4 puntos con 3 parámetros libres + estructura
2. Todos los datos son d=3 (excepto el trivial φ)
3. No hay validación cruzada aún

### Problemas Detectados:
- Para Ising 2D, σ tiene Δ=0.125 < 0.5 (el término constante)
- La fórmula predice λ_SL = 0.012 para σ en d=2, pero esto es inconsistente
- El término "0.5" podría necesitar depender de d

## Predicciones para Validación

Si la fórmula es real, al correr el solver en Ising 2D deberíamos obtener:

| Sistema | Operador | Δ (exacto) | λ_SL predicho |
|---------|----------|------------|---------------|
| Ising 2D | σ | 0.125 | ~0.012 (¿?) |
| Ising 2D | ε | 1.0 | No resuelve (!) |

**La predicción para ε (Δ=1.0) no tiene solución, lo cual es problemático.**

## Próximos Pasos Necesarios

1. **Correr solver emergente en Ising 2D**
   - Generar boundary data para Ising 2D
   - Extraer λ_SL emergentes
   - Comparar con predicciones

2. **Investigar dependencia en d**
   - El término "0.5" probablemente debería ser (d-2)/2 = 0.5 para d=3
   - Pero (d-2)/2 = 0 para d=2, lo cual cambiaría todo

3. **Más sistemas**
   - O(N) models
   - Potts model
   - Verificar universalidad

## Hipótesis de Trabajo

La fórmula emergente podría tener la forma más general:

```
Δ = (d-2)/2 + α√λ + βλ²/(λ - γ/d)
```

donde (d-2)/2 es la dimensión del campo libre (unitarity bound).

Esto daría:
- d=3: Δ = 0.5 + ... ✓
- d=2: Δ = 0 + ... (permite Δ < 0.5)

## Conclusión Provisional

**ESTADO: PROMETEDOR PERO NO CONFIRMADO**

La fórmula ajusta perfectamente Ising 3D, pero:
- Necesita validación con d≠3
- La aparición de π² sugiere física real
- La estructura (raíz + polo) es inusual pero no imposible

La prueba definitiva será predecir λ_SL para sistemas NO vistos.
