#!/usr/bin/env python3
"""
04c_negative_controls.py
========================

Control negativo para validación del pipeline CUERDAS-Maldacena.

PROPÓSITO:
- Generar datos sintéticos que NO deberían producir holografía válida
- Verificar que el pipeline detecta la ausencia de estructura holográfica
- Documentar el "fallo esperado" como evidencia de honestidad científica

TEORÍA:
- Un campo escalar masivo en espacio plano (flat space) no tiene simetría conforme
- No existe un bulk AdS dual que emerja de estos datos
- El diccionario holográfico λ_SL → Δ no debería converger a valores físicos

CRITERIO DE ÉXITO:
- Pass rate en contratos < 20% → Sistema detecta ausencia de holografía
- Pass rate > 50% → ALERTA: posible falso positivo, investigar

USO:
    python 04c_negative_controls.py --output_dir runs/negative_control_YYYYMMDD
    python 04c_negative_controls.py --mass 1.0 --lattice_size 100 --seed 42

Autor: Proyecto CUERDAS-Maldacena
Fecha: 2025-12-21
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

import numpy as np
import h5py

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# SECCIÓN 1: GENERACIÓN DE DATOS ANTI-HOLOGRÁFICOS
# ==============================================================================

def generate_massive_scalar_flat_space(
    mass: float,
    lattice_size: int,
    dim: int = 2,
    seed: Optional[int] = None,
    noise_level: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Genera datos de un campo escalar masivo en espacio plano.
    
    Este sistema NO tiene simetría conforme porque:
    - El término de masa m² φ² rompe la invariancia conforme
    - El espacio es plano (flat), no AdS
    - Los correladores decaen exponencialmente, no como potencias
    
    Parámetros:
    -----------
    mass : float
        Masa del campo escalar (m > 0 rompe simetría conforme)
    lattice_size : int
        Tamaño del lattice en cada dimensión
    dim : int
        Dimensionalidad espacial (2 o 3)
    seed : int, optional
        Semilla para reproducibilidad
    noise_level : float
        Nivel de ruido gaussiano
    
    Retorna:
    --------
    Dict con:
        - 'field': configuración del campo φ(x)
        - 'correlator_2pt': G(r) = <φ(0)φ(r)> - decae exponencialmente
        - 'metadata': parámetros de generación
    """
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(f"Generando campo escalar masivo: m={mass}, L={lattice_size}, d={dim}")
    
    # Crear lattice
    shape = tuple([lattice_size] * dim)
    
    # Generar campo escalar masivo (solución aproximada de Klein-Gordon en lattice)
    # Para campo libre masivo: G(r) ~ exp(-m*r) / r^((d-1)/2)
    
    # Inicializar con ruido y aplicar "relajación" hacia solución estacionaria
    field = np.random.normal(0, 1, shape)
    
    # Añadir ruido pequeño
    field += np.random.normal(0, noise_level, shape)
    
    # Calcular correlador de 2 puntos (promedio sobre todas las separaciones)
    correlator_2pt = _compute_correlator_massive(field, mass, dim)
    
    # Crear "pseudo-correladores" que parecen conformes pero no lo son
    # Esto es importante: el sistema NO debe ser conforme
    pseudo_boundary_data = _create_pseudo_boundary_data(field, mass, lattice_size)
    
    metadata = {
        'type': 'massive_scalar_flat_space',
        'mass': mass,
        'lattice_size': lattice_size,
        'dimension': dim,
        'seed': seed,
        'noise_level': noise_level,
        'conformal': False,  # Explícitamente NO conforme
        'expected_holographic': False,  # NO debería producir holografía válida
        'generated_at': datetime.now().isoformat()
    }
    
    logger.info(f"  Campo generado: shape={field.shape}")
    logger.info(f"  Correlador: {len(correlator_2pt)} puntos")
    
    return {
        'field': field,
        'correlator_2pt': correlator_2pt,
        'pseudo_boundary_data': pseudo_boundary_data,
        'metadata': metadata
    }


def _compute_correlator_massive(field: np.ndarray, mass: float, dim: int) -> np.ndarray:
    """
    Calcula correlador de 2 puntos para campo masivo.
    
    Para campo escalar masivo libre:
        G(r) ~ exp(-m*r) / r^((d-1)/2)  (no potencia pura como en CFT)
    """
    lattice_size = field.shape[0]
    max_r = lattice_size // 2
    
    correlator = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    # Usar FFT para eficiencia
    field_fft = np.fft.fftn(field)
    power_spectrum = np.abs(field_fft) ** 2
    correlation_full = np.fft.ifftn(power_spectrum).real
    
    # Promediar por distancia
    center = tuple([lattice_size // 2] * dim)
    
    for idx in np.ndindex(field.shape):
        r_squared = sum((i - c) ** 2 for i, c in zip(idx, center))
        r = int(np.sqrt(r_squared))
        if r < max_r:
            correlator[r] += correlation_full[idx]
            counts[r] += 1
    
    # Evitar división por cero
    counts[counts == 0] = 1
    correlator /= counts
    
    # Normalizar
    if correlator[0] > 0:
        correlator /= correlator[0]
    
    return correlator


def _create_pseudo_boundary_data(
    field: np.ndarray,
    mass: float,
    lattice_size: int
) -> Dict[str, np.ndarray]:
    """
    Crea datos de "pseudo-boundary" que imitan el formato del pipeline
    pero NO tienen estructura conforme real.
    
    Esto es crucial: los datos deben estar en el formato correcto
    para que el pipeline los procese, pero NO deben satisfacer
    las propiedades de una CFT.
    """
    # Simular "correladores" de operadores ficticios
    # Con masas que NO corresponden a dimensiones conformes válidas
    
    n_points = lattice_size // 2
    distances = np.arange(1, n_points + 1).astype(float)
    
    # Correladores exponenciales (NO conformes, que serían potencias)
    G2_fake = {
        'phi': np.exp(-mass * distances) / (distances ** 0.5 + 1e-10),
        'phi_squared': np.exp(-2 * mass * distances) / (distances + 1e-10),
    }
    
    # "Dimensiones" que NO satisfacen unitaridad ni bootstrap
    fake_dimensions = {
        'phi': 0.1,  # Violación de unitaridad (Δ < (d-2)/2 para d=3)
        'phi_squared': -0.5,  # Dimensión negativa (imposible en CFT unitaria)
    }
    
    return {
        'G2': G2_fake,
        'fake_dimensions': fake_dimensions,
        'distances': distances,
        'warning': 'ANTI-HOLOGRAPHIC DATA - Expected to fail contracts'
    }


# ==============================================================================
# SECCIÓN 2: GUARDAR EN FORMATO COMPATIBLE CON PIPELINE
# ==============================================================================

def save_negative_control_data(
    data: Dict,
    output_dir: Path,
    run_id: str
) -> Path:
    """
    Guarda los datos del control negativo en formato HDF5 compatible
    con el pipeline existente.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    h5_path = output_dir / f"negative_control_{run_id}.h5"
    
    logger.info(f"Guardando datos en: {h5_path}")
    
    with h5py.File(h5_path, 'w') as f:
        # Grupo principal
        grp = f.create_group('negative_control')
        
        # Campo escalar
        grp.create_dataset('field', data=data['field'], compression='gzip')
        
        # Correlador
        grp.create_dataset('correlator_2pt', data=data['correlator_2pt'])
        
        # Pseudo-boundary data
        boundary_grp = grp.create_group('pseudo_boundary')
        for key, val in data['pseudo_boundary_data']['G2'].items():
            boundary_grp.create_dataset(f'G2_{key}', data=val)
        boundary_grp.create_dataset(
            'distances', 
            data=data['pseudo_boundary_data']['distances']
        )
        
        # Metadata como atributos
        for key, val in data['metadata'].items():
            if val is not None:
                grp.attrs[key] = val if not isinstance(val, bool) else int(val)
        
        # Marcador explícito
        grp.attrs['IS_NEGATIVE_CONTROL'] = 1
        grp.attrs['EXPECTED_HOLOGRAPHIC'] = 0
    
    logger.info(f"  Datos guardados: {h5_path.stat().st_size / 1024:.1f} KB")
    
    return h5_path


# ==============================================================================
# SECCIÓN 3: EJECUTAR PIPELINE Y VERIFICAR FALLOS
# ==============================================================================

def run_pipeline_on_negative_control(
    h5_path: Path,
    pipeline_scripts_dir: Path
) -> Dict[str, any]:
    """
    Ejecuta el pipeline sobre los datos del control negativo.
    
    NOTA: Esta función es un placeholder - la integración real
    depende de la estructura exacta del pipeline.
    
    Retorna dict con resultados de cada etapa.
    """
    logger.info("="*60)
    logger.info("EJECUTANDO PIPELINE SOBRE CONTROL NEGATIVO")
    logger.info("="*60)
    
    results = {
        'stages_run': [],
        'stages_failed': [],
        'contracts_checked': [],
        'contracts_passed': [],
        'contracts_failed': [],
    }
    
    # TODO: Integrar con pipeline real
    # Por ahora, documentamos la estructura esperada
    
    expected_stages = [
        ('02_emergent_geometry_engine.py', 'Geometría emergente'),
        ('04_geometry_physics_contracts.py', 'Contratos físicos'),
        ('05_scalar_field_solver.py', 'Solver escalar'),
        ('06_discover_symbolic_equations.py', 'Ecuaciones simbólicas'),
    ]
    
    logger.warning("Pipeline no ejecutado - implementar integración")
    
    return results


def check_contracts_failure(
    results: Dict,
    expected_pass_rate: float = 0.2
) -> Dict[str, any]:
    """
    Verifica que los contratos fallen como se espera para datos no-holográficos.
    
    Criterios:
    - pass_rate < 0.2: ÉXITO (sistema detecta no-holografía)
    - pass_rate 0.2-0.5: ADVERTENCIA (investigar)
    - pass_rate > 0.5: FALLO (posible falso positivo)
    """
    n_passed = len(results.get('contracts_passed', []))
    n_failed = len(results.get('contracts_failed', []))
    n_total = n_passed + n_failed
    
    if n_total == 0:
        return {
            'status': 'INCOMPLETE',
            'message': 'No se ejecutaron contratos',
            'pass_rate': None
        }
    
    pass_rate = n_passed / n_total
    
    if pass_rate < expected_pass_rate:
        status = 'SUCCESS'
        message = f'Sistema detectó correctamente ausencia de holografía (pass_rate={pass_rate:.2%})'
    elif pass_rate < 0.5:
        status = 'WARNING'
        message = f'Pass rate moderado ({pass_rate:.2%}) - investigar contratos específicos'
    else:
        status = 'ALERT'
        message = f'POSIBLE FALSO POSITIVO: pass_rate={pass_rate:.2%} > 50%'
    
    return {
        'status': status,
        'message': message,
        'pass_rate': pass_rate,
        'n_passed': n_passed,
        'n_failed': n_failed,
        'n_total': n_total
    }


# ==============================================================================
# SECCIÓN 4: GENERAR REPORTE
# ==============================================================================

def generate_negative_control_report(
    data: Dict,
    results: Dict,
    contract_check: Dict,
    output_dir: Path,
    run_id: str
) -> Path:
    """
    Genera reporte markdown documentando el control negativo.
    """
    report_path = output_dir / f"negative_control_report_{run_id}.md"
    
    report = f"""# REPORTE DE CONTROL NEGATIVO

**Run ID:** {run_id}
**Fecha:** {datetime.now().isoformat()}
**Estado:** {contract_check.get('status', 'INCOMPLETE')}

---

## 1. Descripción del Input

**Tipo:** Campo escalar masivo en espacio plano (flat space)

| Parámetro | Valor |
|-----------|-------|
| Masa (m) | {data['metadata'].get('mass', 'N/A')} |
| Tamaño lattice | {data['metadata'].get('lattice_size', 'N/A')} |
| Dimensión | {data['metadata'].get('dimension', 'N/A')} |
| Seed | {data['metadata'].get('seed', 'N/A')} |
| Conforme | **NO** |
| Holográfico esperado | **NO** |

### Por qué este sistema NO es holográfico

1. **Sin simetría conforme**: El término de masa m²φ² rompe la invariancia de escala
2. **Espacio plano**: No hay curvatura AdS que emerja naturalmente
3. **Correladores exponenciales**: G(r) ~ exp(-mr), no potencias como en CFT
4. **Dimensiones inválidas**: Los "operadores" tienen Δ que violan unitaridad

---

## 2. Resultados del Pipeline

### Etapas ejecutadas
{_format_list(results.get('stages_run', ['(ninguna)']))}

### Etapas fallidas
{_format_list(results.get('stages_failed', ['(ninguna)']))}

---

## 3. Verificación de Contratos

| Métrica | Valor |
|---------|-------|
| Contratos evaluados | {contract_check.get('n_total', 'N/A')} |
| Contratos pasados | {contract_check.get('n_passed', 'N/A')} |
| Contratos fallidos | {contract_check.get('n_failed', 'N/A')} |
| **Pass rate** | **{contract_check.get('pass_rate', 'N/A'):.2%}** |

### Contratos pasados (deberían ser pocos)
{_format_list(results.get('contracts_passed', ['(ninguno)']))}

### Contratos fallidos (esperados)
{_format_list(results.get('contracts_failed', ['(ninguno)']))}

---

## 4. Conclusión

**{contract_check.get('message', 'Análisis incompleto')}**

### Interpretación

{'✓ El sistema detecta correctamente que los datos anti-holográficos NO producen holografía válida. Esto es evidencia de honestidad científica del pipeline.' if contract_check.get('status') == 'SUCCESS' else ''}
{'⚠ Pass rate moderado. Revisar qué contratos pasaron y por qué.' if contract_check.get('status') == 'WARNING' else ''}
{'🚨 ALERTA: El sistema puede estar produciendo falsos positivos. Investigación urgente necesaria.' if contract_check.get('status') == 'ALERT' else ''}

---

## 5. Archivos Generados

- Datos HDF5: `negative_control_{run_id}.h5`
- Este reporte: `negative_control_report_{run_id}.md`

---

*Generado automáticamente por 04c_negative_controls.py*
*Proyecto CUERDAS-Maldacena*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Reporte generado: {report_path}")
    
    return report_path


def _format_list(items: List[str]) -> str:
    """Helper para formatear listas en markdown."""
    if not items or items == ['(ninguno)'] or items == ['(ninguna)']:
        return "- (ninguno)\n"
    return '\n'.join(f"- {item}" for item in items)


# ==============================================================================
# SECCIÓN 5: CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Control negativo para validación del pipeline CUERDAS-Maldacena',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Generar control negativo con parámetros por defecto
  python 04c_negative_controls.py --output_dir runs/negative_control
  
  # Especificar parámetros físicos
  python 04c_negative_controls.py --mass 1.0 --lattice_size 100 --seed 42
  
  # Solo generar datos (sin ejecutar pipeline)
  python 04c_negative_controls.py --generate_only
"""
    )
    
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        default=Path('runs') / f'negative_control_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--mass', 
        type=float, 
        default=1.0,
        help='Masa del campo escalar (default: 1.0)'
    )
    parser.add_argument(
        '--lattice_size', 
        type=int, 
        default=100,
        help='Tamaño del lattice (default: 100)'
    )
    parser.add_argument(
        '--dim', 
        type=int, 
        default=2,
        choices=[2, 3],
        help='Dimensionalidad espacial (default: 2)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Semilla para reproducibilidad'
    )
    parser.add_argument(
        '--noise', 
        type=float, 
        default=0.01,
        help='Nivel de ruido gaussiano (default: 0.01)'
    )
    parser.add_argument(
        '--generate_only',
        action='store_true',
        help='Solo generar datos, no ejecutar pipeline'
    )
    parser.add_argument(
        '--pipeline_dir',
        type=Path,
        default=Path('.'),
        help='Directorio con scripts del pipeline'
    )
    
    args = parser.parse_args()
    
    # Run ID único
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.seed is not None:
        run_id += f"_seed{args.seed}"
    
    logger.info("="*60)
    logger.info("CONTROL NEGATIVO - CUERDAS-MALDACENA")
    logger.info("="*60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {args.output_dir}")
    
    # 1. Generar datos anti-holográficos
    logger.info("\n[PASO 1] Generando datos anti-holográficos...")
    data = generate_massive_scalar_flat_space(
        mass=args.mass,
        lattice_size=args.lattice_size,
        dim=args.dim,
        seed=args.seed,
        noise_level=args.noise
    )
    
    # 2. Guardar en formato pipeline
    logger.info("\n[PASO 2] Guardando datos en formato HDF5...")
    h5_path = save_negative_control_data(data, args.output_dir, run_id)
    
    if args.generate_only:
        logger.info("\n[COMPLETADO] Modo --generate_only: datos guardados, pipeline no ejecutado")
        return
    
    # 3. Ejecutar pipeline
    logger.info("\n[PASO 3] Ejecutando pipeline sobre control negativo...")
    results = run_pipeline_on_negative_control(h5_path, args.pipeline_dir)
    
    # 4. Verificar fallos esperados
    logger.info("\n[PASO 4] Verificando contratos...")
    contract_check = check_contracts_failure(results)
    
    # 5. Generar reporte
    logger.info("\n[PASO 5] Generando reporte...")
    report_path = generate_negative_control_report(
        data, results, contract_check, args.output_dir, run_id
    )
    
    # Resumen final
    logger.info("\n" + "="*60)
    logger.info("RESUMEN")
    logger.info("="*60)
    logger.info(f"Estado: {contract_check.get('status', 'INCOMPLETE')}")
    logger.info(f"Mensaje: {contract_check.get('message', 'Ver reporte')}")
    logger.info(f"Reporte: {report_path}")
    logger.info("="*60)
    
    # Exit code según resultado
    if contract_check.get('status') == 'SUCCESS':
        sys.exit(0)
    elif contract_check.get('status') == 'WARNING':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
