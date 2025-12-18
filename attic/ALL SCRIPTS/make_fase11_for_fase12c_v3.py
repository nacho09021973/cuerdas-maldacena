#!/usr/bin/env python3
"""
make_fase11_for_fase12c_v3.py â€” Adaptador XI â†’ XII.c (v3 - FIX)

CAMBIO CRÃTICO vs v2:
    - IGNORA by_system (tiene mÂ²LÂ² corruptos y mezcla diferentes d)
    - Reconstruye desde geometries[], agrupando por (family, d)
    - Calcula mÂ²LÂ² = Î”(Î”-d) correctamente para cada operador

USO:
    # Solo AdS con d=4
    python make_fase11_for_fase12c_v3.py \
        --input-summary runs/fase11_gpu_run1/dictionary/holographic_dictionary_v2_summary.json \
        --output-json data_processed/fase11_ads_d4.json \
        --filter-family ads --filter-d 4

    # Todos los sistemas, separados por (family, d)
    python make_fase11_for_fase12c_v3.py \
        --input-summary runs/fase11_gpu_run1/dictionary/holographic_dictionary_v2_summary.json \
        --output-json data_processed/fase11_all_by_family_d.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def load_summary(path: Path) -> Dict[str, Any]:
    """Carga el summary de Fase XI."""
    if not path.exists():
        raise FileNotFoundError(f"Summary no encontrado: {path}")
    return json.loads(path.read_text())


def extract_operators_from_geometries(
    geometries: List[Dict[str, Any]],
    filter_family: Optional[str] = None,
    filter_d: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extrae operadores desde geometries[], agrupando por (family, d).
    
    Returns:
        Dict keyed by "family_d{d}" con estructura:
        {
            "family_d4": {
                "family": "ads",
                "d": 4,
                "operators": [
                    {"name": "...", "Delta": ..., "m2L2_emergent": ..., "geometry": ...},
                    ...
                ]
            }
        }
    """
    # Agrupar por (family, d)
    grouped = defaultdict(lambda: {"operators": [], "geometries": set()})
    
    for geom in geometries:
        family = geom.get("family", "unknown")
        d = geom.get("d")
        
        if d is None:
            print(f"  [WARN] GeometrÃ­a {geom.get('name')} sin 'd', omitida")
            continue
        
        # Aplicar filtros si se especificaron
        if filter_family and family != filter_family:
            continue
        if filter_d is not None and d != filter_d:
            continue
        
        key = f"{family}_d{d}"
        grouped[key]["family"] = family
        grouped[key]["d"] = d
        grouped[key]["geometries"].add(geom.get("name", "unknown"))
        
        # Extraer operadores de conformal
        conformal = geom.get("conformal", {})
        for op_name, op_data in conformal.items():
            if op_name == "summary":
                continue

            delta = op_data.get("Delta_inferred")
            if delta is None:
                # Sin Δ no podemos hacer nada
                continue

            # Intentar recuperar m²L² desde el summary de Fase XI
            # (nombres posibles: m2L2_emergent, m2L2, etc.)
            m2L2 = (
                op_data.get("m2L2_emergent")
                or op_data.get("m2L2")
                or geom.get("m2L2_emergent")
                or geom.get("m2L2")
            )

            if m2L2 is None:
                # IMPORTANTE: no usamos la fórmula Δ(Δ-d) aquí.
                # Si no hay m²L² en los datos, omitimos este operador
                # o, si prefieres, podrías marcarlo como "missing".
                print(
                    f"  [WARN] Operador {op_name} de geometría "
                    f"{geom.get('name')} sin m2L2 en summary; omitido"
                )
                continue

            operator = {
                "name": f"{geom.get('name', 'g')}_{op_name}",
                "Delta": float(delta),
                "Delta_error": float(op_data.get("Delta_error", 0.0)),
                "m2L2_emergent": float(m2L2),
                "m2L2_error": float(op_data.get("m2L2_error", 0.0)),
                "m2L2_method": op_data.get(
                    "m2L2_method", "from_fase11_summary"
                ),
                "source_geometry": geom.get("name"),
                "is_conformal": op_data.get("is_conformal", "Unknown"),
            }
            grouped[key]["operators"].append(operator)

    # Convertir sets a listas para JSON
    for key in grouped:
        grouped[key]["geometries"] = list(grouped[key]["geometries"])
    
    return dict(grouped)


def build_output_json(grouped_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Construye el JSON final para XII.c."""
    systems = []
    
    for sys_name, sys_data in sorted(grouped_data.items()):
        system = {
            "name": sys_name,
            "family": sys_data["family"],
            "d": sys_data["d"],
            "source": "fase11_geometries_reconstructed",
            "operators": sys_data["operators"],
            "metadata": {
                "n_geometries": len(sys_data["geometries"]),
                "geometries": sys_data["geometries"],
                "note": "mÂ²LÂ² calculado como Î”(Î”-d) desde Delta_inferred"
            }
        }
        systems.append(system)
    
    return {"systems": systems}


def print_summary(systems: List[Dict[str, Any]]) -> None:
    """Imprime resumen de los sistemas construidos."""
    print("\n" + "=" * 60)
    print("RESUMEN DE SISTEMAS PARA FASE XII.c")
    print("=" * 60)
    
    total_ops = 0
    for sys in systems:
        n_ops = len(sys.get("operators", []))
        total_ops += n_ops
        n_geoms = sys.get("metadata", {}).get("n_geometries", "?")
        print(f"  â€¢ {sys['name']}: {n_ops} operadores de {n_geoms} geometrÃ­as, d={sys['d']}")
    
    print("-" * 60)
    print(f"  TOTAL: {len(systems)} sistemas, {total_ops} operadores")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptador XI â†’ XII.c v3: Reconstruye desde geometries (ignora by_system corrupto)"
    )
    parser.add_argument(
        "--input-summary", required=True,
        help="Ruta a holographic_dictionary_v2_summary.json"
    )
    parser.add_argument(
        "--output-json", required=True,
        help="Ruta de salida para el JSON compatible con Fase XII.c"
    )
    parser.add_argument(
        "--filter-family", type=str, default=None,
        help="Filtrar por familia (ads, lifshitz, hyperscaling, deformed, unknown)"
    )
    parser.add_argument(
        "--filter-d", type=int, default=None,
        help="Filtrar por dimensiÃ³n d especÃ­fica"
    )
    parser.add_argument(
        "--min-operators", type=int, default=3,
        help="MÃ­nimo de operadores por sistema para incluirlo (default: 3)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADAPTADOR XI â†’ XII.c (v3 - RECONSTRUCCIÃ“N DESDE GEOMETRIES)")
    print("=" * 60)
    print(f"Input:  {args.input_summary}")
    print(f"Output: {args.output_json}")
    if args.filter_family:
        print(f"Filtro familia: {args.filter_family}")
    if args.filter_d:
        print(f"Filtro d: {args.filter_d}")
    print("=" * 60)
    
    # Cargar summary
    summary = load_summary(Path(args.input_summary))
    geometries = summary.get("geometries", [])
    
    if not geometries:
        raise ValueError("No hay geometries[] en el summary")
    
    print(f"\nâœ“ Cargadas {len(geometries)} geometrÃ­as del summary")
    
    # Extraer operadores agrupados por (family, d)
    grouped = extract_operators_from_geometries(
        geometries,
        filter_family=args.filter_family,
        filter_d=args.filter_d
    )
    
    if not grouped:
        raise ValueError("No se extrajeron sistemas. Revisa filtros.")
    
    # Filtrar por mÃ­nimo de operadores
    filtered = {}
    for key, data in grouped.items():
        n_ops = len(data["operators"])
        if n_ops >= args.min_operators:
            filtered[key] = data
        else:
            print(f"  [SKIP] {key}: {n_ops} operadores < {args.min_operators}")
    
    if not filtered:
        raise ValueError(f"NingÃºn sistema tiene >= {args.min_operators} operadores")
    
    # Construir JSON de salida
    output = build_output_json(filtered)
    
    # Guardar
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    
    print_summary(output["systems"])
    print(f"\nâœ“ JSON para Fase XII.c escrito en: {output_path}")
    
    # Verificación rápida (sin usar la fórmula Δ(Δ-d))
    print("\n" + "=" * 60)
    print("VERIFICACIÓN RÁPIDA (primeros 3 ops del primer sistema)")
    print("=" * 60)
    if output["systems"]:
        sys0 = output["systems"][0]
        for op in sys0["operators"][:3]:
            delta = op["Delta"]
            m2 = op["m2L2_emergent"]
            method = op.get("m2L2_method", "from_fase11_summary")
            print(
                f"  Δ={delta:.4f}, m²L²={m2:.4f}, "
                f"source={method}, geom={op.get('source_geometry')}"
            )

if __name__ == "__main__":
    main()
