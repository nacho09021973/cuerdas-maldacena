#!/usr/bin/env python3
# 08_build_holographic_dictionary.py
# CUERDAS — Bloque C: Atlas holográfico (diccionario geométrico)
#
# OBJETIVO
#   Construir un atlas holográfico interno a partir de la información de geometría
#   y operadores, organizado por sistema/familia/dimensión:
#     - Listar operadores relevantes por sistema (nombres, Δ, etiquetas).
#     - Agregar metadatos de geometría y clasificación (ads, lifshitz, hvlf, ...).
#
# ENTRADAS
#   - runs/emergent_geometry/emergent_geometry_summary.json
#   - (opcional) otros resúmenes del bloque A/B según implementación.
#
# SALIDAS
#   runs/holographic_dictionary/
#     holographic_dictionary_summary.json
#       - Estructura by_system[(family, d, ...)] con lista de operadores,
#         Δ, categorías, etc.
#
# OPCIONAL: CHECKS DE m²L²
#   - Con flags explícitas, puede calcular m²L² = Δ(Δ-d) como diagnóstico.
#   - IMPORTANTE: estos cálculos son post-hoc y no entran en entrenamiento.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Proporciona el "atlas" interno que se cruza con:
#       * 09_real_data_and_dictionary_contracts.py
#
# HISTÓRICO
#   - Anteriormente conocido como: 03_holographic_dictionary_v3.py

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from pysr import PySRRegressor

try:
    from run_context import RunContext, add_experiment_args
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_context import RunContext, add_experiment_args

SCRIPT_NAME = "08_build_holographic_dictionary.py"

# Import local IO module for run manifest support
try:
    from cuerdas_io import resolve_geometry_emergent_dir, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase XI: construir diccionario holografico agrupando por (family, d)"
    )
    add_experiment_args(parser)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directorio con los .h5 de geometría (p.ej. fase11_output_v2/geometry)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directorio raíz con run_manifest.json (IO v2). Resuelve geometry_emergent automáticamente.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="Fichero JSON de salida con el diccionario resumido",
    )
    parser.add_argument(
        "--mass-source",
        type=str,
        default="hdf5",
        choices=["hdf5", "emergent"],
        help="Fuente de masas: 'hdf5' (ground truth/control) o 'emergent' (extraer solo Delta de correladores)",
    )
    parser.add_argument(
        "--compute-m2-from-delta",
        action="store_true",
        help="(MODO CONTROL, solo con mass_source=hdf5) Si no hay m2L2 en HDF5, calcula m²L² = Delta(Delta-d)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para PySR",
    )
    return parser.parse_args()


def extract_delta_from_correlator(x, G2):
    """
    Extrae Delta de un correlador de 2 puntos, asumiendo:
        G2(x) ~ 1/x^{2Delta}

    Devuelve un dict con:
        - Delta
        - fit_r2
        - status
    """
    x = np.array(x)
    G2 = np.array(G2)

    mask = (x > 0) & (G2 > 0)
    if mask.sum() < 5:
        return {"status": "insufficient_data"}

    logx = np.log(x[mask])
    logG2 = np.log(G2[mask])

    A = np.vstack([logx, np.ones_like(logx)]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, logG2, rcond=None)
    slope, intercept = coeffs
    Delta = -slope / 2.0

    if residuals.size > 0:
        ss_res = residuals[0]
    else:
        ss_res = np.sum((logG2 - (slope * logx + intercept)) ** 2)
    ss_tot = np.sum((logG2 - np.mean(logG2)) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot

    return {"status": "ok", "Delta": Delta, "fit_r2": r2}


def discover_mass_dimension_relation(Deltas, m2L2, d, seed=42):
    """
    Usa PySR para descubrir la relacion entre Delta y m²L².
    Devuelve un dict con:
        - discovered_equation (str)
        - r2 (float)
        - status
        - holographic_r2 (ajuste si forzamos m²L² = Delta(Delta-d))
    """
    results = {}
    Deltas = np.array(Deltas).reshape(-1, 1)
    m2L2 = np.array(m2L2).reshape(-1, 1)

    X = np.hstack([Deltas, np.full_like(Deltas, d)])
    y = m2L2

    if len(X) < 5:
        results["status"] = "insufficient_data"
        return results

    model = PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        elementwise_loss="L2DistLoss()",
        maxsize=12,
        model_selection="best",
        progress=False,
        verbosity=0,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
    )
    model.fit(X, y)
    best = model.get_best()
    y_pred = model.predict(X)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    results["discovered_equation"] = str(best["equation"])
    results["r2"] = float(r2)
    results["status"] = "ok"

    # Comparacion con Delta(Delta-d) (chequeo teórico, no label)
    Deltas_flat = Deltas.reshape(-1)
    m2L2_flat = m2L2.reshape(-1)
    valid = ~np.isnan(Deltas_flat) & ~np.isnan(m2L2_flat)
    if valid.sum() > 3:
        Deltas_valid = Deltas_flat[valid]
        m2L2_valid = m2L2_flat[valid]
        y_holo = Deltas_valid * (Deltas_valid - d)

        ss_res_holo = np.sum((m2L2_valid - y_holo) ** 2)
        ss_tot_holo = np.sum((m2L2_valid - np.mean(m2L2_valid)) ** 2) + 1e-10
        r2_holo = 1 - ss_res_holo / ss_tot_holo
        results["holographic_r2"] = float(r2_holo)
    else:
        results["holographic_r2"] = None

    return results


def main():
    args = parse_args()
    
    ctx = RunContext.from_args(args, script_name=SCRIPT_NAME)
    output_dir = ctx.stage_dir()
    
    # === RESOLVER RUTAS ===
    geometry_dir = None
    
    # Prioridad 1: --run-dir con cuerdas_io
    if args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir)
        geometry_dir = resolve_geometry_emergent_dir(run_dir=run_dir)
    
    # Prioridad 2: --data-dir explícito
    if geometry_dir is None and args.data_dir:
        geometry_dir = Path(args.data_dir)
    
    if geometry_dir is None:
        raise ValueError("Debe proporcionar --run-dir o --data-dir")
    
    # Resolver output
    if args.output_summary:
        output_file = Path(args.output_summary)
    elif args.run_dir:
        out_dir = Path(args.run_dir) / "holographic_dictionary"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "holographic_dictionary_v3_summary.json"
    else:
        output_file = Path("holographic_dictionary_v3_summary.json")

    if not geometry_dir.exists() or not geometry_dir.is_dir():
        raise FileNotFoundError(f"--data-dir no es un directorio valido: {geometry_dir}")

    print("=" * 70)
    print("FASE XI - DICCIONARIO HOLOGRAFICO v3.1 (FIX EMERGENT, sin manifest)")
    print("=" * 70)
    print(f"Mass source: {args.mass_source.upper()}")
    if args.mass_source == "hdf5":
        print("   MODO CONTROL: Usando ground-truth de HDF5 (Delta_mass_dict)")
        if args.compute_m2_from_delta:
            print("   -> [CONTROL] Si falta m2L2, se calcula m²L² = Delta(Delta-d)")
    else:
        print("   MODO EMERGENTE: Extrayendo Delta de correladores")
        print("   -> NO se calcula m²L² en esta fase (solo atlas de Δ)")
    print("=" * 70)

    # Agrupar por (family, d)
    data_by_family_d = defaultdict(
        lambda: {
            "family": None,
            "d": None,
            "Deltas": [],
            "m2L2": [],
            "geometries": [],
            "operators": [],
        }
    )

    geometry_results = []

    # Recorremos todos los .h5 de la carpeta de geometria
    h5_files = sorted(geometry_dir.glob("*.h5"))
    if not h5_files:
        print(f"[WARN] No se encontraron .h5 en {geometry_dir}")

    for h5_path in h5_files:
        name = h5_path.stem

        with h5py.File(h5_path, "r") as f:
            family = f.attrs.get("family", "unknown")
            try:
                d = int(f.attrs.get("d", 4))
            except Exception:
                d = 4

            key = f"{family}_d{d}"
            data_by_family_d[key]["family"] = family
            data_by_family_d[key]["d"] = d

            geo_result = {"name": name, "family": family, "d": d, "operators_extracted": []}

            boundary = f.get("boundary", None)
            if boundary is None:
                print(f"[WARN] {h5_path} sin grupo 'boundary', se omite")
                continue

            if "x_grid" not in boundary:
                print(f"[WARN] {h5_path} sin 'boundary/x_grid', se omite")
                continue
            x_grid = boundary["x_grid"][:]

            if args.mass_source == "hdf5":
                # ---------------------------
                # MODO CONTROL: Leer Delta_mass_dict
                # ---------------------------
                if "Delta_mass_dict" in f.attrs:
                    delta_mass_dict = json.loads(f.attrs["Delta_mass_dict"])
                else:
                    delta_mass_dict = {}

                for op_name, dm in delta_mass_dict.items():
                    Delta = dm.get("Delta")
                    m2L2 = dm.get("m2L2")

                    if m2L2 is None and args.compute_m2_from_delta:
                        m2L2 = Delta * (Delta - d)
                        m2_source = "from_delta_formula_control"
                    elif m2L2 is None:
                        print(
                            f"   [WARN] {name}/{op_name} sin m2L2 y sin compute_m2_from_delta; omitido"
                        )
                        continue
                    else:
                        m2_source = "from_hdf5"

                    print(
                        f"   {name}/{op_name}: Delta={Delta:.3f}, m²L²={m2L2:.3f} [{m2_source}]"
                    )

                    geo_result["operators_extracted"].append(
                        {
                            "name": op_name,
                            "Delta": Delta,
                            "Delta_fit_r2": 1.0,
                            "m2L2": m2L2,
                        }
                    )

                    data_by_family_d[key]["Deltas"].append(Delta)
                    data_by_family_d[key]["m2L2"].append(m2L2)
                    data_by_family_d[key]["operators"].append(
                        {
                            "name": f"{name}_{op_name}",
                            "Delta": Delta,
                            "m2L2": m2L2,
                            "source_geometry": name,
                            "m2L2_method": m2_source,
                        }
                    )

            else:
                # ---------------------------
                # MODO EMERGENTE: SOLO Δ
                # ---------------------------
                operators_attr = f.attrs.get("operators", "[]")
                try:
                    operators = json.loads(operators_attr)
                except Exception:
                    operators = []

                for op in operators:
                    op_name = op.get("name")
                    if op_name is None:
                        continue

                    G2_key = f"G2_{op_name}"
                    if G2_key not in boundary:
                        continue

                    G2 = boundary[G2_key][:]
                    result = extract_delta_from_correlator(x_grid, G2)
                    if result["status"] != "ok":
                        continue

                    Delta = result["Delta"]

                    print(
                        f"   {name}/{op_name}: Delta={Delta:.3f} "
                        f"[emergent, m²L² NO calculado en esta fase]"
                    )

                    geo_result["operators_extracted"].append(
                        {
                            "name": op_name,
                            "Delta": Delta,
                            "Delta_fit_r2": result.get("fit_r2"),
                            # m2L2 no se guarda en modo emergent
                        }
                    )

                    data_by_family_d[key]["Deltas"].append(Delta)
                    data_by_family_d[key]["operators"].append(
                        {
                            "name": f"{name}_{op_name}",
                            "Delta": Delta,
                            "source_geometry": name,
                            "m2L2_method": "not_available",
                        }
                    )

            data_by_family_d[key]["geometries"].append(name)
            geometry_results.append(geo_result)

    # Construir by_system solo cuando tenemos m2L2 disponible
    by_system = {}
    for key, fdata in sorted(data_by_family_d.items()):
        n = min(len(fdata["Deltas"]), len(fdata["m2L2"]))
        if n == 0:
            print(f"   {key}: 0 puntos con m2L2 disponible [SKIP]")
            continue

        Deltas = fdata["Deltas"][:n]
        m2L2_list = fdata["m2L2"][:n]

        by_system[key] = {
            "family": fdata["family"],
            "d": fdata["d"],
            "n_points": n,
            "Delta": Deltas,
            "m2L2_emergent": m2L2_list,
            "geometries_included": fdata["geometries"],
            "source": args.mass_source,
        }
        print(
            f"   {key}: {n} puntos con m2L2, "
            f"d={fdata['d']}, {len(fdata['geometries'])} geometrias"
        )

    # Descubrir relaciones masa-dimension donde sea posible
    discovery_results = {}
    for key, sdata in by_system.items():
        if sdata["n_points"] < 3:
            print(f"   {key}: datos insuficientes ({sdata['n_points']} < 3)")
            discovery_results[key] = {"status": "insufficient_data"}
            continue

        print(f"\n>> Descubriendo para '{key}' (d={sdata['d']})...")
        result = discover_mass_dimension_relation(
            np.array(sdata["Delta"]),
            np.array(sdata["m2L2_emergent"]),
            sdata["d"],
            seed=args.seed,
        )
        discovery_results[key] = result
        if result["status"] == "ok":
            print(f"   Mejor ecuacion: {result['discovered_equation']}")
            print(f"   R²(PySR): {result['r2']:.4f}")
            if result.get("holographic_r2") is not None:
                print(f"   R²(m²L²=Delta(Delta-d)): {result['holographic_r2']:.4f}")
        else:
            print(f"   Status: {result['status']}")

    summary = {
        "by_system": by_system,
        "discoveries": discovery_results,
        "geometry_results": geometry_results,
        "mass_source": args.mass_source,
        "compute_m2_from_delta": args.compute_m2_from_delta,
        "notes": [
            "v3.1: FIX MODO EMERGENT (no mezcla d ni masas entre geometrías)",
            "rev.honestidad: en modo emergent no se calcula m²L² en este script; "
            "las masas deben venir de datos externos (HDF5 u otros módulos).",
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))
    print(f"\nResumen guardado en: {output_file}")
    
    # === ACTUALIZAR RUN_MANIFEST (IO v2) ===
    if args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir)
            update_run_manifest(
                run_dir,
                {
                    "holographic_dictionary_dir": str(output_file.parent.relative_to(run_dir)
                                                      if output_file.parent.is_relative_to(run_dir)
                                                      else output_file.parent),
                    "holographic_dictionary_summary": str(output_file.relative_to(run_dir)
                                                         if output_file.is_relative_to(run_dir)
                                                         else output_file),
                }
            )
            print(f"Manifest actualizado")
        except Exception as e:
            print(f"[WARN] No se pudo actualizar manifest: {e}")


    
    # === V3: Registrar outputs ===
    ctx.register_outputs({"holographic_summary": "holographic_dictionary_v3_summary.json"})
    ctx.create_aliases()
    ctx.save_manifest()

if __name__ == "__main__":
    main()
