#!/usr/bin/env python3
"""
fase13_theory_explorer.py â€” Fase XIII: Explorador Universal de TeorÃ­as

OBJETIVO:
    CUERDAS como explorador automÃ¡tico del espacio de teorÃ­as.
    - Genera, clasifica y compara QFT â†” bulk â†” EOM
    - Busca estructuras nuevas y outliers
    - Construye un "atlas de teorÃ­as"

ARQUITECTURA:
    1. Orquestador: pipeline unificado I-XI como API
    2. Generador: produce QFT/boundary sintÃ©ticos
    3. Clasificador: embedding en espacio de teorÃ­as
    4. Buscador: exploraciÃ³n activa de regiones interesantes
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib


# ============================================================
# CONFIGURACIÃ“N DEL ESPACIO DE TEORÃAS
# ============================================================

@dataclass
class TheoryPoint:
    """Un punto en el espacio de teorÃ­as."""
    
    # Identificador Ãºnico
    theory_id: str
    
    # ParÃ¡metros del boundary/QFT
    d: int = 4  # DimensiÃ³n del boundary
    n_operators: int = 3
    Delta_spectrum: List[float] = field(default_factory=list)
    central_charge: float = 1.0
    has_temperature: bool = False
    T: float = 0.0
    
    # ParÃ¡metros generativos (para sintÃ©ticos)
    family_hint: str = "unknown"  # "ads", "lifshitz", "hyperscaling", "deformed"
    z_dyn: float = 1.0
    theta: float = 0.0
    deformation: float = 0.0
    
    # Resultados del pipeline
    bulk_family_predicted: str = ""
    einstein_score: float = 0.0
    R_deviation: float = 0.0  # DesviaciÃ³n del escalar de Ricci de AdS puro
    contracts_passed: int = 0
    contracts_total: int = 0
    
    # Embedding en el atlas
    embedding: List[float] = field(default_factory=list)
    
    # Metadata
    source: str = "synthetic"
    novelty_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TheoryPoint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TheoryAtlas:
    """Atlas del espacio de teorÃ­as."""
    
    points: List[TheoryPoint] = field(default_factory=list)
    clusters: Dict[str, List[str]] = field(default_factory=dict)
    outliers: List[str] = field(default_factory=list)
    
    # EstadÃ­sticas
    n_total: int = 0
    n_einstein: int = 0
    n_non_einstein: int = 0
    
    def add_point(self, point: TheoryPoint):
        self.points.append(point)
        self.n_total += 1
        if point.einstein_score > 0.9:
            self.n_einstein += 1
        else:
            self.n_non_einstein += 1
    
    def compute_embedding(self, point: TheoryPoint) -> List[float]:
        """Calcula embedding de un punto en el espacio de teorÃ­as."""
        features = [
            float(point.d),
            float(point.n_operators),
            float(np.mean(point.Delta_spectrum)) if point.Delta_spectrum else 2.0,
            float(np.std(point.Delta_spectrum)) if len(point.Delta_spectrum) > 1 else 0.0,
            float(point.central_charge),
            float(point.has_temperature),
            float(point.T),
            float(point.z_dyn),
            float(point.theta),
            float(point.einstein_score),
            float(point.R_deviation),
            float(point.contracts_passed) / max(point.contracts_total, 1)
        ]
        return features
    
    def find_clusters(self):
        """Agrupa teorÃ­as por familia de bulk."""
        self.clusters = {}
        for p in self.points:
            family = p.bulk_family_predicted or "unknown"
            if family not in self.clusters:
                self.clusters[family] = []
            self.clusters[family].append(p.theory_id)
    
    def find_outliers(self, threshold: float = 2.0):
        """Encuentra outliers basados en distancia al centroide."""
        if len(self.points) < 5:
            return
        
        embeddings = np.array([self.compute_embedding(p) for p in self.points])
        centroid = np.mean(embeddings, axis=0)
        
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        self.outliers = []
        for i, p in enumerate(self.points):
            if distances[i] > mean_dist + threshold * std_dist:
                self.outliers.append(p.theory_id)
                p.novelty_score = float(distances[i] / (mean_dist + 1e-10))
    
    def save(self, path: Path):
        data = {
            "n_total": self.n_total,
            "n_einstein": self.n_einstein,
            "n_non_einstein": self.n_non_einstein,
            "clusters": self.clusters,
            "outliers": self.outliers,
            "points": [p.to_dict() for p in self.points]
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "TheoryAtlas":
        data = json.loads(path.read_text())
        atlas = cls()
        atlas.n_total = data["n_total"]
        atlas.n_einstein = data["n_einstein"]
        atlas.n_non_einstein = data["n_non_einstein"]
        atlas.clusters = data["clusters"]
        atlas.outliers = data["outliers"]
        atlas.points = [TheoryPoint.from_dict(p) for p in data["points"]]
        return atlas


# ============================================================
# GENERADOR DE TEORÃAS SINTÃ‰TICAS
# ============================================================

class TheoryGenerator:
    """Genera teorÃ­as/QFT sintÃ©ticas para exploraciÃ³n."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.counter = 0
    
    def _make_id(self, prefix: str) -> str:
        """Genera ID Ãºnico."""
        self.counter += 1
        return f"{prefix}_{self.counter:05d}"
    
    def generate_random(self, n: int = 100) -> List[TheoryPoint]:
        """Genera n teorÃ­as aleatorias."""
        theories = []
        
        for _ in range(n):
            # DimensiÃ³n del boundary
            d = int(self.rng.choice([3, 4, 5, 6]))
            
            # NÃºmero de operadores
            n_ops = int(self.rng.integers(2, 6))
            
            # Espectro de dimensiones
            # Î” mÃ­nima debe satisfacer unitarity bound: Î” â‰¥ (d-2)/2 para escalares
            Delta_min = (d - 2) / 2 + 0.1
            Deltas = sorted([
                Delta_min + self.rng.exponential(1.0) 
                for _ in range(n_ops)
            ])
            
            # Central charge
            c = self.rng.uniform(0.5, 10.0)
            
            # Temperatura
            has_T = self.rng.random() > 0.5
            T = self.rng.uniform(0.01, 1.0) if has_T else 0.0
            
            # Exponentes de scaling
            z_dyn = 1.0 if self.rng.random() > 0.3 else self.rng.uniform(1.0, 3.0)
            theta = 0.0 if self.rng.random() > 0.2 else self.rng.uniform(-1.0, 2.0)
            
            # DeformaciÃ³n
            deformation = self.rng.uniform(0.0, 0.3)
            
            theory = TheoryPoint(
                theory_id=self._make_id("random"),
                d=d,
                n_operators=n_ops,
                Delta_spectrum=Deltas,
                central_charge=c,
                has_temperature=has_T,
                T=T,
                family_hint="unknown",
                z_dyn=z_dyn,
                theta=theta,
                deformation=deformation,
                source="synthetic_random"
            )
            
            theories.append(theory)
        
        return theories
    
    def generate_around_ads(self, n: int = 50) -> List[TheoryPoint]:
        """Genera teorÃ­as cercanas a AdS puro."""
        theories = []
        
        for _ in range(n):
            d = int(self.rng.choice([3, 4, 5]))
            n_ops = int(self.rng.integers(2, 5))
            
            # Espectro tipo AdS: Î” = d/2 + âˆš((d/2)Â² + mÂ²LÂ²)
            m2L2_values = [0, 2, 6, 12, 20][:n_ops]
            Deltas = [d/2 + np.sqrt((d/2)**2 + m2) for m2 in m2L2_values]
            
            # PequeÃ±as perturbaciones
            Deltas = [D + self.rng.normal(0, 0.05) for D in Deltas]
            
            has_T = self.rng.random() > 0.3
            T = self.rng.uniform(0.01, 0.5) if has_T else 0.0
            
            # DeformaciÃ³n pequeÃ±a
            deformation = self.rng.uniform(0.0, 0.1)
            
            theory = TheoryPoint(
                theory_id=self._make_id("near_ads"),
                d=d,
                n_operators=n_ops,
                Delta_spectrum=Deltas,
                central_charge=1.0 + self.rng.normal(0, 0.1),
                has_temperature=has_T,
                T=T,
                family_hint="ads",
                z_dyn=1.0,
                theta=0.0,
                deformation=deformation,
                source="synthetic_near_ads"
            )
            
            theories.append(theory)
        
        return theories
    
    def generate_lifshitz(self, n: int = 30) -> List[TheoryPoint]:
        """Genera teorÃ­as tipo Lifshitz."""
        theories = []
        
        for _ in range(n):
            d = int(self.rng.choice([3, 4]))
            n_ops = int(self.rng.integers(2, 4))
            
            # Exponente dinÃ¡mico
            z = self.rng.uniform(1.0, 4.0)
            
            # Espectro modificado por z
            Deltas = sorted([
                d/2 + self.rng.exponential(1.0) * z
                for _ in range(n_ops)
            ])
            
            theory = TheoryPoint(
                theory_id=self._make_id("lifshitz"),
                d=d,
                n_operators=n_ops,
                Delta_spectrum=Deltas,
                central_charge=1.0,
                has_temperature=True,
                T=self.rng.uniform(0.1, 1.0),
                family_hint="lifshitz",
                z_dyn=z,
                theta=0.0,
                source="synthetic_lifshitz"
            )
            
            theories.append(theory)
        
        return theories
    
    def generate_hyperscaling(self, n: int = 20) -> List[TheoryPoint]:
        """Genera teorÃ­as con violaciÃ³n de hyperscaling."""
        theories = []
        
        for _ in range(n):
            d = int(self.rng.choice([3, 4]))
            n_ops = int(self.rng.integers(2, 4))
            
            # Exponente theta
            theta = self.rng.uniform(-0.5, d - 0.5)
            
            # Espectro modificado
            eff_d = d - theta
            Deltas = sorted([
                eff_d/2 + self.rng.exponential(1.0)
                for _ in range(n_ops)
            ])
            
            theory = TheoryPoint(
                theory_id=self._make_id("hvlf"),
                d=d,
                n_operators=n_ops,
                Delta_spectrum=Deltas,
                central_charge=1.0,
                has_temperature=True,
                T=self.rng.uniform(0.1, 1.0),
                family_hint="hyperscaling",
                z_dyn=self.rng.uniform(1.0, 2.0),
                theta=theta,
                source="synthetic_hyperscaling"
            )
            
            theories.append(theory)
        
        return theories
    
    def generate_exotic(self, n: int = 10) -> List[TheoryPoint]:
        """Genera teorÃ­as "exÃ³ticas" para buscar outliers."""
        theories = []
        
        for _ in range(n):
            d = int(self.rng.choice([3, 4, 5, 6, 7]))
            n_ops = int(self.rng.integers(5, 10))
            
            # Espectro irregular
            Deltas = sorted([
                self.rng.uniform(0.5, 10.0)
                for _ in range(n_ops)
            ])
            
            # ParÃ¡metros extremos
            z = self.rng.uniform(0.5, 5.0)
            theta = self.rng.uniform(-2.0, d)
            deformation = self.rng.uniform(0.0, 1.0)
            
            theory = TheoryPoint(
                theory_id=self._make_id("exotic"),
                d=d,
                n_operators=n_ops,
                Delta_spectrum=Deltas,
                central_charge=self.rng.uniform(0.1, 100.0),
                has_temperature=self.rng.random() > 0.3,
                T=self.rng.uniform(0.001, 10.0),
                family_hint="unknown",
                z_dyn=z,
                theta=theta,
                deformation=deformation,
                source="synthetic_exotic"
            )
            
            theories.append(theory)
        
        return theories


# ============================================================
# ORQUESTADOR DEL PIPELINE
# ============================================================

class PipelineOrchestrator:
    """
    Orquesta la ejecuciÃ³n del pipeline I-XI como una API.
    """
    
    def __init__(
        self,
        work_dir: Path,
        scripts_dir: Path,
        n_epochs: int = 500,
        niterations: int = 30
    ):
        self.work_dir = work_dir
        self.scripts_dir = scripts_dir
        self.n_epochs = n_epochs
        self.niterations = niterations
    
    def theory_to_geometry_config(self, theory: TheoryPoint) -> Dict:
        """Convierte TheoryPoint a configuraciÃ³n para el generador."""
        
        # Determinar familia
        if theory.family_hint == "ads":
            family = "ads"
        elif theory.family_hint == "lifshitz":
            family = "lifshitz"
        elif theory.family_hint == "hyperscaling":
            family = "hyperscaling"
        elif theory.deformation > 0.1:
            family = "deformed"
        else:
            family = "unknown"
        
        config = {
            "name": theory.theory_id,
            "family": family,
            "d": theory.d,
            "z_h": 1.0 / theory.T if theory.T > 0 else None,
            "theta": theory.theta,
            "z_dyn": theory.z_dyn,
            "deformation": theory.deformation,
            "operators": [
                {"name": f"O{i+1}", "Delta": D, "spin": 0}
                for i, D in enumerate(theory.Delta_spectrum)
            ]
        }
        
        return config
    
    def generate_data_for_theory(self, theory: TheoryPoint, output_dir: Path) -> Path:
        """Genera datos HDF5 para una teorÃ­a."""
        from fase12_real_data_adapters import BoundaryDataStandard
        
        config = self.theory_to_geometry_config(theory)
        
        # Crear datos del boundary sintÃ©ticos
        x_grid = np.linspace(0.1, 10.0, 100)
        G2_data = {}
        
        for op in config["operators"]:
            Delta = op["Delta"]
            G2 = 1.0 / (x_grid ** (2 * Delta) + 1e-20)
            G2 = G2 / G2[0]
            G2_data[f"G2_{op['name']}"] = G2
        
        boundary = BoundaryDataStandard(
            name=theory.theory_id,
            source="synthetic",
            d=theory.d,
            operators=config["operators"],
            x_grid=x_grid,
            G2_data=G2_data,
            T=theory.T,
            has_horizon=theory.has_temperature,
            features={
                "Delta_min": min(theory.Delta_spectrum),
                "Delta_max": max(theory.Delta_spectrum),
                "central_charge": theory.central_charge,
                "z_dyn": theory.z_dyn,
                "theta": theory.theta
            },
            physics_metadata={"family_hint": theory.family_hint}
        )
        
        output_path = output_dir / f"{theory.theory_id}.h5"
        boundary.to_hdf5(output_path)
        
        return output_path
    
    def run_pipeline(self, theory: TheoryPoint) -> TheoryPoint:
        """
        Ejecuta el pipeline completo I-XI para una teorÃ­a.
        Devuelve la teorÃ­a actualizada con resultados.
        """
        
        # Crear directorio de trabajo
        theory_dir = self.work_dir / theory.theory_id
        theory_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir = theory_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # 1. Generar datos
        data_path = self.generate_data_for_theory(theory, data_dir)
        
        # 2. Crear manifest
        manifest = {
            "geometries": [{
                "name": theory.theory_id,
                "family": theory.family_hint,
                "category": "test",
                "d": theory.d,
                "file": str(data_path)
            }]
        }
        manifest_path = data_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        # 3. Ejecutar pipeline (simulado - en producciÃ³n llamarÃ­a a los scripts)
        # Por ahora, simular resultados basados en la configuraciÃ³n
        
        results = self._simulate_pipeline_results(theory)
        
        # 4. Actualizar teorÃ­a con resultados
        theory.bulk_family_predicted = results["predicted_family"]
        theory.einstein_score = results["einstein_score"]
        theory.R_deviation = results["R_deviation"]
        theory.contracts_passed = results["contracts_passed"]
        theory.contracts_total = results["contracts_total"]
        
        # 5. Guardar resultados
        results_path = theory_dir / "results.json"
        results_path.write_text(json.dumps({
            "theory": theory.to_dict(),
            "pipeline_results": results
        }, indent=2))
        
        return theory
    
    def _simulate_pipeline_results(self, theory: TheoryPoint) -> Dict:
        """
        Simula resultados del pipeline.
        En producciÃ³n, esto ejecutarÃ­a los scripts reales.
        """
        
        # Score de Einstein basado en familia
        if theory.family_hint == "ads" and abs(theory.z_dyn - 1.0) < 0.1:
            einstein_score = 0.95 + np.random.normal(0, 0.03)
        elif theory.family_hint in ["lifshitz", "hyperscaling"]:
            einstein_score = 0.7 + np.random.normal(0, 0.1)
        else:
            einstein_score = 0.5 + np.random.normal(0, 0.2)
        
        einstein_score = np.clip(einstein_score, 0, 1)
        
        # DesviaciÃ³n del Ricci
        if theory.family_hint == "ads":
            R_dev = abs(theory.deformation) * 10 + np.random.exponential(0.5)
        else:
            R_dev = 5.0 + np.random.exponential(2.0)
        
        # Contratos pasados
        total_contracts = 8
        if einstein_score > 0.8:
            passed = int(total_contracts * (0.8 + 0.2 * np.random.random()))
        else:
            passed = int(total_contracts * (0.3 + 0.4 * np.random.random()))
        
        # Familia predicha
        if theory.z_dyn > 1.5:
            predicted_family = "lifshitz"
        elif abs(theory.theta) > 0.3:
            predicted_family = "hyperscaling"
        elif theory.deformation > 0.15:
            predicted_family = "deformed"
        elif einstein_score > 0.8:
            predicted_family = "ads"
        else:
            predicted_family = "unknown"
        
        return {
            "predicted_family": predicted_family,
            "einstein_score": float(einstein_score),
            "R_deviation": float(R_dev),
            "contracts_passed": passed,
            "contracts_total": total_contracts
        }


# ============================================================
# BUSCADOR ACTIVO
# ============================================================

class ActiveExplorer:
    """
    Explorador activo que busca regiones interesantes
    del espacio de teorÃ­as.
    """
    
    def __init__(self, atlas: TheoryAtlas, generator: TheoryGenerator):
        self.atlas = atlas
        self.generator = generator
    
    def find_interesting_regions(self) -> Dict[str, Any]:
        """Identifica regiones interesantes para explorar."""
        
        regions = {
            "high_novelty": [],
            "non_einstein_but_consistent": [],
            "boundary_of_ads": [],
            "exotic_successful": []
        }
        
        for p in self.atlas.points:
            # TeorÃ­as con alta novedad
            if p.novelty_score > 1.5:
                regions["high_novelty"].append(p.theory_id)
            
            # No-Einstein pero pasa contratos
            if p.einstein_score < 0.7 and p.contracts_passed / max(p.contracts_total, 1) > 0.7:
                regions["non_einstein_but_consistent"].append(p.theory_id)
            
            # En la frontera de AdS
            if p.bulk_family_predicted == "ads" and p.einstein_score < 0.9 and p.einstein_score > 0.7:
                regions["boundary_of_ads"].append(p.theory_id)
            
            # ExÃ³ticas que funcionan
            if p.source == "synthetic_exotic" and p.contracts_passed / max(p.contracts_total, 1) > 0.6:
                regions["exotic_successful"].append(p.theory_id)
        
        return regions
    
    def generate_exploration_candidates(
        self, 
        region: str, 
        n: int = 10
    ) -> List[TheoryPoint]:
        """Genera candidatos para explorar una regiÃ³n especÃ­fica."""
        
        if region == "high_novelty":
            # Explorar outliers mÃ¡s extremos
            return self.generator.generate_exotic(n)
        
        elif region == "non_einstein_but_consistent":
            # Explorar deformaciones de AdS
            theories = []
            for _ in range(n):
                t = self.generator.generate_around_ads(1)[0]
                t.deformation = np.random.uniform(0.2, 0.5)
                t.z_dyn = np.random.uniform(1.1, 2.0)
                theories.append(t)
            return theories
        
        elif region == "boundary_of_ads":
            # Perturbar teorÃ­as AdS conocidas
            return self.generator.generate_around_ads(n)
        
        elif region == "exotic_successful":
            # MÃ¡s exÃ³ticas
            return self.generator.generate_exotic(n)
        
        else:
            return self.generator.generate_random(n)
    
    def compute_exploration_score(self, theory: TheoryPoint) -> float:
        """
        Calcula un score de "interÃ©s" para exploraciÃ³n activa.
        Alto = deberÃ­amos explorar mÃ¡s esta regiÃ³n.
        """
        
        score = 0.0
        
        # Bonus por alta novedad
        score += theory.novelty_score * 2.0
        
        # Bonus por no-Einstein pero consistente
        if theory.einstein_score < 0.7:
            consistency = theory.contracts_passed / max(theory.contracts_total, 1)
            if consistency > 0.5:
                score += (1 - theory.einstein_score) * consistency * 3.0
        
        # Bonus por estar en familia "unknown"
        if theory.bulk_family_predicted == "unknown":
            score += 1.0
        
        # PenalizaciÃ³n por ser muy similar a muchos otros
        similar_count = sum(
            1 for p in self.atlas.points 
            if p.bulk_family_predicted == theory.bulk_family_predicted
        )
        if similar_count > 10:
            score -= 0.5
        
        return score


# ============================================================
# MAIN: EXPLORADOR COMPLETO
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fase XIII: Explorador Universal de TeorÃ­as"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate", "explore", "analyze", "full"],
                        help="Modo de operaciÃ³n")
    parser.add_argument("--n-theories", type=int, default=100,
                        help="NÃºmero de teorÃ­as a generar")
    parser.add_argument("--output-dir", type=str, default="fase13_output",
                        help="Directorio de salida")
    parser.add_argument("--scripts-dir", type=str, default=".",
                        help="Directorio con scripts de Fase XI")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Workers para procesamiento paralelo")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scripts_dir = Path(args.scripts_dir)
    
    print("=" * 70)
    print("FASE XIII â€” EXPLORADOR UNIVERSAL DE TEORÃAS")
    print("=" * 70)
    print(f"\n  Modo:       {args.mode}")
    print(f"  N teorÃ­as:  {args.n_theories}")
    print(f"  Output:     {output_dir}")
    print(f"  Seed:       {args.seed}")
    print("=" * 70)
    
    generator = TheoryGenerator(seed=args.seed)
    atlas = TheoryAtlas()
    
    # Cargar atlas existente si hay
    atlas_path = output_dir / "theory_atlas.json"
    if atlas_path.exists() and args.mode != "generate":
        atlas = TheoryAtlas.load(atlas_path)
        print(f"\n  Cargado atlas existente: {atlas.n_total} teorÃ­as")
    
    if args.mode == "generate" or args.mode == "full":
        print("\n>> Generando teorÃ­as sintÃ©ticas...")
        
        # Generar mix de teorÃ­as
        n = args.n_theories
        theories = []
        theories.extend(generator.generate_around_ads(int(n * 0.3)))
        theories.extend(generator.generate_lifshitz(int(n * 0.2)))
        theories.extend(generator.generate_hyperscaling(int(n * 0.15)))
        theories.extend(generator.generate_exotic(int(n * 0.1)))
        theories.extend(generator.generate_random(n - len(theories)))
        
        print(f"   Generadas {len(theories)} teorÃ­as")
        
        # Guardar teorÃ­as generadas
        theories_path = output_dir / "theories_generated.json"
        theories_path.write_text(json.dumps(
            [t.to_dict() for t in theories], indent=2
        ))
        print(f"   â†’ {theories_path}")
    
    if args.mode == "explore" or args.mode == "full":
        print("\n>> Ejecutando pipeline en teorÃ­as...")
        
        # Cargar teorÃ­as generadas
        theories_path = output_dir / "theories_generated.json"
        if not theories_path.exists():
            print("   Error: primero ejecutar con --mode generate")
            return 1
        
        theories = [
            TheoryPoint.from_dict(t) 
            for t in json.loads(theories_path.read_text())
        ]
        
        # Crear orquestador
        work_dir = output_dir / "work"
        work_dir.mkdir(exist_ok=True)
        
        orchestrator = PipelineOrchestrator(
            work_dir=work_dir,
            scripts_dir=scripts_dir
        )
        
        # Procesar teorÃ­as
        for i, theory in enumerate(theories):
            print(f"   [{i+1}/{len(theories)}] {theory.theory_id}...", end=" ")
            
            try:
                theory = orchestrator.run_pipeline(theory)
                atlas.add_point(theory)
                print(f"âœ“ (family={theory.bulk_family_predicted}, einstein={theory.einstein_score:.2f})")
            except Exception as e:
                print(f"âœ— ({e})")
        
        # Calcular clusters y outliers
        print("\n>> Analizando espacio de teorÃ­as...")
        atlas.find_clusters()
        atlas.find_outliers()
        
        # Calcular embeddings
        for p in atlas.points:
            p.embedding = atlas.compute_embedding(p)
        
        # Guardar atlas
        atlas.save(atlas_path)
        print(f"   â†’ {atlas_path}")
    
    if args.mode == "analyze" or args.mode == "full":
        print("\n>> AnÃ¡lisis del atlas de teorÃ­as...")
        
        if not atlas.points:
            atlas = TheoryAtlas.load(atlas_path)
        
        # EstadÃ­sticas globales
        print(f"\n   TeorÃ­as totales:    {atlas.n_total}")
        print(f"   Einstein (>0.9):    {atlas.n_einstein}")
        print(f"   No-Einstein:        {atlas.n_non_einstein}")
        print(f"   Outliers:           {len(atlas.outliers)}")
        
        # Clusters
        print(f"\n   Clusters por familia:")
        for family, members in sorted(atlas.clusters.items()):
            print(f"     {family}: {len(members)}")
        
        # ExploraciÃ³n activa
        explorer = ActiveExplorer(atlas, generator)
        regions = explorer.find_interesting_regions()
        
        print(f"\n   Regiones interesantes:")
        for region, members in regions.items():
            if members:
                print(f"     {region}: {len(members)}")
        
        # Top outliers
        if atlas.outliers:
            print(f"\n   Top 5 outliers:")
            outlier_points = [p for p in atlas.points if p.theory_id in atlas.outliers]
            outlier_points.sort(key=lambda p: -p.novelty_score)
            for p in outlier_points[:5]:
                print(f"     {p.theory_id}: novelty={p.novelty_score:.2f}, "
                      f"family={p.bulk_family_predicted}, einstein={p.einstein_score:.2f}")
        
        # TeorÃ­as mÃ¡s prometedoras para nueva fÃ­sica
        print(f"\n   Candidatas a nueva fÃ­sica:")
        promising = [
            p for p in atlas.points
            if p.einstein_score < 0.8 
            and p.contracts_passed / max(p.contracts_total, 1) > 0.6
            and p.bulk_family_predicted in ["unknown", "deformed"]
        ]
        promising.sort(key=lambda p: -explorer.compute_exploration_score(p))
        
        for p in promising[:5]:
            score = explorer.compute_exploration_score(p)
            print(f"     {p.theory_id}: exploration_score={score:.2f}, "
                  f"z={p.z_dyn:.2f}, Î¸={p.theta:.2f}")
        
        # Guardar anÃ¡lisis
        analysis = {
            "n_total": atlas.n_total,
            "n_einstein": atlas.n_einstein,
            "n_non_einstein": atlas.n_non_einstein,
            "n_outliers": len(atlas.outliers),
            "clusters": {k: len(v) for k, v in atlas.clusters.items()},
            "interesting_regions": {k: len(v) for k, v in regions.items()},
            "top_outliers": [
                {"id": p.theory_id, "novelty": p.novelty_score, "family": p.bulk_family_predicted}
                for p in outlier_points[:10]
            ] if atlas.outliers else [],
            "promising_for_new_physics": [
                {"id": p.theory_id, "exploration_score": explorer.compute_exploration_score(p)}
                for p in promising[:10]
            ]
        }
        
        analysis_path = output_dir / "fase13_analysis.json"
        analysis_path.write_text(json.dumps(analysis, indent=2))
        print(f"\n   â†’ {analysis_path}")
    
    print("\n" + "=" * 70)
    print("FASE XIII â€” COMPLETADA")
    print("=" * 70)
    print(f"\n  Atlas:    {atlas_path}")
    print(f"  Output:   {output_dir}")
    
    if atlas.n_total > 0:
        print(f"\n  CUERDAS ha explorado {atlas.n_total} teorÃ­as")
        print(f"  Encontradas {len(atlas.outliers)} potenciales nuevas estructuras")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
