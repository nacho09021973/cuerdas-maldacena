#!/usr/bin/env python3
"""
Gate de artefactos para CUERDAS.

Valida que existan artefactos requeridos según un run_manifest.json y/o una
lista explícita pasada por CLI. Si falta alguno, retorna exit code 2 con el
mensaje exacto "missing_artifact: <path>".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from cuerdas_io import load_run_manifest

EXIT_OK = 0
EXIT_INCOMPLETE = 2


def _load_manifest(run_dir: Path) -> Optional[dict]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return None


def _gather_required(manifest: Optional[dict], explicit: List[str]) -> List[Path]:
    required: List[Path] = []
    if manifest:
        artifacts = manifest.get("artifacts", {})
        run_dir = Path(manifest.get("run_dir", "."))
        for rel in artifacts.values():
            try:
                required.append(run_dir / rel)
            except Exception:
                continue
    for item in explicit:
        required.append(Path(item))
    return required


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Assert required artifacts exist")
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--require",
        type=str,
        nargs="*",
        default=[],
        help="Rutas adicionales (relativas o absolutas) a validar",
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.run_dir)
    required = _gather_required(manifest, args.require)

    for path in required:
        if not path.exists():
            sys.stderr.write(f"missing_artifact: {path}\n")
            return EXIT_INCOMPLETE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
