#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description="Genera manifest.json (legacy) en un data-dir a partir de .h5.")
    ap.add_argument("--data-dir", required=True, help="Directorio que contiene .h5 (raíz del dataset)")
    ap.add_argument("--pattern", default="*.h5", help="Patrón de ficheros (default: *.h5)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"ERROR: no existe data-dir: {data_dir}")

    h5s = sorted(data_dir.glob(args.pattern))
    if not h5s:
        raise SystemExit(f"ERROR: no encontré .h5 con patrón {args.pattern} en {data_dir}")

    systems = []
    for p in h5s:
        name = p.stem
        rel = p.name  # ficheros en raíz del data-dir
        systems.append({
            "name": name,
            "file": rel,          # alias común
            "path": rel,          # alias común
            "h5": rel,            # alias común
            "h5_file": rel,       # alias común
        })

    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": ".",
        "n_systems": len(systems),

        # Campos redundantes a propósito para compatibilidad con lectores distintos
        "systems": systems,
        "h5_files": [p.name for p in h5s],
        "systems_h5": {s["name"]: s["h5_file"] for s in systems},
    }

    out = data_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OK: escrito {out} con {len(systems)} sistemas")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
