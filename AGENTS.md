# Repository Guidelines

## Project Structure & Modules
- Core pipeline scripts live at repo root (`01_generate_sandbox_geometries.py` → `09_real_data_and_dictionary_contracts.py`), plus `bulk_scalar_solver.py` shared by steps B/C. Keep numbering and signatures stable.
- Extensions: `fase11_ising3d_emergent_geometry.py`, `make_fase12_report_from_emergent.py`, `extended_physics_contracts_fase12_13.py`. Controles negativos y versiones legacy viven en `attic/` (ver `attic/README.md`) y no forman parte del pipeline activo.
- Data/layout: `runs/` holds generated artifacts (sandbox, emergent geometry, equations, eigenmodes, dictionaries, contracts). `data/`, `fase12_data_boundary/`, `real_data_sandbox/`, and `outputs/` store inputs or reports; `config/` stores helper configs. Avoid committing heavy outputs unless required.

## Build, Test, and Development Commands
- Use Python 3.11; CPU path must work. Quick env check (versions only): `python 00_validate_io_contracts.py --print-env`.
- Run-mínimo source of truth is `README.md`; keep this quick reference in sync when changing commands or flags.
- Minimal sandbox flow (CPU-safe):
  - `python 01_generate_sandbox_geometries.py --output-dir runs/sandbox_geometries --n-known 3 --n-test 2 --n-unknown 1`
  - `python 02_emergent_geometry_engine.py --data-dir runs/sandbox_geometries --output-dir runs/emergent_geometry --n-epochs 200 --device cpu --mode train --seed 42`
  - `python 03_discover_bulk_equations.py --geometry-dir runs/emergent_geometry --output-dir runs/bulk_equations --d 4 --niterations 50 --maxsize 12`
  - `python 06_build_bulk_eigenmodes_dataset.py --geometry-dir runs/emergent_geometry/geometry_emergent --output-csv runs/bulk_eigenmodes/bulk_modes_dataset.csv --output-json runs/bulk_eigenmodes/bulk_modes_dataset_v2.json`
  - `python 07_emergent_lambda_sl_dictionary.py --input-file runs/bulk_eigenmodes/bulk_modes_dataset_v2.json --output-dir runs/emergent_dictionary --iterations 200 --seed 42 --ops-minimal`
- Optional: `08_build_holographic_dictionary.py` to refresh atlas; `09_real_data_and_dictionary_contracts.py` with a stub report to smoke-test contracts (see paths in `README.md`).

## Coding Style & Naming Conventions
- Python-first; follow PEP 8, 4-space indents, snake_case for functions/vars, UpperCamelCase for classes. Prefer explicit imports and type hints where practical.
- Do not rename existing data keys/columns (`provenance`, `dictionary_source`, dataset fields) or change file formats without a backwards-compatible path.
- Keep CLI flags stable; add new ones with safe defaults and update README sections that document runs.

## Testing Guidelines
- Primary check: the minimal sandbox pipeline above runs end-to-end on CPU and writes summaries to `runs/`. If touching contracts or formats, also run `04_geometry_physics_contracts.py` and confirm outputs still validate existing runs.
- Use small seeds and `--device cpu` when adding examples; document any GPU-only optimizations as optional.
- When modifying IO schemas, add a quick validation helper or extend `00_validate_io_contracts.py` rather than relying on silent assumptions.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative, and scoped (`02: improve loader error handling`, `contracts: extend phase13 bounds`). Group related edits; avoid mixing formatting-only changes with logic.
- PRs/change summaries should state purpose, scripts touched, data format impact, and reproduction steps (commands run, expected outputs in `runs/` paths). Link related issues or research notes; include screenshots only for report/visual changes.
- Note any compatibility considerations (CPU/GPU, PySR/JuliaCall versions) and migration steps when altering dependencies.

## Security & Configuration Notes
- Default to CPU-safe execution; gate GPU or heavy options behind flags. Avoid adding large dependencies without a clear benefit.
- Honesty rule: do not inject known theoretical formulas/priors (e.g., Δ(Δ−d), assumed mass relations) into losses, features, or regularizers of training/discovery scripts (01–09). Post-hoc checks and contract scripts may compare against theory, but must not leak those formulas into model fitting.
- Treat provided datasets as read-only; avoid committing large generated files unless they are minimal examples.

## Cleanup / Deprecation Policy
- Decommissioned scripts or configs move to `attic/` instead of deletion; add a short mapping in `attic/README.md` that explains replacements and status.
- Update references in docs/scripts when moving files, and keep CLI aliases or shims if needed for a short transition.
- CPU run-mínimo must pass after relocations before merging; prefer removing dead code only after a stable replacement is documented.
