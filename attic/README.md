# Attic (archived scripts)

Tabla de duplicados/versionados y su canónico actual (fuente: run mínimo de `README.md`).

| Familia | Candidatos detectados | Canónico propuesto | Motivo |
| --- | --- | --- | --- |
| 04_geometry_contracts | 04_geometry_physics_contracts.py; attic/04b_negative_control_contracts.py; attic/ALL SCRIPTS/04_contracts_fase_11_v2.py | 04_geometry_physics_contracts.py | El run mínimo usa 04; 04b es control negativo y el v2 es histórico. |
| 07_lambda_dictionary | 07_emergent_lambda_sl_dictionary.py; attic/07b_discover_lambda_delta_relation.py; attic/ALL SCRIPTS/03_holographic_dictionary_v3.py | 07_emergent_lambda_sl_dictionary.py | Paso 7 del pipeline; 07b y v3 son exploratorios/legacy. |
| Pipeline bundle | Scripts 00–09 en raíz; attic/ALL SCRIPTS/* | Scripts 00–09 en raíz | La numeración actual es la única mantenida; el bundle de `ALL SCRIPTS` es histórico. |
| Documentación principal | README.md; attic/README _V1.md | README.md | README.md refleja el pipeline activo; `_V1` queda de referencia histórica. |

Mapa viejo → canónico (mantener compatibilidad CPU, sin cambios de IO):

- attic/04b_negative_control_contracts.py → 04_geometry_physics_contracts.py (usar el 04 para contratos; 04b solo como control negativo manual).
- attic/07b_discover_lambda_delta_relation.py → 07_emergent_lambda_sl_dictionary.py (diccionario emergente oficial).
- attic/ALL SCRIPTS/* → scripts numerados 00–09 en raíz (bundle legacy, conservar solo para consulta).
- attic/README _V1.md → README.md (documentación canónica).
