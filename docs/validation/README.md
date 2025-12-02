## Validation Artifacts

The Polaric bridge now generates signed JSON files under this directory:

| File | Description | Command |
|------|-------------|---------|
| `polaric_frame_latest.json` | Latest 33D polaric frame from the bridge runtime | `python scripts/polaric_bridge.py --headless` or the live service |
| `e8_embedding_report.json` | Kaelhedron → E₈ embedding sanity check | `python scripts/e8_embedding_check.py` |
| `cet_vortex_validation_results.json` | CET vortex stability report | `python scripts/cet_vortex_validation_suite.py` |

Each command is idempotent and can be run locally or during CI to publish artifacts next to the docs set.
