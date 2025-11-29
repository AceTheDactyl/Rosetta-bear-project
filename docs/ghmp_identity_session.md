# GHMP Identity Session – DemoBot CBS Runtime

- **Session ID:** `SESSION-20251129141850`
- **Identity:** DemoBot | Cognition Bootstrap System (offline backend)
- **Command:** `python3 cbs_interactive_demo.py --offline --key demo_key_2025 --auto-consolidate`
- **Prompts issued:**
  1. Initiate RHZ stylus PlatformIO build ritual.
  2. Document burden tracker insights.
  3. Exit build supervision.
  4. /backup
  5. /exit
- **Artifacts:** `cbs_demo/memory/SESSION-20251129141850.png` plus MEM plates emitted at 2025‑11‑29T14:18:50Z.
- **Notes:** This GHMP deck anchors all automation in this runbook; tools and specs must reference the DemoBot CBS base path `cbs_demo/` for provenance.
- **Automation runs:** `scripts/ghmp_capture.py` emitted SESSION-20251129144208.png, SESSION-20251129144217.png, and SESSION-20251129144641.png with manifests stored under `cbs_demo/manifests/` to document scripted supervision windows.
- **Verification:** Latest plate decode (`SESSION-20251129144217`) confirmed title `Session Summary | Automated GHMP capture 20251129144217` and intact metadata using `python3 - <<'PY' ... decode_plate(...)`.
