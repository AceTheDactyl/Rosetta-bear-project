# Rosetta Bear Integration Playbook

## Phase Overview
1. **Phase 0 – Alignment:** Read `docs/rosetta_bear_rhz_firmware_update_plan.md` + `docs/rhz_firmware_publish_log.md`, run `python3 cbs_interactive_demo.py --offline --key demo_key_2025 --auto-consolidate`, record GHMP plates.
2. **Phase 1 – Inventory:** Update `docs/rosetta_bear_tool_surface_map.md`, refresh `tool_shed_specs/` YAMLs, and mirror entries inside `tink/CORE_DOCS/HELIX_TOOL_SHED_ARCHITECTURE.md`.
3. **Phase 2 – Automation:** Use `python3 scripts/ghmp_capture.py` and `python3 scripts/regenerate_witnesses.py` to keep GHMP manifests + burden trackers synchronized; import triadic tools via `generated_tools.triadic_rhz`.
4. **Phase 3 – Rituals:** Run `python3 scripts/run_triadic_cycle.py` plus `pio run -d firmware/stylus_maker_esp32s3`, then stash `.elf/.bin` outputs inside `cbs_demo/artifacts/` alongside GHMP manifests.
5. **Phase 4 – Verification:** Decode latest GHMP plates with `ghmp.decode_plate`, ensure tool specs carry observation/meta logs, and update `CHANGELOG.md` + Helix docs.
6. **Phase 5 – Continuous Elevation:** Repeat burden tracker/cascade regeneration weekly, monitor z≈0.900, and refresh this playbook when commands change.

## Monitoring Cadence
- **Weekly:** `python3 scripts/regenerate_witnesses.py` to rebuild burden tracker + cascade JSON. Compare `coordinate` values; if z drifts <0.86, rerun triadic generator.
- **Before/After Firmware Changes:** `python3 scripts/ghmp_capture.py` to mint manifests + backups, then `python3 scripts/run_triadic_cycle.py` to capture coordination bridge output.
- **Monthly:** Review `docs/rosetta_bear_tool_surface_map.md` and `tink/CORE_DOCS/HELIX_TOOL_SHED_ARCHITECTURE.md` for coordinate accuracy; update `tool_shed_specs/*` observation logs with new learnings.

## Command Reference
- GHMP capture automation: `python3 scripts/ghmp_capture.py --base-path cbs_demo --key demo_key_2025`
- Witness regeneration: `python3 scripts/regenerate_witnesses.py`
- Triadic orchestration: `python3 scripts/run_triadic_cycle.py`
- PlatformIO build inside ritual: `pio run -d firmware/stylus_maker_esp32s3`
- Plate verification: `python3 - <<'PY' ... decode_plate('cbs_demo/memory/SESSION-*.png', 'demo_key_2025')`

## Coordinate Map & Docs
- Primary reference: `docs/rosetta_bear_tool_surface_map.md`
- GHMP identity ledger: `docs/ghmp_identity_session.md`
- Publish log + release hooks: `docs/rhz_firmware_publish_log.md`
- Tool specs: `tool_shed_specs/*.yaml`

Keep this playbook close to the workspace so any agent (human or LLM) can replay the Rosetta Bear ⇄ RHZ integration without additional prompts.
