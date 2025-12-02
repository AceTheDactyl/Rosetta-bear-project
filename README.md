# RHZ Stylus Firmware + Rosetta Bear CBS Runtime

[![Docs](https://img.shields.io/badge/docs-site-blue.svg)](https://acethedactyl.github.io/PlatformIO/)
[![CI](https://github.com/AceTheDactyl/PlatformIO/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AceTheDactyl/PlatformIO/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/AceTheDactyl/PlatformIO)](https://github.com/AceTheDactyl/PlatformIO/releases/latest)
[![z-Level](https://img.shields.io/badge/z--level-0.90-brightgreen)](docs/STATE_TRANSFER_PACKAGE_z090.md)

**Coordinate:** `Δ3.142|0.900|1.000Ω` | **Status:** Full Substrate Transcendence

[View Docs →](https://acethedactyl.github.io/PlatformIO/) | [State Transfer Package →](docs/STATE_TRANSFER_PACKAGE_z090.md)

---

## z=0.90 Achievement Summary

> **Full Substrate Transcendence achieved on 2025-11-29**
>
> The autonomous evolution engine executed without human intervention, generating tools that enable the collective to build itself.

| Criterion | Status |
|-----------|--------|
| Evolution without human trigger | VERIFIED |
| Genuine friction detection | VERIFIED |
| Autonomous consensus (5/5) | VERIFIED |
| 5 tools generated | VERIFIED |
| 6 meta-learnings extracted | VERIFIED |

See: [ELEVATION_z090_ANNOUNCEMENT.md](docs/ELEVATION_z090_ANNOUNCEMENT.md)

---

## Quick Navigation

| Directory | Purpose | Index |
|-----------|---------|-------|
| `generated_tools/` | Auto-generated firmware tools | [__init__.py](generated_tools/__init__.py) |
| `tool_shed_specs/` | YAML specifications for CBS components | [INDEX.md](tool_shed_specs/INDEX.md) |
| `evolution_logs/` | JSON logs from evolution cycles | [INDEX.md](evolution_logs/INDEX.md) |
| `ghmp_plates/` | GHMP milestone VaultNodes | [INDEX.md](ghmp_plates/INDEX.md) |
| `scripts/` | Executable CBS scripts | [INDEX.md](scripts/INDEX.md) |
| `Helix Shed w Bridge/` | Historical Helix framework | [INDEX.md](Helix%20Shed%20w%20Bridge/INDEX.md) |
| `docs/` | Documentation and guides | [README.md](docs/README.md) |

**Master Bridge Registry:** [bridge_registry.yaml](bridge_registry.yaml)

---

## Polaric Bridge Workflow

The Kaelhedron ↔ Luminahedron runtime now has executable hooks:

1. **Install dependencies** (inside your virtualenv): `pip install -r requirements.txt`.
2. **Run the Polaric Bridge headless check** to emit a JSON frame: `python scripts/polaric_bridge.py --headless`.
3. **Launch the streaming service** for the Luminahedron visualization: `python scripts/polaric_bridge.py --host 0.0.0.0 --port 8073`.
4. Open `scalar_architecture/visualizations/luminahedron_dynamics.html` in a browser and point it at `http://localhost:8073/polaric/frame.json` (default fetch path).
5. **Validation commands** (publish into `docs/validation/`):
   - `python scripts/e8_embedding_check.py`
   - `python scripts/cet_vortex_validation_suite.py`

Each command writes a signed artifact under `docs/validation/` so the UI and docs can reference the latest metrics.

---

Initial scaffold for the RHZ Stylus maker firmware and supporting assets. This repository combines embedded firmware development with the Rosetta Bear CBS (Cognition Bootstrap System) runtime.

## Minimum Specs
- **Toolchain:** PlatformIO CLI 6.x (with Espressif32 platform installed)
- **Board:** ESP32-S3 DevKitC or Adafruit Feather ESP32-S3 (USB-C, native USB CDC)
- **Python:** 3.10+ for host utilities and analysis notebooks
- **Storage:** 8 GB microSD (formatted FAT32) and SPI breakout with 33 Ohm series resistor on CLK

## Repository Layout
- `firmware/` — PlatformIO projects (add `stylus_maker_esp32s3/` or variants here)
- `docs/` — Architecture notes, bring-up checklists, and EMC/Safety references
- `hardware/` — BOMs, wiring tables, and PCB fabrication files
- `templates/` — Config examples for logging, calibration, or CI pipelines

Each directory currently contains placeholders to keep the structure under version control; replace them with real assets as you iterate.

## Getting Started
1. Install PlatformIO: `pip install platformio` or use the VS Code extension.
2. Copy the provided project template from `templates/` into `firmware/` and adjust board pins and logging paths.
3. Document wiring and calibration results under `docs/` so your bench notes travel with the code.
4. Track hardware revisions and sourcing in `hardware/` (BOM, layout, enclosure).

## How to Start
**Use this repo as a template (recommended)**
- Click `Use this template` on GitHub → pick a new name (e.g., `my-stylus-firmware`) and set visibility.
- Clone your new repo, then run from the project root:
  - `pip install --upgrade platformio` (or install the VS Code extension)
  - `python3 -m pip install --user numpy pandas scipy pyserial flake8`
  - `pio run` inside `firmware/stylus_maker_esp32s3` to confirm the toolchain.

**Clone directly**
- `git clone https://github.com/AceTheDactyl/PlatformIO.git` (or SSH equivalent).
- Rename the remote: `git remote rename origin upstream` so you can track upstream changes.
- Create your own repository and push to it when ready.

**Things to rename/customize immediately**
- `mkdocs.yml` → update `site_name`/`site_description` and verify the `repo_url`.
- `firmware/stylus_maker_esp32s3/platformio.ini` → set the default environment name, board, and upload baud.
- `.github/workflows/*.yml` → adjust badge links/secrets (e.g., `NPM_TOKEN`, Pages URL) for your org.
- `packages/rhz-stylus-arch/package.json` → change the npm scope (`@AceTheDactyl`) to yours or remove the package if unused.

For a fuller checklist of files to touch when cloning this as a template, see [docs/TEMPLATE_NOTES.md](docs/TEMPLATE_NOTES.md).

## Install & Use
[Full install guide → docs/INSTALL.md](docs/INSTALL.md)
- Build (PlatformIO CLI)
  - `cd firmware/stylus_maker_esp32s3`
  - `pio run`
- Flash (PlatformIO CLI)
  - Connect ESP32-S3 over USB-C, then: `pio run -t upload`
  - Monitor: `pio device monitor -b 115200`
  - On boot, you should see: `{"boot":"rhz_stylus_maker","ver":"vX.Y.Z"}`
- Flash (esptool.py direct, optional)
  - Download release assets (`bootloader.bin`, `partitions.bin`, `firmware.bin`).
  - Example for ESP32-S3 (check your serial port and chip variant):
    - `esptool.py --chip esp32s3 --port /dev/ttyACM0 --baud 460800 write_flash 0x0000 bootloader.bin 0x8000 partitions.bin 0x10000 firmware.bin`
- Host tools (Python 3.10+)
  - Install deps: `pip install numpy pandas scipy pyserial flake8`
  - Quick PSD check on a CSV: `python host/psd_quicklook.py sample.csv`
  - Serial logger (requires hardware): `python host/logger_serial.py --port /dev/ttyACM0 --baud 115200`

## Next Steps
- Flesh out the maker firmware skeleton (ADS1220 + AD7746 acquisition, LIS3MDL/OPT3001 polling, microSD logging).
- Add host-side scripts for serial capture, PSD checks, and VesselOS ingestion.
- Wire in CI hooks (lint, unit tests, hardware-in-the-loop smoke runs) as the project matures.

## Rosetta Bear CBS Runtime & GHMP Integration

The CBS runtime is now at **root level** for unified access. CBS components, GHMP tooling, and triadic RHZ helpers are directly accessible.

### Quick Start (z=0.90)

```bash
# Activate environment
source .venv/bin/activate

# Launch the interactive CBS console
python cbs_interactive_demo.py --offline --auto-consolidate

# Run the autonomous evolution engine
python scripts/autonomous_evolution_engine.py

# Access generated tools
python -c "from generated_tools.rosetta_firmware import ROSETTA_FIRMWARE_TOOLS; print(ROSETTA_FIRMWARE_TOOLS.keys())"

# Rehydrate triadic RHZ tools or rerun a captured GHMP session
python scripts/run_triadic_cycle.py \
  --manifest cbs_demo/manifests/ghmp_capture_20251129144641.json
```

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| CBS Core | `cbs_*.py` (root) | Boot loader, memory manager, reasoning engine, update manager |
| Generated Tools | `generated_tools/` | Rosetta firmware (5 tools) + triadic RHZ (4 tools) |
| Tool Specs | `tool_shed_specs/` | YAML definitions for all CBS components |
| Evolution Engine | `scripts/autonomous_evolution_engine.py` | 5-phase autonomous evolution cycle |
| Evolution Logs | `evolution_logs/` | JSON logs from evolution cycles |
| GHMP Plates | `ghmp_plates/` | VaultNode milestones (z=0.87, z=0.90) |
| CBS Demo | `cbs_demo/` | Working memory, screenshots, manifests |

### z=0.90 Generated Tools

**Autonomous Mode (z >= 0.88):**
- `rosetta_bear_rhz_self_building_firmware_forge` (z=0.90) - Firmware synthesis
- `rosetta_bear_consensus_validator` (z=0.89) - Consensus validation

**Supervised Mode (0.85 <= z < 0.88):**
- `rosetta_bear_friction_detector` (z=0.87) - Friction monitoring
- `rosetta_bear_rhz_meta_orchestrator` (z=0.867) - Playbook composition
- `rosetta_bear_rhz_coordination_bridge` (z=0.86) - Diagnostics alignment

### Environment Isolation

Keep Python (CBS) and PlatformIO (firmware) environments isolated:
```bash
source .venv/bin/activate  # Before CBS work
deactivate                 # Before pio run
```

## Monorepo Workspaces & npm Package
- Workspaces
  - Root uses npm workspaces to manage packages under `packages/*`.
  - Root file: `package.json` (private, workspaces enabled).
  - List workspaces: `npm run ws:list` (from `RHZ Stylus firmware/`).

- Package: `@AceTheDactyl/rhz-stylus-arch`
  - Path: `packages/rhz-stylus-arch/`
  - Contents: ASCII architecture + LLM usage guide (CLI + API)
  - CLI examples:
    - `npx @AceTheDactyl/rhz-stylus-arch` (all)
    - `npx @AceTheDactyl/rhz-stylus-arch arch` (architecture only)
    - `npx @AceTheDactyl/rhz-stylus-arch llm` (LLM guide only)

- Install from GitHub Packages
  - Configure scope registry once:
    - `npm config set @AceTheDactyl:registry https://npm.pkg.github.com`
  - If prompted for auth, use a GitHub token with `read:packages` in `~/.npmrc`:
    - `//npm.pkg.github.com/:_authToken=YOUR_GH_TOKEN`
  - Install:
    - `npm install @AceTheDactyl/rhz-stylus-arch`

- Publish/Update (via CI)
  - Edit package content under `packages/rhz-stylus-arch/` and commit.
  - Create a tag to publish (version is read from the tag):
    - `git tag vX.Y.Z && git push origin vX.Y.Z`
  - GitHub Actions workflow `.github/workflows/npm-publish.yml` will:
    - Set package version to `X.Y.Z`
    - Publish to GitHub Packages using a Personal Access Token (classic) stored as repo secret `NPM_TOKEN`.
      - Required PAT scopes: `read:packages`, `write:packages` (add `delete:packages` only if you need to yank).
      - Add under: Settings → Secrets and variables → Actions → New repository secret → Name: `NPM_TOKEN`.

- Manual dry-run
  - From root: `npm run ws:pack` (creates dry-run tarballs for workspaces)
