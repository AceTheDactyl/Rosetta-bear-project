# CBS Demo Index

**Status:** Active Runtime Data

This directory contains runtime artifacts, state snapshots, and session data from CBS operations.

## Structure

```
cbs_demo/
├── INDEX.md          # This file
├── boot_log.txt      # Boot sequence logs
├── config.json       # Runtime configuration
├── identity.png      # CBS collective identity plate
├── artifacts/        # Generated artifacts
├── backups/          # State backups
├── manifests/        # GHMP capture manifests
├── memory/           # Working memory state
├── skills/           # Loaded skill modules
└── updates/          # Applied updates
```

## Key Files

| File | Purpose |
|------|---------|
| `identity.png` | Visual signature for GHMP encoding |
| `config.json` | Runtime configuration |
| `boot_log.txt` | CBS boot sequence log |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `artifacts/` | Generated outputs from CBS sessions |
| `backups/` | Timestamped state snapshots |
| `manifests/` | GHMP session capture files |
| `memory/` | Working memory artifacts |
| `skills/` | Loaded skill definitions |
| `updates/` | Update packages and logs |

## Usage Notes

- This directory is **runtime state** - contents change during CBS operation
- Delete or rename this directory for a clean slate between experiments
- The `identity.png` is referenced by GHMP for plate encoding
- Manifests in `manifests/` can be replayed via `scripts/run_triadic_cycle.py`

## Clean Start

```bash
# Backup and reset
mv cbs_demo cbs_demo_backup_$(date +%Y%m%d)
mkdir -p cbs_demo/{artifacts,backups,manifests,memory,skills,updates}
```
