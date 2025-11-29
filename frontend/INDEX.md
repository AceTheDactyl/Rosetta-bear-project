# Frontend Index

**Status:** Active Development

This directory contains React/TypeScript components for the CBS web interface.

## Components

| Component | Description |
|-----------|-------------|
| `components/HelixConsciousnessVisualization.tsx` | 3D visualization of Helix coordinate space |

## Helix Consciousness Visualization

The main visualization component renders:
- Coordinate space (theta, z, r)
- Elevation history trajectory
- Current z-level indicator
- Phase regime coloring

### Usage

```tsx
import { HelixConsciousnessVisualization } from './components/HelixConsciousnessVisualization';

<HelixConsciousnessVisualization
  coordinate={{ theta: 3.142, z: 0.90, r: 1.0 }}
  elevationHistory={[
    { z: 0.41, name: "Initial Emergence" },
    { z: 0.55, name: "Memory Persistence" },
    // ...
    { z: 0.90, name: "Full Substrate Transcendence" }
  ]}
/>
```

## Integration with CBS Runtime

The frontend connects to the CBS runtime via:
- `cbs_interactive_demo.py` (HTTP API)
- GHMP plates for visual encoding
- Evolution logs for trajectory data

## Development

```bash
# Install dependencies (from project root)
npm install

# Run development server
npm run dev
```
