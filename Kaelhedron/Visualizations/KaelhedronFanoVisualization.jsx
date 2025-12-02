import React, { useState, useEffect, useCallback, useRef } from 'react';

// Sacred Constants
const PHI = (1 + Math.sqrt(5)) / 2;
const PHI_INV = 2 / (1 + Math.sqrt(5));

// Seal data
const SEALS = [
  { id: 1, symbol: 'Ω', name: 'Ground', color: '#4A90A4' },
  { id: 2, symbol: 'Δ', name: 'Change', color: '#7B68EE' },
  { id: 3, symbol: 'Τ', name: 'Form', color: '#20B2AA' },
  { id: 4, symbol: 'Ψ', name: 'Mind', color: '#FFD700' },
  { id: 5, symbol: 'Σ', name: 'Sum', color: '#FF6B6B' },
  { id: 6, symbol: 'Ξ', name: 'Bridge', color: '#9370DB' },
  { id: 7, symbol: 'Κ', name: 'Key', color: '#00CED1' },
];

const FACES = [
  { id: 0, symbol: 'Λ', name: 'Logos', mode: 'Structure' },
  { id: 1, symbol: 'Β', name: 'Bios', mode: 'Process' },
  { id: 2, symbol: 'Ν', name: 'Nous', mode: 'Awareness' },
];

// Fano Lines
const FANO_LINES = [
  { id: 0, points: [1, 2, 3], name: 'Foundation', color: '#FF6B6B' },
  { id: 1, points: [1, 4, 5], name: 'Self-Reference', color: '#4ECDC4' },
  { id: 2, points: [1, 6, 7], name: 'Completion', color: '#45B7D1' },
  { id: 3, points: [2, 4, 6], name: 'Even Path', color: '#96CEB4' },
  { id: 4, points: [2, 5, 7], name: 'Prime Path', color: '#FFEAA7' },
  { id: 5, points: [3, 4, 7], name: 'Growth', color: '#DDA0DD' },
  { id: 6, points: [3, 5, 6], name: 'Balance', color: '#98D8C8' },
];

// Fano plane point positions (hexagonal with center)
const getFanoPosition = (pointId, centerX, centerY, radius) => {
  if (pointId === 4) { // Ψ is center
    return { x: centerX, y: centerY };
  }
  
  // Arrange other 6 points in hexagon
  const hexOrder = [1, 2, 3, 7, 6, 5]; // Clockwise from top
  const idx = hexOrder.indexOf(pointId);
  if (idx === -1) return { x: centerX, y: centerY };
  
  const angle = (idx * Math.PI / 3) - Math.PI / 2; // Start from top
  return {
    x: centerX + radius * Math.cos(angle),
    y: centerY + radius * Math.sin(angle),
  };
};

// F8 multiplication
const F8_MUL = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 2, 4, 6, 3, 1, 7, 5],
  [0, 3, 6, 5, 7, 4, 1, 2],
  [0, 4, 3, 7, 6, 2, 5, 1],
  [0, 5, 1, 4, 2, 7, 3, 6],
  [0, 6, 7, 1, 5, 3, 2, 4],
  [0, 7, 5, 2, 1, 6, 4, 3],
];

const f8Mul = (a, b) => F8_MUL[a][b];

// PSL(3,2) generators
const SIGMA = { 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 1 };
const TAU = { 1: 1, 2: 4, 3: 7, 4: 2, 5: 6, 6: 5, 7: 3 };

const applyPerm = (perm, point) => perm[point] || point;

// Initialize cell state
const initCells = () => {
  const cells = {};
  for (let seal = 1; seal <= 7; seal++) {
    for (let face = 0; face < 3; face++) {
      cells[`${seal}-${face}`] = {
        coherence: seal === 7 ? 0.67 : 0.5,
        phase: Math.random() * Math.PI * 2,
        f8Value: seal,
      };
    }
  }
  return cells;
};

// Main Component
export default function KaelhedronFanoVisualization() {
  const [cells, setCells] = useState(initCells);
  const [time, setTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedSeal, setSelectedSeal] = useState(null);
  const [selectedLine, setSelectedLine] = useState(null);
  const [viewMode, setViewMode] = useState('fano'); // 'fano', 'grid', 'lines'
  const [coupling, setCoupling] = useState(0.15);
  const animationRef = useRef();

  // Calculate metrics
  const totalCoherence = Object.values(cells).reduce((sum, c) => sum + c.coherence, 0) / 21;
  const kFormed = totalCoherence > PHI_INV;
  const gap = Math.max(0, PHI_INV - totalCoherence);
  const gapPercent = (gap / PHI_INV * 100).toFixed(1);

  const getSealCoherence = (sealId) => {
    let sum = 0;
    for (let face = 0; face < 3; face++) {
      sum += cells[`${sealId}-${face}`]?.coherence || 0;
    }
    return sum / 3;
  };

  const getLineCoherence = (line) => {
    let sum = 0;
    let count = 0;
    for (const seal of line.points) {
      for (let face = 0; face < 3; face++) {
        sum += cells[`${seal}-${face}`]?.coherence || 0;
        count++;
      }
    }
    return sum / count;
  };

  // Evolution step
  const evolveStep = useCallback((dt = 0.05) => {
    setCells(prevCells => {
      const newCells = { ...prevCells };
      
      for (let seal = 1; seal <= 7; seal++) {
        for (let face = 0; face < 3; face++) {
          const key = `${seal}-${face}`;
          const cell = prevCells[key];
          let dCoh = 0;
          let dPhase = 0.1 * seal / 7;

          // Line-based coupling
          for (const line of FANO_LINES) {
            if (line.points.includes(seal)) {
              const lineCoh = getLineCoherence(line);
              dCoh += coupling * (lineCoh - cell.coherence);

              // Phase coupling
              for (const otherSeal of line.points) {
                if (otherSeal !== seal) {
                  const otherCell = prevCells[`${otherSeal}-${face}`];
                  if (otherCell) {
                    dPhase += coupling * Math.sin(otherCell.phase - cell.phase) * 0.3;
                  }
                }
              }
            }
          }

          // Central hub (Ψ)
          if (seal === 4) {
            const total = Object.values(prevCells).reduce((s, c) => s + c.coherence, 0) / 21;
            dCoh += coupling * 0.5 * (total - cell.coherence);
          } else {
            const psiCell = prevCells[`4-${face}`];
            if (psiCell) {
              dCoh += coupling * 0.3 * (psiCell.coherence - cell.coherence);
            }
          }

          newCells[key] = {
            ...cell,
            coherence: Math.max(0, Math.min(1, cell.coherence + dCoh * dt)),
            phase: (cell.phase + dPhase * dt) % (Math.PI * 2),
          };
        }
      }
      
      return newCells;
    });
    setTime(t => t + dt);
  }, [coupling]);

  // Animation loop
  useEffect(() => {
    if (isRunning) {
      const animate = () => {
        evolveStep();
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRunning, evolveStep]);

  // Inject coherence
  const injectAt = (sealId, amount = 0.3) => {
    setCells(prev => {
      const newCells = { ...prev };
      for (let face = 0; face < 3; face++) {
        const key = `${sealId}-${face}`;
        if (newCells[key]) {
          newCells[key] = {
            ...newCells[key],
            coherence: Math.min(1, newCells[key].coherence + amount),
          };
        }
      }
      return newCells;
    });
  };

  // Apply symmetry
  const applySymmetry = (type) => {
    const perm = type === 'rotate' ? SIGMA : TAU;
    setCells(prev => {
      const newCells = {};
      for (let seal = 1; seal <= 7; seal++) {
        for (let face = 0; face < 3; face++) {
          const oldKey = `${seal}-${face}`;
          const newSeal = applyPerm(perm, seal);
          const newKey = `${newSeal}-${face}`;
          newCells[newKey] = { ...prev[oldKey] };
        }
      }
      return newCells;
    });
  };

  // Reset
  const reset = () => {
    setCells(initCells());
    setTime(0);
    setIsRunning(false);
  };

  // Fano Plane View
  const FanoPlaneView = () => {
    const centerX = 200;
    const centerY = 180;
    const radius = 120;

    return (
      <svg viewBox="0 0 400 360" className="w-full h-64">
        {/* Background */}
        <rect width="400" height="360" fill="#0a0a1a" rx="8" />
        
        {/* Draw lines first */}
        {FANO_LINES.map(line => {
          const positions = line.points.map(p => getFanoPosition(p, centerX, centerY, radius));
          const isSelected = selectedLine?.id === line.id;
          const lineCoh = getLineCoherence(line);
          
          // For the inscribed circle (line through center), draw as arc
          if (line.points.includes(4)) {
            // Line through center - draw as two segments
            return (
              <g key={line.id}>
                <line
                  x1={positions[0].x} y1={positions[0].y}
                  x2={positions[1].x} y2={positions[1].y}
                  stroke={line.color}
                  strokeWidth={isSelected ? 3 : 2}
                  strokeOpacity={0.3 + lineCoh * 0.7}
                  onClick={() => setSelectedLine(isSelected ? null : line)}
                  className="cursor-pointer"
                />
                <line
                  x1={positions[1].x} y1={positions[1].y}
                  x2={positions[2].x} y2={positions[2].y}
                  stroke={line.color}
                  strokeWidth={isSelected ? 3 : 2}
                  strokeOpacity={0.3 + lineCoh * 0.7}
                  onClick={() => setSelectedLine(isSelected ? null : line)}
                  className="cursor-pointer"
                />
              </g>
            );
          }
          
          // Outer lines - draw as curved paths
          const [p1, p2, p3] = positions;
          const midX = (p1.x + p2.x + p3.x) / 3;
          const midY = (p1.y + p2.y + p3.y) / 3;
          
          return (
            <g key={line.id}>
              <path
                d={`M ${p1.x} ${p1.y} Q ${midX} ${midY} ${p2.x} ${p2.y}`}
                fill="none"
                stroke={line.color}
                strokeWidth={isSelected ? 3 : 2}
                strokeOpacity={0.3 + lineCoh * 0.7}
                onClick={() => setSelectedLine(isSelected ? null : line)}
                className="cursor-pointer"
              />
              <path
                d={`M ${p2.x} ${p2.y} Q ${midX} ${midY} ${p3.x} ${p3.y}`}
                fill="none"
                stroke={line.color}
                strokeWidth={isSelected ? 3 : 2}
                strokeOpacity={0.3 + lineCoh * 0.7}
                onClick={() => setSelectedLine(isSelected ? null : line)}
                className="cursor-pointer"
              />
              <path
                d={`M ${p3.x} ${p3.y} Q ${midX} ${midY} ${p1.x} ${p1.y}`}
                fill="none"
                stroke={line.color}
                strokeWidth={isSelected ? 3 : 2}
                strokeOpacity={0.3 + lineCoh * 0.7}
                onClick={() => setSelectedLine(isSelected ? null : line)}
                className="cursor-pointer"
              />
            </g>
          );
        })}

        {/* Draw points */}
        {SEALS.map(seal => {
          const pos = getFanoPosition(seal.id, centerX, centerY, radius);
          const coh = getSealCoherence(seal.id);
          const isSelected = selectedSeal === seal.id;
          const pointRadius = 20 + coh * 15;
          
          return (
            <g key={seal.id}>
              {/* Glow effect */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={pointRadius + 10}
                fill={seal.color}
                opacity={coh * 0.3}
              />
              {/* Main circle */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={pointRadius}
                fill={seal.color}
                opacity={0.3 + coh * 0.7}
                stroke={isSelected ? '#fff' : seal.color}
                strokeWidth={isSelected ? 3 : 1}
                onClick={() => setSelectedSeal(isSelected ? null : seal.id)}
                className="cursor-pointer transition-all duration-200"
              />
              {/* Symbol */}
              <text
                x={pos.x}
                y={pos.y + 6}
                textAnchor="middle"
                fill="#fff"
                fontSize="18"
                fontWeight="bold"
                className="pointer-events-none"
              >
                {seal.symbol}
              </text>
              {/* Coherence indicator */}
              <text
                x={pos.x}
                y={pos.y + 45}
                textAnchor="middle"
                fill="#aaa"
                fontSize="10"
              >
                {(coh * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}

        {/* Title */}
        <text x="200" y="25" textAnchor="middle" fill="#fff" fontSize="14" fontWeight="bold">
          FANO PLANE PG(2,2)
        </text>
      </svg>
    );
  };

  // Cell Grid View
  const CellGridView = () => (
    <div className="grid grid-cols-8 gap-1 p-2 bg-gray-900 rounded-lg">
      {/* Header row */}
      <div className="text-center text-xs text-gray-500 p-1">Face</div>
      {SEALS.map(seal => (
        <div 
          key={seal.id} 
          className="text-center text-xs font-bold p-1"
          style={{ color: seal.color }}
        >
          {seal.symbol}
        </div>
      ))}
      
      {/* Cell rows */}
      {FACES.map(face => (
        <React.Fragment key={face.id}>
          <div className="text-center text-xs text-gray-400 p-1 flex items-center justify-center">
            {face.symbol}
          </div>
          {SEALS.map(seal => {
            const cell = cells[`${seal.id}-${face.id}`];
            const coh = cell?.coherence || 0;
            return (
              <div
                key={`${seal.id}-${face.id}`}
                className="aspect-square rounded flex items-center justify-center text-xs cursor-pointer transition-all hover:scale-110"
                style={{
                  backgroundColor: seal.color,
                  opacity: 0.3 + coh * 0.7,
                }}
                onClick={() => injectAt(seal.id, 0.1)}
                title={`${seal.symbol}${face.symbol}: ${(coh * 100).toFixed(0)}%`}
              >
                {(coh * 100).toFixed(0)}
              </div>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );

  // Line Coherence View
  const LineCoherenceView = () => (
    <div className="space-y-2 p-2 bg-gray-900 rounded-lg">
      <div className="text-xs text-gray-400 mb-2">Fano Line Coherences</div>
      {FANO_LINES.map(line => {
        const coh = getLineCoherence(line);
        const isSelected = selectedLine?.id === line.id;
        return (
          <div 
            key={line.id}
            className={`flex items-center gap-2 cursor-pointer p-1 rounded ${isSelected ? 'bg-gray-800' : ''}`}
            onClick={() => setSelectedLine(isSelected ? null : line)}
          >
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: line.color }}
            />
            <div className="text-xs text-gray-300 w-24">{line.name}</div>
            <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className="h-full transition-all duration-300"
                style={{ 
                  width: `${coh * 100}%`,
                  backgroundColor: line.color,
                }}
              />
            </div>
            <div className="text-xs text-gray-400 w-10 text-right">
              {(coh * 100).toFixed(0)}%
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4">
      {/* Header */}
      <div className="text-center mb-4">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
          KAELHEDRON-FANO VISUALIZATION
        </h1>
        <p className="text-xs text-gray-500">
          21 Cells × 7 Seals × 3 Faces | Line-Based Evolution
        </p>
      </div>

      {/* K-Formation Status */}
      <div className={`mb-4 p-3 rounded-lg text-center ${kFormed ? 'bg-green-900/50 border border-green-500' : 'bg-gray-900 border border-gray-700'}`}>
        <div className="text-sm mb-1">
          {kFormed ? '✓ K-FORMATION ACHIEVED' : '○ Approaching K-Formation'}
        </div>
        <div className="flex justify-center items-center gap-4 text-xs">
          <span>η = {(totalCoherence * 100).toFixed(1)}%</span>
          <span className="text-gray-500">|</span>
          <span>Threshold = {(PHI_INV * 100).toFixed(1)}%</span>
          <span className="text-gray-500">|</span>
          <span>Gap = {gapPercent}%</span>
        </div>
        {/* Progress bar */}
        <div className="mt-2 h-2 bg-gray-800 rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-300 ${kFormed ? 'bg-green-500' : 'bg-cyan-500'}`}
            style={{ width: `${Math.min(100, totalCoherence / PHI_INV * 100)}%` }}
          />
        </div>
      </div>

      {/* Main View */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <FanoPlaneView />
          {selectedLine && (
            <div className="mt-2 p-2 bg-gray-900 rounded text-xs text-center">
              <span style={{ color: selectedLine.color }}>{selectedLine.name}</span>
              <span className="text-gray-500 mx-2">|</span>
              <span>{selectedLine.points.map(p => SEALS[p-1].symbol).join(' → ')}</span>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <CellGridView />
          <LineCoherenceView />
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`p-2 rounded font-bold text-sm ${isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
        >
          {isRunning ? '⏸ Pause' : '▶ Run'}
        </button>
        <button
          onClick={() => evolveStep(0.5)}
          disabled={isRunning}
          className="p-2 rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-sm"
        >
          Step →
        </button>
        <button
          onClick={reset}
          className="p-2 rounded bg-gray-600 hover:bg-gray-700 text-sm"
        >
          Reset
        </button>
        <button
          onClick={() => injectAt(7, 0.3)}
          className="p-2 rounded bg-purple-600 hover:bg-purple-700 text-sm"
        >
          Inject at Κ
        </button>
      </div>

      {/* Symmetry Controls */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        <button
          onClick={() => applySymmetry('rotate')}
          className="p-2 rounded bg-indigo-600 hover:bg-indigo-700 text-sm"
        >
          σ Rotate
        </button>
        <button
          onClick={() => applySymmetry('reflect')}
          className="p-2 rounded bg-pink-600 hover:bg-pink-700 text-sm"
        >
          τ Reflect
        </button>
        <button
          onClick={() => injectAt(4, 0.2)}
          className="p-2 rounded bg-yellow-600 hover:bg-yellow-700 text-sm"
        >
          Inject at Ψ
        </button>
        <button
          onClick={() => injectAt(1, 0.2)}
          className="p-2 rounded bg-cyan-600 hover:bg-cyan-700 text-sm"
        >
          Inject at Ω
        </button>
      </div>

      {/* Coupling Slider */}
      <div className="p-3 bg-gray-900 rounded-lg mb-4">
        <div className="flex items-center gap-4">
          <span className="text-xs text-gray-400">Coupling:</span>
          <input
            type="range"
            min="0"
            max="50"
            value={coupling * 100}
            onChange={(e) => setCoupling(e.target.value / 100)}
            className="flex-1"
          />
          <span className="text-xs text-gray-300 w-12">{(coupling * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Info Panel */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="p-2 bg-gray-900 rounded">
          <div className="text-gray-500">Time</div>
          <div className="font-mono">{time.toFixed(1)}</div>
        </div>
        <div className="p-2 bg-gray-900 rounded">
          <div className="text-gray-500">φ⁻¹</div>
          <div className="font-mono">{PHI_INV.toFixed(4)}</div>
        </div>
        <div className="p-2 bg-gray-900 rounded">
          <div className="text-gray-500">Cells</div>
          <div className="font-mono">21</div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 p-3 bg-gray-900 rounded-lg text-xs">
        <div className="text-gray-500 mb-2">Interaction Guide:</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-gray-400">
          <div>• Click seal to select</div>
          <div>• Click line to highlight</div>
          <div>• Click cell to inject</div>
          <div>• σ = 7-cycle rotation</div>
          <div>• τ = reflection fixing Ω</div>
          <div>• Ψ (4) is central hub</div>
          <div>• Coherence flows along lines</div>
          <div>• K-form at η &gt; φ⁻¹</div>
        </div>
      </div>
    </div>
  );
}
