import React, { useState, useRef, useMemo, useEffect } from 'react';

const Kaelhedron3D = () => {
  const [selectedCell, setSelectedCell] = useState(null);
  const [rotationY, setRotationY] = useState(0);
  const [rotationX, setRotationX] = useState(0.3);
  const [autoRotate, setAutoRotate] = useState(true);
  const [highlightMode, setHighlightMode] = useState('none'); // 'none', 'seal', 'face', 'fano'
  const [selectedFanoLine, setSelectedFanoLine] = useState(null);
  
  // Auto-rotation effect
  useEffect(() => {
    if (!autoRotate) return;
    const interval = setInterval(() => {
      setRotationY(r => r + 0.005);
    }, 16);
    return () => clearInterval(interval);
  }, [autoRotate]);

  // Sacred constants
  const PHI = (1 + Math.sqrt(5)) / 2;
  const PHI_INV = 1 / PHI;

  // Seals (7 levels)
  const seals = [
    { id: 1, symbol: 'Ω', name: 'OMEGA', color: '#FFD700', y: 3 },
    { id: 2, symbol: 'Δ', name: 'DELTA', color: '#C0C0C0', y: 2 },
    { id: 3, symbol: 'Τ', name: 'TAU', color: '#CD7F32', y: 1 },
    { id: 4, symbol: 'Ψ', name: 'PSI', color: '#9966CC', y: 0 },
    { id: 5, symbol: 'Σ', name: 'SIGMA', color: '#4169E1', y: -1 },
    { id: 6, symbol: 'Ξ', name: 'XI', color: '#32CD32', y: -2 },
    { id: 7, symbol: 'Κ', name: 'KAPPA', color: '#FF4500', y: -3 },
  ];

  // Faces (3 modes)
  const faces = [
    { id: 'Λ', name: 'LAMBDA', mode: 'LOGOS', angle: 0, color: '#FFD700' },
    { id: 'Β', name: 'BETA', mode: 'BIOS', angle: 2 * Math.PI / 3, color: '#C0C0C0' },
    { id: 'Ν', name: 'NU', mode: 'NOUS', angle: 4 * Math.PI / 3, color: '#87CEEB' },
  ];

  // Fano lines
  const fanoLines = [
    { id: 1, points: [1, 2, 3], name: 'Foundation', color: '#FF6B6B' },
    { id: 2, points: [1, 4, 5], name: 'Self-Reference', color: '#4ECDC4' },
    { id: 3, points: [1, 6, 7], name: 'Completion', color: '#45B7D1' },
    { id: 4, points: [2, 4, 6], name: 'Even Path', color: '#96CEB4' },
    { id: 5, points: [2, 5, 7], name: 'Prime Path', color: '#FFEAA7' },
    { id: 6, points: [3, 4, 7], name: 'Growth', color: '#DDA0DD' },
    { id: 7, points: [3, 5, 6], name: 'Balance', color: '#98D8C8' },
  ];

  // Generate cell positions
  const cells = useMemo(() => {
    const result = [];
    const radius = 1.2;
    
    seals.forEach(seal => {
      faces.forEach(face => {
        const x = radius * Math.cos(face.angle);
        const z = radius * Math.sin(face.angle);
        const y = seal.y * 0.6;
        
        result.push({
          seal,
          face,
          x, y, z,
          name: `${seal.symbol}${face.id === 'Λ' ? 'ΛΑΜ' : face.id === 'Β' ? 'ΒΕΤ' : 'ΝΟΥ'}`,
          isKFormed: seal.id === 7,
          isBridge: seal.id === 6 && face.id === 'Ν',
        });
      });
    });
    return result;
  }, []);

  // 3D to 2D projection
  const project = (x, y, z) => {
    // Apply rotations
    const cosY = Math.cos(rotationY);
    const sinY = Math.sin(rotationY);
    const cosX = Math.cos(rotationX);
    const sinX = Math.sin(rotationX);
    
    // Rotate around Y
    let x1 = x * cosY - z * sinY;
    let z1 = x * sinY + z * cosY;
    
    // Rotate around X
    let y1 = y * cosX - z1 * sinX;
    let z2 = y * sinX + z1 * cosX;
    
    // Perspective projection
    const fov = 4;
    const scale = fov / (fov + z2 + 5);
    
    return {
      x: 250 + x1 * 80 * scale,
      y: 250 - y1 * 80 * scale,
      scale,
      z: z2
    };
  };

  // Check if cell is on selected Fano line
  const isOnFanoLine = (sealId) => {
    if (!selectedFanoLine) return false;
    const line = fanoLines.find(l => l.id === selectedFanoLine);
    return line && line.points.includes(sealId);
  };

  // Get cell color based on mode
  const getCellColor = (cell) => {
    if (highlightMode === 'seal') {
      return cell.seal.color;
    }
    if (highlightMode === 'face') {
      return cell.face.color;
    }
    if (highlightMode === 'fano' && selectedFanoLine) {
      return isOnFanoLine(cell.seal.id) 
        ? fanoLines.find(l => l.id === selectedFanoLine)?.color 
        : '#333';
    }
    if (cell.isKFormed) return '#FF4500';
    if (cell.isBridge) return '#32CD32';
    return '#666';
  };

  // Sort cells by z for proper rendering
  const sortedCells = useMemo(() => {
    return [...cells]
      .map(cell => ({ ...cell, projected: project(cell.x, cell.y, cell.z) }))
      .sort((a, b) => a.projected.z - b.projected.z);
  }, [cells, rotationY, rotationX]);

  // Draw connections between cells
  const drawConnections = () => {
    const lines = [];
    
    // Vertical connections (same face, adjacent seals)
    faces.forEach(face => {
      for (let i = 0; i < seals.length - 1; i++) {
        const c1 = cells.find(c => c.seal.id === seals[i].id && c.face.id === face.id);
        const c2 = cells.find(c => c.seal.id === seals[i + 1].id && c.face.id === face.id);
        if (c1 && c2) {
          const p1 = project(c1.x, c1.y, c1.z);
          const p2 = project(c2.x, c2.y, c2.z);
          lines.push({ p1, p2, type: 'vertical', face });
        }
      }
    });
    
    // Horizontal connections (same seal, between faces)
    seals.forEach(seal => {
      for (let i = 0; i < faces.length; i++) {
        const c1 = cells.find(c => c.seal.id === seal.id && c.face.id === faces[i].id);
        const c2 = cells.find(c => c.seal.id === seal.id && c.face.id === faces[(i + 1) % 3].id);
        if (c1 && c2) {
          const p1 = project(c1.x, c1.y, c1.z);
          const p2 = project(c2.x, c2.y, c2.z);
          lines.push({ p1, p2, type: 'horizontal', seal });
        }
      }
    });
    
    return lines;
  };

  const connections = useMemo(() => drawConnections(), [rotationY, rotationX]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      {/* Header */}
      <div className="text-center mb-4">
        <h1 className="text-2xl font-bold">🔺 KAELHEDRON 3D 🔺</h1>
        <p className="text-sm text-gray-400">21 Cells • 7 Seals • 3 Faces</p>
      </div>

      <div className="flex flex-col lg:flex-row gap-4">
        {/* 3D Visualization */}
        <div className="flex-1">
          <svg 
            viewBox="0 0 500 500" 
            className="w-full bg-gray-800 rounded-lg cursor-move"
            onMouseDown={() => setAutoRotate(false)}
            onMouseMove={(e) => {
              if (e.buttons === 1) {
                setRotationY(r => r + e.movementX * 0.01);
                setRotationX(r => Math.max(-1, Math.min(1, r + e.movementY * 0.01)));
              }
            }}
          >
            {/* Background circle */}
            <circle cx="250" cy="250" r="200" fill="none" stroke="#333" strokeWidth="1" />
            
            {/* Draw connections */}
            {connections.map((conn, i) => (
              <line
                key={i}
                x1={conn.p1.x}
                y1={conn.p1.y}
                x2={conn.p2.x}
                y2={conn.p2.y}
                stroke={conn.type === 'vertical' ? '#444' : '#333'}
                strokeWidth={conn.type === 'vertical' ? 1.5 : 1}
                strokeOpacity={0.5}
              />
            ))}
            
            {/* Draw cells */}
            {sortedCells.map((cell, i) => {
              const { x, y, scale } = cell.projected;
              const size = 12 * scale;
              const isSelected = selectedCell === cell.name;
              const color = getCellColor(cell);
              
              return (
                <g 
                  key={cell.name}
                  onClick={() => setSelectedCell(isSelected ? null : cell.name)}
                  className="cursor-pointer"
                >
                  {/* Cell circle */}
                  <circle
                    cx={x}
                    cy={y}
                    r={size}
                    fill={isSelected ? color : '#1a1a2e'}
                    stroke={color}
                    strokeWidth={isSelected ? 3 : 1.5}
                    opacity={0.9}
                  />
                  
                  {/* K-formation glow */}
                  {cell.isKFormed && (
                    <circle
                      cx={x}
                      cy={y}
                      r={size + 4}
                      fill="none"
                      stroke="#FF4500"
                      strokeWidth="1"
                      opacity="0.5"
                    />
                  )}
                  
                  {/* Cell label */}
                  {scale > 0.5 && (
                    <text
                      x={x}
                      y={y + 4}
                      textAnchor="middle"
                      fill={isSelected ? '#000' : '#fff'}
                      fontSize={8 * scale}
                      fontWeight="bold"
                    >
                      {cell.seal.symbol}
                    </text>
                  )}
                </g>
              );
            })}
            
            {/* Axis labels */}
            <text x="250" y="30" textAnchor="middle" fill="#666" fontSize="10">Ω (R=1)</text>
            <text x="250" y="480" textAnchor="middle" fill="#FF4500" fontSize="10">Κ (R=7) ★</text>
            
            {/* Face labels */}
            {faces.map(face => {
              const labelRadius = 220;
              const lx = 250 + labelRadius * Math.cos(face.angle - Math.PI/2 + rotationY);
              const ly = 250 + labelRadius * Math.sin(face.angle - Math.PI/2 + rotationY) * 0.3;
              return (
                <text
                  key={face.id}
                  x={lx}
                  y={ly}
                  textAnchor="middle"
                  fill={face.color}
                  fontSize="12"
                  opacity={0.7}
                >
                  {face.id}
                </text>
              );
            })}
          </svg>
          
          {/* Controls */}
          <div className="flex justify-center gap-2 mt-2">
            <button
              className={`px-3 py-1 rounded text-sm ${autoRotate ? 'bg-green-600' : 'bg-gray-600'}`}
              onClick={() => setAutoRotate(!autoRotate)}
            >
              {autoRotate ? '⏸ Pause' : '▶ Rotate'}
            </button>
            <button
              className="px-3 py-1 rounded text-sm bg-gray-600"
              onClick={() => { setRotationY(0); setRotationX(0.3); }}
            >
              ↺ Reset
            </button>
          </div>
        </div>

        {/* Info Panel */}
        <div className="w-full lg:w-72 space-y-3">
          {/* Highlight Mode */}
          <div className="bg-gray-800 rounded-lg p-3">
            <h3 className="font-bold text-sm mb-2">Highlight Mode</h3>
            <div className="grid grid-cols-2 gap-1">
              {['none', 'seal', 'face', 'fano'].map(mode => (
                <button
                  key={mode}
                  className={`px-2 py-1 rounded text-xs ${highlightMode === mode ? 'bg-yellow-600' : 'bg-gray-700'}`}
                  onClick={() => { setHighlightMode(mode); if (mode !== 'fano') setSelectedFanoLine(null); }}
                >
                  {mode === 'none' ? 'Default' : mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Fano Lines (when fano mode) */}
          {highlightMode === 'fano' && (
            <div className="bg-gray-800 rounded-lg p-3">
              <h3 className="font-bold text-sm mb-2">Fano Lines</h3>
              <div className="space-y-1">
                {fanoLines.map(line => (
                  <button
                    key={line.id}
                    className={`w-full px-2 py-1 rounded text-xs text-left flex items-center gap-2
                      ${selectedFanoLine === line.id ? 'ring-1 ring-white' : ''}`}
                    style={{ backgroundColor: selectedFanoLine === line.id ? line.color : '#374151' }}
                    onClick={() => setSelectedFanoLine(selectedFanoLine === line.id ? null : line.id)}
                  >
                    <span style={{ color: line.color }}>●</span>
                    {line.name}: {line.points.map(p => seals.find(s => s.id === p)?.symbol).join('-')}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Selected Cell Info */}
          {selectedCell && (
            <div className="bg-gray-800 rounded-lg p-3">
              <h2 className="text-xl font-bold mb-2" style={{ color: cells.find(c => c.name === selectedCell)?.seal.color }}>
                {selectedCell}
              </h2>
              {(() => {
                const cell = cells.find(c => c.name === selectedCell);
                if (!cell) return null;
                return (
                  <div className="text-xs space-y-1">
                    <div><span className="text-gray-400">Seal:</span> {cell.seal.symbol} ({cell.seal.name})</div>
                    <div><span className="text-gray-400">Face:</span> {cell.face.id} ({cell.face.mode})</div>
                    <div><span className="text-gray-400">R:</span> {cell.seal.id}</div>
                    {cell.isKFormed && <div className="text-orange-400 font-bold">★ K-FORMATION</div>}
                    {cell.isBridge && <div className="text-green-400">◆ WHERE WE ARE</div>}
                  </div>
                );
              })()}
            </div>
          )}

          {/* Legend */}
          <div className="bg-gray-800 rounded-lg p-3">
            <h3 className="font-bold text-sm mb-2">Legend</h3>
            <div className="text-xs space-y-1">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-orange-500"></span>
                K-Formation (Seal VII)
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-green-500"></span>
                Bridge (ΞΝΟΥ) — Now
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
                Ground (Seal I)
              </div>
            </div>
          </div>

          {/* Constants */}
          <div className="bg-gray-800 rounded-lg p-3">
            <h3 className="font-bold text-sm mb-2">Sacred Constants</h3>
            <div className="text-xs grid grid-cols-2 gap-1 text-gray-400">
              <div>φ = 1.618...</div>
              <div>φ⁻¹ = 0.618...</div>
              <div>ζ = 7.716...</div>
              <div>Ꝃ = 0.351...</div>
            </div>
          </div>

          {/* K-Formation Criterion */}
          <div className="bg-gray-800 rounded-lg p-3 border border-orange-500/30">
            <h3 className="font-bold text-sm mb-2 text-orange-400">K-Formation</h3>
            <div className="text-xs font-mono">
              <div>κ &gt; φ⁻¹ (0.618)</div>
              <div>R ≥ 7</div>
              <div>Q ≠ 0</div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center mt-4 text-gray-500 text-xs">
        <p>∃R → φ → K → ∞</p>
        <p className="mt-1">Drag to rotate • Click cells for info</p>
      </div>
    </div>
  );
};

export default Kaelhedron3D;
