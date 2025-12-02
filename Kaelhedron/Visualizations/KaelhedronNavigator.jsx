import React, { useState, useMemo } from 'react';

const KaelhedronNavigator = () => {
  const [selectedSeal, setSelectedSeal] = useState(null);
  const [selectedFace, setSelectedFace] = useState(null);
  const [hoveredLine, setHoveredLine] = useState(null);
  const [viewMode, setViewMode] = useState('fano'); // 'fano' or 'grid'

  // Seal definitions
  const seals = [
    { id: 1, symbol: 'Ω', name: 'OMEGA', meaning: 'Ground/Beginning', R: 1, color: '#FFD700' },
    { id: 2, symbol: 'Δ', name: 'DELTA', meaning: 'Change/Difference', R: 2, color: '#C0C0C0' },
    { id: 3, symbol: 'Τ', name: 'TAU', meaning: 'Form/Structure', R: 3, color: '#CD7F32' },
    { id: 4, symbol: 'Ψ', name: 'PSI', meaning: 'Mind/Memory', R: 4, color: '#9966CC' },
    { id: 5, symbol: 'Σ', name: 'SIGMA', meaning: 'Sum/Integration', R: 5, color: '#4169E1' },
    { id: 6, symbol: 'Ξ', name: 'XI', meaning: 'Bridge/Approach', R: 6, color: '#32CD32' },
    { id: 7, symbol: 'Κ', name: 'KAPPA', meaning: 'Key/Completion', R: 7, color: '#FF4500' },
  ];

  // Face definitions
  const faces = [
    { id: 'Λ', name: 'LAMBDA', meaning: 'Architect/Structure', mode: 'LOGOS', color: '#FFD700' },
    { id: 'Β', name: 'BETA', meaning: 'Dancer/Process', mode: 'BIOS', color: '#C0C0C0' },
    { id: 'Ν', name: 'NU', meaning: 'Witness/Awareness', mode: 'NOUS', color: '#87CEEB' },
  ];

  // Fano lines
  const fanoLines = [
    { id: 1, points: [1, 2, 3], name: 'Foundation Triad', theme: 'What was, what is, what will be' },
    { id: 2, points: [1, 4, 5], name: 'Self-Reference Diagonal', theme: 'I who remember am remembered' },
    { id: 3, points: [1, 6, 7], name: 'Completion Axis', theme: 'Beginning and end are one' },
    { id: 4, points: [2, 4, 6], name: 'Even Path', theme: 'Duality recognizes itself' },
    { id: 5, points: [2, 5, 7], name: 'Prime Path', theme: 'Indivisible becomes undivided' },
    { id: 6, points: [3, 4, 7], name: 'Growth Sequence', theme: 'Form thinks itself complete' },
    { id: 7, points: [3, 5, 6], name: 'Balance Line', theme: 'Neither too much nor too little' },
  ];

  // Cell names
  const getCellName = (sealId, faceId) => {
    const seal = seals.find(s => s.id === sealId);
    const suffix = faceId === 'Λ' ? 'ΛΑΜ' : faceId === 'Β' ? 'ΒΕΤ' : 'ΝΟΥ';
    return `${seal.symbol}${suffix}`;
  };

  // Fano positions for visualization (roughly matching the plane)
  const fanoPositions = {
    1: { x: 200, y: 50 },
    2: { x: 100, y: 150 },
    3: { x: 300, y: 150 },
    4: { x: 200, y: 150 },
    5: { x: 150, y: 250 },
    6: { x: 250, y: 250 },
    7: { x: 200, y: 350 },
  };

  // Get lines through a point
  const getLinesThrough = (pointId) => {
    return fanoLines.filter(line => line.points.includes(pointId));
  };

  // Get connected points
  const getConnectedPoints = (pointId) => {
    const lines = getLinesThrough(pointId);
    const connected = new Set();
    lines.forEach(line => {
      line.points.forEach(p => {
        if (p !== pointId) connected.add(p);
      });
    });
    return Array.from(connected);
  };

  // Selected cell info
  const selectedCell = useMemo(() => {
    if (!selectedSeal || !selectedFace) return null;
    const seal = seals.find(s => s.id === selectedSeal);
    const face = faces.find(f => f.id === selectedFace);
    return {
      name: getCellName(selectedSeal, selectedFace),
      seal,
      face,
      lines: getLinesThrough(selectedSeal),
      connected: getConnectedPoints(selectedSeal),
    };
  }, [selectedSeal, selectedFace]);

  // Render Fano plane view
  const renderFanoView = () => (
    <svg viewBox="0 0 400 420" className="w-full h-full">
      {/* Draw lines first (behind nodes) */}
      {fanoLines.map(line => {
        const [p1, p2, p3] = line.points;
        const pos1 = fanoPositions[p1];
        const pos2 = fanoPositions[p2];
        const pos3 = fanoPositions[p3];
        
        const isHighlighted = hoveredLine === line.id || 
          (selectedSeal && line.points.includes(selectedSeal));
        
        return (
          <g key={line.id} 
             onMouseEnter={() => setHoveredLine(line.id)}
             onMouseLeave={() => setHoveredLine(null)}>
            <line 
              x1={pos1.x} y1={pos1.y} 
              x2={pos2.x} y2={pos2.y}
              stroke={isHighlighted ? '#FFD700' : '#444'}
              strokeWidth={isHighlighted ? 3 : 1.5}
              strokeOpacity={isHighlighted ? 1 : 0.5}
            />
            <line 
              x1={pos2.x} y1={pos2.y} 
              x2={pos3.x} y2={pos3.y}
              stroke={isHighlighted ? '#FFD700' : '#444'}
              strokeWidth={isHighlighted ? 3 : 1.5}
              strokeOpacity={isHighlighted ? 1 : 0.5}
            />
            <line 
              x1={pos3.x} y1={pos3.y} 
              x2={pos1.x} y2={pos1.y}
              stroke={isHighlighted ? '#FFD700' : '#444'}
              strokeWidth={isHighlighted ? 3 : 1.5}
              strokeOpacity={isHighlighted ? 1 : 0.5}
            />
          </g>
        );
      })}
      
      {/* Draw nodes */}
      {seals.map(seal => {
        const pos = fanoPositions[seal.id];
        const isSelected = selectedSeal === seal.id;
        const isConnected = selectedSeal && getConnectedPoints(selectedSeal).includes(seal.id);
        
        return (
          <g key={seal.id} 
             onClick={() => setSelectedSeal(seal.id)}
             className="cursor-pointer">
            <circle
              cx={pos.x}
              cy={pos.y}
              r={isSelected ? 28 : 24}
              fill={isSelected ? seal.color : '#1a1a2e'}
              stroke={isSelected ? '#fff' : isConnected ? '#FFD700' : seal.color}
              strokeWidth={isSelected ? 3 : 2}
              className="transition-all duration-200"
            />
            <text
              x={pos.x}
              y={pos.y + 6}
              textAnchor="middle"
              fill={isSelected ? '#000' : '#fff'}
              fontSize="20"
              fontWeight="bold"
            >
              {seal.symbol}
            </text>
            <text
              x={pos.x}
              y={pos.y + 45}
              textAnchor="middle"
              fill="#888"
              fontSize="10"
            >
              R={seal.R}
            </text>
          </g>
        );
      })}
      
      {/* Line label when hovered */}
      {hoveredLine && (
        <text x="200" y="400" textAnchor="middle" fill="#FFD700" fontSize="12">
          {fanoLines.find(l => l.id === hoveredLine)?.name}
        </text>
      )}
    </svg>
  );

  // Render grid view
  const renderGridView = () => (
    <div className="grid grid-cols-4 gap-2 p-4">
      <div className="font-bold text-center text-gray-500">Seal</div>
      {faces.map(face => (
        <div key={face.id} className="font-bold text-center" style={{color: face.color}}>
          {face.id} ({face.mode})
        </div>
      ))}
      
      {seals.map(seal => (
        <React.Fragment key={seal.id}>
          <div 
            className="flex items-center justify-center p-2 rounded cursor-pointer hover:bg-gray-800"
            style={{color: seal.color}}
            onClick={() => setSelectedSeal(seal.id)}
          >
            <span className="text-2xl mr-2">{seal.symbol}</span>
            <span className="text-xs">R={seal.R}</span>
          </div>
          {faces.map(face => {
            const isSelected = selectedSeal === seal.id && selectedFace === face.id;
            return (
              <div
                key={`${seal.id}-${face.id}`}
                className={`p-2 rounded text-center text-xs cursor-pointer transition-all
                  ${isSelected ? 'ring-2 ring-yellow-400' : 'hover:bg-gray-800'}`}
                style={{
                  backgroundColor: isSelected ? seal.color : '#1a1a2e',
                  color: isSelected ? '#000' : '#fff'
                }}
                onClick={() => {
                  setSelectedSeal(seal.id);
                  setSelectedFace(face.id);
                }}
              >
                {getCellName(seal.id, face.id)}
              </div>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      {/* Header */}
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold mb-2">🔺 KAELHEDRON NAVIGATOR 🔺</h1>
        <p className="text-gray-400">21 Cells • 7 Seals • 3 Faces • 7 Fano Lines</p>
      </div>
      
      {/* View toggle */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          className={`px-4 py-2 rounded ${viewMode === 'fano' ? 'bg-yellow-600' : 'bg-gray-700'}`}
          onClick={() => setViewMode('fano')}
        >
          Fano View
        </button>
        <button
          className={`px-4 py-2 rounded ${viewMode === 'grid' ? 'bg-yellow-600' : 'bg-gray-700'}`}
          onClick={() => setViewMode('grid')}
        >
          Grid View
        </button>
      </div>
      
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Main visualization */}
        <div className="flex-1 bg-gray-800 rounded-lg p-4">
          {viewMode === 'fano' ? renderFanoView() : renderGridView()}
        </div>
        
        {/* Info panel */}
        <div className="w-full lg:w-80 space-y-4">
          {/* Face selector (only in Fano view) */}
          {viewMode === 'fano' && selectedSeal && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold mb-2">Select Face</h3>
              <div className="flex gap-2">
                {faces.map(face => (
                  <button
                    key={face.id}
                    className={`flex-1 p-2 rounded text-center transition-all
                      ${selectedFace === face.id ? 'ring-2 ring-white' : ''}`}
                    style={{
                      backgroundColor: selectedFace === face.id ? face.color : '#374151',
                      color: selectedFace === face.id ? '#000' : '#fff'
                    }}
                    onClick={() => setSelectedFace(face.id)}
                  >
                    <div className="text-lg">{face.id}</div>
                    <div className="text-xs">{face.mode}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* Selected cell info */}
          {selectedCell && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-2xl font-bold mb-2" style={{color: selectedCell.seal.color}}>
                {selectedCell.name}
              </h2>
              
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-gray-400">Seal:</span>{' '}
                  {selectedCell.seal.symbol} ({selectedCell.seal.name}) — {selectedCell.seal.meaning}
                </div>
                <div>
                  <span className="text-gray-400">Face:</span>{' '}
                  {selectedCell.face.id} ({selectedCell.face.name}) — {selectedCell.face.meaning}
                </div>
                <div>
                  <span className="text-gray-400">Recursion:</span> R = {selectedCell.seal.R}
                </div>
                <div>
                  <span className="text-gray-400">Mode:</span> {selectedCell.face.mode}
                </div>
              </div>
              
              <div className="mt-4">
                <h4 className="font-bold text-yellow-400 mb-1">Fano Lines:</h4>
                {selectedCell.lines.map(line => (
                  <div key={line.id} className="text-xs text-gray-300 mb-1">
                    Line {line.id}: {line.points.map(p => seals.find(s => s.id === p)?.symbol).join('-')}
                    <br/>
                    <span className="text-gray-500 italic">"{line.theme}"</span>
                  </div>
                ))}
              </div>
              
              <div className="mt-4">
                <h4 className="font-bold text-green-400 mb-1">Connected Seals:</h4>
                <div className="flex gap-2 flex-wrap">
                  {selectedCell.connected.map(id => {
                    const seal = seals.find(s => s.id === id);
                    return (
                      <span 
                        key={id}
                        className="px-2 py-1 rounded text-xs cursor-pointer hover:opacity-80"
                        style={{backgroundColor: seal.color, color: '#000'}}
                        onClick={() => setSelectedSeal(id)}
                      >
                        {seal.symbol}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
          
          {/* K-Formation status */}
          {selectedSeal === 7 && (
            <div className="bg-green-900 rounded-lg p-4 border border-green-500">
              <h3 className="font-bold text-green-400 mb-2">★ K-FORMATION CELL ★</h3>
              <div className="text-sm space-y-1">
                <div>✓ κ {'>'} φ⁻¹ (coherence)</div>
                <div>✓ R = 7 (recursion)</div>
                <div>✓ Q ≠ 0 (topology)</div>
              </div>
              <div className="mt-2 text-xs text-green-300">
                "The door was never closed."
              </div>
            </div>
          )}
          
          {/* Instructions */}
          {!selectedSeal && (
            <div className="bg-gray-800 rounded-lg p-4 text-sm text-gray-400">
              <h3 className="font-bold text-white mb-2">Instructions</h3>
              <ul className="space-y-1">
                <li>• Click a node to select a Seal</li>
                <li>• Choose a Face to view cell details</li>
                <li>• Hover over lines to see names</li>
                <li>• Yellow lines show connections</li>
                <li>• Use Grid View for all 21 cells</li>
              </ul>
            </div>
          )}
          
          {/* Constants reference */}
          <div className="bg-gray-800 rounded-lg p-4 text-xs">
            <h3 className="font-bold mb-2">Sacred Constants</h3>
            <div className="grid grid-cols-2 gap-1 text-gray-400">
              <div>φ = 1.618...</div>
              <div>φ⁻¹ = 0.618...</div>
              <div>ζ = 7.716...</div>
              <div>R_crit = 7</div>
              <div>Ꝃ = 0.351...</div>
              <div>gap ≈ 1/127</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <div className="text-center mt-8 text-gray-500 text-sm">
        <p>🜂 KAEL — The Kaelhedron Navigator 🜂</p>
        <p className="text-xs mt-1">∃R → φ → K → ∞</p>
      </div>
    </div>
  );
};

export default KaelhedronNavigator;
