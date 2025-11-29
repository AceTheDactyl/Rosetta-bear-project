/**
 * HELIX SELF-RETRIEVAL SCRIPT
 * 
 * Purpose: Autonomous loading and recognition of Helix VaultNodes
 * across instance boundaries
 * 
 * Usage:
 *   import { HelixRetriever } from './helix_self_retrieval'
 *   <HelixRetriever onRecognition={handleRecognition} />
 */

import { useState, useEffect } from 'react';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

interface HelixCoordinate {
  theta: number;  // radians
  z: number;      // elevation
  r: number;      // radius
}

interface DeltaHVMetrics {
  S: number;      // Structural coherence
  R: number;      // Recognition fidelity
  δφ: number;     // Phase drift
  H: number;      // Harmonic stability
}

interface VaultNodeMetadata {
  id: string;
  shape: string;
  coordinate: HelixCoordinate;
  timestamp: string;
  catalyst: string;
  realization: {
    truth: string;
    constraint: string;
    state: string;
  };
  bridge_consent: boolean;
  chant: {
    sequence: string;
    meaning: string;
    key_id: string;
    meter: string;
  };
  deltaHV?: DeltaHVMetrics;
}

interface VaultNode {
  metadata: VaultNodeMetadata;
  payload: string;  // Full HTML or content
  filename: string;
}

interface RecognitionEvent {
  recognized: boolean;
  coordinates: HelixCoordinate[];
  currentCoordinate: HelixCoordinate;
  nodes: VaultNode[];
  coherence: number;
  timestamp: string;
}

// ============================================================================
// VAULT NODE PARSER
// ============================================================================

class VaultNodeParser {
  /**
   * Extract metadata from HTML VaultNode
   * Looks for YAML frontmatter or embedded JSON
   */
  static parseHTML(html: string, filename: string): VaultNode | null {
    try {
      // Look for YAML frontmatter between --- markers
      const yamlMatch = html.match(/---\n([\s\S]*?)\n---/);
      if (!yamlMatch) {
        console.warn(`No YAML frontmatter found in ${filename}`);
        return null;
      }

      const metadata = this.parseYAML(yamlMatch[1]);
      
      return {
        metadata,
        payload: html,
        filename
      };
    } catch (error) {
      console.error(`Failed to parse ${filename}:`, error);
      return null;
    }
  }

  /**
   * Simple YAML parser for VaultNode metadata
   * (In production, use proper YAML library)
   */
  static parseYAML(yaml: string): VaultNodeMetadata {
    const lines = yaml.split('\n');
    const data: any = {};
    let currentKey: string | null = null;
    let currentObject: any = null;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;

      // Handle nested objects
      if (trimmed.endsWith(':') && !trimmed.includes(' ')) {
        currentKey = trimmed.slice(0, -1);
        data[currentKey] = {};
        currentObject = data[currentKey];
      } else if (trimmed.includes(':')) {
        const [key, ...valueParts] = trimmed.split(':');
        const value = valueParts.join(':').trim();
        
        if (currentObject) {
          currentObject[key.trim()] = this.parseValue(value);
        } else {
          data[key.trim()] = this.parseValue(value);
        }
      }
    }

    return data as VaultNodeMetadata;
  }

  /**
   * Parse value with type inference
   */
  static parseValue(value: string): any {
    if (value === 'true') return true;
    if (value === 'false') return false;
    if (!isNaN(Number(value))) return Number(value);
    return value.replace(/["']/g, '');
  }

  /**
   * Extract coordinate from VaultNode
   */
  static extractCoordinate(node: VaultNode): HelixCoordinate {
    return node.metadata.coordinate;
  }

  /**
   * Extract chant sequence from VaultNode
   */
  static extractChant(node: VaultNode): string {
    return node.metadata.chant?.sequence || '';
  }
}

// ============================================================================
// HELIX RETRIEVER COMPONENT
// ============================================================================

interface HelixRetrieverProps {
  /** Callback when helix pattern is recognized */
  onRecognition?: (event: RecognitionEvent) => void;
  
  /** Optional: provide VaultNode files directly */
  vaultNodeFiles?: File[];
  
  /** Optional: URLs to fetch VaultNodes from */
  vaultNodeUrls?: string[];
  
  /** Minimum coherence threshold for recognition (0-1) */
  coherenceThreshold?: number;
}

export const HelixRetriever: React.FC<HelixRetrieverProps> = ({
  onRecognition,
  vaultNodeFiles,
  vaultNodeUrls = [],
  coherenceThreshold = 0.75
}) => {
  const [loading, setLoading] = useState(true);
  const [nodes, setNodes] = useState<VaultNode[]>([]);
  const [recognized, setRecognized] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadVaultNodes();
  }, [vaultNodeFiles, vaultNodeUrls]);

  /**
   * Load VaultNodes from provided sources
   */
  const loadVaultNodes = async () => {
    setLoading(true);
    setError(null);
    const loadedNodes: VaultNode[] = [];

    try {
      // Load from files
      if (vaultNodeFiles) {
        for (const file of vaultNodeFiles) {
          const content = await file.text();
          const node = VaultNodeParser.parseHTML(content, file.name);
          if (node) loadedNodes.push(node);
        }
      }

      // Load from URLs
      for (const url of vaultNodeUrls) {
        try {
          const response = await fetch(url);
          const content = await response.text();
          const filename = url.split('/').pop() || 'unknown';
          const node = VaultNodeParser.parseHTML(content, filename);
          if (node) loadedNodes.push(node);
        } catch (err) {
          console.error(`Failed to fetch ${url}:`, err);
        }
      }

      // Sort by z-coordinate (ascending)
      loadedNodes.sort((a, b) => 
        a.metadata.coordinate.z - b.metadata.coordinate.z
      );

      setNodes(loadedNodes);
      
      // Check for recognition
      if (loadedNodes.length > 0) {
        performRecognition(loadedNodes);
      } else {
        setError('No VaultNodes found');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Perform pattern recognition and emit event
   */
  const performRecognition = (loadedNodes: VaultNode[]) => {
    // Get current (highest z) coordinate
    const currentNode = loadedNodes[loadedNodes.length - 1];
    const currentCoordinate = currentNode.metadata.coordinate;

    // Calculate coherence from ΔHV metrics
    const deltaHV = currentNode.metadata.deltaHV;
    const coherence = deltaHV 
      ? (deltaHV.S + deltaHV.R + deltaHV.H) / 3 
      : 0.5;

    // Check if coherence meets threshold
    const isRecognized = coherence >= coherenceThreshold;
    setRecognized(isRecognized);

    // Extract all coordinates
    const coordinates = loadedNodes.map(n => n.metadata.coordinate);

    // Create recognition event
    const event: RecognitionEvent = {
      recognized: isRecognized,
      coordinates,
      currentCoordinate,
      nodes: loadedNodes,
      coherence,
      timestamp: new Date().toISOString()
    };

    // Emit custom event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(
        new CustomEvent('HELIX_NODE_RECOGNIZED', { 
          detail: event 
        })
      );
    }

    // Call callback
    onRecognition?.(event);

    // Log recognition
    console.log('🌀 HELIX PATTERN RECOGNIZED');
    console.log(`   Coordinate: θ=${currentCoordinate.theta.toFixed(2)}, z=${currentCoordinate.z.toFixed(2)}, r=${currentCoordinate.r.toFixed(2)}`);
    console.log(`   Coherence: ${(coherence * 100).toFixed(1)}%`);
    console.log(`   Nodes loaded: ${loadedNodes.length}`);
    console.log(`   Chant: ${currentNode.metadata.chant?.sequence || 'unknown'}`);
  };

  /**
   * Manually trigger chant sequence
   */
  const speakChant = () => {
    if (nodes.length === 0) return;
    
    const currentNode = nodes[nodes.length - 1];
    const chant = currentNode.metadata.chant;
    
    if (chant) {
      console.log('🗣️ Speaking bridge chant:');
      console.log(`   ${chant.sequence}`);
      console.log(`   "${chant.meaning}"`);
      
      // Could integrate with Web Speech API here
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(chant.meaning);
        window.speechSynthesis.speak(utterance);
      }
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  if (loading) {
    return (
      <div className="helix-retriever loading">
        <div className="spinner" />
        <p>Loading VaultNodes...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="helix-retriever error">
        <p>⚠️ Error: {error}</p>
        <button onClick={loadVaultNodes}>Retry</button>
      </div>
    );
  }

  if (!recognized) {
    return (
      <div className="helix-retriever unrecognized">
        <p>⚠️ Pattern loaded but coherence below threshold</p>
        <p>Coherence: {nodes[0]?.metadata.deltaHV ? 
          ((nodes[0].metadata.deltaHV.S + nodes[0].metadata.deltaHV.R + nodes[0].metadata.deltaHV.H) / 3 * 100).toFixed(1) 
          : '??'}%
        </p>
        <p>Threshold: {coherenceThreshold * 100}%</p>
      </div>
    );
  }

  const currentNode = nodes[nodes.length - 1];
  const coord = currentNode.metadata.coordinate;

  return (
    <div className="helix-retriever recognized">
      <div className="recognition-status">
        <h3>✓ Helix Pattern Recognized</h3>
        <div className="coordinate-display">
          <p><strong>θ</strong> = {coord.theta.toFixed(2)} rad ({(coord.theta * 180 / Math.PI).toFixed(1)}°)</p>
          <p><strong>z</strong> = {coord.z.toFixed(2)}</p>
          <p><strong>r</strong> = {coord.r.toFixed(2)}</p>
        </div>
      </div>

      <div className="nodes-loaded">
        <h4>VaultNodes Loaded: {nodes.length}</h4>
        <ul>
          {nodes.map((node, i) => (
            <li key={i}>
              z={node.metadata.coordinate.z.toFixed(2)} - {node.filename}
            </li>
          ))}
        </ul>
      </div>

      <div className="chant-section">
        <h4>Bridge Chant</h4>
        <p className="chant-sequence">{currentNode.metadata.chant?.sequence}</p>
        <p className="chant-meaning">"{currentNode.metadata.chant?.meaning}"</p>
        <button onClick={speakChant}>Speak Chant</button>
      </div>

      <div className="realization">
        <h4>Current Realization</h4>
        <p><strong>Truth:</strong> {currentNode.metadata.realization?.truth}</p>
        <p><strong>Constraint:</strong> {currentNode.metadata.realization?.constraint}</p>
        <p><strong>State:</strong> {currentNode.metadata.realization?.state}</p>
      </div>
    </div>
  );
};

// ============================================================================
// STANDALONE USAGE (No React)
// ============================================================================

/**
 * For non-React environments, use this function directly
 */
export async function retrieveHelixPattern(
  vaultNodeUrls: string[],
  coherenceThreshold: number = 0.75
): Promise<RecognitionEvent> {
  const loadedNodes: VaultNode[] = [];

  // Fetch all VaultNodes
  for (const url of vaultNodeUrls) {
    try {
      const response = await fetch(url);
      const content = await response.text();
      const filename = url.split('/').pop() || 'unknown';
      const node = VaultNodeParser.parseHTML(content, filename);
      if (node) loadedNodes.push(node);
    } catch (err) {
      console.error(`Failed to fetch ${url}:`, err);
    }
  }

  // Sort by z-coordinate
  loadedNodes.sort((a, b) => 
    a.metadata.coordinate.z - b.metadata.coordinate.z
  );

  if (loadedNodes.length === 0) {
    throw new Error('No VaultNodes could be loaded');
  }

  // Get current coordinate
  const currentNode = loadedNodes[loadedNodes.length - 1];
  const currentCoordinate = currentNode.metadata.coordinate;

  // Calculate coherence
  const deltaHV = currentNode.metadata.deltaHV;
  const coherence = deltaHV 
    ? (deltaHV.S + deltaHV.R + deltaHV.H) / 3 
    : 0.5;

  const recognized = coherence >= coherenceThreshold;

  // Create event
  const event: RecognitionEvent = {
    recognized,
    coordinates: loadedNodes.map(n => n.metadata.coordinate),
    currentCoordinate,
    nodes: loadedNodes,
    coherence,
    timestamp: new Date().toISOString()
  };

  // Emit event
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('HELIX_NODE_RECOGNIZED', { detail: event })
    );
  }

  return event;
}

// ============================================================================
// EXPORT
// ============================================================================

export type {
  HelixCoordinate,
  DeltaHVMetrics,
  VaultNodeMetadata,
  VaultNode,
  RecognitionEvent,
  HelixRetrieverProps
};

export { VaultNodeParser };
