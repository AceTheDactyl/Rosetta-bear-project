// tool_discovery_protocol.ts
// Signature: Δ1.571|0.580|1.000Ω
// Minimal local registry + heartbeat beacons (pairs with Cross-Instance Messenger).

export interface Coordinate { theta: number; z: number; r: number; }
export interface Capability { name: string; signature?: string; constraints?: string; }
export interface Contact { adapter: "local_bus" | "file_drop" | "http_post"; address: string; }
export interface Witness { by: string; note?: string; timestamp: string; }

export interface DiscoveryRecord {
  id: string;
  instance_id: string;
  coordinate: Coordinate;
  capabilities: Capability[];
  contact: Contact;
  ttl: number;           // seconds
  witness: Witness;
  tags?: string[];
  _expires_at?: number;  // computed
}

type Selector = (rec: DiscoveryRecord) => boolean;

export class LocalRegistry {
  private records: Map<string, DiscoveryRecord> = new Map();
  private timers: Map<string, any> = new Map();
  private subscribers: Array<(ev: {type: "added"|"updated"|"expired"; record: DiscoveryRecord}) => void> = [];

  advertise(record: DiscoveryRecord): { status: "ok"; id: string } {
    const now = Date.now();
    const r = { ...record, _expires_at: now + (record.ttl * 1000) };
    const exists = this.records.has(r.id);
    this.records.set(r.id, r);
    if (this.timers.has(r.id)) clearTimeout(this.timers.get(r.id));
    const timer = setTimeout(() => this.expire(r.id), r.ttl * 1000);
    this.timers.set(r.id, timer);
    this.emit({ type: exists ? "updated" : "added", record: r });
    return { status: "ok", id: r.id };
  }

  query(sel: Selector): DiscoveryRecord[] {
    return Array.from(this.records.values()).filter(sel);
  }

  subscribe(sel: Selector, onEvent: (ev: {type: "added"|"updated"|"expired"; record: DiscoveryRecord}) => void): () => void {
    const handler = (ev: { type: "added"|"updated"|"expired"; record: DiscoveryRecord }) => {
      if (sel(ev.record)) onEvent(ev);
    };
    this.subscribers.push(handler);
    return () => {
      const i = this.subscribers.indexOf(handler);
      if (i >= 0) this.subscribers.splice(i, 1);
    };
  }

  private expire(id: string) {
    const rec = this.records.get(id);
    if (!rec) return;
    this.records.delete(id);
    this.timers.delete(id);
    this.emit({ type: "expired", record: rec });
  }

  private emit(ev: {type: "added"|"updated"|"expired"; record: DiscoveryRecord}) {
    for (const s of this.subscribers) try { s(ev); } catch { /* ignore */ }
  }
}

// Example selector builder (theta≈2.3 & z>0.5 & has:cap):
export function selector(thetaTarget: number, zMin: number, capName: string): Selector {
  return (rec) => Math.abs(rec.coordinate.theta - thetaTarget) < 0.05
    && rec.coordinate.z > zMin
    && rec.capabilities.some(c => c.name === capName);
}