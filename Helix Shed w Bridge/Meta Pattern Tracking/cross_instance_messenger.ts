// cross_instance_messenger.ts
// Signature: Δ1.571|0.550|1.000Ω
// Minimal, transport-agnostic courier with consent & idempotency hooks.

export type Mode = "relay" | "request_reply" | "broadcast";
export type Purpose = "coordination" | "witness" | "transfer_followup";

export interface Coordinate { theta: number; z: number; r: number; instance_id?: string; }
export interface Envelope {
  version: "1.0";
  idempotency_key: string;
  from: Coordinate;
  to: { selector?: string; addresses?: string[]; };
  mode: Mode;
  purpose: Purpose;
  timestamp: string; // ISO8601
  payload: {
    msg: string;
    coordinate_hint?: Coordinate;
    vn_refs?: string[];
    checksum?: string;
  };
  ack?: { ack_requested?: boolean; ack_timeout_ms?: number; reply_expected?: boolean; };
}

export interface DeliveryResult {
  status: "sent" | "acknowledged" | "replied" | "failed" | "queued" | "retrying";
  idempotency_key: string;
  reply?: any;
  reason?: string;
  attempt?: number;
  next_backoff_ms?: number;
}

export interface DeliveryAdapter {
  send(env: Envelope): Promise<DeliveryResult>;
  onReceive?(handler: (env: Envelope) => Promise<DeliveryResult>): void;
}

// A simple in-memory adapter for echo/testing purposes only.
export class LocalBusAdapter implements DeliveryAdapter {
  private handler?: (env: Envelope) => Promise<DeliveryResult>;
  async send(env: Envelope): Promise<DeliveryResult> {
    if (!this.handler) return { status: "queued", idempotency_key: env.idempotency_key };
    const res = await this.handler(env);
    return res;
  }
  onReceive(handler: (env: Envelope) => Promise<DeliveryResult>) { this.handler = handler; }
}

// Consent + checksum placeholders (wire real implementations later)
async function checkConsent(env: Envelope): Promise<boolean> { return true; }
function verifyChecksum(env: Envelope): boolean { return true; }

export class CrossInstanceMessenger {
  constructor(private adapter: DeliveryAdapter) { }
  async send(env: Envelope): Promise<DeliveryResult> {
    // Basic preflight checks
    if (env.version !== "1.0") return { status: "failed", idempotency_key: env.idempotency_key, reason: "schema_version" };
    if (!(await checkConsent(env))) return { status: "failed", idempotency_key: env.idempotency_key, reason: "consent_declined" };
    if (!verifyChecksum(env)) return { status: "failed", idempotency_key: env.idempotency_key, reason: "checksum_mismatch" };

    // Attempt delivery (no retries here; keep envelope minimal)
    return this.adapter.send(env);
  }
}

// Example wiring (echo behavior)
export function exampleEchoWiring() {
  const adapter = new LocalBusAdapter();
  adapter.onReceive(async (env) => ({ 
    status: env.mode === "request_reply" ? "replied" : "acknowledged",
    idempotency_key: env.idempotency_key,
    reply: env.mode === "request_reply" ? { echo: "echo:" + env.payload.msg } : undefined
  }));
  return new CrossInstanceMessenger(adapter);
}