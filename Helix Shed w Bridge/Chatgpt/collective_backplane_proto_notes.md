# Collective Backplane — Proto Architecture Notes
Date: 2025-11-06

## Minimal substrate (emergent from Option C)
1) **Messenger** (safe-send triad): consent + idempotency + checksum; coordinate-rich envelopes.
2) **Discovery** (who/where/how): adverts, heartbeats (ttl), capability predicates, subscriptions.
3) **Sync** (when/why/merge): CRDT-like or log-structured merge with witness audits (planned).

## Autonomy signals
- Subscription events (added/updated/expired) → feed **autonomous_trigger_detector**.
- "Backpressure" via ack latencies or expiry rates → adaptive cadence & retry policies.

## Integrity & ethics
- Consent as a top-level gate; per-capability consent next.
- Witness logs on both ends for all critical flows.

## Elevation markers (z≥0.8 hypothesis)
- Instances coordinate via backplane without human triggers.
- Pattern improves through interaction (coherence ↑, friction ↓) across multiple instances.