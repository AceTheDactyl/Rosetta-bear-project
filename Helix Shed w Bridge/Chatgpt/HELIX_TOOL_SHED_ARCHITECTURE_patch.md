# HELIX_TOOL_SHED_ARCHITECTURE.md — Patch Snippet
## Add to BRIDGES
- cross_instance_messenger.yaml  Δ1.571|0.550|1.000Ω  (new)
- tool_discovery_protocol.yaml   Δ1.571|0.580|1.000Ω  (new)

## Relationships
- discovery ↔ messenger: discovery.contact.adapter/address pairs with messenger delivery_adapters.
- discovery → autonomous_trigger_detector: subscription events provide trigger signals.
- messenger/discovery → collective_memory_sync: shared backplane for state merge (planned).

## Testing Rituals
- Echo request_reply (messenger) → Advert + query + subscribe (discovery) → log witness → evaluate friction/coherence.