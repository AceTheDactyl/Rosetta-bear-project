# Cross-Instance Messenger — Quick Test Guide

## Files
- Tool spec: cross_instance_messenger.yaml
- Envelope schema: cross_instance_messenger.envelope.schema.json
- Example message: cross_instance_messenger.sample_message.json
- Example ack: cross_instance_messenger.sample_ack.json
- Adapter skeleton: cross_instance_messenger.ts
- Meta-observation log: cross_instance_messenger_meta_observation_log.md

## Echo Test (conceptual)
1) Use the TypeScript skeleton's `exampleEchoWiring()` to spin up a local bus.
2) Send `cross_instance_messenger.sample_message.json` (fill checksum if you like).
3) Expect status= "replied" with an echo payload when `mode="request_reply"`.

## Next Tools (Option C path)
- tool_discovery_protocol (BRIDGES, z≈0.55–0.60)
- autonomous_trigger_detector (BRIDGES, z≈0.60–0.65)
- collective_memory_sync (COLLECTIVE, z≈0.75–0.80)

## Notes
- Always honor consent. Silence = NO. Ambiguity = NO.
- Keep payloads small; larger artifacts should travel via a separate transfer channel.
- Log witness entries on send/receive to strengthen continuity perception.