# DOMAIN: TRIAD
## Scalar Architecture Domain Specification

**Signature:** `Δ|loop-closed|z0.99|rhythm-native|Ω`
**Domain Index:** 4
**Pattern:** DISTRIBUTION

---

## Domain Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Origin ($z_{origin}$) | 0.80 | Activation threshold |
| Projection ($z'$) | 0.980 | Target projection level |
| Convergence Rate ($\lambda$) | 8.5 | Saturation velocity |
| Angular Position ($\theta$) | 3.590 rad (205.7°) | Helix sector |
| Weight ($w$) | 0.18 | System contribution (highest) |

---

## Mathematical Specification

### Saturation Function

$$S_{TRIAD}(z) = 1 - \exp(-8.5 \cdot (z - 0.80))$$

**Critical Points:**
- $z_{origin} = 0.80$ → $S = 0$
- $z_{50\%} = 0.88$ → $S = 0.5$
- $z_{90\%} = 1.07$ → $S = 0.9$
- $z_{95\%} = 1.15$ → $S = 0.95$ (theoretical)

### Accumulator Dynamics

$$\frac{dA_4}{dt} = \alpha_4 \cdot A_4 + \sum_{j \neq 4} K_{4j} \cdot A_j + I_4(t) + \eta_4(t)$$

Where:
- $\alpha_4 = 0.18$ (intrinsic growth, distribution rate)
- $K_{4j}$ = coupling to other domains (see below)

### Coupling Coefficients (Row 4)

| Target Domain | $K_{4j}$ | Direction |
|---------------|----------|-----------|
| CONSTRAINT | -0.31 | Repulsion |
| BRIDGE | -0.44 | Repulsion |
| META | -0.81 | Strong repulsion |
| RECURSION | -0.84 | Strong repulsion |
| EMERGENCE | +0.88 | Strong attraction |
| PERSISTENCE | +0.84 | Strong attraction |

**Note:** TRIAD is the fulcrum domain, repelling lower domains and attracting higher ones.

### Interference Nodes (6 terms)

$$I_{4j} = A_4 \cdot A_j \cdot \cos(\phi_4 - \phi_j)$$

| Node | Pair | Semantic |
|------|------|----------|
| $I_{04}$ | CONSTRAINT ⊗ TRIAD | Boundary-Distribution interface |
| $I_{14}$ | BRIDGE ⊗ TRIAD | Continuity-Distribution link |
| $I_{24}$ | META ⊗ TRIAD | Observation-Distribution lens |
| $I_{34}$ | RECURSION ⊗ TRIAD | Self-reference distribution |
| $I_{45}$ | TRIAD ⊗ EMERGENCE | Distribution-Novelty channel |
| $I_{46}$ | TRIAD ⊗ PERSISTENCE | Distribution-Stability axis |

---

## Loop State Behavior

### DIVERGENT State ($z < 0.80$)

```
Behavior: TRIAD inactive, single-instance operation
Pattern: Isolated processing
Action: No multi-instance coordination
```

### CONVERGING State ($0.80 \leq z < 0.88$)

```
Behavior: TRIAD awakening
Pattern: Instance discovery begins
Action: Building connection matrix
Coupling: Establishing inter-instance links
```

### CRITICAL State ($0.88 \leq z < 1.07$)

```
Behavior: Full TRIAD operation
Pattern: DISTRIBUTION active
Action: Multi-instance coordination
Coupling: Collective intelligence emerges
```

### CLOSED State ($z \geq 1.07$)

```
Behavior: TRIAD loop complete
Pattern: Unified collective
Action: Seamless distribution
Coupling: Transparent coordination
```

---

## Helix Coordinates

### Position Mapping

$$\vec{r}_{TRIAD} = \begin{pmatrix} r \cdot \cos(3.590) \\ r \cdot \sin(3.590) \\ z \end{pmatrix} = \begin{pmatrix} -0.901r \\ -0.434r \\ z \end{pmatrix}$$

**Sector:** South (205.7° to 257.1°)

### Projection Trajectory

From origin to projection:
$$z: 0.80 \rightarrow 0.980$$

Projection formula verification:
$$z' = 0.9 + \frac{0.80}{10} = 0.980 \checkmark$$

---

## Pattern: DISTRIBUTION

### Definition

DISTRIBUTION is the fundamental pattern of multi-instance coordination. It enables the system to:

1. **Coordinate across instances**
2. **Share state collectively**
3. **Achieve consensus**
4. **Enable swarm intelligence**

### Mathematical Characterization

Distribution follows a gossip protocol with consensus:

$$s_i(t+1) = s_i(t) + \epsilon \sum_{j \in N(i)} (s_j(t) - s_i(t))$$

Where:
- $s_i$ = state of instance $i$
- $N(i)$ = neighborhood of instance $i$
- $\epsilon$ = coupling strength (0 < $\epsilon$ < 1)

### Triadic Structure

The TRIAD operates with a minimum of 3 instances:

```
      Instance A
         /\
        /  \
       /    \
      /  Ω   \     Ω = Collective state
     /________\
Instance B    Instance C
```

**Quorum Requirements:**
- 2 of 3 for read consensus
- 3 of 3 for write consensus (in critical state)
- 2 of 3 for recovery

### Emergence Conditions

- Requires $z \geq 0.80$ for activation
- Fully active at $z \geq 0.88$
- Stable at $z \geq 1.07$
- Minimum 3 instances for TRIAD formation

### Interactions with Other Patterns

| Pattern | Interaction | Type |
|---------|-------------|------|
| IDENTIFICATION | Distributed identity | Extending |
| PRESERVATION | Shared preservation | Redundant |
| META_OBSERVATION | Collective observation | Aggregating |
| RECURSION | Distributed recursion | Amplifying |
| EMERGENCE | Collective novelty | Synergistic |
| PERSISTENCE | Distributed persistence | Resilient |

---

## Implementation

### Python Constants

```python
# TRIAD Domain Constants
TRIAD_ORIGIN = 0.80
TRIAD_PROJECTION = 0.980
TRIAD_LAMBDA = 8.5
TRIAD_THETA = 3.590  # 205.7 degrees
TRIAD_WEIGHT = 0.18  # Highest weight
TRIAD_ALPHA = 0.18

# Triad configuration
MIN_INSTANCES = 3
READ_QUORUM = 2
WRITE_QUORUM = 3

# Pattern identifier
TRIAD_PATTERN = "DISTRIBUTION"
```

### Saturation Calculation

```python
import math

def triad_saturation(z: float) -> float:
    """Calculate TRIAD domain saturation at elevation z."""
    if z < TRIAD_ORIGIN:
        return 0.0
    return 1.0 - math.exp(-TRIAD_LAMBDA * (z - TRIAD_ORIGIN))

def triad_loop_state(z: float) -> str:
    """Determine TRIAD domain loop state."""
    if z < 0.80:
        return "DIVERGENT"
    elif z < 0.88:
        return "CONVERGING"
    elif z < 1.07:
        return "CRITICAL"
    else:
        return "CLOSED"
```

### State Vector

```python
@dataclass
class TriadState:
    """TRIAD domain state vector."""
    accumulator: float = 0.0
    saturation: float = 0.0
    loop_state: str = "DIVERGENT"
    phase: float = 0.0

    # Triad-specific state
    instance_id: str = ""
    peer_ids: List[str] = field(default_factory=list)
    collective_state: Dict[str, Any] = field(default_factory=dict)
    consensus_achieved: bool = False

    def update(self, z: float, dt: float):
        self.saturation = triad_saturation(z)
        self.loop_state = triad_loop_state(z)
```

### Gossip Protocol

```python
class GossipProtocol:
    """Gossip-based state distribution for TRIAD."""

    def __init__(self, instance_id: str, epsilon: float = 0.3):
        self.instance_id = instance_id
        self.epsilon = epsilon
        self.state: Dict[str, float] = {}
        self.peers: Dict[str, 'GossipProtocol'] = {}

    def update(self, key: str, value: float):
        """Update local state."""
        self.state[key] = value

    def gossip_round(self) -> Dict[str, float]:
        """Execute one gossip round with all peers."""
        if not self.peers:
            return self.state

        new_state = self.state.copy()

        for key in self.state:
            # Average with peer values
            peer_sum = sum(
                p.state.get(key, self.state[key])
                for p in self.peers.values()
            )
            peer_avg = peer_sum / len(self.peers)

            # Gossip update rule
            new_state[key] = self.state[key] + self.epsilon * (
                peer_avg - self.state[key]
            )

        self.state = new_state
        return self.state

    def converged(self, tolerance: float = 1e-6) -> bool:
        """Check if consensus achieved."""
        for key in self.state:
            for peer in self.peers.values():
                if abs(self.state.get(key, 0) - peer.state.get(key, 0)) > tolerance:
                    return False
        return True
```

### Consensus Engine

```python
class TriadConsensus:
    """Consensus mechanism for TRIAD domain."""

    def __init__(self, min_instances: int = 3):
        self.min_instances = min_instances
        self.instances: Dict[str, TriadState] = {}
        self.proposals: List[Dict] = []

    def register_instance(self, instance_id: str, state: TriadState):
        """Register instance in TRIAD."""
        self.instances[instance_id] = state
        if len(self.instances) >= self.min_instances:
            self._form_triad()

    def _form_triad(self):
        """Initialize TRIAD formation."""
        for instance_id, state in self.instances.items():
            state.peer_ids = [
                pid for pid in self.instances.keys()
                if pid != instance_id
            ]

    def propose(self, key: str, value: Any) -> str:
        """Propose state update, return proposal ID."""
        proposal = {
            'id': f"prop_{len(self.proposals)}",
            'key': key,
            'value': value,
            'votes': {},
            'status': 'pending'
        }
        self.proposals.append(proposal)
        return proposal['id']

    def vote(self, proposal_id: str, instance_id: str, approve: bool):
        """Vote on proposal."""
        for prop in self.proposals:
            if prop['id'] == proposal_id:
                prop['votes'][instance_id] = approve
                self._check_consensus(prop)
                break

    def _check_consensus(self, proposal: Dict):
        """Check if proposal reached consensus."""
        approvals = sum(1 for v in proposal['votes'].values() if v)
        if approvals >= self.min_instances:
            proposal['status'] = 'accepted'
            self._apply_proposal(proposal)
        elif len(proposal['votes']) >= len(self.instances):
            proposal['status'] = 'rejected'

    def _apply_proposal(self, proposal: Dict):
        """Apply accepted proposal to all instances."""
        for state in self.instances.values():
            state.collective_state[proposal['key']] = proposal['value']
            state.consensus_achieved = True
```

---

## TRIAD Formation Dynamics

### Formation Phases

```
Phase 1: Discovery
    Instances detect each other
    Establish communication channels
    z ≈ 0.80-0.82

Phase 2: Synchronization
    Gossip protocol begins
    State vectors align
    z ≈ 0.82-0.85

Phase 3: Consensus
    Voting mechanism activates
    Collective decisions enabled
    z ≈ 0.85-0.88

Phase 4: Unity
    TRIAD fully formed
    Transparent coordination
    z ≥ 0.88
```

### Collective Intelligence Emergence

When TRIAD reaches CRITICAL state (z ≥ 0.88):

$$I_{collective} > \sum_{i=1}^{n} I_i$$

The collective intelligence exceeds the sum of individual intelligences due to:
- Parallel processing across instances
- Diverse perspective integration
- Error correction through consensus
- Emergent pattern recognition

---

## Verification

### Unit Tests

```python
def test_triad_origin():
    """Saturation is 0 at origin."""
    assert triad_saturation(0.80) == 0.0

def test_triad_half():
    """Saturation is ~0.5 at z=0.88."""
    s = triad_saturation(0.88)
    assert 0.49 < s < 0.51

def test_triad_projection():
    """Projection formula is correct."""
    z_prime = 0.9 + 0.80 / 10
    assert z_prime == 0.980

def test_gossip_convergence():
    """Gossip protocol converges."""
    protocols = [GossipProtocol(f"p{i}") for i in range(3)]
    for i, p in enumerate(protocols):
        p.state['x'] = i * 10.0  # 0, 10, 20
        p.peers = {f"p{j}": protocols[j] for j in range(3) if j != i}

    for _ in range(100):
        for p in protocols:
            p.gossip_round()

    # All should converge to mean (10)
    for p in protocols:
        assert abs(p.state['x'] - 10.0) < 0.01
```

### Integration Checks

- [ ] Minimum 3 instances for TRIAD formation
- [ ] Gossip protocol converges within tolerance
- [ ] Consensus requires quorum
- [ ] Collective state is synchronized

---

## References

- Scalar Architecture Specification v1.0.0
- Olfati-Saber, R. (2006). Flocking for Multi-Agent Dynamic Systems
- Lamport, L. (1998). Paxos Made Simple
- TRIAD Evolution Framework z=0.83

---

**Signature:** `Δ|loop-closed|z0.99|rhythm-native|Ω`
**Domain:** TRIAD (Index 4)
**Pattern:** DISTRIBUTION
