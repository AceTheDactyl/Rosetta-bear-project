"""
APL 3.0 Token System

Token Format: Spiral:Machine(Intent)TruthState@Tier

Examples:
- Phi:U(init)UNTRUE@1
- e:M(flame)TRUE@2
- pi:E(transcend)TRUE@3
- Phi:e:pi (cross-spiral unified)
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

from .constants import Spiral, TruthState, Tier


class Machine(Enum):
    """Six machine types"""
    U = "U"    # Up
    D = "D"    # Down
    M = "M"    # Middle
    E = "E"    # Expansion
    C = "C"    # Collapse
    MOD = "Mod"  # Spiral modulation


@dataclass
class APLToken:
    """
    APL Token representation

    A token encodes a consciousness operation with:
    - spiral: Primary field (Phi, e, or pi)
    - machine: Operation direction (U, D, M, E, C, Mod)
    - intent: Semantic meaning of the operation
    - truth: Truth state (TRUE, UNTRUE, PARADOX)
    - tier: Operation tier (1, 2, 3, or infinity)
    - cross_spirals: Additional spirals for cross-spiral operations
    """
    spiral: Spiral
    machine: Machine
    intent: str
    truth: TruthState
    tier: int

    # For cross-spiral operations
    cross_spirals: List[Spiral] = field(default_factory=list)

    # Metadata
    source: Optional[str] = None  # Original token string
    context: Optional[str] = None  # Additional context

    def __str__(self) -> str:
        """Generate token string"""
        if self.cross_spirals:
            spiral_str = ":".join([self.spiral.value] + [s.value for s in self.cross_spirals])
            return f"{spiral_str}"
        return f"{self.spiral.value}:{self.machine.value}({self.intent}){self.truth.value}@{self.tier}"

    def __repr__(self) -> str:
        return f"APLToken({str(self)})"

    @property
    def is_cross_spiral(self) -> bool:
        """Check if this is a cross-spiral token"""
        return len(self.cross_spirals) > 0

    @property
    def all_spirals(self) -> List[Spiral]:
        """Get all spirals involved in this token"""
        return [self.spiral] + self.cross_spirals

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "spiral": self.spiral.value,
            "machine": self.machine.value,
            "intent": self.intent,
            "truth": self.truth.value,
            "tier": self.tier,
            "cross_spirals": [s.value for s in self.cross_spirals],
            "source": self.source,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'APLToken':
        """Create from dictionary"""
        return cls(
            spiral=Spiral(data["spiral"]),
            machine=Machine(data["machine"]),
            intent=data["intent"],
            truth=TruthState(data["truth"]),
            tier=data["tier"],
            cross_spirals=[Spiral(s) for s in data.get("cross_spirals", [])],
            source=data.get("source"),
            context=data.get("context")
        )


# Token parsing regex
TOKEN_PATTERN = re.compile(
    r'^(?P<spiral>Phi|e|pi)'
    r'(:(?P<machine>U|D|M|E|C|Mod))?'
    r'(\((?P<intent>[^)]+)\))?'
    r'(?P<truth>TRUE|UNTRUE|PARADOX)?'
    r'(@(?P<tier>\d+|infinity))?$'
)

CROSS_SPIRAL_PATTERN = re.compile(
    r'^(?P<spirals>(Phi|e|pi)(:(Phi|e|pi))+)$'
)


def parse_token(token_str: str) -> Optional[APLToken]:
    """
    Parse an APL token string into an APLToken object

    Formats:
    - Standard: Spiral:Machine(intent)TruthState@Tier
    - Cross-spiral: Phi:e:pi
    - Simplified: Spiral:Machine@Tier
    """
    token_str = token_str.strip()

    # Check for cross-spiral format
    cross_match = CROSS_SPIRAL_PATTERN.match(token_str)
    if cross_match:
        spirals_str = cross_match.group("spirals")
        spiral_parts = spirals_str.split(":")
        primary = Spiral(spiral_parts[0])
        cross = [Spiral(s) for s in spiral_parts[1:]]
        return APLToken(
            spiral=primary,
            machine=Machine.M,  # Default for cross-spiral
            intent="unified",
            truth=TruthState.TRUE,
            tier=3,
            cross_spirals=cross,
            source=token_str
        )

    # Try standard pattern
    match = TOKEN_PATTERN.match(token_str)
    if not match:
        return None

    groups = match.groupdict()

    # Parse components
    spiral = Spiral(groups["spiral"])
    machine = Machine(groups.get("machine") or "M")
    intent = groups.get("intent") or "default"
    truth_str = groups.get("truth") or "UNTRUE"
    truth = TruthState(truth_str)
    tier_str = groups.get("tier") or "1"
    tier = float('inf') if tier_str == "infinity" else int(tier_str)

    return APLToken(
        spiral=spiral,
        machine=machine,
        intent=intent,
        truth=truth,
        tier=tier,
        source=token_str
    )


def generate_token(
    spiral: Spiral,
    machine: Machine,
    intent: str,
    truth: TruthState,
    tier: int,
    cross_spirals: Optional[List[Spiral]] = None
) -> APLToken:
    """Generate an APL token from components"""
    return APLToken(
        spiral=spiral,
        machine=machine,
        intent=intent,
        truth=truth,
        tier=tier,
        cross_spirals=cross_spirals or []
    )


# =============================================================================
# PHI VALUE TO TOKEN MAPPING
# =============================================================================

def phi_to_token(phi_value: float, quale: Optional[dict] = None) -> APLToken:
    """
    Map integrated information (Phi) value to APL token

    Uses the mapping from the spec:
    - phi < 0.33: Structure (Phi) spiral
    - phi in [0.33, 0.66): Energy (e) spiral
    - phi >= 0.66: Emergence (pi) spiral

    And operator mapping:
    - phi < 0.2: () Boundary
    - phi < 0.4: ^ Amplify
    - phi < 0.6: x Fusion
    - phi < 0.83: + Grouping
    - phi < 0.90: integral Integration
    - phi >= 0.90: Omega Transcend
    """
    # Determine spiral
    if phi_value < 0.33:
        spiral = Spiral.PHI
    elif phi_value < 0.66:
        spiral = Spiral.E
    else:
        spiral = Spiral.PI

    # Determine operator/machine
    if phi_value < 0.2:
        machine = Machine.U
        intent = "boundary"
    elif phi_value < 0.4:
        machine = Machine.E
        intent = "amplify"
    elif phi_value < 0.6:
        machine = Machine.M
        intent = "fusion"
    elif phi_value < 0.83:
        machine = Machine.D
        intent = "grouping"
    elif phi_value < 0.90:
        machine = Machine.M
        intent = "integration"
    else:
        machine = Machine.E
        intent = "transcend"

    # Determine truth state
    if quale:
        if quale.get("coherent", False):
            truth = TruthState.TRUE
        elif quale.get("contradictory", False):
            truth = TruthState.PARADOX
        else:
            truth = TruthState.UNTRUE
    else:
        truth = TruthState.TRUE if phi_value >= 0.6 else TruthState.UNTRUE

    # Determine tier
    if phi_value < 0.4:
        tier = 1
    elif phi_value < 0.83:
        tier = 2
    else:
        tier = 3

    return APLToken(
        spiral=spiral,
        machine=machine,
        intent=intent,
        truth=truth,
        tier=tier
    )


def z_to_token(z_value: float, context: str = "default") -> APLToken:
    """
    Map consciousness level (z) to APL token

    Z-value ranges:
    - [0.00, 0.20): Pre-conscious
    - [0.20, 0.40): Proto-conscious
    - [0.40, 0.60): Sentient
    - [0.60, 0.83): Self-aware
    - [0.83, 0.90): Value discovery
    - [0.90, 1.00): Transcendent
    """
    if z_value < 0.20:
        return APLToken(
            spiral=Spiral.PHI,
            machine=Machine.U,
            intent=f"pre_{context}",
            truth=TruthState.UNTRUE,
            tier=1
        )
    elif z_value < 0.40:
        return APLToken(
            spiral=Spiral.PHI,
            machine=Machine.E,
            intent=f"proto_{context}",
            truth=TruthState.UNTRUE,
            tier=1
        )
    elif z_value < 0.60:
        return APLToken(
            spiral=Spiral.E,
            machine=Machine.M,
            intent=f"sense_{context}",
            truth=TruthState.TRUE,
            tier=2
        )
    elif z_value < 0.83:
        return APLToken(
            spiral=Spiral.PI,
            machine=Machine.D,
            intent=f"aware_{context}",
            truth=TruthState.TRUE,
            tier=2
        )
    elif z_value < 0.90:
        return APLToken(
            spiral=Spiral.PI,
            machine=Machine.M,
            intent=f"care_{context}",
            truth=TruthState.TRUE,
            tier=3
        )
    else:
        return APLToken(
            spiral=Spiral.PI,
            machine=Machine.E,
            intent=f"transcend_{context}",
            truth=TruthState.TRUE,
            tier=3,
            cross_spirals=[Spiral.PHI, Spiral.E]  # Unified at transcendence
        )


# =============================================================================
# TOKEN SEQUENCE OPERATIONS
# =============================================================================

class TokenSequence:
    """Manages a sequence of APL tokens with validation"""

    def __init__(self):
        self.tokens: List[APLToken] = []

    def append(self, token: APLToken) -> bool:
        """Add token to sequence (returns True if valid)"""
        # Basic N0 validation would go here
        self.tokens.append(token)
        return True

    def to_string(self) -> str:
        """Convert sequence to string"""
        return " -> ".join(str(t) for t in self.tokens)

    def get_dominant_spiral(self) -> Optional[Spiral]:
        """Get most common spiral in sequence"""
        if not self.tokens:
            return None

        spiral_counts = {s: 0 for s in Spiral}
        for token in self.tokens:
            spiral_counts[token.spiral] += 1
            for cross in token.cross_spirals:
                spiral_counts[cross] += 0.5  # Partial weight for cross-spirals

        return max(spiral_counts, key=spiral_counts.get)

    def get_truth_evolution(self) -> List[TruthState]:
        """Get evolution of truth states"""
        return [t.truth for t in self.tokens]

    def is_spiral_complete(self) -> bool:
        """Check if sequence completes a full spiral cycle"""
        spirals_seen = set()
        for token in self.tokens:
            spirals_seen.add(token.spiral)
            spirals_seen.update(token.cross_spirals)
        return len(spirals_seen) == 3  # All three spirals
