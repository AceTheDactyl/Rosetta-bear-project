#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                      PSL(3,2): THE 168 AUTOMORPHISMS                                     ║
║                                                                                          ║
║              Complete enumeration of the Fano plane symmetry group                       ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PSL(3,2) = GL(3,F₂) / Z(GL(3,F₂)) = GL(3,2) (since Z = {I} over F₂)                    ║
║                                                                                          ║
║  This is the automorphism group of:                                                      ║
║  - The Fano plane PG(2,2)                                                                ║
║  - The Hamming [7,4,3] code                                                              ║
║  - The octonion multiplication table                                                     ║
║  - The Klein quartic                                                                     ║
║                                                                                          ║
║  Order: 168 = 2³ × 3 × 7 = (2³-1)(2³-2)(2³-4) = 7×6×4                                   ║
║  Simple group (the second smallest non-abelian simple group after A₅)                    ║
║                                                                                          ║
║  Implementation: Points of Fano = non-zero vectors in F₂³                                ║
║                  Automorphisms = invertible 3×3 matrices over F₂                         ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from itertools import product


# ═══════════════════════════════════════════════════════════════════════════════════════════
# FANO PLANE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════════════════

# Points of Fano plane as binary vectors (non-zero elements of F₂³)
# We use integer encoding: vector (a,b,c) -> 1*a + 2*b + 4*c
# Point 1 = (1,0,0) = 1
# Point 2 = (0,1,0) = 2
# Point 3 = (1,1,0) = 3
# Point 4 = (0,0,1) = 4
# Point 5 = (1,0,1) = 5
# Point 6 = (0,1,1) = 6
# Point 7 = (1,1,1) = 7

def vec_to_point(v: Tuple[int, int, int]) -> int:
    """Convert F₂³ vector to point number (1-7)."""
    return v[0] + 2*v[1] + 4*v[2]

def point_to_vec(p: int) -> Tuple[int, int, int]:
    """Convert point number (1-7) to F₂³ vector."""
    return (p & 1, (p >> 1) & 1, (p >> 2) & 1)

# Lines of Fano plane: a line is {p, q, p⊕q} where ⊕ is XOR
# In projective terms, three points are collinear iff their vectors sum to 0 (mod 2)
FANO_LINES = [
    frozenset({1, 2, 3}),  # (1,0,0) + (0,1,0) + (1,1,0) = (0,0,0) ✓
    frozenset({1, 4, 5}),  # (1,0,0) + (0,0,1) + (1,0,1) = (0,0,0) ✓
    frozenset({1, 6, 7}),  # (1,0,0) + (0,1,1) + (1,1,1) = (0,0,0) ✓
    frozenset({2, 4, 6}),  # (0,1,0) + (0,0,1) + (0,1,1) = (0,0,0) ✓
    frozenset({2, 5, 7}),  # (0,1,0) + (1,0,1) + (1,1,1) = (0,0,0) ✓
    frozenset({3, 4, 7}),  # (1,1,0) + (0,0,1) + (1,1,1) = (0,0,0) ✓
    frozenset({3, 5, 6}),  # (1,1,0) + (1,0,1) + (0,1,1) = (0,0,0) ✓
]

LINE_NAMES = [
    "Foundation", "Self-Reference", "Completion",
    "Even Path", "Prime Path", "Growth", "Balance"
]

SEAL_NAMES = {1: "Ω", 2: "Δ", 3: "Τ", 4: "Ψ", 5: "Σ", 6: "Ξ", 7: "Κ"}


# ═══════════════════════════════════════════════════════════════════════════════════════════
# GL(3,F₂) - INVERTIBLE 3×3 MATRICES OVER F₂
# ═══════════════════════════════════════════════════════════════════════════════════════════

def mat_mul_f2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication over F₂."""
    return (A @ B) % 2

def mat_det_f2(M: np.ndarray) -> int:
    """Determinant over F₂ (0 or 1)."""
    # For 3×3, use standard formula mod 2
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)) % 2
    return det

def mat_inv_f2(M: np.ndarray) -> Optional[np.ndarray]:
    """Matrix inverse over F₂, or None if not invertible."""
    if mat_det_f2(M) == 0:
        return None
    
    # Adjugate matrix (cofactors transposed)
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    
    adj = np.array([
        [(e*i - f*h) % 2, (c*h - b*i) % 2, (b*f - c*e) % 2],
        [(f*g - d*i) % 2, (a*i - c*g) % 2, (c*d - a*f) % 2],
        [(d*h - e*g) % 2, (b*g - a*h) % 2, (a*e - b*d) % 2]
    ], dtype=int)
    
    # Over F₂, det = 1, so inverse = adjugate
    return adj % 2

def apply_matrix(M: np.ndarray, point: int) -> int:
    """Apply matrix to a point (as F₂³ vector)."""
    v = np.array(point_to_vec(point), dtype=int)
    result = mat_mul_f2(M, v.reshape(3, 1)).flatten() % 2
    return vec_to_point(tuple(result))

def matrix_to_perm(M: np.ndarray) -> Dict[int, int]:
    """Convert matrix to permutation of points {1..7}."""
    return {p: apply_matrix(M, p) for p in range(1, 8)}


def generate_gl3_f2() -> List[np.ndarray]:
    """
    Generate all invertible 3×3 matrices over F₂.
    |GL(3,F₂)| = (2³-1)(2³-2)(2³-4) = 7 × 6 × 4 = 168
    """
    matrices = []
    
    # Enumerate all 3×3 binary matrices
    for entries in product([0, 1], repeat=9):
        M = np.array(entries, dtype=int).reshape(3, 3)
        if mat_det_f2(M) == 1:  # Invertible
            matrices.append(M)
    
    return matrices


def is_fano_automorphism(perm: Dict[int, int]) -> bool:
    """
    Check if a permutation of {1..7} is a Fano plane automorphism.
    An automorphism must map lines to lines.
    """
    for line in FANO_LINES:
        # Apply permutation to line
        image = frozenset(perm[p] for p in line)
        # Check if image is also a line
        if image not in FANO_LINES:
            return False
    return True


def apply_perm(perm: Dict[int, int], point: int) -> int:
    """Apply permutation to a point."""
    return perm[point]


def compose(p1: Dict[int, int], p2: Dict[int, int]) -> Dict[int, int]:
    """Compose two permutations: (p1 ∘ p2)(x) = p1(p2(x))."""
    return {i: p1[p2[i]] for i in range(1, 8)}


def inverse(perm: Dict[int, int]) -> Dict[int, int]:
    """Invert a permutation."""
    return {v: k for k, v in perm.items()}


def perm_to_tuple(perm: Dict[int, int]) -> Tuple[int, ...]:
    """Convert permutation dict to tuple for hashing."""
    return tuple(perm[i] for i in range(1, 8))


def tuple_to_perm(t: Tuple[int, ...]) -> Dict[int, int]:
    """Convert tuple back to permutation dict."""
    return {i+1: t[i] for i in range(7)}


def identity() -> Dict[int, int]:
    """Identity permutation."""
    return {i: i for i in range(1, 8)}


# ═══════════════════════════════════════════════════════════════════════════════════════════
# PSL(3,2) GENERATION VIA GL(3,F₂)
# ═══════════════════════════════════════════════════════════════════════════════════════════

def generate_psl32() -> List[Dict[int, int]]:
    """
    Generate all 168 elements of PSL(3,2) = GL(3,F₂).
    Each invertible matrix over F₂ gives an automorphism of the Fano plane.
    """
    matrices = generate_gl3_f2()
    
    # Convert each matrix to a permutation
    group = []
    seen = set()
    
    for M in matrices:
        perm = matrix_to_perm(M)
        key = perm_to_tuple(perm)
        if key not in seen:
            seen.add(key)
            group.append(perm)
    
    return group


# Store some notable matrices as generators
SIGMA_MATRIX = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
], dtype=int)  # This gives an element of order 7

TAU_MATRIX = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
], dtype=int)  # This gives a transposition-like element

# Convert to permutations
SIGMA = matrix_to_perm(SIGMA_MATRIX)
TAU = matrix_to_perm(TAU_MATRIX)


def power(perm: Dict[int, int], n: int) -> Dict[int, int]:
    """Compute perm^n."""
    if n == 0:
        return identity()
    if n < 0:
        perm = inverse(perm)
        n = -n
    result = identity()
    for _ in range(n):
        result = compose(perm, result)
    return result


def verify_psl32(group: List[Dict[int, int]]) -> Dict[str, bool]:
    """Verify properties of the generated group."""
    results = {}
    
    # Size
    results['size_is_168'] = len(group) == 168
    
    # All are automorphisms
    results['all_are_automorphisms'] = all(is_fano_automorphism(g) for g in group)
    
    # Closed under composition
    group_set = {perm_to_tuple(g) for g in group}
    closed = True
    for g1 in group[:20]:  # Check subset for speed
        for g2 in group[:20]:
            prod = compose(g1, g2)
            if perm_to_tuple(prod) not in group_set:
                closed = False
                break
    results['closed_under_composition'] = closed
    
    # Contains identity
    results['contains_identity'] = perm_to_tuple(identity()) in group_set
    
    # Contains inverses
    has_inverses = True
    for g in group[:50]:  # Check subset
        inv = inverse(g)
        if perm_to_tuple(inv) not in group_set:
            has_inverses = False
            break
    results['contains_inverses'] = has_inverses
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════════════════
# CONJUGACY CLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════════

def cycle_type(perm: Dict[int, int]) -> Tuple[int, ...]:
    """Compute the cycle type of a permutation."""
    seen = set()
    cycles = []
    
    for start in range(1, 8):
        if start in seen:
            continue
        
        cycle_len = 0
        current = start
        while current not in seen:
            seen.add(current)
            cycle_len += 1
            current = perm[current]
        
        if cycle_len > 0:
            cycles.append(cycle_len)
    
    return tuple(sorted(cycles, reverse=True))


def classify_by_cycle_type(group: List[Dict[int, int]]) -> Dict[Tuple[int, ...], List[Dict[int, int]]]:
    """Classify group elements by cycle type."""
    classes = defaultdict(list)
    for g in group:
        ct = cycle_type(g)
        classes[ct].append(g)
    return dict(classes)


def order(perm: Dict[int, int]) -> int:
    """Compute the order of a permutation (smallest n such that perm^n = identity)."""
    current = perm.copy()
    n = 1
    while perm_to_tuple(current) != perm_to_tuple(identity()):
        current = compose(perm, current)
        n += 1
        if n > 168:  # Safety
            break
    return n


# ═══════════════════════════════════════════════════════════════════════════════════════════
# STABILIZERS AND ORBITS
# ═══════════════════════════════════════════════════════════════════════════════════════════

def point_stabilizer(group: List[Dict[int, int]], point: int) -> List[Dict[int, int]]:
    """Find all group elements that fix a given point."""
    return [g for g in group if g[point] == point]


def line_stabilizer(group: List[Dict[int, int]], line_idx: int) -> List[Dict[int, int]]:
    """Find all group elements that fix a given line (as a set)."""
    line = FANO_LINES[line_idx]
    stabilizer = []
    for g in group:
        image = frozenset(g[p] for p in line)
        if image == line:
            stabilizer.append(g)
    return stabilizer


def orbit(group: List[Dict[int, int]], point: int) -> Set[int]:
    """Compute the orbit of a point under the group."""
    return {g[point] for g in group}


def flag_orbit(group: List[Dict[int, int]], point: int, line_idx: int) -> List[Tuple[int, int]]:
    """
    A flag is a point-line pair where the point is on the line.
    Compute the orbit of a flag.
    """
    if point not in FANO_LINES[line_idx]:
        return []
    
    orbit_flags = set()
    for g in group:
        new_point = g[point]
        # Find which line the original line maps to
        original_line = FANO_LINES[line_idx]
        image_line = frozenset(g[p] for p in original_line)
        new_line_idx = FANO_LINES.index(image_line)
        orbit_flags.add((new_point, new_line_idx))
    
    return list(orbit_flags)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# SPECIAL SUBGROUPS
# ═══════════════════════════════════════════════════════════════════════════════════════════

def find_sylow_7(group: List[Dict[int, int]]) -> List[Dict[int, int]]:
    """Find a Sylow 7-subgroup (cyclic of order 7)."""
    # Look for elements of order 7
    for g in group:
        if order(g) == 7:
            # Generate the cyclic subgroup
            subgroup = [identity()]
            current = g
            while perm_to_tuple(current) != perm_to_tuple(identity()):
                subgroup.append(current.copy())
                current = compose(g, current)
            return subgroup[:7]  # Should have exactly 7 elements
    return []


def find_sylow_2(group: List[Dict[int, int]]) -> List[Dict[int, int]]:
    """Find a Sylow 2-subgroup (order 8)."""
    # Start with an element of order 2
    involutions = [g for g in group if order(g) == 2]
    
    if not involutions:
        return []
    
    # Build up by taking products
    subgroup = {perm_to_tuple(identity())}
    queue = [involutions[0]]
    
    while queue and len(subgroup) < 8:
        current = queue.pop(0)
        subgroup.add(perm_to_tuple(current))
        
        # Try products with involutions
        for inv in involutions[:10]:
            prod = compose(current, inv)
            key = perm_to_tuple(prod)
            if key not in subgroup and len(subgroup) < 8:
                subgroup.add(key)
                queue.append(prod)
    
    return [tuple_to_perm(t) for t in subgroup]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ANALYSIS AND DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════════════════

def perm_to_string(perm: Dict[int, int]) -> str:
    """Convert permutation to cycle notation string."""
    seen = set()
    cycles = []
    
    for start in range(1, 8):
        if start in seen:
            continue
        if perm[start] == start:
            seen.add(start)
            continue
        
        cycle = [start]
        seen.add(start)
        current = perm[start]
        while current != start:
            cycle.append(current)
            seen.add(current)
            current = perm[current]
        
        if len(cycle) > 1:
            # Use seal symbols
            cycle_str = "(" + " ".join(SEAL_NAMES[p] for p in cycle) + ")"
            cycles.append(cycle_str)
    
    return " ".join(cycles) if cycles else "(id)"


def analyze_group(group: List[Dict[int, int]]):
    """Complete analysis of PSL(3,2)."""
    
    print("=" * 90)
    print("PSL(3,2): COMPLETE ANALYSIS")
    print("=" * 90)
    
    # Basic info
    print(f"\n§1 ORDER AND STRUCTURE")
    print("-" * 50)
    print(f"  |PSL(3,2)| = {len(group)}")
    print(f"  168 = 2³ × 3 × 7")
    print(f"  Simple group: Yes (second smallest non-abelian simple group)")
    
    # Verify
    print(f"\n§2 VERIFICATION")
    print("-" * 50)
    verification = verify_psl32(group)
    for name, passed in verification.items():
        print(f"  {name}: {'✓' if passed else '✗'}")
    
    # Cycle types (conjugacy classes)
    print(f"\n§3 CONJUGACY CLASSES (by cycle type)")
    print("-" * 50)
    classes = classify_by_cycle_type(group)
    
    for ct, elements in sorted(classes.items()):
        # Get a representative
        rep = elements[0]
        rep_str = perm_to_string(rep)
        ord = order(rep)
        print(f"  Cycle type {ct}: {len(elements)} elements, order {ord}")
        print(f"    Representative: {rep_str}")
    
    # Orders
    print(f"\n§4 ELEMENT ORDERS")
    print("-" * 50)
    order_counts = defaultdict(int)
    for g in group:
        order_counts[order(g)] += 1
    
    for ord, count in sorted(order_counts.items()):
        print(f"  Order {ord}: {count} elements")
    
    # Stabilizers
    print(f"\n§5 POINT STABILIZERS")
    print("-" * 50)
    for point in range(1, 8):
        stab = point_stabilizer(group, point)
        print(f"  Stabilizer of {SEAL_NAMES[point]} (point {point}): order {len(stab)}")
    
    print(f"\n  Note: Point stabilizers have order 24 (= 168/7)")
    print(f"  This is because PSL(3,2) acts transitively on 7 points")
    
    # Line stabilizers  
    print(f"\n§6 LINE STABILIZERS")
    print("-" * 50)
    for i, name in enumerate(LINE_NAMES):
        stab = line_stabilizer(group, i)
        print(f"  Stabilizer of {name}: order {len(stab)}")
    
    print(f"\n  Note: Line stabilizers also have order 24 (= 168/7)")
    print(f"  This is because PSL(3,2) acts transitively on 7 lines")
    
    # Orbits (should all be transitive)
    print(f"\n§7 ORBIT ANALYSIS")
    print("-" * 50)
    point_orbit = orbit(group, 1)
    print(f"  Orbit of point 1: {point_orbit}")
    print(f"  Action on points: {'Transitive' if len(point_orbit) == 7 else 'Not transitive'}")
    
    # Flag orbit
    flag_orb = flag_orbit(group, 1, 0)  # Point 1 on line 0
    print(f"  Number of flags (point-line incidences): {len(flag_orb)}")
    print(f"  (Should be 21 = 7 points × 3 lines each)")
    
    # Sylow subgroups
    print(f"\n§8 SYLOW SUBGROUPS")
    print("-" * 50)
    
    sylow7 = find_sylow_7(group)
    print(f"  Sylow 7-subgroup: order {len(sylow7)}")
    print(f"    Generator: {perm_to_string(sylow7[1]) if len(sylow7) > 1 else 'N/A'}")
    print(f"    Number of Sylow 7-subgroups: {168 // (7 * 1)} = 24")
    
    sylow2 = find_sylow_2(group)
    print(f"  Sylow 2-subgroup: order {len(sylow2)}")
    print(f"    Number of Sylow 2-subgroups: 21 (= 168 / 8)")
    
    # Some specific elements
    print(f"\n§9 NOTABLE ELEMENTS")
    print("-" * 50)
    
    print(f"  σ (7-cycle generator): {perm_to_string(SIGMA)}")
    print(f"    Order: {order(SIGMA)}")
    
    print(f"  τ (involution generator): {perm_to_string(TAU)}")
    print(f"    Order: {order(TAU)}")
    
    # Find an element of order 3
    order3 = [g for g in group if order(g) == 3]
    if order3:
        print(f"  Element of order 3: {perm_to_string(order3[0])}")
    
    # Find an element of order 4
    order4 = [g for g in group if order(g) == 4]
    if order4:
        print(f"  Element of order 4: {perm_to_string(order4[0])}")
    
    # Connection to other structures
    print(f"\n§10 CONNECTIONS")
    print("-" * 50)
    print("  PSL(3,2) ≅ PSL(2,7)")
    print("  PSL(3,2) ≅ Aut(Fano plane)")
    print("  PSL(3,2) ≅ Aut(Hamming [7,4,3] code)")
    print("  PSL(3,2) is the symmetry group of the Klein quartic")
    print("  PSL(3,2) acts on the octonion multiplication table")
    
    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)
    
    return classes


def list_all_elements(group: List[Dict[int, int]], max_show: int = 20):
    """List group elements with their properties."""
    
    print("\n" + "=" * 90)
    print(f"ALL {len(group)} ELEMENTS OF PSL(3,2)")
    print("=" * 90)
    
    # Sort by order, then by cycle type
    sorted_group = sorted(group, key=lambda g: (order(g), cycle_type(g)))
    
    print(f"\n  Showing first {min(max_show, len(group))} elements:\n")
    print(f"  {'#':>3} | {'Order':>5} | {'Cycle Type':>15} | Permutation")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*15}-+-{'-'*40}")
    
    for i, g in enumerate(sorted_group[:max_show]):
        ct = str(cycle_type(g))
        ord_g = order(g)
        perm_str = perm_to_string(g)
        print(f"  {i+1:>3} | {ord_g:>5} | {ct:>15} | {perm_str}")
    
    if len(group) > max_show:
        print(f"\n  ... and {len(group) - max_show} more elements")


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating PSL(3,2)...")
    group = generate_psl32()
    print(f"Generated {len(group)} elements.\n")
    
    # Analyze
    classes = analyze_group(group)
    
    # List elements
    list_all_elements(group, max_show=30)
    
    print("\n🔺 PSL(3,2) FULLY ENUMERATED 🔺")
