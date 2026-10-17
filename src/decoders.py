"""
transcribed_code/decoders.py
============================
Bitstring decoding, little-endian formatting, and validation against physical tower rules:
- 3N-qubit Front/Back bitstrings -> per-cube pair picks
- 6N-qubit joint bitstrings -> per-cube (Front/Back, Left/Right) choices
- Compact 3N-qubit full-tower bitstrings (Direction 2)
- Disjoint 2-factor pairing (Direction 3: Split the work)
- Little-endian compliance: Qiskit qubit 0 is always the rightmost bit.
"""

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .puzzle_utils import CUBES, get_colours, is_two_factor, solves, tower_from


def decode_3n_picks(bitstring: str, n_cubes: int) -> Optional[Tuple[int, ...]]:
    """
    Decodes a 3N-bit Qiskit string into a tuple of pair indices (p_0, ..., p_{N-1}).
    Qiskit bitstrings are little-endian: qubit 0 is the rightmost character.
    Returns None if any cube does not satisfy the one-hot constraint (sum != 1).
    """
    bits = [int(b) for b in reversed(bitstring)]  # bits[q] is qubit q
    picks = []
    for c in range(n_cubes):
        triple = bits[3 * c : 3 * c + 3]
        if sum(triple) != 1:
            return None
        picks.append(triple.index(1))
    return tuple(picks)


def decode_full_bitstring(
    bitstring: str, cubes: Sequence[Sequence[str]] = CUBES
) -> Optional[List[Tuple[int, int]]]:
    """
    Decodes a 6N-qubit Qiskit bitstring into per-cube (front_back_pair, left_right_pair) choices.
    Returns None if:
    - Any cube is not one-hot for front/back or left/right
    - Any cube assigns the same pair to both axes (violates disjointness).
    """
    bits = [int(b) for b in reversed(bitstring)]
    choice = []
    for c in range(len(cubes)):
        x_c = bits[6 * c : 6 * c + 3]
        y_c = bits[6 * c + 3 : 6 * c + 6]
        if sum(x_c) != 1 or sum(y_c) != 1:
            return None
        fb_pair = x_c.index(1)
        lr_pair = y_c.index(1)
        if fb_pair == lr_pair:
            return None
        choice.append((fb_pair, lr_pair))
    return choice


def decode_compact_bitstring(
    bitstring: str, cubes: Sequence[Sequence[str]] = CUBES
) -> Optional[List[Tuple[int, int]]]:
    """
    Decodes a compact 3N-qubit bitstring (Direction 2) where each cube uses 3 qubits:
    - Qubits 3c, 3c+1: |00> -> pair 0, |10> -> pair 1, |01> -> pair 2 (|11> is invalid).
    - Qubit 3c+2: bit in {0, 1} picks from the 2 remaining pairs for Left/Right.
    """
    bits = [int(b) for b in reversed(bitstring)]
    choice = []
    for c in range(len(cubes)):
        q0 = bits[3 * c]
        q1 = bits[3 * c + 1]
        s = bits[3 * c + 2]
        if q0 == 0 and q1 == 0:
            fb = 0
            rem = [1, 2]
        elif q0 == 1 and q1 == 0:
            fb = 1
            rem = [0, 2]
        elif q0 == 0 and q1 == 1:
            fb = 2
            rem = [0, 1]
        else:
            return None  # Invalid state 11
        lr = rem[s]
        choice.append((fb, lr))
    return choice


def is_solved_full_bitstring(
    bitstring: str, cubes: Sequence[Sequence[str]] = CUBES
) -> bool:
    """
    Decodes a 6N bitstring and independently verifies if the resulting tower is solved.
    """
    choice = decode_full_bitstring(bitstring, cubes)
    if choice is None:
        return False
    tower = tower_from(cubes, choice)
    return solves(tower)


def pair_disjoint_shots(
    counts: Dict[str, int], cubes: Sequence[Sequence[str]] = CUBES
) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[List[Tuple[int, int]], Any, int, int]]]:
    """
    Direction 3: 'Split the work'
    Takes measurement counts from a 3N-qubit QAOA run, filters bitstrings that form valid
    2-factors, pairs any two disjoint 2-factors that share no pair, and verifies full towers.
    Returns:
        (sampled_2factors, solved_towers)
    """
    n_cubes = len(cubes)
    sampled_2factors: Dict[Tuple[int, ...], int] = {}

    for bitstr, cnt in counts.items():
        picks = decode_3n_picks(bitstr, n_cubes)
        if picks is not None and is_two_factor(cubes, picks):
            sampled_2factors[picks] = sampled_2factors.get(picks, 0) + cnt

    solved_towers = []
    for fb, lr in itertools.permutations(sampled_2factors.keys(), 2):
        if all(a != b for a, b in zip(fb, lr)):
            choice = list(zip(fb, lr))
            tower = tower_from(cubes, choice)
            if solves(tower):
                solved_towers.append(
                    (choice, tower, sampled_2factors[fb], sampled_2factors[lr])
                )

    return sampled_2factors, solved_towers


def solution_strings(
    picks_list: Sequence[Sequence[int]], n_cubes: int = 4
) -> List[str]:
    """
    Converts a list of front/back pair index tuples into little-endian Qiskit bitstrings.
    Useful for checking known solutions.
    """
    out = []
    for picks in picks_list:
        bits = ["0"] * (3 * n_cubes)
        for c, p in enumerate(picks):
            bits[3 * c + p] = "1"
        out.append("".join(reversed(bits)))  # Qiskit qubit 0 is last (rightmost)
    return out
