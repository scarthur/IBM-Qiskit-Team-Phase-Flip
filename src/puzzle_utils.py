"""
transcribed_code/puzzle_utils.py
================================
Core puzzle geometry, graph utilities, validation functions, and N-cube puzzle generator.
Refactored from the Argon conversation and aligned with starter_notebook.ipynb conventions.
"""

import itertools
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Default constants for the 4-cube Instant Insanity puzzle
SEED: int = 7
COLOURS: str = "RGBY"
CUBES: List[List[str]] = [
    ["GR", "BR", "GY"],  # Cube 1
    ["GY", "RB", "BY"],  # Cube 2
    ["BR", "RG", "RY"],  # Cube 3
    ["YR", "RG", "BG"],  # Cube 4
]

# Each cube has 3 opposite-face pairs (0, 1, 2).
# An ordered choice of (front/back pair, left/right pair) has 3 * 2 = 6 possibilities:
CHOICES: List[Tuple[int, int]] = list(itertools.permutations(range(3), 2))

# Planted puzzle seeds verified to yield exactly 1 distinct tower (2 cases: tower & quarter turn)
BONUS_SEEDS: Dict[int, int] = {5: 16, 6: 86, 7: 68, 8: 4}


def get_colours(cubes: Sequence[Sequence[str]]) -> str:
    """
    Extract the sorted unique colour characters from an arbitrary cube set.
    Fixes the starter notebook's hardcoded 'RGBY' assumption to support N-cube bonus puzzles.
    """
    return "".join(sorted({ch for cube in cubes for pair in cube for ch in pair}))


def orientations(cube: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    """
    Returns all 24 orientations of a cube as (front, back, left, right) visible colours.
    """
    views = []
    for a, b in itertools.permutations(range(3), 2):  # FB pair, LR pair
        for fa in (0, 1):
            for fb in (0, 1):
                fb_pair, lr_pair = cube[a], cube[b]
                views.append((fb_pair[fa], fb_pair[1 - fa], lr_pair[fb], lr_pair[1 - fb]))
    return views


def solves(tower: Sequence[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]) -> bool:
    """
    Checks if a stacked tower is solved: every side (front, right, back, left)
    must display each colour exactly once without duplicate clashes.
    """
    return all(
        len({view[side] for view in tower if view[side] is not None}) == len(tower)
        for side in range(4)
    )


def degrees(cubes: Sequence[Sequence[str]], picks: Sequence[int]) -> Dict[str, int]:
    """
    Counts how many times each colour appears among the picked pairs.
    """
    colours = get_colours(cubes)
    count = dict.fromkeys(colours, 0)
    for cube, p in zip(cubes, picks):
        for colour in cube[p]:
            count[colour] += 1
    return count


def is_two_factor(cubes: Sequence[Sequence[str]], picks: Sequence[int]) -> bool:
    """
    True when the picked pairs form a 2-factor: each colour appears exactly twice.
    """
    return all(d == 2 for d in degrees(cubes, picks).values())


def orient(cubes: Sequence[Sequence[str]], picks: Sequence[int]) -> Dict[int, Tuple[str, str]]:
    """
    Walks the cycles of a 2-factor to determine front and back face assignments.
    Returns {cube_index: (front_colour, back_colour)}.
    """
    lines = {c: cubes[c][p] for c, p in enumerate(picks)}
    placed: Dict[int, Tuple[str, str]] = {}
    while len(placed) < len(lines):
        c = next(c for c in lines if c not in placed)
        here = lines[c][0]
        while c is not None:
            u, v = lines[c]
            placed[c] = (here, v if u == here else u)
            here = placed[c][1]
            c = next(
                (d for d, (a, b) in lines.items() if d not in placed and here in (a, b)),
                None,
            )
    return placed


def tower_from(
    cubes: Sequence[Sequence[str]], choice: Sequence[Tuple[int, int]]
) -> List[Tuple[str, str, str, str]]:
    """
    Converts per-cube choices ((fb_pair, lr_pair), ...) into a 4-sided physical tower:
    (front, left, back, right) colours per cube level.
    """
    front = orient(cubes, [fb for fb, _ in choice])
    side = orient(cubes, [lr for _, lr in choice])
    return [(front[c][0], side[c][0], front[c][1], side[c][1]) for c in range(len(cubes))]


def one_hot(p: int, size: int = 3) -> List[int]:
    """Returns a one-hot binary list of length `size` with a 1 at index `p`."""
    return [int(i == p) for i in range(size)]


def to_bits(x: Sequence[Sequence[int]], y: Sequence[Sequence[int]]) -> List[int]:
    """Flattens 2D arrays of front/back and left/right choice variables into a bit array."""
    bits = []
    for c in range(len(x)):
        bits += list(x[c]) + list(y[c])
    return bits


def cost(
    cubes: Sequence[Sequence[str]],
    x: Sequence[Sequence[int]],
    y: Sequence[Sequence[int]],
    lam: float = 2.0,
    mu: float = 2.0,
) -> float:
    """
    QUBO cost function: evaluates total penalty for front/back (x) and left/right (y) picks.
    Evaluates to 0.0 if and only if the assignment forms a valid solved tower.
    """
    colours = get_colours(cubes)
    total = 0.0
    for c in range(len(cubes)):
        # One pair per axis penalty
        total += lam * ((sum(x[c]) - 1) ** 2 + (sum(y[c]) - 1) ** 2)
        # No double duty penalty (cannot pick same pair for both axes)
        total += mu * sum(x[c][p] * y[c][p] for p in range(3))

    # Degree constraints for each axis
    for picks in (x, y):
        for k in colours:
            d = sum(
                cubes[c][p].count(k) * picks[c][p]
                for c in range(len(cubes))
                for p in range(3)
            )
            total += (d - 2) ** 2
    return float(total)


def make_puzzle(n: int, seed: int) -> List[List[str]]:
    """
    Generates an N-cube, N-colour puzzle with a planted solution consisting of two
    Hamiltonian cycles through all N colour letters.
    Cleaned and transcribed from the Argon discussion.
    """
    rng = random.Random(seed)
    letters = "".join(chr(ord("A") + i) for i in range(n))
    cycles = []
    for _ in range(2):
        order = rng.sample(letters, n)
        lines = [order[i] + order[(i + 1) % n] for i in range(n)]
        rng.shuffle(lines)
        cycles.append(lines)

    cubes = []
    for i in range(n):
        third = rng.choice(letters) + rng.choice(letters)
        pairs = ["".join(rng.sample(pair, 2)) for pair in (cycles[0][i], cycles[1][i], third)]
        rng.shuffle(pairs)
        cubes.append(pairs)
    return cubes


def count_answers(cubes: Sequence[Sequence[str]]) -> int:
    """
    Brute-force verification: counts how many of the 6**N per-cube choices yield valid 2-factors
    on both the front/back and left/right axes.
    """
    letters = get_colours(cubes)
    total = 0
    for choice in itertools.product(CHOICES, repeat=len(cubes)):
        for side in (0, 1):
            deg = dict.fromkeys(letters, 0)
            for cube, pick in zip(cubes, choice):
                for ch in cube[pick[side]]:
                    deg[ch] += 1
            if any(d != 2 for d in deg.values()):
                break
        else:
            total += 1
    return total
