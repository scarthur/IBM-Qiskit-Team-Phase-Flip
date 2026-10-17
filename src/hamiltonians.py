"""
transcribed_code/hamiltonians.py
================================
QUBO and Ising Hamiltonian generators for the Instant Insanity puzzle:
- 3N-qubit single-side (Front/Back) Hamiltonian
- 6N-qubit joint (Front/Back + Left/Right) Hamiltonian
- N-qubit conditional Stage-2 (Left/Right) Hamiltonian
- Compact 3N-qubit full-tower Hamiltonian (Direction 2: 3 qubits per cube)
- Matrix/Ising converters and classical diagonal energy evaluators.
"""

from typing import Dict, List, Sequence, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .puzzle_utils import get_colours


def build_single_side_qubo(
    cubes: Sequence[Sequence[str]],
    penalty_onehot: float = 4.0,
    penalty_color: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Builds the 3N-qubit QUBO matrix Q and constant offset for one side pair (e.g. Front/Back).
    Binary variables: x_{i, p} at index q = 3*i + p.
    """
    n = len(cubes)
    num_qubits = 3 * n
    letters = get_colours(cubes)
    Q = np.zeros((num_qubits, num_qubits))
    offset = 0.0

    # 1. One-hot constraint per cube: penalty_onehot * (sum_p x_{i,p} - 1)^2
    for i in range(n):
        offset += penalty_onehot
        for p1 in range(3):
            q1 = 3 * i + p1
            Q[q1, q1] -= penalty_onehot  # x^2 - 2x = -x for x in {0, 1}
            for p2 in range(p1 + 1, 3):
                q2 = 3 * i + p2
                Q[q1, q2] += 2 * penalty_onehot

    # 2. Color degree constraint: penalty_color * (sum_{i,p} c_{i,p,k} * x_{i,p} - 2)^2
    for ch in letters:
        offset += 4.0 * penalty_color
        coeffs: Dict[int, int] = {}
        for i, cube in enumerate(cubes):
            for p, pair in enumerate(cube):
                cnt = pair.count(ch)
                if cnt > 0:
                    coeffs[3 * i + p] = cnt

        for q1, c1 in coeffs.items():
            Q[q1, q1] += penalty_color * (c1**2 - 4 * c1)
            for q2, c2 in coeffs.items():
                if q1 < q2:
                    Q[q1, q2] += penalty_color * (2 * c1 * c2)

    return Q, offset


def build_full_6n_qubo(
    cubes: Sequence[Sequence[str]],
    penalty_onehot: float = 4.0,
    penalty_color: float = 1.0,
    penalty_disjoint: float = 4.0,
) -> Tuple[np.ndarray, float]:
    """
    Builds the 6N-qubit QUBO matrix for Front/Back (first 3N qubits) + Left/Right (next 3N qubits)
    with a disjointness coupling penalty (no pair can be selected on both axes).
    """
    n = len(cubes)
    Q_side, offset_side = build_single_side_qubo(cubes, penalty_onehot, penalty_color)
    Q_full = np.zeros((6 * n, 6 * n))
    Q_full[: 3 * n, : 3 * n] = Q_side
    Q_full[3 * n :, 3 * n :] = Q_side
    offset_full = 2.0 * offset_side

    # Disjointness penalty: D * x_{i,p} * y_{i,p}
    for q in range(3 * n):
        Q_full[q, 3 * n + q] += penalty_disjoint

    return Q_full, offset_full


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], float]:
    """
    Converts a QUBO matrix Q (x in {0, 1}) to Ising formulation (z in {+1, -1})
    via x_q = (1 - z_q) / 2.
    Returns:
        h: 1D array of single-qubit Z coefficients
        J: dict of {(i, j): weight} for two-qubit ZZ interactions
        const: constant energy offset shift
    """
    num_qubits = Q.shape[0]
    h = np.zeros(num_qubits)
    J: Dict[Tuple[int, int], float] = {}
    const = offset

    for i in range(num_qubits):
        const += Q[i, i] / 2.0
        h[i] -= Q[i, i] / 2.0
        for j in range(i + 1, num_qubits):
            w = Q[i, j]
            if abs(w) > 1e-9:
                const += w / 4.0
                h[i] -= w / 4.0
                h[j] -= w / 4.0
                J[(i, j)] = w / 4.0

    return h, J, const


def front_back_hamiltonian(cubes: Sequence[Sequence[str]], lam: float = 0.0) -> SparsePauliOp:
    """
    Constructs the 3N-qubit Front/Back Hamiltonian using Qiskit's SparsePauliOp.
    Supports dynamic colour alphabets for arbitrary N-cube instances.
    """
    colours = get_colours(cubes)
    n = 3 * len(cubes)
    one = SparsePauliOp("I" * n)
    var = lambda q: SparsePauliOp.from_sparse_list([("", [], 0.5), ("Z", [q], -0.5)], n)
    H = 0 * one

    for k in colours:
        d = -2 * one
        for c, cube in enumerate(cubes):
            for p in range(3):
                if cube[p].count(k):
                    d += cube[p].count(k) * var(3 * c + p)
        H += d @ d

    if lam:
        for c in range(len(cubes)):
            pick = sum((var(3 * c + p) for p in range(3)), -1 * one)
            H += lam * (pick @ pick)

    return H.simplify()


def build_hamiltonian_full(
    cubes: Sequence[Sequence[str]], lam: float = 2.0, mu: float = 2.0
) -> SparsePauliOp:
    """
    Constructs the full 6N-qubit one-hot Hamiltonian for Front/Back and Left/Right.
    Qubit 6c + p is x_{c,p}; Qubit 6c + 3 + p is y_{c,p}.
    """
    colours = get_colours(cubes)
    n = 6 * len(cubes)
    one = SparsePauliOp("I" * n)
    var = lambda q: SparsePauliOp.from_sparse_list([("", [], 0.5), ("Z", [q], -0.5)], n)

    xs = [[var(6 * c + p) for p in range(3)] for c in range(len(cubes))]
    ys = [[var(6 * c + 3 + p) for p in range(3)] for c in range(len(cubes))]
    H = 0 * one

    for c in range(len(cubes)):
        if lam:
            for row in (xs[c], ys[c]):
                pick = sum(row, -1 * one)
                H += lam * (pick @ pick)
        if mu:
            H += mu * sum(xs[c][p] @ ys[c][p] for p in range(3))

    for picks in (xs, ys):
        for k in colours:
            d = -2 * one
            for c, cube in enumerate(cubes):
                for p in range(3):
                    if cube[p].count(k):
                        d += cube[p].count(k) * picks[c][p]
            H += d @ d

    return H.simplify()


def left_right_hamiltonian_n(
    cubes: Sequence[Sequence[str]], fb_picks: Sequence[int]
) -> Tuple[SparsePauliOp, List[List[int]]]:
    """
    Stage 2 Hamiltonian for the Left/Right axis given fixed Front/Back pair picks.
    Each cube c has exactly 2 unused pairs rem[c] = [p0, p1].
    Qubit c in {0, 1} selects rem[c][bit_c].
    Returns (H_lr on N qubits, rem_pairs list).
    """
    colours = get_colours(cubes)
    n_cubes = len(cubes)
    one = SparsePauliOp("I" * n_cubes)
    var = lambda q: SparsePauliOp.from_sparse_list([("", [], 0.5), ("Z", [q], -0.5)], n_cubes)
    rem_pairs = [[p for p in range(3) if p != fb_picks[c]] for c in range(n_cubes)]

    H = 0 * one
    for k in colours:
        d = -2 * one
        for c, cube in enumerate(cubes):
            p0, p1 = rem_pairs[c]
            c0, c1 = cube[p0].count(k), cube[p1].count(k)
            if c0:
                d += c0 * one
            if c1 != c0:
                d += (c1 - c0) * var(c)
        H += d @ d

    return H.simplify(), rem_pairs


def build_compact_3n_hamiltonian(
    cubes: Sequence[Sequence[str]], lam: float = 8.0
) -> SparsePauliOp:
    """
    Direction 2: Encodes all 6 valid (FB, LR) choices into 3 qubits per cube (3N total):
    - Qubits (3c, 3c+1) in {00, 01, 10} select FB pair a in {0, 1, 2}; state 11 is penalized.
    - Qubit (3c+2) in {0, 1} selects which of the 2 remaining pairs goes to LR.
    Generates multi-qubit Pauli Z interaction terms up to order 6.
    """
    colours = get_colours(cubes)
    n_cubes = len(cubes)
    n_qubits = 3 * n_cubes
    one = SparsePauliOp("I" * n_qubits)
    var = lambda q: SparsePauliOp.from_sparse_list([("", [], 0.5), ("Z", [q], -0.5)], n_qubits)

    x_ops = []
    y_ops = []
    H = 0 * one

    for c in range(n_cubes):
        q0, q1, s = var(3 * c), var(3 * c + 1), var(3 * c + 2)
        not_q0, not_q1, not_s = one - q0, one - q1, one - s

        # FB pair indicators: 0 -> |00>, 1 -> |10>, 2 -> |01>
        x0 = (not_q0 @ not_q1).simplify()
        x1 = (q0 @ not_q1).simplify()
        x2 = (not_q0 @ q1).simplify()
        x_ops.append([x0, x1, x2])

        # Penalty for invalid state |11> on qubits 3c, 3c+1
        H += lam * (q0 @ q1)

        # LR pair indicators conditioned on remaining pairs:
        y0 = (x1 @ not_s + x2 @ s).simplify()
        y1 = (x0 @ not_s + x2 @ not_s).simplify()
        y2 = (x0 @ s + x1 @ s).simplify()
        y_ops.append([y0, y1, y2])

    for picks in (x_ops, y_ops):
        for k in colours:
            d = -2 * one
            for c, cube in enumerate(cubes):
                for p in range(3):
                    if cube[p].count(k):
                        d += cube[p].count(k) * picks[c][p]
            H += d @ d

    return H.simplify()


def energy(H: SparsePauliOp, bits: Sequence[int]) -> float:
    """
    Calculates the expectation value of a diagonal Z-only Hamiltonian on a classical bitstring.
    bits[q] corresponds to qubit q. Qiskit's Pauli labels order qubit 0 at the end (reverse).
    """
    total = 0.0
    for label, coeff in zip(H.paulis.to_labels(), H.coeffs):
        sign = 1
        for q, op in enumerate(reversed(label)):
            if op == "Z" and bits[q]:
                sign = -sign
        total += coeff.real * sign
    return float(total)


def diagonal(H: SparsePauliOp) -> np.ndarray:
    """
    Returns the vector of eigenvalues of a Z-only Hamiltonian across all 2**N basis states.
    Indexed identically to Statevector.probabilities().
    """
    idx = np.arange(2**H.num_qubits)
    values = np.zeros(len(idx), dtype=np.float64)
    for label, coeff in zip(H.paulis.to_labels(), H.coeffs):
        sign = np.ones(len(idx))
        for q, op in enumerate(reversed(label)):
            if op == "Z":
                sign *= 1 - 2 * ((idx >> q) & 1)
        values += coeff.real * sign
    return values
