"""
transcribed_code/solvers.py
===========================
Classical and hybrid quantum-classical solvers for the Instant Insanity puzzle:
- Direction 5: CVaR (Conditional Value-at-Risk) optimization & layer-by-layer warm starts
- Two-Stage Hybrid Solver (Stage 1: 3N qubits FB -> Stage 2: N qubits LR)
- Classical ground truth solver for baseline comparisons.
"""

import itertools
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector

from .circuits import constrained_qaoa, lr_qaoa_circuit
from .decoders import decode_3n_picks
from .hamiltonians import diagonal, left_right_hamiltonian_n
from .puzzle_utils import CHOICES, CUBES, is_two_factor, solves, tower_from


def cvar_cost(probs: np.ndarray, costs: np.ndarray, alpha: float = 0.20) -> float:
    """
    Direction 5: Conditional Value at Risk (CVaR) objective function.
    Computes the expected energy of the lowest alpha fraction of the probability distribution.
    Focuses classical optimization directly on amplifying the ground state probability.
    """
    order = np.argsort(costs)
    sorted_probs = probs[order]
    sorted_costs = costs[order]
    cum_probs = np.cumsum(sorted_probs)
    idx = int(np.searchsorted(cum_probs, alpha))

    if idx == 0:
        return float(sorted_costs[0])

    weight_head = sorted_probs[:idx]
    rem = alpha - cum_probs[idx - 1]
    val = (np.dot(weight_head, sorted_costs[:idx]) + rem * sorted_costs[idx]) / alpha
    return float(val)


def optimize_cvar_layerwise(
    H: Any,
    cubes: Sequence[Sequence[str]],
    p_max: int = 2,
    alpha: float = 0.25,
    maxiter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Direction 5: Layer-by-layer QAOA optimization using CVaR.
    Warm-starts layer p from layer p-1, progressively refining the ansatz.
    Returns:
        (optimal_gammas, optimal_betas, history_evals)
    """
    costs = diagonal(H)
    gammas = [np.pi / 4]
    betas = [np.pi / 8]
    history: List[float] = []

    for p in range(1, p_max + 1):
        if p == 1:
            init_angles = gammas + betas
        else:
            init_angles = gammas + [gammas[-1]] + betas + [betas[-1]]

        def objective(angles: np.ndarray) -> float:
            g = angles[:p]
            b = angles[p:]
            qc = constrained_qaoa(H, g, b, cubes)
            probs = Statevector(qc).probabilities()
            val = cvar_cost(probs, costs, alpha=alpha)
            history.append(val)
            return val

        res = minimize(objective, init_angles, method="COBYLA", options={"maxiter": maxiter})
        gammas = list(res.x[:p])
        betas = list(res.x[p:])

    return np.array(gammas), np.array(betas), history


def two_stage_hybrid_solve(
    cubes: Sequence[Sequence[str]],
    fb_counts: Dict[str, int],
    shots_lr: int = 1000,
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[str, str, str, str]]]]:
    """
    Two-Stage Hybrid Solver:
    - Stage 1: Takes sampled counts from 3N Front/Back QAOA
    - Stage 2: For each sampled 2-factor, constructs the N-qubit Left/Right Hamiltonian,
      optimizes the N-qubit circuit, samples shots, and checks for full-tower solutions.
    Returns:
        (per_cube_choices, solved_tower) or None if no partner found in sample.
    """
    n_cubes = len(cubes)
    # Collect valid sampled 2-factors sorted by shot count
    candidate_2factors = []
    for bitstr, cnt in sorted(fb_counts.items(), key=lambda kv: kv[1], reverse=True):
        picks = decode_3n_picks(bitstr, n_cubes)
        if picks is not None and is_two_factor(cubes, picks):
            if picks not in candidate_2factors:
                candidate_2factors.append(picks)

    for fb_pick in candidate_2factors:
        H_lr, rem_pairs = left_right_hamiltonian_n(cubes, fb_pick)
        costs_lr = diagonal(H_lr)

        # Optimize p=2 angles on the N-qubit statevector
        res_lr = minimize(
            lambda a: Statevector(lr_qaoa_circuit(H_lr, a[:2], a[2:])).probabilities() @ costs_lr,
            [0.5, 0.8, 0.4, 0.2],
            method="COBYLA",
            options={"maxiter": 120},
        )
        qc_lr = lr_qaoa_circuit(H_lr, res_lr.x[:2], res_lr.x[2:])
        probs_lr = Statevector(qc_lr).probabilities()

        # Sample or check basis states
        best_states = np.argsort(probs_lr)[::-1]
        for state_idx in best_states[:10]:
            bits_lr = [(state_idx >> q) & 1 for q in range(n_cubes)]
            lr_pick = [rem_pairs[c][bits_lr[c]] for c in range(n_cubes)]
            candidate_choice = list(zip(fb_pick, lr_pick))
            tower = tower_from(cubes, candidate_choice)
            if solves(tower):
                return candidate_choice, tower

    return None


def classical_exhaustive_solver(
    cubes: Sequence[Sequence[str]]
) -> List[List[Tuple[int, int]]]:
    """
    Exhaustively searches all 6^N valid per-cube choices to find all solved towers.
    Used to establish exact baseline counts and verify correctness.
    """
    solutions = []
    for choice in itertools.product(CHOICES, repeat=len(cubes)):
        tower = tower_from(cubes, choice)
        if solves(tower):
            solutions.append(list(choice))
    return solutions
