"""
transcribed_code/benchmarks.py
==============================
N-cube scaling benchmark suite (N = 4, 5, 6, 7, 8) and GRADING.md reporting generators:
- Evaluates resource metrics (qubits, transpiled CX gates, depth)
- Compares QAOA success rates against analytical random-guess baselines
- Generates formatted Markdown results tables matching GRADING.md submission requirements.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from qiskit import transpile

from .circuits import constrained_qaoa, get_circuit_metrics
from .decoders import decode_3n_picks
from .hamiltonians import front_back_hamiltonian
from .puzzle_utils import BONUS_SEEDS, CUBES, is_two_factor, make_puzzle, solves, tower_from
from .qbraid_runtime import SEED, run_on_qbraid


def run_bonus_scaling_benchmark(
    seeds: Optional[Dict[int, int]] = None,
    angles: Optional[Sequence[float]] = None,
    shots_base: int = 2000,
) -> List[Dict[str, Any]]:
    """
    Executes the N-cube scaling benchmark across N in {4, 5, 6, 7, 8}.
    Uses transferred QAOA angles from N=4 (per AGENTS.md Rule 10) to avoid exponential
    classical optimization times on deeper bonus puzzles.
    """
    if seeds is None:
        seeds = BONUS_SEEDS
    if angles is None:
        # Pre-tuned warm-start angles from Section 6 (g1, g2, b1, b2)
        angles = [2.8015, 2.9102, 0.2854, 0.1983]

    gammas = angles[:2]
    betas = angles[2:]

    experiments = [(4, None, CUBES)] + [
        (n, seed, make_puzzle(n, seed)) for n, seed in sorted(seeds.items())
    ]
    results: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"{'N':<4} | {'Qubits':<7} | {'CX':<6} | {'Depth':<6} | {'P(2-factor)':<12} | {'Random (2/3^N)':<15} | {'Full Tower?':<12}")
    print("-" * 80)

    for n, seed, cubes_n in experiments:
        H_n = front_back_hamiltonian(cubes_n, lam=0.0)
        qc_n = constrained_qaoa(H_n, gammas, betas, cubes=cubes_n)

        # Standardized metrics after transpilation to basic gates
        metrics = get_circuit_metrics(qc_n, seed_transpiler=SEED)
        cx_gates = metrics["two_qubit_gates"]
        depth = metrics["depth"]

        shots_n = shots_base if n <= 6 else shots_base * 2
        counts_n = run_on_qbraid(qc_n, shots=shots_n)

        # Analyze 2-factor sampling frequency
        two_factors: Dict[tuple, int] = {}
        for bitstr, cnt in counts_n.items():
            picks = decode_3n_picks(bitstr, n)
            if picks is not None and is_two_factor(cubes_n, picks):
                two_factors[picks] = two_factors.get(picks, 0) + cnt

        hits = sum(two_factors.values())
        p_tf = hits / shots_n

        # Verify if two disjoint 2-factors solve the full tower
        full_tower_found = False
        import itertools

        for fb1, fb2 in itertools.permutations(two_factors.keys(), 2):
            if all(a != b for a, b in zip(fb1, fb2)):
                choice_n = list(zip(fb1, fb2))
                if solves(tower_from(cubes_n, choice_n)):
                    full_tower_found = True
                    break

        rand_w_subspace = 2.0 / (3.0**n)
        rand_pair_space = 2.0 / (6.0**n)
        rand_physical_3d = 8.0 / (24.0**n)

        row = {
            "n": n,
            "seed": seed,
            "qubits": 3 * n,
            "cx": cx_gates,
            "depth": depth,
            "shots": shots_n,
            "p_tf": p_tf,
            "full_tower_found": full_tower_found,
            "rand_w_subspace": rand_w_subspace,
            "rand_pair_space": rand_pair_space,
            "rand_physical_3d": rand_physical_3d,
        }
        results.append(row)

        print(
            f"{n:<4} | {3*n:<7} | {cx_gates:<6} | {depth:<6} | "
            f"{p_tf:<12.2%} | {rand_w_subspace:<15.4%} | {str(full_tower_found):<12}"
        )

    print("=" * 80)
    return results


def format_markdown_table(results: List[Dict[str, Any]]) -> str:
    """
    Formats benchmark results into a GitHub-style Markdown table matching GRADING.md.
    """
    lines = [
        "| Puzzle (N cubes) | Encoding | Device | Qubits | Two-qubit gates (CX) | Depth | Shots | P(valid 2-factor) | Random Baseline (2/3^N) | Amplification |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        label = f"N={r['n']} (Core)" if r["seed"] is None else f"N={r['n']} (seed={r['seed']})"
        amplification = r["p_tf"] / r["rand_w_subspace"] if r["rand_w_subspace"] > 0 else 0
        lines.append(
            f"| {label} | 3N W-state + XY (p=2) | aws:aws:sim:sv1 / Aer | {r['qubits']} | "
            f"{r['cx']} | {r['depth']} | {r['shots']} | {r['p_tf']:.2%} | "
            f"{r['rand_w_subspace']:.4%} | {amplification:.1f}x |"
        )
    return "\n".join(lines)
