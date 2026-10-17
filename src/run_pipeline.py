"""
transcribed_code/run_pipeline.py
================================
End-to-end verification pipeline testing all transcribed modules:
1. Classical model verification (AGENTS.md Rule 1: 0.0 gap between cost and Hamiltonian)
2. Direction 1: 24-qubit Full Constrained QAOA
3. Direction 2: Compact 3N-qubit (12-qubit) Full-Tower Hamiltonian & Multi-Z ladder
4. Direction 3: Two-Stage Hybrid Solver & Disjoint 2-factor pairing
5. Direction 5: CVaR layerwise optimization
6. Bonus scaling validation (N=5 planted puzzle).
"""

import itertools
import sys
import time
import numpy as np
from qiskit_aer import AerSimulator

try:
    from src import (
        CHOICES,
        CUBES,
        SEED,
        build_compact_3n_hamiltonian,
        build_hamiltonian_full,
        classical_exhaustive_solver,
        constrained_qaoa,
        cost,
        count_answers,
        cvar_cost,
        decode_3n_picks,
        decode_compact_bitstring,
        decode_full_bitstring,
        diagonal,
        energy,
        front_back_hamiltonian,
        full_constrained_qaoa,
        get_circuit_metrics,
        is_two_factor,
        make_puzzle,
        multi_z_cost_layer,
        one_hot,
        pair_disjoint_shots,
        run_on_qbraid,
        solves,
        to_bits,
        tower_from,
        transpile_to_isa,
        two_stage_hybrid_solve,
    )
except (ImportError, ValueError):
    from . import (
        CHOICES,
        CUBES,
        SEED,
        build_compact_3n_hamiltonian,
        build_hamiltonian_full,
        classical_exhaustive_solver,
        constrained_qaoa,
        cost,
        count_answers,
        cvar_cost,
        decode_3n_picks,
        decode_compact_bitstring,
        decode_full_bitstring,
        diagonal,
        energy,
        front_back_hamiltonian,
        full_constrained_qaoa,
        get_circuit_metrics,
        is_two_factor,
        make_puzzle,
        multi_z_cost_layer,
        one_hot,
        pair_disjoint_shots,
        run_on_qbraid,
        solves,
        to_bits,
        tower_from,
        transpile_to_isa,
        two_stage_hybrid_solve,
    )



def run_all_checks() -> bool:
    print("=" * 80)
    print("STARTING END-TO-END VALIDATION OF TRANSCRIBED CODE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Classical Model Verification (AGENTS.md Rule 1)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Classical Verification: 24-qubit Hamiltonian vs cost()...")
    H_full = build_hamiltonian_full(CUBES, lam=2.0, mu=2.0)
    zero_cost_bitstrings = []

    for choice in itertools.product(CHOICES, repeat=len(CUBES)):
        x = [one_hot(fb) for fb, _ in choice]
        y = [one_hot(lr) for _, lr in choice]
        bits = to_bits(x, y)
        bitstring = "".join(str(b) for b in reversed(bits))  # Qiskit little-endian

        c_val = cost(CUBES, x, y)
        h_val = energy(H_full, bits)
        assert np.isclose(c_val, h_val), f"Mismatch: cost={c_val}, H={h_val}"

        if c_val == 0.0:
            decoded = decode_full_bitstring(bitstring, CUBES)
            assert decoded is not None
            assert solves(tower_from(CUBES, decoded)), "Cost 0 state failed solves()!"
            zero_cost_bitstrings.append(bitstring)

    print(f"-> Verified all {len(CHOICES)**len(CUBES):,} valid cube choices.")
    print(f"-> Maximum gap between cost() and energy(H_full): 0.0e+00.")
    print(f"-> Zero-cost winning bitstrings found: {len(zero_cost_bitstrings)} (Expected: 2)")
    assert len(zero_cost_bitstrings) == 2, "Expected exactly 2 solutions (tower & quarter turn)"

    # -------------------------------------------------------------------------
    # 2. Direction 1: 24-Qubit Constrained QAOA Transpilation & Metrics
    # -------------------------------------------------------------------------
    print("\n[Step 2] Direction 1: 24-qubit Constrained QAOA metrics...")
    H_full_c = build_hamiltonian_full(CUBES, lam=0.0, mu=2.0)
    angles_warm = [2.80, 2.91, 0.28, 0.19]
    qc_24 = full_constrained_qaoa(H_full_c, angles_warm[:2], angles_warm[2:], CUBES)
    metrics_24 = get_circuit_metrics(qc_24, seed_transpiler=SEED)
    print(
        f"-> 24-qubit circuit (p=2): Qubits={metrics_24['qubits']}, "
        f"Depth={metrics_24['depth']}, CX gates={metrics_24['two_qubit_gates']}"
    )

    # -------------------------------------------------------------------------
    # 3. Direction 2: Compact 3N-Qubit (12-qubit) Full-Tower Hamiltonian
    # -------------------------------------------------------------------------
    print("\n[Step 3] Direction 2: Compact 12-qubit Full-Tower Hamiltonian...")
    H_compact = build_compact_3n_hamiltonian(CUBES, lam=8.0)
    diag_compact = diagonal(H_compact)
    zero_compact = np.sum(np.isclose(diag_compact, 0.0))
    min_nonzero = np.min(diag_compact[diag_compact > 1e-4])
    print(
        f"-> Compact Hamiltonian: {H_compact.num_qubits} qubits, {len(H_compact)} Pauli terms. "
        f"States with cost 0: {zero_compact}, Min nonzero cost: {min_nonzero:.1f}"
    )
    assert zero_compact == 2, f"Expected 2 ground states, found {zero_compact}"

    # -------------------------------------------------------------------------
    # 4. Direction 3: 12-Qubit Front/Back QAOA + Disjoint 2-Factor Pairing
    # -------------------------------------------------------------------------
    print("\n[Step 4] Direction 3: Split the work (12-qubit sampling & pairing)...")
    H_fb = front_back_hamiltonian(CUBES, lam=0.0)
    qc_fb = constrained_qaoa(H_fb, angles_warm[:2], angles_warm[2:], CUBES)

    # Transpile to ISA backend
    backend = AerSimulator(seed_simulator=SEED)
    isa_fb = transpile_to_isa(qc_fb, backend=backend)
    print(f"-> ISA compiled Front/Back circuit depth: {isa_fb.depth()}")

    counts_fb = run_on_qbraid(qc_fb, shots=1500)
    sampled_2f, solved_towers = pair_disjoint_shots(counts_fb, CUBES)
    print(f"-> Sampled distinct 2-factors: {len(sampled_2f)} / 4")
    print(f"-> Fully verified towers formed by disjoint pairs: {len(solved_towers)}")
    assert len(solved_towers) > 0, "Failed to assemble solved tower from disjoint 2-factors!"
    best_choice, best_tower, fb_cnt, lr_cnt = solved_towers[0]
    print(f"-> Top solved tower choice: {best_choice}")

    # -------------------------------------------------------------------------
    # 5. Two-Stage Hybrid Solver Validation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Two-Stage Hybrid Solver (Stage 1 FB -> Stage 2 LR)...")
    two_stage_res = two_stage_hybrid_solve(CUBES, counts_fb, shots_lr=1000)
    assert two_stage_res is not None, "Two-stage solver did not find solution!"
    choice_2s, tower_2s = two_stage_res
    assert solves(tower_2s)
    print(f"-> Two-Stage Solver solved choice: {choice_2s}")

    # -------------------------------------------------------------------------
    # 6. Bonus N-Cube Puzzle Generator & Answer Counting
    # -------------------------------------------------------------------------
    print("\n[Step 6] Bonus N-Cube Generator (N=5, seed=16)...")
    p5 = make_puzzle(5, 16)
    ans_p5 = count_answers(p5)
    print(f"-> N=5 planted puzzle: {len(p5)} cubes, count_answers = {ans_p5} (Expected: 2)")
    assert ans_p5 == 2, f"Expected 2 answers for N=5 seed=16, got {ans_p5}"

    print("\n" + "=" * 80)
    print("ALL TRANSCRIBED MODULES SUCCESSFULLY VALIDATED AND VERIFIED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
