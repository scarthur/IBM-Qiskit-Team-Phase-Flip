"""
transcribed_code
================
Modular package containing the refactored, Qiskit 2.x compliant quantum algorithms
for the UConn Fall Fest 2026 Instant Insanity QAOA challenge.
"""

from .puzzle_utils import (
    BONUS_SEEDS,
    CHOICES,
    COLOURS,
    CUBES,
    SEED,
    cost,
    count_answers,
    degrees,
    get_colours,
    is_two_factor,
    make_puzzle,
    one_hot,
    orient,
    orientations,
    solves,
    to_bits,
    tower_from,
)

from .hamiltonians import (
    build_compact_3n_hamiltonian,
    build_full_6n_qubo,
    build_hamiltonian_full,
    build_single_side_qubo,
    diagonal,
    energy,
    front_back_hamiltonian,
    left_right_hamiltonian_n,
    qubo_to_ising,
)

from .circuits import (
    constrained_qaoa,
    constrained_vqe_ansatz,
    cost_layer,
    full_constrained_qaoa,
    get_circuit_metrics,
    lr_qaoa_circuit,
    multi_z_cost_layer,
    standard_qaoa_circuit,
    w_state,
    xy_mixer,
)

from .decoders import (
    decode_3n_picks,
    decode_compact_bitstring,
    decode_full_bitstring,
    is_solved_full_bitstring,
    pair_disjoint_shots,
    solution_strings,
)

from .solvers import (
    classical_exhaustive_solver,
    cvar_cost,
    optimize_cvar_layerwise,
    two_stage_hybrid_solve,
)

from .qbraid_runtime import (
    prepare_circuit,
    run_on_qbraid,
    run_v2_sampler,
    strip_marker_counts,
    submit_job,
    transpile_to_isa,
)

from .benchmarks import (
    format_markdown_table,
    run_bonus_scaling_benchmark,
)

__all__ = [
    # puzzle_utils
    "SEED",
    "COLOURS",
    "CUBES",
    "CHOICES",
    "BONUS_SEEDS",
    "get_colours",
    "orientations",
    "solves",
    "degrees",
    "is_two_factor",
    "orient",
    "tower_from",
    "one_hot",
    "to_bits",
    "cost",
    "make_puzzle",
    "count_answers",
    # hamiltonians
    "build_single_side_qubo",
    "build_full_6n_qubo",
    "qubo_to_ising",
    "front_back_hamiltonian",
    "build_hamiltonian_full",
    "left_right_hamiltonian_n",
    "build_compact_3n_hamiltonian",
    "energy",
    "diagonal",
    # circuits
    "w_state",
    "cost_layer",
    "multi_z_cost_layer",
    "xy_mixer",
    "standard_qaoa_circuit",
    "constrained_qaoa",
    "full_constrained_qaoa",
    "lr_qaoa_circuit",
    "constrained_vqe_ansatz",
    "get_circuit_metrics",
    # decoders
    "decode_3n_picks",
    "decode_full_bitstring",
    "decode_compact_bitstring",
    "is_solved_full_bitstring",
    "pair_disjoint_shots",
    "solution_strings",
    # solvers
    "cvar_cost",
    "optimize_cvar_layerwise",
    "two_stage_hybrid_solve",
    "classical_exhaustive_solver",
    # qbraid_runtime
    "submit_job",
    "prepare_circuit",
    "strip_marker_counts",
    "run_on_qbraid",
    "transpile_to_isa",
    "run_v2_sampler",
    # benchmarks
    "run_bonus_scaling_benchmark",
    "format_markdown_table",
]
