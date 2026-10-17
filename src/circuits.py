"""
transcribed_code/circuits.py
============================
Parameterized Quantum Circuit ansätze for QAOA and VQE:
- Dicke / W-state initialization (|W> = (|100> + |010> + |001>)/sqrt(3))
- Standard Z/ZZ cost layer and multi-qubit Z parity ladder cost layer
- Intra-cube XY mixer (XXPlusYYGate)
- 3N Constrained QAOA & 6N Full-Tower Constrained QAOA
- Direction 4: Constrained-subspace VQE ansatz
- Circuit metric extraction (qubits, depth, CX count) per GRADING.md rules.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RYGate, XGate, XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp


def w_state(qc: QuantumCircuit, a: int, b: int, c: int) -> None:
    """
    Prepares a W-state equal superposition (|100> + |010> + |001>)/sqrt(3) on qubits a, b, c.
    Enforces the one-hot constraint exactly inside the quantum circuit.
    """
    qc.ry(2 * np.arcsin(1.0 / np.sqrt(3)), a)
    qc.append(RYGate(np.pi / 2).control(1, ctrl_state=0), [a, b])
    qc.append(XGate().control(2, ctrl_state="00"), [a, b, c])


def cost_layer(qc: QuantumCircuit, H: SparsePauliOp, gamma: Any) -> None:
    """
    Applies exp(-i * gamma * H) for a Hamiltonian composed of Z and ZZ Pauli terms.
    Uses rz for 1-qubit terms and rzz for 2-qubit interactions.
    """
    for label, coeff in zip(H.paulis.to_labels(), H.coeffs):
        qubits = [q for q, op in enumerate(reversed(label)) if op == "Z"]
        angle = 2.0 * gamma * coeff.real
        if len(qubits) == 1:
            qc.rz(angle, qubits[0])
        elif len(qubits) == 2:
            qc.rzz(angle, qubits[0], qubits[1])


def multi_z_cost_layer(qc: QuantumCircuit, H: SparsePauliOp, gamma: Any) -> None:
    """
    Extends cost_layer to support arbitrary multi-qubit Pauli Z operators (Z, ZZ, ZZZ, ...)
    via a CNOT parity ladder + single RZ rotation. Essential for Direction 2 (compact 3N encoding).
    """
    for label, coeff in zip(H.paulis.to_labels(), H.coeffs):
        qubits = [q for q, op in enumerate(reversed(label)) if op == "Z"]
        if not qubits:
            continue
        angle = 2.0 * gamma * coeff.real
        if len(qubits) == 1:
            qc.rz(angle, qubits[0])
        elif len(qubits) == 2:
            qc.rzz(angle, qubits[0], qubits[1])
        else:
            for i in range(len(qubits) - 1):
                qc.cx(qubits[i], qubits[i + 1])
            qc.rz(angle, qubits[-1])
            for i in reversed(range(len(qubits) - 1)):
                qc.cx(qubits[i], qubits[i + 1])


def xy_mixer(qc: QuantumCircuit, beta: Any, cubes: Sequence[Sequence[str]]) -> None:
    """
    Applies intra-cube XY mixers using XXPlusYYGate(2 * beta) across all pairs of each cube's
    three qubits. Preserves total excitation number (Hamming weight = 1).
    """
    for c in range(len(cubes)):
        q = [3 * c, 3 * c + 1, 3 * c + 2]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            qc.append(XXPlusYYGate(2 * beta), [q[i], q[j]])


def standard_qaoa_circuit(
    H: SparsePauliOp, p_layers: int = 1
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Builds a standard textbook QAOA circuit:
    - Equal superposition across all qubits using Hadamard (H)
    - p_layers of alternating cost_layer and transverse-field mixer (rx)
    """
    gammas = ParameterVector("gamma", p_layers)
    betas = ParameterVector("beta", p_layers)
    qc = QuantumCircuit(H.num_qubits)
    qc.h(range(H.num_qubits))

    for k in range(p_layers):
        cost_layer(qc, H, gammas[k])
        qc.rx(2 * betas[k], range(H.num_qubits))

    return qc, gammas, betas


def constrained_qaoa(
    H: SparsePauliOp,
    gammas: Sequence[float],
    betas: Sequence[float],
    cubes: Sequence[Sequence[str]],
) -> QuantumCircuit:
    """
    Constructs the 3N-qubit Constrained QAOA circuit:
    W-state start on each cube triple + alternating cost layers and XY mixers.
    """
    qc = QuantumCircuit(3 * len(cubes))
    for c in range(len(cubes)):
        w_state(qc, 3 * c, 3 * c + 1, 3 * c + 2)

    for g, b in zip(gammas, betas):
        qc.barrier(label="cost")
        cost_layer(qc, H, g)
        qc.barrier(label="mixer")
        xy_mixer(qc, b, cubes)

    return qc


def full_constrained_qaoa(
    H: SparsePauliOp,
    gammas: Sequence[float],
    betas: Sequence[float],
    cubes: Sequence[Sequence[str]],
) -> QuantumCircuit:
    """
    Direction 1: 6N-qubit (24 qubits for N=4) Constrained QAOA circuit.
    Initializes two W-states per cube (Front/Back triple and Left/Right triple),
    then applies alternating cost steps and XY-mixers on all triples.
    """
    n_cubes = len(cubes)
    qc = QuantumCircuit(6 * n_cubes)

    # 1. Initialize two W-states per cube
    for c in range(n_cubes):
        w_state(qc, 6 * c, 6 * c + 1, 6 * c + 2)
        w_state(qc, 6 * c + 3, 6 * c + 4, 6 * c + 5)

    # 2. Alternating cost and XY mixer layers
    for g, b in zip(gammas, betas):
        qc.barrier(label="cost")
        cost_layer(qc, H, g)
        qc.barrier(label="mixer")
        for c in range(n_cubes):
            for offset in (6 * c, 6 * c + 3):
                q = [offset, offset + 1, offset + 2]
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    qc.append(XXPlusYYGate(2 * b), [q[i], q[j]])

    return qc


def lr_qaoa_circuit(
    H_lr: SparsePauliOp, gammas: Sequence[float], betas: Sequence[float]
) -> QuantumCircuit:
    """
    Constructs the Stage-2 N-qubit QAOA circuit for Left/Right pair selection.
    """
    qc = QuantumCircuit(H_lr.num_qubits)
    qc.h(range(H_lr.num_qubits))
    for g, b in zip(gammas, betas):
        cost_layer(qc, H_lr, g)
        qc.rx(2 * b, range(H_lr.num_qubits))
    return qc


def constrained_vqe_ansatz(
    thetas: Sequence[float], cubes: Sequence[Sequence[str]], layers: int = 2
) -> QuantumCircuit:
    """
    Direction 4: Symmetry-preserving VQE ansatz.
    Starts with W-states, entangles adjacent cubes with CZ gates (preserving Hamming weight),
    and applies independent parameterized XY rotations per cube pair.
    """
    n_cubes = len(cubes)
    qc = QuantumCircuit(3 * n_cubes)
    for c in range(n_cubes):
        w_state(qc, 3 * c, 3 * c + 1, 3 * c + 2)

    idx = 0
    for _ in range(layers):
        for c in range(n_cubes):
            next_c = (c + 1) % n_cubes
            for p in range(3):
                qc.cz(3 * c + p, 3 * next_c + p)
        for c in range(n_cubes):
            q = [3 * c, 3 * c + 1, 3 * c + 2]
            for i, j in ((0, 1), (0, 2), (1, 2)):
                qc.append(XXPlusYYGate(thetas[idx]), [q[i], q[j]])
                idx += 1
    return qc


def get_circuit_metrics(
    qc: QuantumCircuit,
    basis_gates: Optional[List[str]] = None,
    seed_transpiler: int = 7,
) -> Dict[str, Any]:
    """
    Standardizes circuit metrics (qubits, depth, and two-qubit CX gate count)
    after transpilation to basis_gates=["rz", "sx", "x", "cx"], as required by GRADING.md.
    """
    if basis_gates is None:
        basis_gates = ["rz", "sx", "x", "cx"]
    transpiled = transpile(
        qc, basis_gates=basis_gates, seed_transpiler=seed_transpiler
    )
    ops = transpiled.count_ops()
    return {
        "qubits": transpiled.num_qubits,
        "depth": transpiled.depth(),
        "two_qubit_gates": ops.get("cx", 0),
        "total_ops": dict(ops),
    }
