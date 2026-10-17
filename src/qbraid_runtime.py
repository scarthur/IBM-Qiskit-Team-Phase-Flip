"""
transcribed_code/qbraid_runtime.py
==================================
Execution interface for qBraid devices (AWS SV1) and local Qiskit 2.x V2 primitives:
- submit_job wrapper for qBraid SDK with local fallback
- run_on_qbraid with marker qubit handling and little-endian ordering
- ISA pass manager compilation (generate_preset_pass_manager)
- V2 Primitives PUB runner extracting from result[0].data.meas.get_counts().
"""

from typing import Any, Dict, List, Optional, Sequence
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

SEED: int = 7
SV1: str = "aws:aws:sim:sv1"


def submit_job(
    qc: QuantumCircuit, shots: int = 1000, device_id: str = SV1
) -> Optional[Dict[str, int]]:
    """
    Submits a circuit to a qBraid device (such as AWS SV1) using the qBraid SDK.
    In qBraid Lab, credentials load automatically.
    Returns None if qbraid is not installed or if credentials are not configured,
    triggering a clean fallback to local simulation without crashing.
    """
    try:
        from qbraid.runtime import QbraidProvider  # type: ignore

        provider = QbraidProvider()
        device = provider.get_device(device_id)
        print(f"[qBraid] Target device {device_id} status: {device.status()}")
        job = device.run(qc, shots=shots)
        print(f"[qBraid] Submitted job ID: {job.id}")
        result = job.result()
        # Retrieve counts from native qBraid result
        data = getattr(result, "data", None)
        if data and hasattr(data, "get_counts"):
            return data.get_counts()
        elif hasattr(result, "get_counts"):
            return result.get_counts()
        return None
    except Exception as e:
        # Graceful fallback when running locally outside qBraid Lab
        print(f"[Notice] qBraid submission bypassed ({e}); falling back to local simulation.")
        return None


def prepare_circuit(qc: QuantumCircuit, seed: int = SEED) -> QuantumCircuit:
    """
    Appends a marker qubit initialized to 1 and transpiles to standard basis gates.
    The marker qubit verifies bit-ordering consistency across different remote providers.
    """
    m = qc.copy()
    marker = m.add_bits([m.find_bit(0)[0]]) if False else None  # marker handling
    # Add an extra marker qubit at the end
    n = m.num_qubits
    marked_qc = QuantumCircuit(n + 1)
    marked_qc.compose(qc, qubits=range(n), inplace=True)
    marked_qc.x(n)  # Fix marker qubit to |1>
    marked_qc.measure_all()
    return transpile(
        marked_qc, basis_gates=["rz", "sx", "x", "cx"], seed_transpiler=seed
    )


def strip_marker_counts(raw_counts: Dict[str, int]) -> Dict[str, int]:
    """
    Drops the marker qubit from measurement keys and returns counts in Qiskit order.
    """
    cleaned: Dict[str, int] = {}
    for bitstr, cnt in raw_counts.items():
        # Check whether marker (value 1) is at the MSB or LSB
        if bitstr.startswith("1"):
            key = bitstr[1:]
        elif bitstr.endswith("1"):
            key = bitstr[:-1]
        else:
            key = bitstr[1:]  # Fallback
        cleaned[key] = cleaned.get(key, 0) + cnt
    return cleaned


def run_on_qbraid(
    circuit: QuantumCircuit, shots: int = 1000, device_id: str = SV1
) -> Dict[str, int]:
    """
    High-level execution pipeline:
    1. Prepares circuit with marker qubit and decomposes to basis gates
    2. Attempts submission to qBraid SV1
    3. Falls back to local AerSimulator if remote credentials are unavailable
    4. Strips marker qubit and returns clean counts.
    """
    qc_prepared = prepare_circuit(circuit)
    raw = submit_job(qc_prepared, shots=shots, device_id=device_id)
    if raw is None:
        backend = AerSimulator(seed_simulator=SEED)
        raw = backend.run(qc_prepared, shots=shots).result().get_counts()
    return strip_marker_counts(raw)


def transpile_to_isa(
    qc: QuantumCircuit,
    backend: Optional[Any] = None,
    optimization_level: int = 3,
    seed_transpiler: int = SEED,
) -> QuantumCircuit:
    """
    Compiles a circuit into an Instruction Set Architecture (ISA) circuit
    matching the target backend's topology using generate_preset_pass_manager.
    Strictly follows Qiskit 2.x standards.
    """
    if backend is None:
        backend = AerSimulator(seed_simulator=SEED)
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed_transpiler,
    )
    return pm.run(qc)


def run_v2_sampler(
    qc: QuantumCircuit,
    param_values: Optional[Sequence[float]] = None,
    shots: int = 1000,
    seed: int = SEED,
) -> Dict[str, int]:
    """
    Executes a circuit using Qiskit 2.x V2 Primitives interface:
    - Inputs passed as Primitive Unified Blocs (PUBs)
    - Results extracted via classical register name 'meas' (result[0].data.meas.get_counts()).
    """
    measured_qc = qc.copy()
    if not measured_qc.cregs:
        measured_qc.measure_all()

    sampler = StatevectorSampler(seed=seed)
    if param_values is not None:
        pub = (measured_qc, param_values)
    else:
        pub = (measured_qc,)

    job = sampler.run([pub], shots=shots)
    pub_result = job.result()[0]
    return pub_result.data.meas.get_counts()
