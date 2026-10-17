# Instant Insanity on a Quantum Computer
### UConn Fall Fest 2026 · IBM Qiskit & qBraid Challenge
**Team Name:** Team Phase Flip (`IBM-Qiskit-Team-Phase-Flip`)  
**Team Members:** Santiago Arthur ([@scarthur](https://github.com/scarthur)), Cınar TG ([@cinartg](https://github.com/cinartg))

---

## Overview & Level Reached

We reached **all three levels**: **Warm-Up**, **01 Core**, and **02 Bonus (Full Scaling to $N=5, 6, 7, 8$ Cubes & Colours)**.

Our solution implements constrained Quantum Approximate Optimization Algorithm (QAOA) techniques using Qiskit 2.x and qBraid:
1. **Warm-Up (12 Qubits):** Constrained $W$-state initialization with $XY$-mixers ($XX+YY$) for the front/back 2-factor, reaching **28.2%** success rate ($>5.7\times$ over random guessing).
2. **01 Core (Full 4-Cube Tower):** 
   - **Classical Verification:** Complete verification of the 24-qubit Hamiltonian against the classical objective over all 1,296 valid configurations, proving an exact energy gap of **0.0e+00**.
   - **Method A (24-Qubit QAOA):** Full 24-qubit Double $W$-state + $XY$-mixer QAOA achieving **1.60%** success rate on the complete tower ($>50\times$ over the random $W$-subspace baseline $0.0305\%$ and $>130,000\times$ over the unconstrained $2/2^{24}$ space).
   - **Method B (Two-Stage Hybrid QAOA):** 12-qubit front/back QAOA followed by a 4-qubit left/right QAOA for the remaining degrees of freedom, achieving **28.2%** in Stage 1 and **26.2%** in Stage 2, yielding a verified solved tower.
3. **02 Bonus (Generalization to $N$ Cubes & Colours):** Generalization across planted-solution puzzles ($N \in \{4, 5, 6, 7, 8\}$), reporting depth, CX count, and success rate scaling against exponential random baselines.

---

## Repository Layout

```
.
├── README.md              # Team, levels reached, how to run, results table, job IDs, AI use note
├── solution.ipynb         # Self-contained Jupyter notebook, saved with all outputs & plots
├── requirements.txt       # Environment dependencies (Qiskit 2.x, qBraid, qiskit-aer)
├── src/                   # Modular Python package for Hamiltonians, circuits, solvers, and pipeline
│   ├── __init__.py
│   ├── puzzle_utils.py    # Instant Insanity definitions, checks, and planted puzzle generators
│   ├── hamiltonians.py    # QUBO, 24-qubit, 12-qubit, and left/right dynamic Hamiltonians
│   ├── circuits.py        # W-state prep, XX+YY mixers, QAOA ansatzes, transpile metrics
│   ├── decoders.py        # Little-endian bitstring decoders and 2-factor pairing logic
│   ├── solvers.py         # Full 24q, compact 12q, and two-stage hybrid solvers
│   ├── qbraid_runtime.py  # qBraid device submission wrapper with local Aer fallback
│   ├── benchmarks.py      # Resource counting and N-puzzle scaling benchmarks
│   └── run_pipeline.py    # End-to-end command-line verification pipeline
└── resources/             # Shared starter helpers copied from challenge repository
    ├── __init__.py
    └── helpers.py         # Visualizers, graph net drawings, and transpilation helpers
```

---

## How to Run

### Option 1: Run the Complete Notebook (`solution.ipynb`)
The primary deliverable is [`solution.ipynb`](solution.ipynb). It is completely self-contained and pre-executed with all cell outputs, charts, and verification checks visible:
1. Launch JupyterLab or open the repository in qBraid Lab.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `solution.ipynb` and select the Python kernel.
4. Run all cells from top to bottom.
   - **Local Mode (Default):** Runs reproducibly via `AerSimulator` with fixed seed (`SEED = 7`).
   - **qBraid SV1 Mode:** Set `USE_SV1 = True` in Cell 3 to submit jobs to `aws:aws:sim:sv1` via the configured `QbraidProvider()`.

### Option 2: Run the Command-Line Verification Pipeline
You can also run the modular package verification via terminal:
```bash
python -m src.run_pipeline
```
This runs classical verification across all 1,296 states, tests circuit transpilation and gate counts, solves the puzzle, and validates the $N=5$ bonus problem.

---

## Results Table

All two-qubit gate counts are measured after standard transpilation: `transpile(qc, basis_gates=["rz", "sx", "x", "cx"])`. Every reported tower is strictly verified by classical `solves()`.

| Puzzle (N cubes) | Encoding | Device | Qubits | Two-qubit gates (after transpiling) | Depth | Shots | P(correct tower) | Random-guess baseline | qBraid job ID |
|---|---|---|---|---|---|---|---|---|---|
| 4 (Warm-up FB) | 3N W+XY (p=12) | `AerSimulator (local)` | 12 | 1424 | 781 | 1000 | 89.9% (FB 2-factor) | 4.94% (4/81 W) | `local-aer` |
| 4 (Core Full 24q) | 6N Double W+XY (p=12) | `AerSimulator (local)` | 24 | 3136 | 820 | 2000 | 19.55% | 0.0305% (2/6561 W) / 0.000012% (2/2^24) | `local-aer` |
| 4 (Core 2-Stage) | 12q W+XY (FB) + 4q QAOA (LR) (p=12) | `AerSimulator (local)` | 12 + 4 | 1424 + 72 | 781 + 136 | 1000 | 89.9% FB / 92.1% LR | 0.154% (2/1296 pairs) | `local-aer` |
| 4 (CUBES) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 12 | 1424 | 781 | 1000 | 89.90% (2-factor, solved=True) | 2.469% (2/3^N) / 0.1543% (2/6^N) | `local-aer` |
| 5 (seed=16) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 15 | 1888 | 779 | 1000 | 93.00% (2-factor, solved=True) | 0.823% (2/3^N) / 0.0257% (2/6^N) | `local-aer` |
| 6 (seed=86) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 18 | 2304 | 783 | 2000 | 51.70% (2-factor, solved=True) | 0.274% (2/3^N) / 0.0043% (2/6^N) | `local-aer` |
| 7 (seed=68) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 21 | 2864 | 890 | 4000 | 45.75% (2-factor, solved=True) | 0.091% (2/3^N) / 0.0007% (2/6^N) | `local-aer` |
| 8 (seed=4) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 24 | 3304 | 888 | 4000 | 45.40% (2-factor, solved=True) | 0.030% (2/3^N) / 0.0001% (2/6^N) | `local-aer` |

---

## qBraid Job IDs & Device Execution

- **Deterministic Local Baseline:** To guarantee consistent grading and review without consuming runtime credits during offline evaluation, the reported numbers in `solution.ipynb` were generated using `AerSimulator` with `seed_simulator = 7` (recorded as `local-aer`).
- **qBraid Cloud Support:** The notebook includes live integration with qBraid's `aws:aws:sim:sv1` simulator through `QbraidProvider().get_device("aws:aws:sim:sv1")`. Setting `USE_SV1 = True` in Cell 3 directs circuit execution to qBraid and records remote job IDs into `JOB_LOG`.

---

## Classical Checks & Verification

As emphasized by the grading rubric (*checks before claims*):
1. **Hamiltonian Energy Equivalence:** We tested the full 24-qubit Hamiltonian against `cost(CUBES, x, y)` across all $3^4 \times 2^4 = 1,296$ valid cube assignments. The maximum absolute gap is **`0.0e+00`**, proving that the Pauli $Z$ and $ZZ$ mappings preserve the cost landscape with zero penalty distortions.
2. **Solution Set Verification:** The Hamiltonian has exactly 2 zero-cost ground states in the feasible space, matching the two known solutions for the puzzle.
3. **Independent Solution Verification:** Decoded bitstrings are checked against the independent classical validator `solves(tower)`. For both Method A (24q full) and Method B (two-stage), the decoded configurations produce the solved tower:
   ```python
   [('G', 'Y', 'B', 'R'), ('Y', 'B', 'Y', 'G'), ('B', 'R', 'R', 'Y'), ('R', 'G', 'G', 'B')]
   ```

---

## AI Use Note

In accordance with Deliverable 5 and the challenge guidelines:
- **Tools Used:** We utilized **Argon & Claude** (paired with Antigravity and CC CLI).
- **Application Areas:**
  1. *Encoding & Circuit Synthesis:* Structuring the $W$-state preparation circuit and $XX+YY$ parameterization for preserving the 1-hot constraint per cube, eliminating infeasible Hamming weights.
  2. *Hamiltonian Formulation:* Translating the $N$-cube graph-degree matching penalties into Pauli operator representations for both one-hot and two-stage decompositions.
  3. *Benchmarking & Generalization:* Building the automated evaluation loop for bonus puzzles ($N=5..8$) using the starter's `make_puzzle` and plotting scaling performance.
- **Verification Methodology:**
  - Every formulation proposed by the AI was benchmarked against brute-force classical search over the 1,296 configurations to verify zero energy discrepancies.
  - Bitstring decoders were audited for Qiskit little-endian ordering to prevent indexing inversions.
  - All quantum outputs were verified by classical graph and tower checkers before reporting.
