# Instant Insanity on a Quantum Computer
### UConn Fall Fest 2026 · IBM Qiskit & qBraid Challenge
**Team Name:** Team Phase Flip (`IBM-Qiskit-Team-Phase-Flip`)  
**Team Members:** Santiago Arthur ([@scarthur](https://github.com/scarthur)), Cınar TG ([@cinartg](https://github.com/cinartg))

---

## Overview & Level Reached

We reached **all three levels**: **Warm-Up**, **01 Core**, and **02 Bonus (Full Scaling to $N=5, 6, 7, 8$ Cubes & Colours)**.

Our solution implements constrained Quantum Approximate Optimization Algorithm (QAOA) techniques using Qiskit 2.x and qBraid. Every QAOA circuit in the main notebook, [`solution12.ipynb`](solution12.ipynb), uses **p = 12 layers**, with all 24 angles optimized directly for the success probability:
1. **Warm-Up (12 Qubits):** Constrained $W$-state initialization with $XY$-mixers ($XX+YY$) for the front/back 2-factor, reaching **89.9%** sampled success (exact 91.1%), $18\times$ over random one-hot guessing (4/81 = 4.94%).
2. **01 Core (Full 4-Cube Tower):** 
   - **Classical Verification:** The 24-qubit Hamiltonian equals the classical objective `cost(CUBES, x, y)` on all 1,296 valid configurations, and exactly 2 of them have zero cost (the tower and its quarter turn).
   - **Method A (24-Qubit QAOA):** Full 24-qubit Double $W$-state + $XY$-mixer QAOA achieving **19.55%** success rate on the complete tower (391/2000 shots; exact 18.45%). That is about $640\times$ the random $W$-subspace baseline (2/6561 = 0.0305%) and about $1.6$ million $\times$ the unconstrained baseline ($2/2^{24}$).
   - **Method B (Two-Stage Hybrid QAOA):** 12-qubit front/back QAOA followed by a 4-qubit left/right QAOA for the remaining degrees of freedom, achieving **89.9%** in Stage 1 and **92.1%** in Stage 2 (random: 4.94% and 1/16 = 6.25%), yielding a verified solved tower.
3. **02 Bonus (Generalization to $N$ Cubes & Colours):** Generalization across planted-solution puzzles ($N \in \{4, 5, 6, 7, 8\}$, seeds `{5: 16, 6: 86, 7: 68, 8: 4}`), reporting depth, CX count, and success rate scaling against exponential random baselines. A solved tower is found and checked for every $N$.

### What changed from our first version (p = 2 → p = 12)

Our first notebook, [`solution.ipynb`](solution.ipynb), used p = 2 with COBYLA on the average cost. `solution12.ipynb` keeps the same encodings and adds:
- **p = 12 layers** (`P = 12`), optimized with L-BFGS-B to maximize **P(lowest cost)**, the number we report, from 6 annealing-ramp starting points (γ grows, β shrinks) with a fixed seed.
- **An exact one-hot-subspace simulator.** The $W$-state start and $XY$ mixer keep every 3-qubit group one-hot, so the state lives on $3^{\text{groups}}$ basis states: 81 for the 12-qubit warm-up and 6,561 for the 24-qubit core, instead of $2^{24} \approx 16.8$ million amplitudes. Gradients come from the adjoint method (one forward and one backward pass for all 24 angles).
- **A one-hot-reduced cost Hamiltonian** (`one_hot_reduce`). Inside a one-hot group $Z_iZ_j = Z_i + Z_j - 1$, so those $ZZ$ terms become single-$Z$ terms and each removed term saves 2 CX: 1,640 → 1,424 CX on 12 qubits and 3,568 → 3,136 CX on 24 qubits.
- **Packed cost layers.** `cost_layer` groups the `rzz` gates into rounds on disjoint qubit pairs, so commuting gates run side by side. The unitary is unchanged.

| Experiment | p = 2 (`solution.ipynb`) | p = 12 (`solution12.ipynb`) |
|---|---|---|
| Warm-up, P(front/back 2-factor) | 28.1% · 300 CX · depth 218 | 89.9% · 1,424 CX · depth 781 |
| Core 24q, P(solved tower) | 1.60% · 648 CX · depth 224 | 19.55% · 3,136 CX · depth 820 |
| Core two-stage, Stage 2 (4q) | 26.1% · 12 CX · depth 29 | 92.1% · 72 CX · depth 136 |
| Bonus N = 8, P(2-factor) | 4.28% · 648 CX · depth 320 | 45.40% · 3,304 CX · depth 888 |

Six times as many layers buys a much higher success rate at roughly 4–6× the two-qubit gate count, which matters for any future run on real hardware.

---

## Repository Layout

```
.
├── README.md              # Team, levels reached, how to run, results table, job IDs, AI use note
├── solution.ipynb         # First version (p = 12), kept for the comparison above
├── requirements.txt       # Environment dependencies (Qiskit 1.3+ / 2.x, qBraid, qiskit-aer)
└── resources/             # Shared starter helpers copied from challenge repository
    ├── __init__.py
    └── helpers.py         # Visualizers, graph net drawings, tower_from/orient, and transpilation helpers
```

---

## How to Run

The primary deliverable is [`solution12.ipynb`](solution12.ipynb). It is self-contained (it imports only `resources/helpers.py`) and saved with all cell outputs, charts, and verification checks visible:
1. Open the repository in qBraid Lab (or JupyterLab).
2. Install the dependencies if your kernel lacks them (the first cell also installs anything missing):
   ```bash
   pip install -r requirements.txt
   ```
3. Open `solution12.ipynb` and select the Python kernel (`Python 3 [UConn 2026]` in qBraid Lab).
4. Run all cells from top to bottom. Headless:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace solution12.ipynb
   ```
   - **Local Mode (Default):** Runs reproducibly via `AerSimulator` with fixed seed (`SEED = 7`). The 24-qubit angle optimization takes about a minute (63 s in the saved run).
   - **qBraid SV1 Mode:** Set `USE_SV1 = True` in the `# Cell 3: qBraid SV1 Runner` cell to submit every circuit to `aws:aws:sim:sv1` through `QbraidProvider()`. Each job ID is printed and stored in `JOB_LOG`. SV1 bills simulation time at 7.5 qBraid credits per minute.

---

## Results Table

All two-qubit gate counts are measured after standard transpilation: `transpile(qc, basis_gates=["rz", "sx", "x", "cx"])`. Every reported tower is verified by the classical `solves()` check.

| Puzzle (N cubes) | Encoding | Device | Qubits | Two-qubit gates (after transpiling) | Depth | Shots | P(correct tower) | Random-guess baseline | qBraid job ID |
|---|---|---|---|---|---|---|---|---|---|
| 4 (Warm-up FB) | 3N W+XY (p=12) | `AerSimulator (local)` | 12 | 1424 | 781 | 1000 | 89.9% (FB 2-factor) | 4.94% (4/81 W) | `local-aer` |
| 4 (Core Full 24q) | 6N Double W+XY (p=12) | `AerSimulator (local)` | 24 | 3136 | 820 | 2000 | 19.55% | 0.0305% (2/6561 W) / 0.000012% (2/2^24) | `local-aer` |
| 4 (Core 2-Stage) | 12q W+XY (FB) + 4q QAOA (LR) (p=12) | `AerSimulator (local)` | 12 + 4 | 1424 + 72 | 781 + 136 | 1000 + 1000 | 89.9% FB / 92.1% LR | 4.94% FB / 6.25% LR; 0.154% overall (2/1296) | `local-aer` |
| 4 (CUBES) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 12 | 1424 | 781 | 1000 | 89.90% (2-factor, solved=True) | 4.938% (4/3^N) / 0.1543% (2/6^N) | `local-aer` |
| 5 (seed=16) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 15 | 1888 | 779 | 1000 | 93.00% (2-factor, solved=True) | 2.469% (6/3^N) / 0.0257% (2/6^N) | `local-aer` |
| 6 (seed=86) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 18 | 2304 | 783 | 2000 | 51.70% (2-factor, solved=True) | 0.549% (4/3^N) / 0.0043% (2/6^N) | `local-aer` |
| 7 (seed=68) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 21 | 2864 | 890 | 4000 | 45.75% (2-factor, solved=True) | 0.183% (4/3^N) / 0.0007% (2/6^N) | `local-aer` |
| 8 (seed=4) | 3N W+XY Disjoint/2-Stage (p=12) | `AerSimulator (local)` | 24 | 3304 | 888 | 4000 | 45.40% (2-factor, solved=True) | 0.091% (6/3^N) / 0.0001% (2/6^N) | `local-aer` |

**How to read the table:**
- For the 3N rows, the per-shot number is **P(front/back 2-factor)**, not P(full tower). The tower is then assembled by pairing two sampled 2-factors that share no edge (a classical step). If no such pair is sampled, an N-qubit left/right QAOA runs as Stage 2. `solved=True` means the assembled tower passed `solves()`.
- The `4 (CUBES)` row is the warm-up circuit run again inside the scaling loop, so its numbers match the warm-up row.
- **Random baselines** use the true number of front/back 2-factors in each puzzle (4, 6, 4, 4, 6 for N = 4…8), found by enumerating all $3^N$ front/back picks. The notebook's auto-printed table assumes 2 per puzzle (2/3^N), which understates the baseline by 2–3×. `count_answers` gives exactly 2 solved choices for every puzzle, so 2/6^N is exact.
- In the bonus, the p = 12 angles are re-optimized for each instance.

---

## qBraid Job IDs & Device Execution

- **Local runs:** Every number reported above and in `solution12.ipynb` was generated with `AerSimulator` and `seed_simulator = 7`, recorded as `local-aer`. No SV1 job IDs are listed yet.
- **qBraid Cloud Support:** The notebook includes live integration with qBraid's `aws:aws:sim:sv1` simulator through `QbraidProvider().get_device("aws:aws:sim:sv1")`. Setting `USE_SV1 = True` directs circuit execution to qBraid and records remote job IDs into `JOB_LOG`, which then fill the job ID column of the auto-generated table.

---

## Classical Checks & Verification

As emphasized by the grading rubric (*checks before claims*):
1. **Hamiltonian Energy Equivalence:** We tested the full 24-qubit Hamiltonian against `cost(CUBES, x, y)` across all $3^4 \times 2^4 = 1,296$ valid cube assignments. The maximum absolute gap is **`0.0e+00`** (an `assert` stops the notebook on any mismatch), so the Pauli $Z$ and $ZZ$ mapping preserves the cost landscape exactly.
2. **Solution Set Verification:** The Hamiltonian has exactly 2 zero-cost ground states in the feasible space, matching the two known solutions for the puzzle, and each one passes `solves()`.
3. **Simulator and Optimizer Checks:** The one-hot-subspace simulator matches Qiskit's `Statevector` on the real 12-qubit, p = 12 circuit (max $|\Delta P| = 4.9\times10^{-16}$; probability outside the one-hot subspace $7.5\times10^{-15}$). The adjoint gradient matches central finite differences (max error $2.3\times10^{-9}$). The one-hot reduction leaves every subspace energy unchanged (asserted for 12 and 24 qubits), and the optimizer's exact warm-up probability matches `Statevector` of the circuit we actually run.
4. **Independent Solution Verification:** Decoded bitstrings are checked against the independent classical validator `solves(tower)`, shot by shot. Decoders read Qiskit's little-endian order (qubit 0 is the rightmost character). For both Method A (24q full) and Method B (two-stage), the decoded configurations produce the solved tower, printed as (front, right, back, left) per cube, top cube first:
   ```python
   [('G', 'B', 'Y', 'R'), ('Y', 'Y', 'B', 'G'), ('B', 'R', 'R', 'Y'), ('R', 'G', 'G', 'B')]
   ```
5. **Classical Comparison:** Classical brute force (`count_answers`) checks all 1,296 choices of the four-cube puzzle in under 0.01 s, and all 1,679,616 choices for N = 8 in about 4 s on one CPU core. We do not claim a quantum speedup; the results show how far the constrained QAOA concentrates probability on solutions compared with random guessing, and what it costs in gates.

---

## AI Use Note

In accordance with Deliverable 5 and the challenge guidelines:
- **Tools Used:** We utilized **Argon & Claude** (paired with Antigravity and CC CLI).
- **Application Areas:**
  1. *Encoding & Circuit Synthesis:* Structuring the $W$-state preparation circuit and $XX+YY$ parameterization for preserving the 1-hot constraint per cube, eliminating infeasible Hamming weights.
  2. *Hamiltonian Formulation:* Translating the $N$-cube graph-degree matching penalties into Pauli operator representations for both one-hot and two-stage decompositions.
  3. *Benchmarking & Generalization:* Building the automated evaluation loop for bonus puzzles ($N=5..8$) using the starter's `make_puzzle` and plotting scaling performance.
  4. *Write-up:* This README was regenerated with Claude Code from the saved outputs of both notebooks.
- **Verification Methodology:**
  - Every formulation proposed by the AI was benchmarked against brute-force classical search over the 1,296 configurations to verify zero energy discrepancies.
  - The fast p = 12 optimizer was checked against Qiskit's `Statevector` and its gradient against finite differences before we trusted it.
  - Bitstring decoders were audited for Qiskit little-endian ordering to prevent indexing inversions.
  - All quantum outputs were verified by classical graph and tower checkers before reporting.
  - Random-guess baselines were recomputed by enumerating every front/back pick of every puzzle.
