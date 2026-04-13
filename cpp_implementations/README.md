# C++ QAP Implementations

This directory contains C++ implementations of the Quadratic Assignment Problem algorithms (both foundational and modifications) alongside their automated benchmarking suite.

## Directory Structure
- `qap_solver.cpp`: The core C++ engine. Features:
  - `bb_base`: Standard Branch & Bound with DFS and Gilmore-Lawler Bound via fast $\mathcal{O}(n^3)$ Hungarian method.
  - `bb_novel`: Dynamically expands using Max-Min Branching and bootstraps bounding operations using a Tabu-Search Warm-Start.
  - `ts_base`: Fast implementation of iterative incremental-Delta Tabu Search.
  - `ts_novel`: Features Elite Path Relinking scaling diversification structurally mapped rather than relying on randomized shake, coupled with Reactive Tenure scaling based on collision iterations.
- `Makefile`: Script to build `qap_solver` utilizing `-O3` gcc optimization hooks.
- `run_cpp_benchmarks.py`: Fully automates `qapLIB` instance data ingestion natively bypassing OS manual intervention. Parses solver constraints and exports metrics including runtime, total traversed nodes/iterations, and Peak Child RSS Memory profiles into `results_cpp.csv`. Automatically maps 6 evaluation distributions into the `plots/` subdirectory.


## How to Compile & Run

### 1. Compile the C++ Binary
```bash
make
```

### 2. Manual Execution
You can manually run specific QAP components pointing any local QAPLIB dataset:
```bash
./qap_solver ../data/nug12.dat <algorithm> <limit>
```
- `<algorithm>`: `bb_base`, `bb_novel`, `ts_base`, `ts_novel`
- `<limit>`: For B&B, an absolute execution timeout (in seconds). For Tabu Search, an absolute iteration bound limit.

**Example for BB:** ` ./qap_solver ../data/chr15a.dat bb_novel 10`
**Example for TS:** ` ./qap_solver ../data/tai30a.dat ts_novel 5000`

### 3. Automated Benchmarking
Generate the entire comparative pipeline directly using Python (requires `matplotlib` and `numpy` optionally through a `venv`):
```bash
python run_cpp_benchmarks.py
```
Outputs will populate `results_cpp.csv` alongside `.pdf` visual comparisons mapping algorithmic superiority under uniform operational restraints natively enclosed within `/plots`.
