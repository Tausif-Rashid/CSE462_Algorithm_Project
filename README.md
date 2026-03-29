# QAP Benchmark Runner

Small project for running QAP benchmarks (Branch & Bound + Tabu Search) and generating result plots.

## Requirements

- Python 3.9+
- numpy
- matplotlib
- SciPy

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

## How to Run

From the project root:

```bash
python main.py
```

What `main.py` does:

1. Checks required QAPLIB `.dat` files and downloads missing ones into `data/`.
2. Runs benchmarks and saves CSV outputs in `results/`.
3. Generates plots in `plots/`.

## Quick Mode Tip (Faster Runs)

To run a smaller instance set (faster), enable quick mode in `main.py` by changing the `quick_mode` argument in `step2_run_benchmarks()`:

```python
results, convergence_data = run_benchmarks(
    data_dir=DATA_DIR, results_dir=RESULTS_DIR, bb_max_n=14, quick_mode=True
)
```

Set it back to `False` for full runs.
