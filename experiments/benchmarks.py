import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.qaplib_parser import parse_qaplib
from experiments.branch_and_bound import branch_and_bound_original, branch_and_bound_improved
from experiments.tabu_search import tabu_search_original, tabu_search_reactive


# Best known solutions from QAPLIB (Burkard, Karisch & Rendl, 1997)
BKS = {
    "nug12": 578, "nug14": 1014, "nug15": 1150, "nug20": 2570,
    "nug25": 3744, "nug30": 6124,
    "chr12a": 9552, "chr15a": 9896,
    "tai12a": 224416, "tai15a": 388214, "tai20a": 703482,
    "tai25a": 1167256, "tai30a": 1818146,
}


def parse_sln(filepath):
    """Parse a QAPLIB .sln file. Returns (n, best_cost, perm_1indexed)."""
    with open(filepath, 'r') as f:
        lines = f.read().split('\n')
    tokens = []
    for line in lines:
        tokens.extend(line.split())
    n = int(tokens[0])
    cost = int(tokens[1])
    perm = [int(x) for x in tokens[2:2+n]]
    return n, cost, perm


def run_benchmarks(data_dir="data", results_dir="results", bb_max_n=14, quick_mode=False):
    """Run all algorithms on QAPLIB instances. B&B only for n <= bb_max_n."""
    os.makedirs(results_dir, exist_ok=True)

    # Use only QAPLIB .dat files (skip random ones)
    qaplib_names = ["nug12", "chr12a", "tai12a", "nug14", "chr15a", "tai15a",
                    "nug20", "tai20a", "tai25a", "nug25", "nug30", "tai30a"]
    qaplib_names_quick = ["nug12", "chr12a", "tai12a", "nug14", "chr15a", "tai15a",
                          "nug20", "tai20a"]
    selected_names = qaplib_names_quick if quick_mode else qaplib_names
    instances = []
    for name in selected_names:
        path = os.path.join(data_dir, name + ".dat")
        if os.path.exists(path):
            instances.append(name)

    results = []
    convergence_data = {}

    for name in instances:
        filepath = os.path.join(data_dir, name + ".dat")
        n, F, D = parse_qaplib(filepath)
        bks = BKS.get(name, None)
        print(f"\n{'=' * 50}")
        print(f"Instance: {name} (n={n}, BKS={bks})")
        print(f"{'=' * 50}")

        # Tabu Search (always run)
        for algo_name, algo_func in [("TS-Original", tabu_search_original),
                                     ("TS-Reactive", tabu_search_reactive)]:
            print(f"  Running {algo_name}...")
            iters = max(3000, n * 150)
            res = algo_func(n, F, D, max_iter=iters)
            gap = 100.0 * (res["cost"] - bks) / bks if bks else 0
            results.append({
                "instance": name, "n": n, "algorithm": algo_name,
                "cost": res["cost"], "bks": bks if bks else res["cost"],
                "gap_pct": round(gap, 3),
                "time": round(res["time"], 4),
                "memory_mb": res["memory_mb"], "nodes": 0
            })
            convergence_data[(name, algo_name)] = res.get("convergence", [])
            print(f"    Cost={res['cost']}, Gap={gap:.2f}%, Time={res['time']:.2f}s")

        # B&B (only for small instances)
        if n <= bb_max_n:
            for algo_name, algo_func in [("BB-Original", branch_and_bound_original),
                                         ("BB-Improved", branch_and_bound_improved)]:
                print(f"  Running {algo_name}...")
                res = algo_func(n, F, D)
                gap = 100.0 * (res["cost"] - bks) / bks if bks else 0
                results.append({
                    "instance": name, "n": n, "algorithm": algo_name,
                    "cost": res["cost"], "bks": bks if bks else res["cost"],
                    "gap_pct": round(gap, 3),
                    "time": round(res["time"], 4),
                    "memory_mb": res["memory_mb"], "nodes": res["nodes"]
                })
                print(f"    Cost={res['cost']}, Gap={gap:.2f}%, Time={res['time']:.2f}s, Nodes={res['nodes']}")

    # Save to CSV
    csv_path = os.path.join(results_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance", "n", "algorithm", "cost", "bks", "gap_pct",
            "time", "memory_mb", "nodes"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Save convergence data
    conv_path = os.path.join(results_dir, "convergence.csv")
    with open(conv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "algorithm", "iteration", "best_cost"])
        for (inst, algo), conv in convergence_data.items():
            for iteration, cost in conv:
                writer.writerow([inst, algo, iteration, cost])

    print(f"\nResults saved to {csv_path}")
    print(f"Convergence data saved to {conv_path}")
    return results, convergence_data


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    run_benchmarks(data_dir=data_dir, results_dir=results_dir)
