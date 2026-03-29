#!/usr/bin/env python3
"""Master script: generate data, run benchmarks, produce plots."""

import os
import sys
import csv
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

sys.path.insert(0, BASE_DIR)


def step1_check_data():
    """Check that QAPLIB instances exist, downloading missing files."""
    print("\n" + "=" * 60)
    print("STEP 1: Checking QAPLIB instances")
    print("=" * 60)
    os.makedirs(DATA_DIR, exist_ok=True)

    base_urls = [
        "https://qaplib.mgi.polymtl.ca/data.d",
        "http://qaplib.mgi.polymtl.ca/data.d",
        "https://anjos.mgi.polymtl.ca/qaplib/data.d",
        "http://anjos.mgi.polymtl.ca/qaplib/data.d",
    ]

    def download_instance(instance_name, out_path):
        for base_url in base_urls:
            url = f"{base_url}/{instance_name}.dat"
            try:
                urllib.request.urlretrieve(url, out_path)
                return url
            except Exception:
                continue
        return None

    required = ["nug12","chr12a","tai12a","nug14","chr15a","tai15a",
                "nug20","tai20a","tai25a","nug25","nug30","tai30a"]
    for name in required:
        path = os.path.join(DATA_DIR, name + ".dat")
        if os.path.exists(path):
            print(f"  {name}.dat  OK")
        else:
            print(f"  {name}.dat  MISSING - downloading...")
            source_url = download_instance(name, path)
            if source_url and os.path.exists(path):
                print(f"  {name}.dat  DOWNLOADED ({source_url})")
            else:
                print(f"  {name}.dat  FAILED - could not download from QAPLIB")


def step2_run_benchmarks():
    """Run all algorithms on all instances."""
    print("\n" + "=" * 60)
    print("STEP 2: Running benchmarks")
    print("=" * 60)
    from experiments.benchmarks import run_benchmarks
    results, convergence_data = run_benchmarks(
        data_dir=DATA_DIR, results_dir=RESULTS_DIR, bb_max_n=14, quick_mode=False
    )
    return results, convergence_data


def load_results_from_csv():
    """Load results from CSV file."""
    results = []
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n"] = int(row["n"])
            row["cost"] = int(row["cost"])
            row["bks"] = int(row.get("bks", row["cost"]))
            row["gap_pct"] = float(row.get("gap_pct", 0))
            row["time"] = float(row["time"])
            row["memory_mb"] = float(row["memory_mb"])
            row["nodes"] = int(row["nodes"])
            results.append(row)
    return results


def load_convergence_from_csv():
    """Load convergence data from CSV file."""
    convergence_data = {}
    conv_path = os.path.join(RESULTS_DIR, "convergence.csv")
    with open(conv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            inst, algo, iteration, cost = row
            key = (inst, algo)
            if key not in convergence_data:
                convergence_data[key] = []
            convergence_data[key].append((int(iteration), int(cost)))
    return convergence_data


def step3_generate_plots(results=None, convergence_data=None):
    """Generate all plots from results."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating plots")
    print("=" * 60)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if results is None:
        results = load_results_from_csv()
    if convergence_data is None:
        convergence_data = load_convergence_from_csv()

    # Color scheme
    colors = {
        "BB-Original": "#2196F3",
        "BB-Improved": "#FF9800",
        "TS-Original": "#4CAF50",
        "TS-Reactive": "#E91E63",
    }

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
    })

    # --- Plot 1: B&B Runtime vs Instance ---
    print("  Generating bb_runtime_vs_n.pdf...")
    fig, ax = plt.subplots()
    bb_instances = sorted(set(
        r["instance"] for r in results if r["algorithm"].startswith("BB")
    ))
    x = np.arange(len(bb_instances))
    width = 0.35
    for i, algo in enumerate(["BB-Original", "BB-Improved"]):
        times = []
        for inst in bb_instances:
            match = [r for r in results if r["instance"] == inst and r["algorithm"] == algo]
            times.append(match[0]["time"] if match else 0)
        ax.bar(x + i * width - width / 2, times, width, label=algo, color=colors[algo])
    ax.set_xlabel("Instance")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Branch & Bound: Runtime per Instance")
    ax.set_xticks(x)
    ax.set_xticklabels(bb_instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    fig.savefig(os.path.join(PLOTS_DIR, "bb_runtime_vs_n.pdf"), bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: B&B Nodes Explored ---
    print("  Generating bb_nodes_explored.pdf...")
    fig, ax = plt.subplots()
    bb_instances = sorted(set(
        r["instance"] for r in results if r["algorithm"].startswith("BB")
    ))
    x = np.arange(len(bb_instances))
    width = 0.35
    for i, algo in enumerate(["BB-Original", "BB-Improved"]):
        nodes = []
        for inst in bb_instances:
            match = [r for r in results if r["instance"] == inst and r["algorithm"] == algo]
            nodes.append(match[0]["nodes"] if match else 0)
        ax.bar(x + i * width - width / 2, nodes, width, label=algo, color=colors[algo])
    ax.set_xlabel("Instance")
    ax.set_ylabel("Nodes Explored")
    ax.set_title("Branch & Bound: Nodes Explored per Instance")
    ax.set_xticks(x)
    ax.set_xticklabels(bb_instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    fig.savefig(os.path.join(PLOTS_DIR, "bb_nodes_explored.pdf"), bbox_inches='tight')
    plt.close(fig)

    # --- Plot 3: TS Convergence (chr15a) ---
    print("  Generating ts_convergence.pdf...")
    fig, ax = plt.subplots()
    for algo in ["TS-Original", "TS-Reactive"]:
        key = ("chr15a", algo)
        if key in convergence_data:
            conv = convergence_data[key]
            iters, costs = zip(*conv)
            ax.plot(iters, costs, '-', label=algo, color=colors[algo], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost")
    ax.set_title("Tabu Search Convergence on chr15a")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOTS_DIR, "ts_convergence.pdf"), bbox_inches='tight')
    plt.close(fig)

    # --- Plot 4: TS Runtime vs Problem Size (avg over instances per n) ---
    print("  Generating ts_runtime_vs_n.pdf...")
    fig, ax = plt.subplots()
    for algo in ["TS-Original", "TS-Reactive"]:
        from collections import defaultdict
        by_n = defaultdict(list)
        for r in results:
            if r["algorithm"] == algo:
                by_n[r["n"]].append(r["time"])
        ns = sorted(by_n)
        times = [sum(by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, times, 'o-', label=algo, color=colors[algo], linewidth=2, markersize=6)
    ax.set_xlabel("Problem Size (n)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Tabu Search: Runtime vs Problem Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOTS_DIR, "ts_runtime_vs_n.pdf"), bbox_inches='tight')
    plt.close(fig)

    # --- Plot 5: TS Gap Comparison (using BKS from QAPLIB) ---
    print("  Generating ts_gap_comparison.pdf...")
    fig, ax = plt.subplots()
    ts_instances = sorted(set(
        r["instance"] for r in results if r["algorithm"].startswith("TS")
    ))

    x = np.arange(len(ts_instances))
    width = 0.35
    for i, algo in enumerate(["TS-Original", "TS-Reactive"]):
        gaps = []
        for inst in ts_instances:
            match = [r for r in results if r["instance"] == inst and r["algorithm"] == algo]
            if match:
                gaps.append(match[0].get("gap_pct", 0))
            else:
                gaps.append(0)
        ax.bar(x + i * width - width / 2, gaps, width, label=algo, color=colors[algo])
    ax.set_xlabel("Instance")
    ax.set_ylabel("Gap to Best Known (%)")
    ax.set_title("Tabu Search: Solution Quality Gap")
    ax.set_xticks(x)
    ax.set_xticklabels(ts_instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(os.path.join(PLOTS_DIR, "ts_gap_comparison.pdf"), bbox_inches='tight')
    plt.close(fig)

    # --- Plot 6: Memory Comparison ---
    print("  Generating memory_comparison.pdf...")
    fig, ax = plt.subplots()
    all_instances = sorted(set(r["instance"] for r in results))
    all_algos = ["BB-Original", "BB-Improved", "TS-Original", "TS-Reactive"]
    x = np.arange(len(all_instances))
    width = 0.2
    for i, algo in enumerate(all_algos):
        mems = []
        for inst in all_instances:
            match = [r for r in results if r["instance"] == inst and r["algorithm"] == algo]
            mems.append(match[0]["memory_mb"] if match else 0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, mems, width, label=algo, color=colors[algo])
    ax.set_xlabel("Instance")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(all_instances, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(os.path.join(PLOTS_DIR, "memory_comparison.pdf"), bbox_inches='tight')
    plt.close(fig)

    # --- Plot 7: Algorithm Comparison (Gap % for instances where all 4 ran) ---
    print("  Generating algorithm_comparison.pdf...")
    fig, ax = plt.subplots()
    # Find instances where all 4 algorithms ran
    common_instances = []
    for inst in all_instances:
        algos_for_inst = set(r["algorithm"] for r in results if r["instance"] == inst)
        if all(a in algos_for_inst for a in all_algos):
            common_instances.append(inst)
    common_instances.sort()

    if common_instances:
        x = np.arange(len(common_instances))
        width = 0.2
        for i, algo in enumerate(all_algos):
            gaps = []
            for inst in common_instances:
                match = [r for r in results if r["instance"] == inst and r["algorithm"] == algo]
                gaps.append(match[0].get("gap_pct", 0) if match else 0)
            offset = (i - 1.5) * width
            ax.bar(x + offset, gaps, width, label=algo, color=colors[algo])
        ax.set_xlabel("Instance")
        ax.set_ylabel("Gap to Best Known (%)")
        ax.set_title("Algorithm Comparison: Solution Quality Gap")
        ax.set_xticks(x)
        ax.set_xticklabels(common_instances, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "No instances with all 4 algorithms",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Algorithm Comparison")

    fig.savefig(os.path.join(PLOTS_DIR, "algorithm_comparison.pdf"), bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    step1_check_data()
    results, convergence_data = step2_run_benchmarks()
    step3_generate_plots(results, convergence_data)
    print("\nDone!")
