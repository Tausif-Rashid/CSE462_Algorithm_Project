import os
import subprocess
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
EXEC_PATH = os.path.join(BASE_DIR, "qap_solver")

instances = ["nug12", "chr12a", "tai12a", "nug14", "chr15a", "tai15a", "nug20", "tai20a", "tai25a", "nug25", "nug30", "tai30a"]

# Best Known Solutions from QAPLIB
BKS = {
    "nug12": 578, "chr12a": 9552, "tai12a": 224416, 
    "nug14": 1014, "chr15a": 9896, "tai15a": 388214,
    "nug20": 2570, "tai20a": 703482,
    "tai25a": 1167256, "nug25": 3744,
    "nug30": 6124, "tai30a": 1818146
}

def run_benchmarks():
    results = []
    
    # B&B limits
    bb_timeout = 60.0 
    ts_max_iter = 1000 
    
    for inst in instances:
        n = int("".join(filter(str.isdigit, inst)))
        data_path = os.path.join(DATA_DIR, f"{inst}.dat")
        if not os.path.exists(data_path):
            continue
            
        print(f"Running {inst} (n={n})")
        
        # Branch and Bound 
        if n <= 12:
            for algo in ["bb_base", "bb_novel"]:
                cmd = [EXEC_PATH, data_path, algo, str(bb_timeout)]
                try:
                    out = subprocess.check_output(cmd, text=True).strip()
                    parts = out.split(",")
                    if len(parts) >= 4:
                        cost, time_taken, nodes, memory = parts[0], parts[1], parts[2], parts[3]
                        cost, time_taken, nodes, memory = int(cost), float(time_taken), int(nodes), float(memory)
                        gap = 100.0 * (cost - BKS[inst]) / BKS[inst]
                        results.append({
                            "instance": inst, "n": n, "algorithm": algo,
                            "cost": cost, "time": time_taken, "nodes": nodes, "memory": memory, "gap_pct": gap
                        })
                except Exception as e:
                    pass
                    
        # Tabu Search
        for algo in ["ts_base", "ts_novel"]:
            iters = ts_max_iter * (n // 10)
            cmd = [EXEC_PATH, data_path, algo, str(iters)]
            try:
                out = subprocess.check_output(cmd, text=True).strip()
                parts = out.split(",")
                if len(parts) >= 4:
                    cost, time_taken, iter_count, memory = parts[0], parts[1], parts[2], parts[3]
                    cost, time_taken, iter_count, memory = int(cost), float(time_taken), int(iter_count), float(memory)
                    gap = 100.0 * (cost - BKS[inst]) / BKS[inst]
                    results.append({
                        "instance": inst, "n": n, "algorithm": algo,
                        "cost": cost, "time": time_taken, "nodes": iter_count, "memory": memory, "gap_pct": gap 
                    })
            except Exception as e:
                 pass

    # Write CSV
    with open(os.path.join(BASE_DIR, "results_cpp.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "n", "algorithm", "cost", "time", "nodes", "memory", "gap_pct"])
        writer.writeheader()
        writer.writerows(results)
    
    return results

def generate_plots(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = {"bb_base": "#2196F3", "bb_novel": "#FF9800", "ts_base": "#4CAF50", "ts_novel": "#E91E63"}
    plt.rcParams.update({'font.size': 11, 'figure.figsize': (8, 5), 'figure.dpi': 150})
    
    # 1. B&B Graphs
    bb_res = [r for r in results if r["algorithm"].startswith("bb")]
    if bb_res:
        bb_insts = sorted(list(set(r["instance"] for r in bb_res)))
        x = np.arange(len(bb_insts))
        width = 0.35
        
        # Gap plot removed since B&B is an exact algorithm
        
        # Nodes Explored
        fig, ax = plt.subplots()
        for i, algo in enumerate(["bb_base", "bb_novel"]):
            nodes = [next((r["nodes"] for r in bb_res if r["instance"] == inst and r["algorithm"] == algo), 0) for inst in bb_insts]
            ax.bar(x + i*width - width/2, nodes, width, label=algo, color=colors[algo])
        ax.set_xlabel("Instance")
        ax.set_ylabel("Nodes Explored")
        ax.set_title("Branch & Bound: Nodes Explored (Timeout 5s)")
        ax.set_xticks(x)
        ax.set_xticklabels(bb_insts, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        fig.savefig(os.path.join(PLOTS_DIR, "bb_nodes_cpp.pdf"), bbox_inches='tight')
        plt.close(fig)

        # Runtime
        fig, ax = plt.subplots()
        for i, algo in enumerate(["bb_base", "bb_novel"]):
            times = [next((r["time"] for r in bb_res if r["instance"] == inst and r["algorithm"] == algo), 0) for inst in bb_insts]
            ax.bar(x + i*width - width/2, times, width, label=algo, color=colors[algo])
        ax.set_xlabel("Instance")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Branch & Bound: Runtime (Timeout 5s)")
        ax.set_xticks(x)
        ax.set_xticklabels(bb_insts, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        fig.savefig(os.path.join(PLOTS_DIR, "bb_runtime_cpp.pdf"), bbox_inches='tight')
        plt.close(fig)

    # 2. TS Graphs
    ts_res = [r for r in results if r["algorithm"].startswith("ts")]
    if ts_res:
        ts_insts = sorted(list(set(r["instance"] for r in ts_res)))
        x = np.arange(len(ts_insts))
        width = 0.35
        
        # Gap%
        fig, ax = plt.subplots()
        for i, algo in enumerate(["ts_base", "ts_novel"]):
            gaps = [next((r["gap_pct"] for r in ts_res if r["instance"] == inst and r["algorithm"] == algo), 0) for inst in ts_insts]
            ax.bar(x + i*width - width/2, gaps, width, label=algo, color=colors[algo])
        ax.set_xlabel("Instance")
        ax.set_ylabel("Optimality Gap (%)")
        ax.set_title("Tabu Search: Optimality Gap Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(ts_insts, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.savefig(os.path.join(PLOTS_DIR, "ts_gap_cpp.pdf"), bbox_inches='tight')
        plt.close(fig)
        
        # Runtime
        fig, ax = plt.subplots()
        for i, algo in enumerate(["ts_base", "ts_novel"]):
            times = [next((r["time"] for r in ts_res if r["instance"] == inst and r["algorithm"] == algo), 0) for inst in ts_insts]
            ax.plot(x, times, 'o-', label=algo, color=colors[algo], linewidth=2)
        ax.set_xlabel("Instance")
        ax.set_ylabel("Time (s)")
        ax.set_title("Tabu Search: Scalability and Runtime")
        ax.set_xticks(x)
        ax.set_xticklabels(ts_insts, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(PLOTS_DIR, "ts_time_cpp.pdf"), bbox_inches='tight')
        plt.close(fig)

    # 3. Memory Comparison
    fig, ax = plt.subplots()
    all_instances = sorted(list(set(r["instance"] for r in results)))
    all_algos = ["bb_base", "bb_novel", "ts_base", "ts_novel"]
    x = np.arange(len(all_instances))
    width = 0.2
    for i, algo in enumerate(all_algos):
        mems = [next((r["memory"] for r in results if r["instance"] == inst and r["algorithm"] == algo), 0) for inst in all_instances]
        offset = (i - 1.5) * width
        ax.bar(x + offset, mems, width, label=algo, color=colors[algo])
    ax.set_xlabel("Instance")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(all_instances, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(os.path.join(PLOTS_DIR, "memory_comparison_cpp.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 4. Algorithm Comparison Gap
    fig, ax = plt.subplots()
    common_instances = []
    for inst in all_instances:
        algos_for_inst = set(r["algorithm"] for r in results if r["instance"] == inst)
        if all(a in all_algos for a in algos_for_inst):
            common_instances.append(inst)
    if common_instances:
        x_c = np.arange(len(common_instances))
        for i, algo in enumerate(all_algos):
            gaps = [next((r["gap_pct"] for r in results if r["instance"] == inst and r["algorithm"] == algo), 0) for inst in common_instances]
            offset = (i - 1.5) * width
            ax.bar(x_c + offset, gaps, width, label=algo, color=colors[algo])
        ax.set_xlabel("Instance")
        ax.set_ylabel("Optimality Gap (%)")
        ax.set_title("Algorithm Comparison: Solution Quality Gap")
        ax.set_xticks(x_c)
        ax.set_xticklabels(common_instances, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        fig.savefig(os.path.join(PLOTS_DIR, "algorithm_comparison_cpp.pdf"), bbox_inches='tight')
        plt.close(fig)

    # TS Convergence removed as per request

if __name__ == "__main__":
    results = run_benchmarks()
    generate_plots(results)
    print("Done generating plots.")
