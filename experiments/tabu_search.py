import time
import tracemalloc
import random
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.qaplib_parser import qap_cost, delta_cost


def tabu_search_original(n, F, D, max_iter=5000, seed=42):
    """Robust Tabu Search (Taillard 1991 style).

    Fixed tenure, 2-swap neighborhood, aspiration criterion.
    Returns dict with keys: cost, perm, time, memory_mb, convergence.
    """
    tracemalloc.start()
    start_time = time.time()
    rng = random.Random(seed)

    # Initialize with random permutation
    perm = list(range(n))
    rng.shuffle(perm)
    current_cost = qap_cost(perm, F, D)

    best_perm = list(perm)
    best_cost = current_cost

    tenure = max(1, n // 3)
    # tabu_matrix[i][j] = iteration until which swapping i,j is tabu
    tabu_matrix = [[0] * n for _ in range(n)]

    convergence = [(0, best_cost)]

    for iteration in range(1, max_iter + 1):
        best_r, best_s = -1, -1
        best_delta = float('inf')

        for r in range(n - 1):
            for s in range(r + 1, n):
                d = delta_cost(perm, F, D, r, s)
                # Check if move is tabu
                is_tabu = (tabu_matrix[r][s] > iteration)
                # Aspiration: accept tabu move if it gives new global best
                aspiration = (current_cost + d < best_cost)

                if (not is_tabu or aspiration) and d < best_delta:
                    best_delta = d
                    best_r, best_s = r, s

        if best_r == -1:
            # All moves are tabu and none satisfies aspiration; pick least tabu
            min_tabu = float('inf')
            for r in range(n - 1):
                for s in range(r + 1, n):
                    if tabu_matrix[r][s] < min_tabu:
                        min_tabu = tabu_matrix[r][s]
                        best_r, best_s = r, s
            best_delta = delta_cost(perm, F, D, best_r, best_s)

        # Apply the swap
        perm[best_r], perm[best_s] = perm[best_s], perm[best_r]
        current_cost += best_delta
        tabu_matrix[best_r][best_s] = iteration + tenure
        tabu_matrix[best_s][best_r] = iteration + tenure

        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = list(perm)

        if iteration % 100 == 0:
            convergence.append((iteration, best_cost))

    elapsed = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    return {
        "cost": int(best_cost),
        "perm": best_perm,
        "time": elapsed,
        "memory_mb": round(peak_mb, 2),
        "convergence": convergence
    }


def tabu_search_reactive(n, F, D, max_iter=5000, seed=42):
    """Reactive Tabu Search (Battiti & Tecchiolli 1994 style).

    Dynamic tenure, cycling detection, diversification.
    Returns dict with keys: cost, perm, time, memory_mb, convergence.
    """
    tracemalloc.start()
    start_time = time.time()
    rng = random.Random(seed)

    # Initialize with random permutation
    perm = list(range(n))
    rng.shuffle(perm)
    current_cost = qap_cost(perm, F, D)

    best_perm = list(perm)
    best_cost = current_cost

    tenure = max(1, n // 4)
    max_tenure = n
    min_tenure = 1
    tabu_matrix = [[0] * n for _ in range(n)]

    # Cycling detection
    visited_hashes = set()
    visited_hashes.add(hash(tuple(perm)))

    non_improving_count = 0
    diversification_threshold = n * 10

    convergence = [(0, best_cost)]

    for iteration in range(1, max_iter + 1):
        best_r, best_s = -1, -1
        best_delta = float('inf')

        for r in range(n - 1):
            for s in range(r + 1, n):
                d = delta_cost(perm, F, D, r, s)
                is_tabu = (tabu_matrix[r][s] > iteration)
                aspiration = (current_cost + d < best_cost)

                if (not is_tabu or aspiration) and d < best_delta:
                    best_delta = d
                    best_r, best_s = r, s

        if best_r == -1:
            min_tabu = float('inf')
            for r in range(n - 1):
                for s in range(r + 1, n):
                    if tabu_matrix[r][s] < min_tabu:
                        min_tabu = tabu_matrix[r][s]
                        best_r, best_s = r, s
            best_delta = delta_cost(perm, F, D, best_r, best_s)

        # Apply the swap
        perm[best_r], perm[best_s] = perm[best_s], perm[best_r]
        current_cost += best_delta
        tabu_matrix[best_r][best_s] = iteration + int(tenure)
        tabu_matrix[best_s][best_r] = iteration + int(tenure)

        # Cycling detection
        perm_hash = hash(tuple(perm))
        if perm_hash in visited_hashes:
            # Cycle detected: increase tenure
            tenure = min(max_tenure, tenure * 1.1)
        else:
            # No cycle: decay tenure
            tenure = max(min_tenure, tenure * 0.995)
            visited_hashes.add(perm_hash)

        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = list(perm)
            non_improving_count = 0
        else:
            non_improving_count += 1

        # Diversification
        if non_improving_count >= diversification_threshold:
            perm = list(best_perm)
            current_cost = best_cost
            # Perturb with 3 random swaps
            for _ in range(3):
                r = rng.randint(0, n - 1)
                s = rng.randint(0, n - 1)
                while s == r:
                    s = rng.randint(0, n - 1)
                d = delta_cost(perm, F, D, r, s)
                perm[r], perm[s] = perm[s], perm[r]
                current_cost += d
            non_improving_count = 0
            # Limit visited hashes size to prevent memory blowup
            if len(visited_hashes) > 100000:
                visited_hashes.clear()

        if iteration % 100 == 0:
            convergence.append((iteration, best_cost))

    elapsed = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    return {
        "cost": int(best_cost),
        "perm": best_perm,
        "time": elapsed,
        "memory_mb": round(peak_mb, 2),
        "convergence": convergence
    }


if __name__ == "__main__":
    from experiments.qaplib_parser import parse_qaplib

    if len(sys.argv) < 2:
        print("Usage: python tabu_search.py <dat_file>")
        sys.exit(1)

    n, F, D = parse_qaplib(sys.argv[1])
    print(f"Instance: n={n}")

    print("\nRunning TS-Original...")
    res = tabu_search_original(n, F, D)
    print(f"  Cost={res['cost']}, Time={res['time']:.2f}s, Mem={res['memory_mb']}MB")

    print("\nRunning TS-Reactive...")
    res = tabu_search_reactive(n, F, D)
    print(f"  Cost={res['cost']}, Time={res['time']:.2f}s, Mem={res['memory_mb']}MB")
