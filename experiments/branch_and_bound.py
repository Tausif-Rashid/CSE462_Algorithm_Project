import time
import tracemalloc
import numpy as np
from scipy.optimize import linear_sum_assignment


def gilmore_lawler_bound(partial_perm, n, F, D):
    """Compute the Gilmore-Lawler lower bound for a partial assignment.

    partial_perm: list of assigned locations for facilities 0..len(partial_perm)-1.
    """
    assigned = {i: partial_perm[i] for i in range(len(partial_perm))}
    free_fac = [i for i in range(n) if i not in assigned]
    free_loc = [k for k in range(n) if k not in assigned.values()]

    # Cost of fixed-fixed interactions
    c_fixed = 0
    for i in assigned:
        for j in assigned:
            c_fixed += F[i][j] * D[assigned[i]][assigned[j]]

    m = len(free_fac)
    if m == 0:
        return c_fixed

    # Build cost matrix for LAP
    cost_matrix = np.zeros((m, m))
    for ii, i in enumerate(free_fac):
        for kk, k in enumerate(free_loc):
            # Fixed-free interaction cost (both directions)
            val = 0
            for j in assigned:
                val += F[i][j] * D[k][assigned[j]] + F[j][i] * D[assigned[j]][k]
            # Free-free: use rearrangement inequality for lower bound.
            # For outgoing: sort F[i][j] ascending, D[k][l] descending => min product sum
            # For incoming: sort F[j][i] ascending, D[l][k] descending
            # Take the max of the two as a tighter single-direction bound
            f_out = sorted([F[i][j2] for j2 in free_fac if j2 != i])
            d_out = sorted([D[k][l2] for l2 in free_loc if l2 != k], reverse=True)
            bound_out = sum(a * b for a, b in zip(f_out, d_out))

            f_in = sorted([F[j2][i] for j2 in free_fac if j2 != i])
            d_in = sorted([D[l2][k] for l2 in free_loc if l2 != k], reverse=True)
            bound_in = sum(a * b for a, b in zip(f_in, d_in))

            val += max(bound_out, bound_in)
            cost_matrix[ii][kk] = val

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return c_fixed + cost_matrix[row_ind, col_ind].sum()


def branch_and_bound_original(n, F, D, timeout=60):
    """Depth-first B&B with Gilmore-Lawler bound.

    Returns dict with keys: cost, perm, nodes, time, memory_mb.
    """
    tracemalloc.start()
    start_time = time.time()
    best_cost = float('inf')
    best_perm = None
    nodes_explored = 0

    # DFS using a stack: each element is a partial permutation (list)
    stack = [[]]  # start with empty assignment

    while stack:
        if time.time() - start_time > timeout:
            break

        partial = stack.pop()
        nodes_explored += 1
        depth = len(partial)

        if depth == n:
            # Evaluate full permutation
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost += F[i][j] * D[partial[i]][partial[j]]
            if cost < best_cost:
                best_cost = cost
                best_perm = list(partial)
            continue

        # Compute lower bound
        lb = gilmore_lawler_bound(partial, n, F, D)
        if lb >= best_cost:
            continue

        # Branch: assign facility 'depth' to each free location
        used_locs = set(partial)
        free_locs = [k for k in range(n) if k not in used_locs]
        # Push in reverse order so smallest index is explored first
        for loc in reversed(free_locs):
            stack.append(partial + [loc])

    elapsed = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    if best_perm is None:
        best_perm = list(range(n))
        best_cost = sum(F[i][j] * D[i][j] for i in range(n) for j in range(n))

    return {
        "cost": int(best_cost),
        "perm": best_perm,
        "nodes": nodes_explored,
        "time": elapsed,
        "memory_mb": round(peak_mb, 2)
    }


def _greedy_initial_solution(n, F, D):
    """Greedy heuristic: assign facility with max flow-sum to location with min distance-sum."""
    flow_sums = [sum(F[i]) + sum(F[:, i]) for i in range(n)]
    dist_sums = [sum(D[k]) + sum(D[:, k]) for k in range(n)]

    fac_order = sorted(range(n), key=lambda i: flow_sums[i], reverse=True)
    loc_order = sorted(range(n), key=lambda k: dist_sums[k])

    perm = [0] * n
    used_locs = set()
    for fac in fac_order:
        for loc in loc_order:
            if loc not in used_locs:
                perm[fac] = loc
                used_locs.add(loc)
                break

    cost = sum(F[i][j] * D[perm[i]][perm[j]] for i in range(n) for j in range(n))
    return perm, cost


def branch_and_bound_improved(n, F, D, timeout=60):
    """DFS B&B with greedy initial upper bound, Gilmore-Lawler bound,
    and LB-ordered child exploration.

    Improvements over original:
      1. Greedy heuristic provides a tight initial UB for early pruning.
      2. Children are sorted by LB so the most promising branch is explored first.

    Returns dict with keys: cost, perm, nodes, time, memory_mb.
    """
    tracemalloc.start()
    start_time = time.time()
    nodes_explored = 0

    best_perm, best_cost = _greedy_initial_solution(n, F, D)

    stack = [[]]

    while stack:
        if time.time() - start_time > timeout:
            break

        partial = stack.pop()
        nodes_explored += 1
        depth = len(partial)

        if depth == n:
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost += F[i][j] * D[partial[i]][partial[j]]
            if cost < best_cost:
                best_cost = cost
                best_perm = list(partial)
            continue

        lb = gilmore_lawler_bound(partial, n, F, D)
        if lb >= best_cost:
            continue

        used_locs = set(partial)
        free_locs = [k for k in range(n) if k not in used_locs]

        children = []
        for loc in free_locs:
            new_partial = partial + [loc]
            child_lb = gilmore_lawler_bound(new_partial, n, F, D)
            if child_lb < best_cost:
                children.append((child_lb, new_partial))

        children.sort(key=lambda x: x[0], reverse=True)
        for _, child_partial in children:
            stack.append(child_partial)

    elapsed = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    return {
        "cost": int(best_cost),
        "perm": best_perm,
        "nodes": nodes_explored,
        "time": elapsed,
        "memory_mb": round(peak_mb, 2)
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.qaplib_parser import parse_qaplib

    if len(sys.argv) < 2:
        print("Usage: python branch_and_bound.py <dat_file>")
        sys.exit(1)

    n, F, D = parse_qaplib(sys.argv[1])
    print(f"Instance: n={n}")

    print("\nRunning BB-Original...")
    res = branch_and_bound_original(n, F, D)
    print(f"  Cost={res['cost']}, Nodes={res['nodes']}, Time={res['time']:.2f}s, Mem={res['memory_mb']}MB")

    print("\nRunning BB-Improved...")
    res = branch_and_bound_improved(n, F, D)
    print(f"  Cost={res['cost']}, Nodes={res['nodes']}, Time={res['time']:.2f}s, Mem={res['memory_mb']}MB")
