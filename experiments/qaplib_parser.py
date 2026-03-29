import numpy as np


def parse_qaplib(filepath):
    """Parse a QAPLIB .dat file. Returns (n, F, D)."""
    with open(filepath, 'r') as f:
        content = f.read()
    nums = []
    for token in content.split():
        try:
            nums.append(int(token))
        except ValueError:
            continue
    n = nums[0]
    F = np.array(nums[1:1 + n * n]).reshape(n, n)
    D = np.array(nums[1 + n * n:1 + 2 * n * n]).reshape(n, n)
    return n, F, D


def qap_cost(perm, F, D):
    """Compute QAP objective for permutation perm (list/array of ints)."""
    n = len(perm)
    P = np.array(perm)
    return int(np.sum(F * D[np.ix_(P, P)]))


def delta_cost(perm, F, D, r, s):
    """O(n) delta evaluation for swapping perm[r] and perm[s].
    Returns cost(perm_after_swap) - cost(perm_before_swap)."""
    n = len(perm)
    p, q = perm[r], perm[s]
    delta = (F[r][r] - F[s][s]) * (D[q][q] - D[p][p]) + \
            (F[r][s] - F[s][r]) * (D[q][p] - D[p][q])
    for k in range(n):
        if k == r or k == s:
            continue
        t_k = perm[k]
        delta += (F[k][r] - F[k][s]) * (D[t_k][q] - D[t_k][p]) + \
                 (F[r][k] - F[s][k]) * (D[q][t_k] - D[p][t_k])
    return delta


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qaplib_parser.py <filepath>")
        sys.exit(1)
    n, F, D = parse_qaplib(sys.argv[1])
    print(f"n = {n}")
    print(f"F =\n{F}")
    print(f"D =\n{D}")
    perm = list(range(n))
    print(f"Identity cost = {qap_cost(perm, F, D)}")
