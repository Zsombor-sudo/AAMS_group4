import itertools
from typing import List, Tuple
import numpy as np
from math import sqrt
import random
# -------------------------------
# Brute-force mTSP (single depot)
# -------------------------------
# Problem:
#   - Nodes: 0..n-1, depot = 0
#   - Visit every customer (1..n-1) exactly once
#   - Exactly m tours, each starts/ends at the depot
#   - Objective: minimize total distance
#
# Approach:
#   - Enumerate every permutation of customers
#   - For each permutation, enumerate every composition of n_cust into m parts
#     (lengths of consecutive segments), allowing empty routes
#   - Compute cost of each set of m routes and track the best

'''
D = [
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31, 18, 23, 17, 25, 9],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41, 12, 28, 19, 32, 25],
    [20, 15, 0, 15, 14, 25, 81, 9, 23, 27, 13, 19, 8, 16, 14],
    [21, 29, 15, 0, 4, 12, 92, 12, 25, 13, 25, 13, 24, 19, 15],
    [16, 28, 14, 4, 0, 16, 94, 9, 20, 16, 22, 19, 21, 23, 17],
    [31, 40, 25, 12, 16, 0, 95, 24, 36, 3, 37, 21, 27, 32, 23],
    [100, 72, 81, 92, 94, 95, 0, 90, 101, 99, 87, 93, 89, 101, 90],
    [12, 21, 9, 12, 9, 24, 90, 0, 15, 25, 13, 10, 8, 20, 9],
    [4, 29, 23, 25, 20, 36, 101, 15, 0, 35, 18, 27, 22, 24, 13],
    [31, 41, 27, 13, 16, 3, 99, 25, 35, 0, 35, 18, 28, 25, 21],
    [18, 12, 13, 25, 22, 37, 87, 13, 18, 35, 0, 15, 12, 16, 14],
    [23, 28, 19, 13, 19, 21, 93, 10, 27, 18, 15, 0, 17, 20, 12],
    [17, 19, 8, 24, 21, 27, 89, 8, 22, 28, 12, 17, 0, 16, 10],
    [25, 32, 16, 19, 23, 32, 101, 20, 24, 25, 16, 20, 16, 0, 14],
    [9, 25, 14, 15, 17, 23, 90, 9, 13, 21, 14, 12, 10, 14, 0],
]'''

def route_cost(D: List[List[float]], route: List[int]) -> float:
    """Cost of a single route including depot at start/end, e.g., [0, a, b, 0]."""
    return sum(D[route[i]][route[i+1]] for i in range(len(route)-1))

def compositions(n: int, m: int):
    """
    Yield all m-tuples of nonnegative integers that sum to n.
    (Stars and bars; allows empty routes.)
    """
    # Choose m-1 cut positions among n + m - 1 slots
    # Equivalent constructive approach:
    # iterate over increasing (m-1) cut indices in range(1..n+m-1)
    # convert to segment lengths.
    from itertools import combinations
    for cuts in combinations(range(n + m - 1), m - 1):
        last = -1
        parts = []
        for c in cuts:
            parts.append(c - last - 1)
            last = c
        parts.append(n + m - 1 - last - 1)
        yield tuple(parts)

def apply_split(perm: List[int], lengths: Tuple[int, ...]) -> List[List[int]]:
    """Split permutation into m segments according to 'lengths'."""
    routes = []
    idx = 0
    for L in lengths:
        seg = perm[idx:idx+L]
        routes.append([0] + seg + [0])  # add depot at both ends
        idx += L
    return routes

def brute_force_mtsp(D: List[List[float]], m: int, depot: int = 0):
    n = len(D)
    assert 0 <= depot < n
    customers = [i for i in range(n) if i != depot]
    n_cust = len(customers)

    assert 1 <= m <= n_cust  # if m>n_cust you’ll force many empty tours; allowed but odd

    best_cost = float("inf")
    best_routes = None
    best_perm = None
    best_lengths = None

    perms_iter = itertools.permutations(customers)
    #length = sum(1 for _ in perms_iter)
    #print(f"Nr of permutations {length}")

    for perm in perms_iter:

        # enumerate all ways to split this permutation into m consecutive segments
        for lens in compositions(n_cust, m):

            routes = apply_split(list(perm), lens)

            total = sum(route_cost(D, r) for r in routes)

            if total < best_cost - 1e-12:
                best_cost = total
                best_routes = routes
                best_perm = list(perm)
                best_lengths = lens

    return best_cost, best_routes, best_perm, best_lengths


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # Tiny demo (brute force explodes fast; keep n small!)
    D = [
        [0, 1, 15, 20, 10, 120],
        [10, 0, 35, 25, 170, 28],
        [15, 15, 0, 30, 30, 160],
        [20, 25, 30, 0, 11, 13],
        [12, 17, 230, 11, 0, 19],
        [18, 28, 160, 13, 19, 0],
    ]

    ### Generate random distance matrix from num_points points
    num_points = 11
    xy_list = [[random.randint(0,100), random.randint(0,100)] for _ in range(num_points)]
    dist = lambda p1, p2: sqrt(((p1-p2)**2).sum())
    D = np.asarray([[dist(np.array(p1), np.array(p2)) for p2 in xy_list] for p1 in xy_list])

    depot = 0
    m = 2  # exactly two tours

    best_cost, best_routes, best_perm, best_lengths = brute_force_mtsp(D, m, depot)
    print(f"Best total cost: {best_cost:.4f}")
    print(f"Permutation of customers: {best_perm}")
    print(f"Split lengths (m parts summing to {len(best_perm)}): {best_lengths}")
    for i, r in enumerate(best_routes, 1):
        print(f"Route {i}: {r}")