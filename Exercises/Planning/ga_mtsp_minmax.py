import random
import math
from typing import List, Tuple
import numpy as np
from math import sqrt
import random
# ===========================
# mTSP GA — Minimize MAKESPAN
# ===========================
# - Nodes: 0..n-1, depot=0 by default
# - Chromosome: permutation of customers (1..n-1)
# - Decoder: DP over consecutive splits to minimize MAX route cost
# - Fitness: (makespan, total_distance)  -> lexicographic
# - Operators: tournament selection, OX crossover, swap & shuffle mutations
# - Optional: 2-opt on the currently longest route (off by default)

# ---------- Helpers ----------

def route_cost_from_perm_segment(D, depot, perm, a, b) -> float:
    """Cost of a route visiting perm[a:b] in order, starting/ending at depot."""
    if a >= b:  # empty route
        return 0.0
    cost = D[depot][perm[a]]
    for k in range(a, b - 1):
        cost += D[perm[k]][perm[k + 1]]
    cost += D[perm[b - 1]][depot]
    return cost

def decode_optimal_split_makespan(D, depot, perm: List[int], m: int):
    """
    Split perm into m routes to minimize makespan (max route cost).
    DP[r][k] = min over t<=k of max(DP[r-1][t], cost(t..k)).
    Returns: makespan, splits, routes, total_distance
    """
    n = len(perm)
    INF = 1e100

    # Precompute segment costs C[i][k] for perm[i:k]
    C = [[0.0]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for k in range(i+1, n+1):
            C[i][k] = route_cost_from_perm_segment(D, depot, perm, i, k)

    DP = [[INF]*(n+1) for _ in range(m+1)]
    PR = [[-1]*(n+1) for _ in range(m+1)]
    DP[0][0] = 0.0

    for r in range(1, m+1):
        for k in range(0, n+1):
            best, arg = INF, -1
            for t in range(0, k+1):
                cand = max(DP[r-1][t], C[t][k])
                if cand < best:
                    best, arg = cand, t
            DP[r][k], PR[r][k] = best, arg

    makespan = DP[m][n]

    # reconstruct splits
    splits = [n]
    r, k = m, n
    while r > 0:
        t = PR[r][k]
        splits.append(t)
        r, k = r-1, t
    splits = list(reversed(splits))
    if splits[0] != 0: splits = [0] + splits

    routes, total = [], 0.0
    for r in range(m):
        a, b = splits[r], splits[r+1]
        if a == b:
            routes.append([depot, depot])
        else:
            routes.append([depot] + perm[a:b] + [depot])
            total += C[a][b]
    return makespan, splits, routes, total

def tournament_select(pop, fit, k=3):
    best_i = min(random.sample(range(len(pop)), k), key=lambda i: fit[i])
    return pop[best_i]

def ox_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(pa, pb):
        child = [None]*n
        child[a:b+1] = pa[a:b+1]
        fill = [g for g in pb if g not in child]
        idxs = list(range(b+1, n)) + list(range(0, a))
        for pos, g in zip(idxs, fill):
            child[pos] = g
        return child
    return ox(p1, p2), ox(p2, p1)

def swap_mutation(chrom: List[int], p=0.2):
    if random.random() < p and len(chrom) >= 2:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]

def shuffle_segment_mutation(chrom: List[int], p=0.1):
    if random.random() < p and len(chrom) >= 3:
        a, b = sorted(random.sample(range(len(chrom)), 2))
        seg = chrom[a:b+1]
        random.shuffle(seg)
        chrom[a:b+1] = seg

def two_opt_route(route: List[int], D) -> List[int]:
    if len(route) < 4: return route[:]
    def rlen(r): return sum(D[r[i]][r[i+1]] for i in range(len(r)-1))
    best = route[:]
    best_cost = rlen(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-3):
            for j in range(i+1, len(best)-1):
                if j == i+1: continue
                cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                c = rlen(cand)
                if c + 1e-9 < best_cost:
                    best, best_cost, improved = cand, c, True
    return best

def nearest_neighbor_perm(D, depot) -> List[int]:
    n = len(D)
    rem = set(range(n)) - {depot}
    perm, cur = [], depot
    while rem:
        nxt = min(rem, key=lambda j: D[cur][j])
        perm.append(nxt)
        rem.remove(nxt)
        cur = nxt
    return perm

# ---------- GA ----------

class MTSP_GA_Makespan:
    def __init__(
        self, D: List[List[float]], m: int, depot: int = 0,
        pop_size=150, generations=600, cx_rate=0.9,
        mut_swap_p=0.25, mut_shuffle_p=0.10, tournament_k=3,
        elitism=2, local_2opt_every=0, seed=0, log=True
    ):
        self.D, self.m, self.depot = D, m, depot
        self.pop_size, self.generations = pop_size, generations
        self.cx_rate = cx_rate
        self.mut_swap_p, self.mut_shuffle_p = mut_swap_p, mut_shuffle_p
        self.tournament_k, self.elitism = tournament_k, elitism
        self.local_2opt_every = local_2opt_every
        self.log = log
        random.seed(seed)
        n = len(D)
        assert 0 <= depot < n and 1 <= m <= n-1

    def _evaluate(self, perm: List[int]):
        mk, splits, routes, total = decode_optimal_split_makespan(
            self.D, self.depot, perm, self.m
        )
        # Primary: makespan; Secondary: total distance
        return (mk, total), splits, routes

    def _init_population(self) -> List[List[int]]:
        base = nearest_neighbor_perm(self.D, self.depot)
        pop = [base[:]]
        customers = list(range(len(self.D))); customers.remove(self.depot)
        while len(pop) < self.pop_size:
            c = base[:] if len(pop) % 2 == 0 else customers[:]
            random.shuffle(c); pop.append(c)
        return pop

    def solve(self):
        pop = self._init_population()
        fitness, decoded = [], []
        for chrom in pop:
            f, s, r = self._evaluate(chrom)
            fitness.append(f); decoded.append((f, s, r))

        best_fit = (math.inf, math.inf); best_routes = None; best_perm = None

        for g in range(1, self.generations+1):
            # track best
            i = min(range(len(pop)), key=lambda t: fitness[t])
            if fitness[i] < best_fit:
                best_fit, best_perm, best_routes = fitness[i], pop[i][:], decoded[i][2]
            if self.log and (g == 1 or g % 50 == 0 or g == self.generations):
                print(f"Gen {g:4d} | best makespan = {best_fit[0]:.4f} | total = {best_fit[1]:.4f}")

            # elitism
            order = sorted(range(len(pop)), key=lambda t: fitness[t])
            new_pop = [pop[idx][:] for idx in order[:self.elitism]]

            # reproduction
            while len(new_pop) < self.pop_size:
                p1 = tournament_select(pop, fitness, self.tournament_k)
                p2 = tournament_select(pop, fitness, self.tournament_k)
                if random.random() < self.cx_rate:
                    c1, c2 = ox_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                swap_mutation(c1, self.mut_swap_p); shuffle_segment_mutation(c1, self.mut_shuffle_p)
                swap_mutation(c2, self.mut_swap_p); shuffle_segment_mutation(c2, self.mut_shuffle_p)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size: new_pop.append(c2)

            pop = new_pop

            # evaluation (+ optional local search on longest route)
            fitness, decoded = [], []
            for chrom in pop:
                if self.local_2opt_every and g % self.local_2opt_every == 0:
                    f, s, routes = self._evaluate(chrom)
                    # 2-opt only the current longest route to target makespan
                    def rlen(r): return sum(self.D[r[i]][r[i+1]] for i in range(len(r)-1))
                    idx_long = max(range(len(routes)), key=lambda ridx: rlen(routes[ridx]))
                    routes[idx_long] = two_opt_route(routes[idx_long], self.D)
                    # flatten back to permutation
                    chrom = [v for r in routes for v in r if v != self.depot]
                f, s, r = self._evaluate(chrom)
                fitness.append(f); decoded.append((f, s, r))

        return best_fit, best_routes, best_perm


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # 15x15 example matrix from earlier
    '''D = [
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
    D = [
        [0, 1, 15, 20, 10, 120],
        [10, 0, 35, 25, 170, 28],
        [15, 15, 0, 30, 30, 160],
        [20, 25, 30, 0, 11, 13],
        [12, 17, 230, 11, 0, 19],
        [18, 28, 160, 13, 19, 0],
    ]
    depot, m = 0, 2

    ### Generate random distance matrix from num_points points
    num_points = 40
    xy_list = [[random.randint(0,100), random.randint(0,100)] for _ in range(num_points)]
    dist = lambda p1, p2: sqrt(((p1-p2)**2).sum())
    D = np.asarray([[dist(np.array(p1), np.array(p2)) for p2 in xy_list] for p1 in xy_list])

    ga = MTSP_GA_Makespan(
        D, m, depot=depot,
        pop_size=120, generations=400, cx_rate=0.9,
        mut_swap_p=0.25, mut_shuffle_p=0.10,
        tournament_k=3, elitism=2,
        local_2opt_every=0,  # try 25+ to refine bottleneck route
        seed=42, log=True
    )

    (best_makespan, best_total), best_routes, best_perm = ga.solve()

    print("\n=== GA Result (Makespan) ===")
    print(f"Makespan (longest tour): {best_makespan:.4f}")
    print(f"Total distance (tie-break): {best_total:.4f}")
    for i, r in enumerate(best_routes, 1):
        print(f"Route {i}: {r}")
    print(f"Chromosome (perm of customers): {best_perm}")
