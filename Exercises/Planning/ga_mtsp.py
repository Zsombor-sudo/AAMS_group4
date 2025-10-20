import random
import math
from typing import List, Tuple

# ===========================
# mTSP GA (single depot)
# ===========================
# - Input:
#     D: n x n distance/cost matrix (0..n-1), depot index = 0
#     m: number of routes (salesmen)
# - Chromosome: permutation of customers (nodes 1..n-1)
# - Decoder: splits permutation into m segments via DP to minimize
#            sum of depot->segment->depot costs.
# - Constraints:
#     * Every customer visited exactly once
#     * Exactly m tours, each starts/ends at depot
# - Objective: minimize total distance
#
# No external libraries required.

# ---------- Utilities ----------

def route_cost_from_perm_segment(D, depot, perm, start, end_exclusive) -> float:
    """Cost of a single route visiting perm[start:end] (in order), starting and ending at depot."""
    if start >= end_exclusive:
        # Empty route: cost 0 (allowed; decoder may create empty routes if m > 1)
        return 0.0
    cost = D[depot][perm[start]]
    for k in range(start, end_exclusive - 1):
        i, j = perm[k], perm[k + 1]
        cost += D[i][j]
    cost += D[perm[end_exclusive - 1]][depot]
    return cost

def decode_optimal_split(D, depot, perm: List[int], m: int):
    """
    Given a permutation of customers, compute the optimal split into m routes
    (each starts/ends at depot) that minimizes total cost.

    Returns:
        total_cost, splits, routes
        - splits: indices [b0=0 < b1 < ... < bm = len(perm)] where route r is perm[br:br+1]
        - routes: list of full routes including depot at start/end, e.g., [ [0, a, b, 0], [0, c, 0], ... ]
    """
    n = len(perm)
    # DP[r][k] = min cost to cover first k customers with r routes
    INF = 1e100
    DP = [[INF] * (n + 1) for _ in range(m + 1)]
    prev = [[-1] * (n + 1) for _ in range(m + 1)]
    DP[0][0] = 0.0

    # Precompute segment costs C[i][k] = cost of route visiting perm[i:k]
    C = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for k in range(i + 1, n + 1):
            C[i][k] = route_cost_from_perm_segment(D, depot, perm, i, k)

    for r in range(1, m + 1):
        for k in range(0, n + 1):
            # last segment begins at t in [0..k], cover perm[t:k] as route r
            best = INF
            best_t = -1
            for t in range(0, k + 1):
                cand = DP[r - 1][t] + C[t][k]
                if cand < best:
                    best = cand
                    best_t = t
            DP[r][k] = best
            prev[r][k] = best_t

    total_cost = DP[m][n]
    # Reconstruct split points
    splits = [n]
    r, k = m, n
    while r > 0:
        t = prev[r][k]
        splits.append(t)
        r, k = r - 1, t
    splits = list(reversed(splits))  # [0=... < ... < n]
    # Ensure first split is 0
    if splits[0] != 0:
        splits = [0] + splits

    # Build explicit routes with depot
    routes = []
    for r in range(m):
        a, b = splits[r], splits[r + 1]
        if a == b:
            routes.append([depot, depot])  # empty route (allowed if desired)
        else:
            seq = [depot] + perm[a:b] + [depot]
            routes.append(seq)

    return total_cost, splits, routes

# ---------- GA operators ----------

def tournament_select(pop, fitness, k=3):
    best = None
    best_fit = math.inf
    for _ in range(k):
        i = random.randrange(len(pop))
        if fitness[i] < best_fit:
            best_fit = fitness[i]
            best = pop[i]
    return best

def ox_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Order Crossover (OX) for permutations."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(parent_a, parent_b):
        child = [None] * n
        # Copy slice from parent_a
        child[a:b+1] = parent_a[a:b+1]
        # Fill remainder in order from parent_b
        pb = [g for g in parent_b if g not in child]
        idxs = list(range(b+1, n)) + list(range(0, a))
        for pos, gene in zip(idxs, pb):
            child[pos] = gene
        return child
    return ox(p1, p2), ox(p2, p1)

def swap_mutation(chrom: List[int], p=0.2):
    """Swap two positions with probability p."""
    if random.random() < p and len(chrom) >= 2:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]

def shuffle_segment_mutation(chrom: List[int], p=0.1):
    """Randomly shuffle a short contiguous segment."""
    if random.random() < p and len(chrom) >= 3:
        a, b = sorted(random.sample(range(len(chrom)), 2))
        seg = chrom[a:b+1]
        random.shuffle(seg)
        chrom[a:b+1] = seg

def two_opt_route(route: List[int], D) -> List[int]:
    """2-opt on a single full route including depot at ends (e.g., [0, a, b, c, 0])."""
    changed = True
    best = route[:]
    def route_len(r):
        return sum(D[r[i]][r[i+1]] for i in range(len(r)-1))
    best_cost = route_len(best)
    while changed:
        changed = False
        # only interior indices can be swapped
        for i in range(1, len(best) - 3):
            for j in range(i + 1, len(best) - 1):
                if j == i + 1:
                    continue
                new_r = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost = route_len(new_r)
                if new_cost + 1e-9 < best_cost:
                    best, best_cost = new_r, new_cost
                    changed = True
    return best

def routes_to_perm(routes: List[List[int]]) -> List[int]:
    """Flatten routes (with depot) back to a permutation of customers."""
    perm = []
    for r in routes:
        perm += [v for v in r if v != r[0]]  # remove leading depot
        if perm and perm[-1] == 0:
            perm.pop()  # remove trailing depot if present
    return [v for v in perm if v != 0]

# ---------- Heuristic initializers ----------

def nearest_neighbor_perm(D, depot) -> List[int]:
    n = len(D)
    customers = set(range(n)) - {depot}
    perm = []
    cur = depot
    while customers:
        nxt = min(customers, key=lambda j: D[cur][j])
        perm.append(nxt)
        customers.remove(nxt)
        cur = nxt
    return perm

# ---------- GA solver ----------

class MTSP_GA:
    def __init__(
        self,
        D: List[List[float]],
        m: int,
        depot: int = 0,
        pop_size: int = 150,
        generations: int = 600,
        cx_rate: float = 0.9,
        mut_swap_p: float = 0.25,
        mut_shuffle_p: float = 0.10,
        tournament_k: int = 3,
        elitism: int = 2,
        local_2opt_every: int = 0,  # set >0 to apply 2-opt every k generations
        seed: int = 0,
        log: bool = True
    ):
        self.D, self.m, self.depot = D, m, depot
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = cx_rate
        self.mut_swap_p = mut_swap_p
        self.mut_shuffle_p = mut_shuffle_p
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.local_2opt_every = local_2opt_every
        random.seed(seed)
        self.log = log

        self.n = len(D)
        assert 0 <= depot < self.n
        assert 1 <= m <= self.n - 1

    def _evaluate(self, perm: List[int]) -> Tuple[float, List[int], List[List[int]]]:
        return decode_optimal_split(self.D, self.depot, perm, self.m)

    def _init_population(self) -> List[List[int]]:
        base = nearest_neighbor_perm(self.D, self.depot)
        pop = [base[:]]
        # random shuffles of base plus fully random perms
        customers = list(range(self.n))
        customers.remove(self.depot)
        while len(pop) < self.pop_size:
            if len(pop) % 2 == 0:
                c = base[:]
                random.shuffle(c)
                pop.append(c)
            else:
                c = customers[:]
                random.shuffle(c)
                pop.append(c)
        return pop

    def solve(self):
        pop = self._init_population()
        fitness = []
        decoded = []  # cache (cost, splits, routes)
        for chrom in pop:
            cost, splits, routes = self._evaluate(chrom)
            fitness.append(cost)
            decoded.append((cost, splits, routes))

        best_cost, best_routes, best_perm = math.inf, None, None

        for g in range(1, self.generations + 1):
            # Track best
            idx = min(range(len(pop)), key=lambda i: fitness[i])
            if fitness[idx] < best_cost - 1e-9:
                best_cost = fitness[idx]
                best_perm = pop[idx][:]
                best_routes = decoded[idx][2]

            if self.log and (g == 1 or g % 50 == 0 or g == self.generations):
                print(f"Gen {g:4d} | best = {best_cost:.4f}")

            # Elites
            order = sorted(range(len(pop)), key=lambda i: fitness[i])
            new_pop = [pop[i][:] for i in order[:self.elitism]]

            # Reproduction
            while len(new_pop) < self.pop_size:
                p1 = tournament_select(pop, fitness, self.tournament_k)
                p2 = tournament_select(pop, fitness, self.tournament_k)
                if random.random() < self.cx_rate:
                    c1, c2 = ox_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                swap_mutation(c1, self.mut_swap_p)
                swap_mutation(c2, self.mut_swap_p)
                shuffle_segment_mutation(c1, self.mut_shuffle_p)
                shuffle_segment_mutation(c2, self.mut_shuffle_p)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop

            # Evaluate
            fitness = []
            decoded = []
            for chrom in pop:
                # Optional local 2-opt every k generations (light intensity)
                if self.local_2opt_every and g % self.local_2opt_every == 0:
                    # Decode, improve each route with 2-opt, then re-flatten
                    cost, splits, routes = self._evaluate(chrom)
                    improved_routes = [two_opt_route(r, self.D) if len(r) > 3 else r for r in routes]
                    chrom = routes_to_perm(improved_routes)
                cost, splits, routes = self._evaluate(chrom)
                fitness.append(cost)
                decoded.append((cost, splits, routes))

        return best_cost, best_routes, best_perm


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # Small demo (asymmetric allowed)
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
    #num_points = 3
    #xy_list = [[random.randint(0,100), random.randint(0,100)] for _ in range(num_points)]
    #dist = lambda p1, p2: sqrt(((p1-p2)**2).sum())
    #dm = np.asarray([[dist(np.array(p1), np.array(p2)) for p2 in xy_list] for p1 in xy_list])


    ga = MTSP_GA(
        D, m, depot=depot,
        pop_size=120,
        generations=400,
        cx_rate=0.9,
        mut_swap_p=0.25,
        mut_shuffle_p=0.10,
        tournament_k=3,
        elitism=2,
        local_2opt_every=0,  # set to, e.g., 25 to enable lightweight local search
        seed=42,
        log=True
    )

    best_cost, best_routes, best_perm = ga.solve()

    print("\n=== GA Result ===")
    print(f"Total cost: {best_cost:.4f}")
    for i, r in enumerate(best_routes, 1):
        print(f"Route {i}: {r}")
    print(f"Chromosome (perm of customers): {best_perm}")