import numpy as np
import csv
import os

class Metrics:
    def __init__(self, n_agents, tick=0.1, csv_dir="metrics_logs"):
        self.n_agents = int(n_agents)
        self.tick = float(tick)
        self.time = 0.0
        self.step = 0

        # scratch
        self.dmat = np.full((self.n_agents, self.n_agents), np.inf, dtype=float)
        self.min_per_agent = np.full(self.n_agents, np.inf, dtype=float)

        self.goal = None

        # rank order for current goal cycle (list of agent indices)
        self.rank_order = list(range(self.n_agents))

        # --- CSV setup ---
        os.makedirs(csv_dir, exist_ok=True)

        # 1) goal distances per agent
        goal_path = os.path.join(csv_dir, "distances_to_goal.csv")
        self._f_goal = open(goal_path, "a", newline="")
        self._w_goal = csv.writer(self._f_goal)
        if os.stat(goal_path).st_size == 0:
            self._w_goal.writerow(["t"] + [f"goalDist_{i}" for i in range(self.n_agents)])
            self._f_goal.flush()

        # 2) adjacent-by-rank distances per step
        rank_path = os.path.join(csv_dir, "distances_adjacent_by_rank.csv")
        self._f_rank = open(rank_path, "a", newline="")
        self._w_rank = csv.writer(self._f_rank)
        if os.stat(rank_path).st_size == 0:
            hdr = ["t"] + [f"d_r{i}_r{i+1}" for i in range(self.n_agents - 1)]
            self._w_rank.writerow(hdr)
            self._f_rank.flush()

        # 3) rank→agent mapping per cycle
        order_path = os.path.join(csv_dir, "rank_orders.csv")
        self._f_order = open(order_path, "a", newline="")
        self._w_order = csv.writer(self._f_order)
        if os.stat(order_path).st_size == 0:
            self._w_order.writerow(["t_cycle_start"] + [f"rank{i}_agentIndex" for i in range(self.n_agents)])
            self._f_order.flush()

    # ------------ configuration ------------
    def set_goal(self, goal_xy):
        g = np.asarray(goal_xy, dtype=float).reshape(-1)
        self.goal = g[:2]

    def set_order_by_leader(self, positions_xy, leader_idx=0, t_cycle_start=None):
        pos = np.asarray(positions_xy, dtype=float).reshape(self.n_agents, 2)
        leader_pos = pos[leader_idx]
        dists = np.linalg.norm(pos - leader_pos[None, :], axis=1)
        order = np.argsort(dists)
        # ensure leader is rank 0 (helps in tie cases)
        if order[0] != leader_idx:
            order = np.array([leader_idx] + [i for i in order if i != leader_idx])
        self.rank_order = order.tolist()

        if t_cycle_start is None:
            t_cycle_start = self.time
        self._w_order.writerow([round(float(t_cycle_start), 3)] + self.rank_order)
        self._f_order.flush()

    # ------------ per-step update ------------
    def update(self, positions_xy):
        pos = np.asarray(positions_xy, dtype=float).reshape(self.n_agents, 2)
        self.time += self.tick
        self.step += 1

        # full pairwise distances (for rank-adjacent logging)
        N = self.n_agents
        dmat = np.empty((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i == j:
                    dmat[i, j] = np.inf
                else:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dmat[i, j] = (dx*dx + dy*dy) ** 0.5
        self.dmat = dmat
        self.min_per_agent = np.min(dmat, axis=1)

        if self.goal is not None:
            gdxdy = pos - self.goal[None, :]
            d_goal = np.sqrt(np.sum(gdxdy * gdxdy, axis=1))
            row_goal = [round(float(x), 3) for x in d_goal.tolist()]
        else:
            row_goal = [""] * self.n_agents
        self._w_goal.writerow([round(self.time, 3)] + row_goal)
        self._f_goal.flush()

        ro = self.rank_order
        adj = [round(float(dmat[ro[k], ro[k+1]]), 3) for k in range(N - 1)]
        self._w_rank.writerow([round(self.time, 3)] + adj)
        self._f_rank.flush()

    # ------------ teardown ------------
    def close(self):
        try:
            self._f_goal.close()
            self._f_rank.close()
            self._f_order.close()
        except Exception:
            pass
