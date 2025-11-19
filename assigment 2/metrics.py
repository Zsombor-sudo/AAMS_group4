import numpy as np
import csv
import os

class Metrics:
    def __init__(self, n_agents, tick=0.1, csv_dir="metrics_logs"):
        self.n_agents = int(n_agents)
        self.tick = float(tick)
        self.time = 0
        self.step = 0

        self.dmat = np.full((self.n_agents, self.n_agents), np.inf)
        self.min_per_agent = np.full(self.n_agents, np.inf)
        self.goal = None

        os.makedirs(csv_dir, exist_ok=True)

        # goal distances per agent
        goal_path = os.path.join(csv_dir, "distances_to_goal.csv")
        self._f_goal = open(goal_path, "a", newline="")
        self._w_goal = csv.writer(self._f_goal)
        if os.stat(goal_path).st_size == 0:
            self._w_goal.writerow(["t"] + [f"goalDist_{i}" for i in range(self.n_agents)])
            self._f_goal.flush()

        #nearest neighbour
        nn_path = os.path.join(csv_dir, "distances_nearest.csv")
        self._f_nn = open(nn_path, "a", newline="")
        self._w_nn = csv.writer(self._f_nn)
        if os.stat(nn_path).st_size == 0:
            self._w_nn.writerow(["t"] + [f"nnDist_{i}" for i in range(self.n_agents)])

    def set_goal(self, goal_xy):
        g = np.asarray(goal_xy).reshape(-1)
        self.goal = g[:2]
    
    def update(self, positions_xy):
        pos = np.asarray(positions_xy, dtype=float).reshape(self.n_agents, 2)
        self.time += self.tick
        self.step += 1

        N = self.n_agents
        dmat = np.empty((N, N))
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

        # nearest neighbour distance
        nn_row = [round(float(x), 3) for x in self.min_per_agent.tolist()]
        self._w_nn.writerow([self.time] + nn_row)
        self._f_nn.flush()

    # ------------ teardown ------------
    def close(self):
        try:
            self._f_goal.close()
            self._f_nn.close()
        except Exception:
            pass
