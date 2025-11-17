# metrics_csv.py
import numpy as np
import csv
import os

class Metrics:
    def __init__(self, n_agents, radii, dt,
                 d_collide_margin=0.0, d_safe_margin=0.2,
                 radius_goal=0.2,
                 csv_dir="metrics_logs",
                 log=1):
        self.n_agents = n_agents
        self.radii = np.array(radii).reshape(n_agents)
        self.dt = dt

        # thresholds
        self.d_collide_margin = d_collide_margin
        self.d_safe_margin = d_safe_margin
        self.radius_goal = radius_goal

        # state
        self.time = 0.0
        self.time_close = np.zeros(n_agents)
        self.time_collision = np.zeros(n_agents)
        self.min_dist_now = np.full(n_agents, np.inf)

        self.arrival_times = [None] * n_agents
        self.use_leader_goal_flag = False
        self.leader_goal_time = None
        self.goal_xy = None

        # logging
        self.log = max(1, log)
        self.step = 0
        os.makedirs(csv_dir, exist_ok=True)
        self.f_prox = open(os.path.join(csv_dir, "proximity.csv"), "a", newline="")
        self.f_arr  = open(os.path.join(csv_dir, "arrival.csv"), "a", newline="")
        self.w_prox = csv.writer(self.f_prox)
        self.w_arr  = csv.writer(self.f_arr)

        self.w_prox.writerow(["t"]
            + [f"minDist_{i}" for i in range(self.n_agents)]
            + [f"timeClose_{i}" for i in range(self.n_agents)]
            + [f"timeCollision_{i}" for i in range(self.n_agents)])
        self.w_arr.writerow(["t"] + [f"arrival_{i}" for i in range(self.n_agents)])
        self.f_prox.flush(); self.f_arr.flush()

    def set_goal(self, goal_xy):
        self.goal_xy = np.array(goal_xy).reshape(2)

    def use_leader_as_goal(self, flag=True):
        self.use_leader_goal_flag = flag

    def mark_leader_goal_reached(self, time_now):
        self.leader_goal_time = time_now

    #if arrived, we stop counting its position for the proximity detection
    def update(self, positions_xy, arrived=None):
        pos = np.asarray(positions_xy).reshape(self.n_agents, 2)
        self.time += self.dt
        self.step += 1

        # euclidean distance(?) not sure if correct
        N = pos.shape[0] # pos is (N, 2)
        dmat = np.empty((N, N))

        for i in range(N):
            for j in range(N):
                if i == j:
                    dmat[i, j] = np.inf  
                else:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dmat[i, j] = (dx*dx + dy*dy) ** 0.5 


        # optionally ignore arrived agents for proximity check
        if arrived is not None:
            arrived_bool = np.asarray(arrived)
            for i in range(self.n_agents):
                if arrived_bool[i]:
                    dmat[i, :] = np.inf
                    dmat[:, i] = np.inf


        self.min_dist_now = np.min(dmat, axis=1)
        d_collide = 2 * self.radii + self.d_collide_margin
        d_safe    = 2 * self.radii + self.d_safe_margin

        for i in range(self.n_agents):
            if self.min_dist_now[i] < d_collide[i]:
                self.time_collision[i] += self.dt
            elif self.min_dist_now[i] < d_safe[i]:
                self.time_close[i] += self.dt

        # arrivals
        ref_point = None
        if self.use_leader_goal_flag:
            if (self.leader_goal_time is not None) and (self.time >= self.leader_goal_time):
                ref_point = pos[0]  # the leader's position

        if ref_point is not None:
            for i in range(self.n_agents):
                if self.arrival_times[i] is None:
                    if np.sum((pos[i] - ref_point)**2) <= self.radius_goal ** 2:
                        self.arrival_times[i] = self.time

        # CSV
        
    def finalize(self):
        arrivals = [a for a in self.arrival_times if a is not None]
        makespan = max(arrivals) if arrivals else None
        med_arr  = float(np.median(arrivals)) if arrivals else None
        p90_arr  = float(np.percentile(arrivals, 90)) if arrivals else None
        return {
            "time_end": self.time,
            "time_close_per_agent": self.time_close,
            "time_collision_per_agent": self.time_collision,
            "arrival_times": self.arrival_times,
            "makespan": makespan,
            "median_arrival": med_arr,
            "p90_arrival": p90_arr,
        }

    def close(self):
        try:
            self.f_prox.close()
            self.f_arr.close()
        except Exception:
            pass
