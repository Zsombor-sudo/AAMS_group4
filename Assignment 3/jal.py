import csv
import os
import random
import itertools

class JointActionLearner:
    def __init__(
        self,
        csv_path,
        agent_ids,
        actions,
        alpha = 0.1,
        gamma = 0.99,
        eps_start = 1.0,
        eps_min = 0.05,
        eps_decay = 0.999):

        self.qcsv_path = csv_path
        self.agent_ids = sorted(agent_ids)
        self.n_agents = len(self.agent_ids)

        self.action_space = actions
        self.n_actions = len(self.action_space)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.q = {}
        if csv_path is not None and os.path.exists(csv_path):
            print(f"Loading Q-table from {csv_path}!")
            self.load(csv_path)
        else:
            print("Starting with empty Q-table!")


    def all_possible_joint_actions(self):
        return itertools.product(range(self.n_actions), repeat=self.n_agents)

    def max_q_next(self, next_state_key):
        best = None
        for ja in self.all_possible_joint_actions():
            val = self.q.get((next_state_key, ja), 0.0)
            if best is None or val > best:
                best = val
        return best if best is not None else 0.0

    def select_joint_action(self, state_key):
        if random.random() < self.epsilon:
            joint_action_idx = list(random.randrange(self.n_actions) for i in range(self.n_agents))
        else:
            best_q =  1e-12
            best_joint_actions = []
            for i in self.all_possible_joint_actions():
                q_val = self.q.get((state_key, i), 0.0)
                if (best_q is 1e-12) or (q_val > best_q):
                    best_q = q_val
                    best_joint_actions = [i]
                elif abs(q_val - best_q) <= 1e-12:
                    best_joint_actions.append(i)

            if not best_joint_actions:
                joint_action_idx = list(
                    random.randrange(self.n_actions) for _ in range(self.n_agents))
            else:
                joint_action_idx = random.choice(best_joint_actions)

        joint_action_str = {
            agent_id: self.action_space[joint_action_idx[i]]
            for i, agent_id in enumerate(self.agent_ids)}
        return joint_action_str, joint_action_idx
    
    def update(self, state_key, joint_action_idx, reward, next_state_key,done):

        #Q(s, a) <-(1-alpha) Q(s,a) + alpha [ r + gamma max_a' Q(s', a') ]
        key = (state_key, joint_action_idx)
        old_q = self.q.get(key, 0.0)

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.max_q_next(next_state_key)

        new_q = (1 - self.alpha) * old_q + self.alpha * target
        self.q[key] = new_q

        # epsilon decay
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path = None):
        if path is None:
            path = self.qcsv_path
        if path is None:
            return

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["state", "joint_action", "q"])
            for (state_key, ja), q_val in self.q.items():
                ja_str = ",".join(str(i) for i in ja)
                writer.writerow([state_key, ja_str, q_val])

    def load(self, path = None):
        if path is None:
            path = self.qcsv_path
        if path is None or not os.path.exists(path):
            return

        self.q.clear()
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                state_key = row["state"]
                ja_str = row["joint_action"]
                q_val = float(row["q"])
                ja = tuple(int(x) for x in ja_str.split(",")) if ja_str else tuple()
                self.q[(state_key, ja)] = q_val

    