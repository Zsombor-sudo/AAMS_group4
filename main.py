import irsim
import numpy as np
import metrics as m

env = irsim.make('basic.yaml')
leader = env.robot_list[0]
followers = env.robot_list[1:]


metrics = m.Metrics(n_agents=5, radii=[0.2]*5, dt=0.1,
                    d_collide_margin=0.0, d_safe_margin=0.2,
                    radius_goal=0.2)
metrics.set_goal(np.array([6.8, 6.8]))

# ---  Leader path ---
leader_path = [
    np.array([[2.0],[2.0]]),
    np.array([[3.0],[2.7]]),
    np.array([[4.2],[3.8]]),
    np.array([[5.0],[4.5]]),
    np.array([[6.0],[5.5]]),
    np.array([[6.5],[6.2]]),
    np.array([[6.8],[6.8]]),
]

path_index = 0
target = leader_path[path_index]

# --- config ---
goal = leader_path[-1].reshape(2)   
goal_radius = 0.5            # arrival radius
arrived_s = 0.5              # must arrive for in seconds
arrival_tolarence = 0.1

arrived = [False]*len(env.robot_list)
dwell_steps = [0]*len(env.robot_list)

def update_arrival(i, pos_xy):
    if arrived[i]:
        return
    # pos_xy is (2,1); make it (2,) and compare to goal (2,)
    if np.linalg.norm(pos_xy.reshape(2) - goal) <= goal_radius:
        dwell_steps[i] += 1
        if dwell_steps[i] * arrival_tolarence >= arrived_s:
            arrived[i] = True
    else:
        dwell_steps[i] = 0

# All agent parameters
max_forward = 0.2
max_turn = 0.2
tolerance = 0.05  # distance tolerance to waypoint

# Pheromone parameters
world_size = (8, 8)
scale = 4 
field_h, field_w = world_size[0] * scale, world_size[1] * scale
pheromone_map = np.zeros((field_h, field_w))

deposit_rate = 0.5

alpha = 3.0   # pheromone influence

# freshness map + time
pheromone_age = np.full((field_h, field_w), -np.inf)  # last time stamp
t = 0.0
DT = 0.1  # matches your step time

# stuck watchdog state for ALL robots (leader at 0, followers start=1)
last_score  = [0.0] * len(env.robot_list)
stuck_steps = [0]   * len(env.robot_list)

def to_grid(p):
    x = int(np.clip(round(p[0,0] * scale), 0, field_h - 1))
    y = int(np.clip(round(p[1,0] * scale), 0, field_w - 1))
    return x, y


def deposit_pheromone(pos, amount=None):
    if amount is None:
        amount = deposit_rate
    x, y = to_grid(pos)
    pheromone_map[x, y] += amount
    pheromone_age[x, y] = t  # freshness stamp


def move_follower(idx, follower):
    pos   = follower.state[:2]
    theta = follower.state[2,0]
    x, y  = to_grid(pos)
    
    if arrived[idx]:
        follower.step(np.array([[0.0],[0.0]]))
        return
    
    # --- 360° scan with freshness ---
    radius_cells = 3
    best_score   = 0.0
    best_dir     = np.zeros((2,1))
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= field_h or ny < 0 or ny >= field_w:
                continue
            tau = pheromone_map[nx, ny]
            if tau <= 0.0:
                continue
            dist = np.hypot(dx, dy)
            age = pheromone_age[nx, ny]
            age_penalty = np.exp(-0.8 * max(0.0, t - age))
            score = (tau / dist) * age_penalty
            if score > best_score:
                best_score = score
                best_dir = np.array([[dx],[dy]]) / dist

    # stuck, explore
    EPS_IMPROVE = 1e-3
    STUCK_K = 5
    EXPL_STEPS = 15

    if best_score <= last_score[idx] + EPS_IMPROVE:
        stuck_steps[idx] += 1
    else:
        stuck_steps[idx] = 0
    last_score[idx] = best_score

    if best_score == 0.0 or stuck_steps[idx] >= STUCK_K or stuck_steps[idx] < 0:
        # exploration burst
        rand = np.random.uniform(-1.0, 1.0, size=(2,1))
        u = rand / (np.linalg.norm(rand)+1e-6)
        if stuck_steps[idx] >= STUCK_K:
            stuck_steps[idx] = -EXPL_STEPS
        else:
            stuck_steps[idx] += 1  # count up to 0
    else:
        u = best_dir


    # separation + slowdown (unchanged)
    neighbor_list = follower.get_fov_detected_objects()
    sep = np.zeros((2,1))
    slowdown = 1.0
    if neighbor_list:
        for n in neighbor_list:
            d = (pos - n.state[:2])
            dist = np.linalg.norm(d) + 1e-6
            if dist < 0.7:
                sep += d / (dist**2)
        min_dist = min(np.linalg.norm(n.state[:2] - pos) for n in neighbor_list)
        slowdown = np.clip(min_dist / 1.0, 0.2, 1.0)

    u = u + 0.8 * sep
    if np.linalg.norm(u) < 1e-6:
        follower.step(np.array([[0.0],[0.0]])); return

    target_angle = np.arctan2(u[1,0], u[0,0])
    angle_diff   = (target_angle - theta + np.pi) % (2*np.pi) - np.pi
    v_angular = np.clip(angle_diff, -max_turn, max_turn)
    v_forward = slowdown * 0.15 * np.exp(-3*abs(angle_diff))
    if best_score > 0:   # speed floor if following scent
        v_forward = max(0.06, v_forward)

    follower.step(np.array([[v_forward],[v_angular]]))



for step in range(5000):
    t = step * DT

    # Leader
    pos   = leader.state[:2]
    theta = leader.state[2,0]

    diff = target - pos
    dist = np.linalg.norm(diff)

    for i, r in enumerate(env.robot_list):
        update_arrival(i, r.state[:2])

    # change waypoint if close enough
    if dist < tolerance and path_index < len(leader_path) - 1:
        path_index += 1
        target = leader_path[path_index]
        diff   = target - pos
        dist   = np.linalg.norm(diff)

    if dist > 0:
        target_angle = np.arctan2(diff[1,0], diff[0,0])
        angle_diff   = (target_angle - theta + np.pi) % (2*np.pi) - np.pi

        v_forward = max_forward * np.exp(-3 * abs(angle_diff))
        v_angular = np.clip(angle_diff, -max_turn, max_turn)
        velocity  = np.array([[v_forward], [v_angular]])
    else:
        v_forward = 0.0
        v_angular = 0.0
        velocity  = np.zeros((2,1))

    leader.step(velocity)

    # Pheromone update
    if np.linalg.norm(velocity) > 1e-3:
        boost = 1.0 + 2.0 * (abs(v_angular) / max_turn)
        deposit_pheromone(pos, amount=deposit_rate * boost)

    # Follower
    for idx, f in enumerate(followers, start=1):  # start=1 if leader is index 0
        move_follower(idx, f)

    positions = np.array([r.state[:2,0] for r in env.robot_list])
    metrics.update(positions) 

    env.render()

env.end()
summary = metrics.finalize()
metrics.close()
print(summary)
