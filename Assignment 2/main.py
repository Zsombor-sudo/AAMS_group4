import irsim
import numpy as np
from irsim.util.util import WrapToPi, relative_position
from irsim.world.object_base import ObjectBase

env = irsim.make('setup.yaml')
leader = env.robot_list[0]
followers = env.robot_list[1:]

# Set random goals for leader, no specific goal for followers
# Random pos between (10,10) and (24,24) 
leader_goal = np.random.uniform(10, 24, size=(2, 1))
leader.set_goal(leader_goal.flatten().tolist() + [0]) # z=0

for follower in followers:
    follower.set_goal([-1,-1,0])  # No specific goal for followers (out of bounds)
env.reset_plot()

# ---  Parameters ---
leader_max_forward = 1  # leader max forward speed
follower_max_forward = 0.9
max_turn = 1            # max angular speed

# --- Pheromone parameters ---
world_size = (25, 25)
pheromone_map = np.zeros(world_size, dtype=float)
deposit_rate = 1.0
evaporation_rate = 0.1
max_pheromone = 10.0

# --- Follower formation preferences ---
desired_spacing = 1.0     # preferred separation between robots
slowdown_min = 0.12
slowdown_max = follower_max_forward

def elect_new_leader_closest_to_goal(current_leader: ObjectBase, followers: list[ObjectBase]):
    """
    Sample a new random goal and pick the robot (leader or follower) closest to it as new leader.
    Resets pheromone map. Returns (new_leader, followers).
    """
    # Reset pheromone map so followers track new leader only
    global pheromone_map
    pheromone_map = np.zeros_like(pheromone_map)

    # Sample new random goal
    new_goal = np.random.uniform(10, 24, size=(2, 1))

    # Find closest robot to new goal
    candidates = [current_leader] + followers
    dists = []
    for r in candidates:
        dist, _ = relative_position(r.state[:2], new_goal)
        dists.append(dist)
    closest_idx = int(np.argmin(dists))
    new_leader = candidates[closest_idx]

    # If the new leader was a follower, adjust lists
    if new_leader is not current_leader:
        followers.remove(new_leader)
        followers.append(current_leader)
        new_leader.color = 'r'

        # Demote current leader to follower
        current_leader.color = 'g'
        current_leader.set_goal([-1,-1,0])  # no specific goal for follower
    # Else leader stays leader; followers unchanged

    # Assign goal to new leader
    new_leader.set_goal(new_goal.flatten().tolist() + [0])
    print(f"New leader (closest to goal) elected: Robot ID {new_leader.id}")
    return new_leader, followers

def elect_new_leader_random(current_leader: ObjectBase, followers: list[ObjectBase]):
    """
    Randomly pick a new leader from the follower list.
    Returns new leader and updated followers list.
    """
    # Reset pheromone map so followers track new leader only
    global pheromone_map
    pheromone_map = np.zeros_like(pheromone_map)

    new_leader_idx = np.random.randint(0, len(followers)-1)
    new_leader = followers.pop(new_leader_idx)
    followers.append(current_leader)  # demote current leader to follower

    new_leader.color = 'r'      # change color to red
    current_leader.color = 'g'  # change color to green

    # Reset goals for new leader and follower
    new_leader_goal = np.random.uniform(0, 24, size=(2, 1))
    new_leader.set_goal(new_leader_goal.flatten().tolist() + [0])  # z=0
    current_leader.set_goal([-1,-1,0])  # no specific goal for follower

    print(f"New leader elected: Robot ID {new_leader.id}")
    return new_leader, followers

# --- Pheromone functions ---
def deposit_pheromone(pos):
    """
    Leader drops pheromone at closest grid cell.
    NOTE: pheromone_map indexed as [row=y, col=x] (numpy convention).
    pos is a 2x1 column vector (x,y).
    """
    x = int(round(float(pos[0, 0])))
    y = int(round(float(pos[1, 0])))
    if 0 <= x < world_size[1] and 0 <= y < world_size[0]:
        pheromone_map[y, x] = min(max_pheromone, pheromone_map[y, x] + deposit_rate)

def evaporate_pheromone():
    """
    Global pheromone evaporation.
    FORMAL: τ_ij ← (1 - ρ) * τ_ij
    where ρ = evaporation_rate
    """
    global pheromone_map
    pheromone_map *= (1 - evaporation_rate)

def compute_repulsion(follower: ObjectBase, max_vel, decay=3.0):
    """
    Smooth Gaussian repulsion from nearby robots.
    The closer they are, the stronger the push away.
    """
    state = follower.state
    linear = 0.0
    neighbor_list = follower.get_fov_detected_objects()

    if len(neighbor_list) == 0:
        return 0.0

    for neighbor in neighbor_list:
        distance, radian = relative_position(state[:2], neighbor.state[:2])
        diff_radian = WrapToPi(radian - state[2, 0])
        
        # Repulsive term — pushes away if neighbor is close
        # The exp(- (distance/decay)^2) makes it strong when distance < decay
        # Math: (v_max * -cos(diff_angle) * exp(- (d/decay)^2)) / (2 * N) where N is number of neighbors
        linear += (max_vel[0, 0] * (-np.cos(diff_radian)) * np.exp(- (distance / decay)**2)) / (2 * len(neighbor_list))

    return linear

def move_follower(follower: ObjectBase):
    """
    Follower samples pheromone concentration in three directions (ahead, left, right)
    and turns toward the strongest scent.
    """
    pos = follower.state[:2]
    theta = float(follower.state[2, 0])

    # --- Expand search for pheromone peak ---
    x = int(round(float(pos[0,0])))
    y = int(round(float(pos[1,0])))
    max_r = 25  # maximum expansion radius
    best_cell = None
    best_val = -1.0
    for r in range(1, max_r + 1):
        found = False
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx = x + dx
                cy = y + dy
                if 0 <= cx < world_size[1] and 0 <= cy < world_size[0]:
                    val = pheromone_map[cy, cx]
                    if val > best_val and val > 0:
                        best_val = val
                        best_cell = (cx, cy)
                        found = True
        if found:
            break

    target = np.array([[best_cell[0]],[best_cell[1]]], dtype=float)
    _, target_angle = relative_position(pos, target)

    # --- Steering towards pheromone peak ---
    angle_diff = WrapToPi(target_angle - theta)
    v_angular = np.clip(angle_diff, -max_turn, max_turn)

    # --- Repulsion (linear term) ---
    v_forward = follower_max_forward + compute_repulsion(follower, np.array([[follower_max_forward]]))
    v_forward = np.clip(v_forward, -follower_max_forward, follower_max_forward)

    follower.step(np.array([[v_forward], [v_angular]]))

for step in range(3000):
    # Check if leader reached its goal
    leader.check_arrive_status()
    if leader.arrive_flag:
        print(f"Leader reached goal at step {step}. Electing new leader.")
        leader, followers = elect_new_leader_closest_to_goal(leader, followers)

    pos = leader.state[:2]
    theta = leader.state[2,0]
    goal_pos = leader.goal[0:2]
    dist, target_angle = relative_position(pos, goal_pos)
    angle_diff = WrapToPi(target_angle - theta)

    if dist > leader.goal_threshold:
        v_forward = min(leader_max_forward, float(dist))
        v_angular = np.clip(angle_diff, -max_turn, max_turn)
        leader.step(np.array([[v_forward],[v_angular]]))
    else:
        leader.step(np.zeros((2,1)))

    # Pheromone update
    deposit_pheromone(pos) # τ_ij ← τ_ij + Δτ_ij
    evaporate_pheromone()  # τ_ij ← (1 - ρ) * τ_ij

    for follower in followers:
        move_follower(follower)

    env.render()

env.end()