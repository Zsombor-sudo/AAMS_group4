
import irsim
import numpy as np
import matplotlib.pyplot as plt
from irsim.util.util import WrapToPi, relative_position
from irsim.world.object_base import ObjectBase
from metrics import Metrics

env = irsim.make('basic_v2.yaml')
leader = env.robot_list[0]
followers = env.robot_list[1:]

leader_max_forward = 1  # leader max forward speed
follower_max_forward = 1
max_turn = 1            # max angular speed

# --- Pheromone parameters ---
world_size = (25, 25)
pheromone_map = np.zeros(world_size)
deposit_rate = 1.0
evaporation_rate = 0.3

m = Metrics(n_agents=len(env.robot_list), tick=0.1, csv_dir="metrics_logs")
m.set_goal(leader.goal)
# --- Functions ---
def elect_new_leader_closest_to_goal(current_leader: ObjectBase, followers: list[ObjectBase]):
    """
    Sample a new random goal and pick the robot (leader or follower) closest to it as new leader.
    Resets pheromone map. Returns (new_leader, followers).
    """
    # Reset pheromone map so followers track new leader only
    global pheromone_map
    pheromone_map = np.zeros_like(pheromone_map)

    # Sample new random goal
    new_goal = np.random.uniform(0, 25, size=(2, 1))
    m.set_goal(new_goal.flatten())

    # Find closest robot to new goal
    candidates = [current_leader] + followers
    dists = []
    for r in candidates:
        dist, _ = relative_position(r.state[:2], new_goal)
        dists.append(dist)
    closest_idx = int(np.argmin(dists))
    new_leader = candidates[closest_idx]
    positions = np.array([r.state[:2,0] for r in env.robot_list])
    m.set_order_by_leader(positions, leader_idx=closest_idx, t_cycle_start=m.time)
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

def deposit_pheromone(pos):
    """
    Leader leaves pheromone at current position.
    FORMAL:  τ_ij ← τ_ij + Δτ_ij
    
    """
    x, y = int(round(pos[0,0])), int(round(pos[1,0]))
    if 0 <= x < world_size[0] and 0 <= y < world_size[1]:
        pheromone_map[x, y] += deposit_rate
    

def evaporate_pheromone():
    
    """
    Global pheromone evaporation.
    FORMAL: τ_ij ← (1 - ρ) * τ_ij
    where ρ = evaporation_rate
    
    """
    global pheromone_map
    pheromone_map *= (1 - evaporation_rate)


def move_follower(follower):
    """
    Follower moves based on pheromone concentrations.
    Slows down if other robots are detected in its FOV.
    """
    # Current position and orientation
    pos = follower.state[:2]
    theta = follower.state[2,0]
    x, y = int(round(pos[0,0])), int(round(pos[1,0]))
    max_r = 25  # maximum expansion radius
    best_cell = None
    best_val = -1.0

    # --- Expand search for pheromone peak ---
    for r in range(1, max_r + 1):
        found = False
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx = x + dx
                cy = y + dy
                if 0 <= cx < world_size[1] and 0 <= cy < world_size[0]:
                    val = pheromone_map[cx, cy]
                    if val > best_val and val > 0:
                        best_val = val
                        best_cell = (cx, cy)
                        found = True
        if found:
            break


    # --- Check neighbors in FOV for slowing down ---
    neighbor_list = follower.get_fov_detected_objects()  # robots in FOV
    slowdown_factor = 1.0  # default (full speed)

    if neighbor_list:
        min_distance = min(
            np.linalg.norm(neighbor.state[:2] - pos) for neighbor in neighbor_list
        )
        # Slowdown factor decreases linearly with distance
        # Closer neighbors -> slower speed, minimum 0.01
        slowdown_factor = max(0.01, min_distance / 1.2)  

  
    target = np.array([[best_cell[0]],[best_cell[1]]], dtype=float)
    _, target_angle = relative_position(pos, target)

    # --- Steering towards pheromone peak ---
    angle_diff = WrapToPi(target_angle - theta)
    v_angular = np.clip(angle_diff, -max_turn, max_turn)

    # --- Repulsion (linear term) ---
    v_forward = follower_max_forward * slowdown_factor

    follower.step(np.array([[v_forward], [v_angular]]))


for step in range(2000):
    
    # Check if leader reached its goal
    leader.check_arrive_status()
    if leader.arrive_flag:
        print(f"Leader reached goal at step {step}.")
        # DEFINE NEW LEADER GOAL
        new_goal = np.random.uniform(0, 25, size=(2,))
        m.set_goal(new_goal.flatten()) 
        leader.set_goal([new_goal[0], new_goal[1], 0])

        #USING SELECT LEADER FUNCTION
        # print(f"Electing new leader.")
        # leader, followers = elect_new_leader_closest_to_goal(leader, followers)
    
    positions = np.array([r.state[:2,0] for r in env.robot_list])  
    m.update(positions)
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

   
    # --- Pheromone update ---
    deposit_pheromone(pos) # τ_ij ← τ_ij + Δτ_ij
    evaporate_pheromone()  # τ_ij ← (1 - ρ) * τ_ij

    # --- Follower behavior ---
    for f in followers:
        move_follower(f)


    env.render()
    
    #if env.done(): break # check if the simulation is done

env.end()


