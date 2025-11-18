
import irsim
import numpy as np
import matplotlib.pyplot as plt
from irsim.util.util import WrapToPi, relative_position
from irsim.world.object_base import ObjectBase

env = irsim.make('basic.yaml')
leader = env.robot_list[0]
followers = env.robot_list[1:]

leader_max_forward = 1  # leader max forward speed
follower_max_forward = 0.9
max_turn = 1            # max angular speed

# --- Pheromone parameters ---
world_size = (25, 25)
pheromone_map = np.zeros(world_size)
deposit_rate = 1.0
evaporation_rate = 0.3

# --- Functions ---


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
    slowdown_factor = 2.0  # default (full speed)

    if neighbor_list:
        min_distance = min(
            np.linalg.norm(neighbor.state[:2] - pos) for neighbor in neighbor_list
        )
        # Slowdown factor decreases linearly with distance
        # Closer neighbors -> slower speed, minimum 0.01
        slowdown_factor = max(0.01, min_distance / 1.0)  

  
    target = np.array([[best_cell[0]],[best_cell[1]]], dtype=float)
    _, target_angle = relative_position(pos, target)

    # --- Steering towards pheromone peak ---
    angle_diff = WrapToPi(target_angle - theta)
    v_angular = np.clip(angle_diff, -max_turn, max_turn)

    # --- Repulsion (linear term) ---
    v_forward = follower_max_forward * slowdown_factor

    follower.step(np.array([[v_forward], [v_angular]]))


for step in range(5000):
    
    # Check if leader reached its goal
    leader.check_arrive_status()
    if leader.arrive_flag:
        print(f"Leader reached goal at step {step}. Electing new leader.")
        # DEFINE NEW LEADER GOAL
        new_goal = np.random.uniform(0, 25, size=(2,))
        leader.set_goal([new_goal[0], new_goal[1], 0])

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

    ax = plt.gca()
   
    ax.grid(True)

env.end()


