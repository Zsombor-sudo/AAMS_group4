
import irsim
import numpy as np

env = irsim.make('basic.yaml')
leader = env.robot_list[0]
followers = env.robot_list[1:]

# ---  Leader path ---
leader_path = [
    np.array([[2],[2]]), 
    np.array([[2],[3]]),
    #np.array([[3],[3]]),
    np.array([[4],[3]]),
    #np.array([[4],[4]]),
    np.array([[4],[5]]),
    np.array([[5],[5]]),
    np.array([[5],[6]]), 
    np.array([[6],[6]])  
]

path_index = 0
target = leader_path[path_index]

# ---  Parameters ---
max_forward = 0.2   # max forward speed
max_turn = 0.2      # max angular speed
tolerance = 0.05    # distance tolerance to waypoint

# --- Pheromone parameters ---
world_size = (8, 8)
pheromone_map = np.zeros(world_size)
deposit_rate = 1.0
evaporation_rate = 0.01

alpha = 1.0   # influence of pheromone
beta = 2.0    # influence of heuristic (distance to leader)

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

    # --- Check neighbors in FOV for slowing down ---
    neighbor_list = follower.get_fov_detected_objects()  # robots in FOV
    slowdown_factor = 2.0  # default (full speed)

    if neighbor_list:
        min_distance = min(
            np.linalg.norm(neighbor.state[:2] - pos) for neighbor in neighbor_list
        )
        # Slowdown factor decreases linearly with distance
        # Closer neighbors -> slower speed, minimum 0.05
        slowdown_factor = max(0.01, min_distance / 1.0)  

    # 4-connected neighbors (up, down, left, right)
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    candidates = []
    probs = []

    # Evaluate pheromone on each neighbor
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < world_size[0] and 0 <= ny < world_size[1]:
            tau = pheromone_map[nx, ny]**alpha
            candidates.append((dx, dy))
            probs.append(tau)

    # No valid move or no pheromone
    if len(probs) == 0 or np.sum(probs) == 0:
        return

    # Normalize probabilities
    probs = np.array(probs) / np.sum(probs)
    choice = candidates[np.random.choice(len(candidates), p=probs)]
    target = pos + np.array([[choice[0]], [choice[1]]])

    # Compute heading and angular difference
    diff = target - pos
    target_angle = np.arctan2(diff[1,0], diff[0,0])
    angle_diff = target_angle - theta
    angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi

    # Apply slowdown factor to forward speed
    v_forward = slowdown_factor * 0.15 * np.exp(-3*abs(angle_diff))
    v_angular = np.clip(angle_diff, -max_turn, max_turn)

    # Step the follower
    follower.step(np.array([[v_forward], [v_angular]]))


for step in range(5000):
    # Current state
    pos = leader.state[:2]        # shape (2,1)
    theta = leader.state[2,0]     # orientation in radians

    # Vector to target
    diff = target - pos
    dist = np.linalg.norm(diff)

    # Check if reached current target
    if dist < tolerance and path_index < len(leader_path) - 1:
        path_index += 1
        target = leader_path[path_index]
        diff = target - pos
        dist = np.linalg.norm(diff)

    if dist > 0:
        # Angle to target
        target_angle = np.arctan2(diff[1,0], diff[0,0])
        angle_diff = target_angle - theta
        # Wrap to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi

        # Forward speed reduces when turning
        v_forward = max_forward * np.exp(-3*abs(angle_diff))
        v_angular = np.clip(angle_diff, -max_turn, max_turn)

        # Create velocity vector (shape (2,1)) [forward, angular]
        velocity = np.array([[v_forward],[v_angular]])
    else:
        velocity = np.zeros((2,1))

    # Step the robot
    leader.step(velocity)
   
    # --- Pheromone update ---
    deposit_pheromone(pos) # τ_ij ← τ_ij + Δτ_ij
    evaporate_pheromone()  # τ_ij ← (1 - ρ) * τ_ij

    # --- Follower behavior ---
    for f in followers:
        move_follower(f)


    env.render()

env.end()
