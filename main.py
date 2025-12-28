import argparse
import irsim
import numpy as np
from apple import Apple
from irsim.world.object_base import ObjectBase
from irsim.world.world import World
from matplotlib import pyplot as plt
from utils import clear_labels, draw_grid, init_labels, update_labels
from irsim.util.util import relative_position
from pathlib import Path
import random
from enum import Enum

CELL_SIZE = 1.0
MOVE_SPEED = 1.0 # max=1.0
EPS = 1e-3
ACTION_SPACE = ["up", "down", "left", "right", "collect", "noop"]
NO_MOVE_ACTIONS = {"collect", "noop"}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "display"], default="display")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--qcsv", default="q_table.csv")
args = parser.parse_args()

NUM_EPISODES, NUM_STEPS = args.episodes, args.steps
# NUM_EPISODES = 2
# NUM_STEPS = 200


# Learning parameters
epsilon = 0.1
alpha = 0.1
gamma = 0.95

GRID_WIDTH = 13 
GRID_HEIGHT = 9  

# --- Initialize environment, agents and apples ---
env = irsim.make('setup.yaml')
apples: list[Apple] = []
next_apple_id: int = 0
agents = env.robot_list
[setattr(a, "level", 1) for a in agents] # Could maybe be removed 

class AgentState(Enum):
    EXPLORING = "exploring"
    WAITING = "waiting"
    MOVING = "moving"

# Per-agent motion state
motion_state = { 
    a.id: { 
        "state": AgentState.EXPLORING, "target_pos": None, "prev_state": None, "prev_action": None, "accumulated_reward": 0.0} for a in agents 
}
agent_labels: dict[int, plt.Text] = {}
apple_labels: dict[int, plt.Text] = {}

# q table for each agent
folder_path = Path(__file__).parent / 'q_tables'
folder_path.mkdir(exist_ok=True)
q_tables = []

def pos_to_state(x, y):
    return int(y * GRID_WIDTH + x)

def state_to_pos(state):
    x = state % GRID_WIDTH
    y = state // GRID_WIDTH
    return (x, y)

NUM_STATES = GRID_WIDTH * GRID_HEIGHT  # 117 state
for a in agents:
    fileName = f'q_table{a.id}.csv'
    file_path = folder_path / fileName
    if not file_path.exists():
        q_tables.append(np.zeros([NUM_STATES, len(ACTION_SPACE)]))
    else:
        q_tables.append(np.loadtxt(file_path, delimiter=','))

def spawn_apple(x, y, level=1):
    global next_apple_id
    apple = Apple(next_apple_id, x, y, level)
    env.add_object(apple)
    apples.append(apple)
    next_apple_id += 1
    return apple

def spawn_random_apple(level=1):
    w: World = env._world
    x = int(np.random.uniform(0, w.width+1))
    y = int(np.random.uniform(0, w.height+1))
    return spawn_apple(x, y, level)

def clear_apples():
    global apples
    for apple in apples:
        if not apple.collected:
            env.delete_object(apple.id)
    apples = []

def cell_pos(agent: ObjectBase):
    return (int(round(agent.state[0,0])), int(round(agent.state[1,0])))

def prev_cell_pos(agent: ObjectBase):
    return (int(round(agent.prev_state[0,0])), int(round(agent.prev_state[1,0])))

def adjacent_to_apple(agent: ObjectBase, apple: Apple):
    ax, ay = int(apple.state[0,0]), int(apple.state[1,0])
    x, y = cell_pos(agent)
    return abs(x - ax) + abs(y - ay) == 1

def get_direction(action: str):
    match action:
        case "up":    return np.array([0.0, 1.0])
        case "down":  return np.array([0.0, -1.0])
        case "left":  return np.array([-1.0, 0.0])
        case "right": return np.array([1.0, 0.0])
        case "collect" | "noop": return np.array([0.0, 0.0])

def get_target_pos(agent: ObjectBase, action: str):
    x, y = cell_pos(agent)
    dir = get_direction(action)
    return (x + int(dir[0]), y + int(dir[1]))

def nearest_dist_from_xy(x, y):
    best = None
    for ap in apples:
        if ap.collected:
            continue
        ax, ay = int(ap.state[0,0]), int(ap.state[1,0])
        d = abs(x-ax) + abs(y-ay)
        best = d if best is None else min(best, d)
    return best

def help_needed(agent: ObjectBase):
    for apple in apples:
        if apple.collected:
            continue
        if adjacent_to_apple(agent, apple):
            total = sum(a.level for a in agents if adjacent_to_apple(a, apple))
            if total < apple.level:
                return True, apple.level, total
            else:
                return False, apple.level, total
    return False, None, None

def is_valid_action(agent: ObjectBase, action: str):
    if action == "collect":
        # Only if next to an uncollected apple
        return any((not ap.collected) and adjacent_to_apple(agent, ap) for ap in apples)
    
    if action in NO_MOVE_ACTIONS:
        return True
    
    occupied_pos = { cell_pos(a) for a in agents if a.id != agent.id }
    
    reserved_targets = { 
        motion_state[a.id]["target_pos"] 
        for a in agents 
        if a.id != agent.id and motion_state[a.id]["target_pos"] is not None
    }
    occupied_pos = occupied_pos.union(reserved_targets)
    
    # can't move into apple cell
    apple_pos = { (int(ap.state[0,0]), int(ap.state[1,0])) for ap in apples if not ap.collected }
    occupied_pos = occupied_pos.union(apple_pos)
    
    target = get_target_pos(agent, action)
    if target in occupied_pos:
        return False
    
    x, y = target
    w: World = env._world
    x0, x1 = w.x_range
    y0, y1 = w.y_range
    return (x0 <= x <= x1) and (y0 <= y <= y1)

def calculate_reward(agent: ObjectBase, action: str):
    reward = -0.1  # Negative reward as time goes
    
    # Need help check for higher level eapples
    need_help, apple_lvl, total_lvl = help_needed(agent)
    if need_help:
        if action == "noop":
            return 1  # Some sort of positive reward to wait next to an apple, may need tweeking 
        if action in ["up", "down", "left", "right"]:
            for ap in apples:
                if not ap.collected and adjacent_to_apple(agent, ap):
                    tx, ty = get_target_pos(agent, action)
                    if abs(tx - int(ap.state[0,0])) + abs(ty - int(ap.state[1,0])) != 1:
                        reward -= 0.3
                    break
    
    if action == "collect":
        collected = False
        for apple in apples:
            if not apple.collected and adjacent_to_apple(agent, apple):
                # Get all adjacent agents
                adjacent_agents = [a for a in agents if adjacent_to_apple(a, apple)]
                total_level = sum(a.level for a in adjacent_agents)
                
                if total_level >= apple.level:
                    apple.collect()
                    env.delete_object(apple.id)
                    collected = True
                    
                    shared_reward = (10.0 * apple.level) / len(adjacent_agents)
                    
                    for a in adjacent_agents:
                        a.level += 1
                        print(f"Agent {a.id} collected apple (lvl {apple.level}, {cell_pos(apple)})")
                        
                        # If this is not the current agent, give them the reward directly
                        if a.id != agent.id:
                            # Store reward for their next Q update
                            motion_state[a.id]["accumulated_reward"] += shared_reward
                    
                    reward = shared_reward
                    break
        
        if not collected:
            reward = -5.0  # Failed collection
    
    elif action == "noop":
        reward = -0.5  # Small penalty for doing nothing, needs experimentation
    
    else:  # Movement 
        x, y = cell_pos(agent)
        tx, ty = get_target_pos(agent, action)

        #oscilliating penality
        px, py = prev_cell_pos(agent)
        if px==tx and py==ty:
            print("Osc")
            return reward - 2
        
        d0 = nearest_dist_from_xy(x, y)
        d1 = nearest_dist_from_xy(tx, ty)

        if d0 is not None and d1 is not None:
            reward += 0.5 * (d0 - d1)  # closer => positive  farther => negative
        
    
    return reward

def begin_action(agent: ObjectBase, action: str):
    curr_state = motion_state[agent.id]
    x, y = cell_pos(agent)
    
    curr_state["prev_state"] = pos_to_state(x, y)
    curr_state["prev_action"] = ACTION_SPACE.index(action)
    
    target = get_target_pos(agent, action)
    curr_state["target_pos"] = target
    curr_state["state"] = AgentState.MOVING

def progress_motion(agent: ObjectBase):
    curr_state = motion_state[agent.id]
    if curr_state["state"] != AgentState.MOVING:
        return
    
    pos = agent.state[:2].flatten()
    target = np.array(curr_state["target_pos"])
    diff = target - pos
    dist = np.linalg.norm(diff)
    
    if dist < EPS:

        if int(agent.state[0, 0]) != int(target[0]) or int(agent.state[1, 0]) != int(target[1]):
            agent.prev_state[0,0] = agent.state[0,0]
            agent.prev_state[1,0] = agent.state[1,0]
        
        # Snap to target position
        agent.state[0, 0] = target[0]
        agent.state[1, 0] = target[1]
        
        if curr_state["prev_state"] is not None:
            prev_state = curr_state["prev_state"]
            prev_action = curr_state["prev_action"]
            
            # Use accumulated reward (includes own action + the help bonuses!!!)
            reward = curr_state["accumulated_reward"]
            
            new_x, new_y = cell_pos(agent)
            new_state = pos_to_state(new_x, new_y)
            
            # Q learning update
            old_value = q_tables[agent.id][prev_state, prev_action]
            next_max = np.max(q_tables[agent.id][new_state, :])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_tables[agent.id][prev_state, prev_action] = new_value
            curr_state["accumulated_reward"] = 0.0
        
        curr_state["state"] = AgentState.EXPLORING
        curr_state["target_pos"] = None
        return
    
    direction = diff / dist
    vel = direction * MOVE_SPEED
    agent.step(vel.reshape(2, 1))

def step_agent(agent: ObjectBase):
    curr_state = motion_state[agent.id]
    
    if curr_state["state"] == AgentState.EXPLORING:
        x, y = cell_pos(agent)
        state = pos_to_state(x, y)
        
        valid_actions = [i for i, a in enumerate(ACTION_SPACE) if is_valid_action(agent, a)]
        
        if not valid_actions:
            return 
        
        if random.random() < epsilon:
            actionNum = random.choice(valid_actions)
        else:
            q_values = q_tables[agent.id][state, :]
            best_q = max(q_values[i] for i in valid_actions)
            best_actions = [i for i in valid_actions if q_values[i] == best_q]
            actionNum = random.choice(best_actions)
        
        action = ACTION_SPACE[actionNum]
        
        reward = calculate_reward(agent, action)
        motion_state[agent.id]["accumulated_reward"] += reward
        
        begin_action(agent, action)
        
        if action in NO_MOVE_ACTIONS:
            progress_motion(agent)
    
    elif curr_state["state"] == AgentState.MOVING:
        progress_motion(agent)
    
    elif curr_state["state"] == AgentState.WAITING:
        pass

# --- Main loop ---
ax = plt.gca()

for ep in range(NUM_EPISODES):
    env.reset()
    
    # Spawn apples
    spawn_apple(5, 5, level=2)
    spawn_apple(8, 3, level=1)
    spawn_apple(2, 7, level=3)
    spawn_apple(10, 6, level=1)
    
    # Initialize agents
    agents[0].state[0,0] = 0
    agents[0].state[1,0] = 0
    agents[1].state[0,0] = 12
    agents[1].state[1,0] = 0
    agents[2].state[0,0] = 0
    agents[2].state[1,0] = 8
    agents[3].state[0,0] = 12
    agents[3].state[1,0] = 8
    
    agents[0].prev_state = agents[0].state
    agents[1].prev_state = agents[1].state
    agents[2].prev_state = agents[2].state
    agents[3].prev_state = agents[3].state
    for agent in agents:
        agent.level = 1
        motion_state[agent.id]["state"] = AgentState.EXPLORING
        motion_state[agent.id]["target_pos"] = None
        motion_state[agent.id]["prev_state"] = None
        motion_state[agent.id]["prev_action"] = None
        motion_state[agent.id]["accumulated_reward"] = 0.0
    
    if args.mode == "display":
        env.reset_plot()
        draw_grid(ax, env, CELL_SIZE)
        env.render()
        agent_labels, apple_labels = init_labels(ax, agents, apples)
    
    for step in range(NUM_STEPS):
        if all(a.collected for a in apples):
            print(f"All apples collected in episode {ep} at step {step}.")
            break
        
        for agent in agents:
            step_agent(agent)
        
        if args.mode == "display":
            env.render()
            ax.set_title(f"Episode {ep+1} | Step {step+1}")
            update_labels(ax, agent_labels, apple_labels, agents, apples)
    
    clear_apples()
    if args.mode == "display":
        clear_labels(agent_labels, apple_labels)
    print(f"Episode {ep+1} finished.")

print("Simulation ended.")

if args.mode == "train":
    for a in agents:
        fileName = f'q_table{a.id}.csv'
        np.savetxt(folder_path / fileName, q_tables[a.id], delimiter=',', fmt='%f')
    print("Q-tables saved.")

env.end()