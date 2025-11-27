import argparse

import irsim
import numpy as np
from apple import Apple
from irsim.world.object_base import ObjectBase
from irsim.world.world import World
from matplotlib import pyplot as plt
from utils import clear_labels, draw_grid, init_labels, update_labels

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

# --- Initialize environment, agents and apples ---
env = irsim.make('setup.yaml')
apples: list[Apple] = []
next_apple_id: int = 0
agents = env.robot_list
[setattr(a, "level", 1) for a in agents] 

# Per-agent motion state: Game Theory assumption??? since all agents know others' states
motion_state = { a.id: { "moving": False, "target_pos": None } for a in agents }
agent_labels: dict[int, plt.Text] = {}
apple_labels: dict[int, plt.Text] = {}

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

# --- Core functions ---
def cell_pos(agent: ObjectBase):
    return (int(round(agent.state[0,0])), int(round(agent.state[1,0])))

def adjacent_to_apple(agent: ObjectBase, apple: Apple):
    ax, ay = int(apple.state[0,0]), int(apple.state[1,0])
    x, y = cell_pos(agent)
    return abs(x - ax) + abs(y - ay) == 1 # Ensures no diagonal adjacency

def get_direction(action: str):
    match action:
        case "up":    return np.array([0.0, 1.0])
        case "down":  return np.array([0.0, -1.0])
        case "left":  return np.array([-1.0, 0.0])
        case "right": return np.array([1.0, 0.0])
        case "collect" | "noop": return np.array([0.0, 0.0])
        case _: raise ValueError(f"Unknown action {action}")

def get_target_pos(agent: ObjectBase, action: str):
    x, y = cell_pos(agent)
    dir = get_direction(action)
    return (x + int(dir[0]), y + int(dir[1])) 

def is_valid_action(agent: ObjectBase, action: str):
    if action in NO_MOVE_ACTIONS:
        #TODO: Check collect validity? For now always valid
        return True  # noop always valid

    # Check other agents' positions and targets
    occupied_pos = { cell_pos(a) for a in agents }
    reserved_targets = { 
        motion_state[a.id]["target_pos"] 
        for a in agents 
        if motion_state[a.id]["moving"] and motion_state[a.id]["target_pos"] is not None
    }
    occupied_pos = occupied_pos.union(reserved_targets)

    # Check apple positions
    apple_pos = { cell_pos(a) for a in apples if not a.collected }
    occupied_pos = occupied_pos.union(apple_pos)
    
    target = get_target_pos(agent, action)
    if target in occupied_pos:
        return False
    
    # Check world boundaries
    x, y = target
    w: World = env._world
    x0, x1 = w.x_range
    y0, y1 = w.y_range
    return (x0 <= x <= x1) and (y0 <= y <= y1)

def begin_action(agent: ObjectBase, action: str):
    curr_state = motion_state[agent.id]
    if action in NO_MOVE_ACTIONS:
        curr_state["moving"] = False

        if action == "collect":
            for apple in apples:
                if not apple.collected and adjacent_to_apple(agent, apple) and agent.level >= apple.level:
                    apple.collect()
                    env.delete_object(apple.id)
                    spawn_random_apple(apple.level)  # Spawn new apple
                    
                    if args.mode == "display": # Fix display after apple collection
                        env.reset_plot()
                        draw_grid(ax, env, CELL_SIZE)

                        global agent_labels, apple_labels
                        clear_labels(agent_labels, apple_labels)
                        agent_labels, apple_labels = init_labels(ax, agents, apples)
                        env.render()
                    
                    print(f"Agent {agent.id} of level {agent.level} collected level {apple.level} apple at {cell_pos(apple)}")
                    #TODO: Simple level up for testing, needs reward system
                    agent.level += 1 
        return
    
    target = get_target_pos(agent, action)
    curr_state["moving"] = True
    curr_state["target_pos"] = target

def progress_motion(agent: ObjectBase):
    curr_state = motion_state[agent.id]
    if not curr_state["moving"]:
        return
    
    pos = agent.state[:2].flatten()
    target = curr_state["target_pos"] 
    diff = target - pos
    dist = np.linalg.norm(diff)
    
    # Reached target
    if dist < EPS:
        # Snap to target position to avoid drift
        agent.state[0, 0] = target[0]
        agent.state[1, 0] = target[1]
        curr_state["moving"] = False
        return

    direction = diff / dist
    vel = direction * MOVE_SPEED
    agent.step(vel.reshape(2, 1)) # [[vx], [vy]]

def step_agent(agent: ObjectBase):
    curr_state = motion_state[agent.id]
    if not curr_state["moving"]:
        action = np.random.choice(ACTION_SPACE)
        while not is_valid_action(agent, action):
            action = np.random.choice(ACTION_SPACE)
        begin_action(agent, action)
    progress_motion(agent)


# --- Main loop ---
ax = plt.gca()
for ep in range(NUM_EPISODES):
    env.reset()
    # Spawn apples
    spawn_random_apple(1)
    spawn_random_apple(1)
    spawn_random_apple(2)
    spawn_random_apple(3)

    # Randomize agent positions
    for agent in agents:
        agent.state[0,0] = int(np.random.uniform(0, env._world.width+1))
        agent.state[1,0] = int(np.random.uniform(0, env._world.height+1))
        agent.level = 1  # reset level
        motion_state[agent.id]["moving"] = False
        motion_state[agent.id]["target_pos"] = None

    if args.mode == "display":
        env.reset_plot()
        draw_grid(ax, env, CELL_SIZE)
        env.render()
        agent_labels, apple_labels = init_labels(ax, agents, apples)

    for step in range(NUM_STEPS):
        if all(a.collected for a in apples):
            print(f"All apples collected in episode {ep} at step {step}.")
            env.done()
            break

        for agent in agents:
            step_agent(agent)

        if args.mode == "display":
            env.render()
            ax.set_title(f"Episode {ep+1} | Step {step+1}")
            update_labels(ax, agent_labels, apple_labels, agents, apples)

    clear_apples()
    clear_labels(agent_labels, apple_labels)
    print(f"Episode {ep+1} finished.")

print("Simulation ended.")
# env.end()