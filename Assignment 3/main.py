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

# Q-learning is fully implemented, i just need to train and tweak stuff

# One problem remains in the form of when a high level apple is collected, 
# it is collected purely based on whether the total level of agents around
# the apple is sufficient, of course one agent is preforming the collect action,
# but all agents get the reward, so some agents will get a reward for performing
# a random other action, which is probably not ideal

# After doing some training, it seems that the agents get stuck between two cells.
# It seems that the agent learns to go back and fourth between two cells and gets 
# stuck like that. So in cell 1 the agent has learned that it gets the best reward
# when moving left, but in the left cell, it has learned that the highest reward is
# achieved by moving right, so it just moves back and fourth between the two.
# This is probably fixable by just tweaking the reward values as well as alpha and gamma,
# but maybe some extra check to punish this specific behaviour is needed.

# Other ideas:
# Maybe some type of decreasing reward could help. Like it starts 
# at negative -1 and then for each simulation step, it decreases 
# by -0.1 or something
#
# Some type of check for if the opposite action has been chosen, 
# could also be implemented. So it would be punished harder if it 
# goes left then right, as it rarely makes sense to go backwards

CELL_SIZE = 1.0
MOVE_SPEED = 1.0 # max=1.0
EPS = 1e-3
# ACTION_SPACE = ["up", "down", "left", "right", "collect", "noop"]
# NO_MOVE_ACTIONS = {"collect", "noop"}
ACTION_SPACE = ["right", "left", "up", "down", "collect"]
NO_MOVE_ACTIONS = {"collect"}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "display"], default="train")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--qcsv", default="q_table.csv")
args = parser.parse_args()
NUM_EPISODES, NUM_STEPS = args.episodes, args.steps
# NUM_EPISODES = 2
# NUM_STEPS = 200

# World size (Make sure these match the values in the .yaml file)
height = 8
width = 12

# Learning parameters
epsilon = 1
alpha = 0.5
gamma = 0.9

# --- Initialize environment, agents and apples ---
env = irsim.make('setup.yaml')
apples: list[Apple] = []
next_apple_id: int = 0
agents = env.robot_list
[setattr(a, "level", 1) for a in agents]    # Could maybe be removed 

# Define agent states:
class AgentState(Enum):
    EXPLORING = "exploring"
    WAITING = "waiting"
    MOVING = "moving"

# Per-agent motion state: Game Theory assumption??? since all agents know others' states
motion_state = { a.id: { "state": AgentState.EXPLORING, "target_pos": None, "reward": -2, "apples": 0 } for a in agents }
agent_labels: dict[int, plt.Text] = {}
apple_labels: dict[int, plt.Text] = {}

# Create/Load Q table for each agent:
folder_path = Path(__file__).parent / 'q_tables'

q_tables = []
for a in agents:
    fileName = f'q_table{a.id}_0.csv'
    # print(fileName)
    # If it doesn't exist, create a Q table
    if not (folder_path / fileName).exists():
        q_tables.append(np.zeros([117, len(ACTION_SPACE)]))  # 13*9=117 states
        # print(f'{fileName} created')
    else: # Otherwise load the Q table:
        q_tables.append(np.loadtxt(folder_path / fileName, delimiter=','))
        # print(f'{fileName} loaded')

def increase_collected_apples(agent: ObjectBase):
    # Save the Q_table used until now
    fileName = f'q_table{agent.id}_{motion_state[agent.id]["apples"]}.csv'
    np.savetxt(folder_path / fileName, q_tables[agent.id], delimiter=',', fmt='%f')
    # print(f'{fileName} saved')

    # Update collected apples and Q_table file name
    motion_state[agent.id]["apples"] += 1
    fileName = f'q_table{agent.id}_{motion_state[agent.id]["apples"]}.csv'

    # If it doesn't exist, create a Q table
    if not (folder_path / fileName).exists():
        q_tables[agent.id] = np.zeros([117, len(ACTION_SPACE)])  # 13*9=117 states
        # print(f'{fileName} created')
    else: # Otherwise load the Q table:
        q_tables[agent.id] = np.loadtxt(folder_path / fileName, delimiter=',')
        # print(f'{fileName} loaded')

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
        if motion_state[a.id]["state"] and motion_state[a.id]["target_pos"] is not None
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
    # curr_state = motion_state[agent.id]
    
    if action in NO_MOVE_ACTIONS:
        if action == "collect":
            # print(f"Agent {agent.id} attempted to collet an apple")
            
            # Reduce reward to punish standing still
            motion_state[agent.id]["reward"] -= 1

            # Check for apples and collect if possible
            for apple in apples:
                if not apple.collected and adjacent_to_apple(agent, apple):
                    motion_state[agent.id]["reward"] = 1
                    total_level = 0
                    adjacent_agents = []
                    # Check for agents surrounding the apple
                    for a in agents:
                        dist, _ = relative_position(a.state, apple.state)
                        if dist <= 1.25:
                            adjacent_agents.append(a)
                            # Calculate total level of agents around the apple
                            total_level += a.level

                    # Check if total level is enough to collect apple
                    if total_level >= apple.level:
                        # Collect if possible
                        apple.collect()
                        env.delete_object(apple.id)
                        
                        for a in adjacent_agents:
                            # print(f"Agent {a.id} of level {a.level} collected level {apple.level} apple at {cell_pos(apple)}")
                            #TODO: Simple level up for testing, needs reward system
                            a.level += 1
                            motion_state[a.id]["state"] = AgentState.EXPLORING
                            
                            # Set agent rewards
                            motion_state[a.id]["reward"] = 100
                            
                            # Update apple state for each agent
                            increase_collected_apples(a)
                            
                    # else:
                    #     # Otherwise set this agent to wait
                    #     curr_state["state"] = AgentState.WAITING
        # else:
            # print(f"Agent {agent.id} choose to stand still")
                    
        return

    target = get_target_pos(agent, action)
    motion_state[agent.id]["state"] = AgentState.MOVING
    motion_state[agent.id]["target_pos"] = target

def progress_motion(agent: ObjectBase):
    # curr_state = motion_state[agent.id]
    if not motion_state[agent.id]["state"] == AgentState.MOVING:
        return
    
    pos = agent.state[:2].flatten()
    target = motion_state[agent.id]["target_pos"] 
    diff = target - pos
    dist = np.linalg.norm(diff)
    
    # Reached target
    if dist < EPS:
        # Snap to target position to avoid drift
        agent.state[0, 0] = target[0]
        agent.state[1, 0] = target[1]
        motion_state[agent.id]["state"] = AgentState.EXPLORING
        return

    direction = diff / dist
    vel = direction * MOVE_SPEED
    agent.step(vel.reshape(2, 1)) # [[vx], [vy]]

def step_agent(agent: ObjectBase):
    curr_state = motion_state[agent.id]["state"]

    match curr_state:
        case AgentState.EXPLORING:
            # Get the current state
            x, y = cell_pos(agent)
            state = x + (y*(width+1))
            # print(state)

            # Pick Q-learning action
            if random.random() < epsilon:
                # Pick random action
                actionNum = random.randrange(len(ACTION_SPACE))
                action = ACTION_SPACE[actionNum]

                # Repick a random action until a valid action is picked
                while not is_valid_action(agent, action):
                    actionNum = random.randrange(len(ACTION_SPACE))
                    action = ACTION_SPACE[actionNum]
            else:
                # Pick the best action according to Q-table
                actionNum = np.argmax(q_tables[agent.id][state, :])
                action = ACTION_SPACE[actionNum]

                # If the action is invalid, punish it heavily and pick a new one
                while not is_valid_action(agent, action):
                    q_tables[agent.id][state, actionNum] -= 100
                    actionNum = np.argmax(q_tables[agent.id][state, :])
                    action = ACTION_SPACE[actionNum]

            # Start the choosen action
            begin_action(agent, action)

            # Calculate reward
            # The reward is saved in the motion_state thingy
            # Here the reward value will be updated if the agent 
            # does something that is worth rewarding (like picking 
            # up an apple), and then when a q_value is being 
            # calculated, the reward is reset

            reward = motion_state[agent.id]["reward"]
            # print(f"Agent {agent.id} got reward: {reward}")

            # Calculate new Q_value:
            next_x, next_y = motion_state[agent.id]["target_pos"]
            next_state = next_x + (next_y*(width+1))
            old_value = q_tables[agent.id][state, actionNum]
            next_max = np.max(q_tables[agent.id][next_state, :])
            # if agent.id == 0:
            #     print(f"Agent {agent.id} old_value was {old_value} and next_max is {next_max}")

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_tables[agent.id][state, actionNum] = new_value

            # Reset reward
            motion_state[agent.id]["reward"] = -2

            # Move towards the new state
            progress_motion(agent)

        case AgentState.WAITING: 
            pass

        case AgentState.MOVING: 
            progress_motion(agent)

        case _: raise ValueError(f"Unknown action: {curr_state}")


# --- Main loop ---
ax = plt.gca()
for ep in range(NUM_EPISODES):
    env.reset()

    # Adjust epsilon
    # epsilon -= 1/NUM_EPISODES
    # epsilon = pow((1 - (ep+1)/NUM_EPISODES), 4) # (1-x)^4
    epsilon = pow((1 - (ep+1)/NUM_EPISODES), 2) # (1-x)^2

    # Spawn apples
    # spawn_random_apple(1)
    # spawn_random_apple(1)
    # spawn_random_apple(2)
    # spawn_random_apple(3)
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
    for agent in agents:
    #     agent.state[0,0] = int(np.random.uniform(0, env._world.width+1))
    #     agent.state[1,0] = int(np.random.uniform(0, env._world.height+1))
        agent.level = 1  # reset level
        motion_state[agent.id]["state"] = AgentState.EXPLORING
        motion_state[agent.id]["target_pos"] = cell_pos(agent)

        # Reset Q tables
        if (motion_state[agent.id]["apples"] != 0):
            # Save current q_table
            fileName = f'q_table{agent.id}_{motion_state[agent.id]["apples"]}.csv'
            np.savetxt(folder_path / fileName, q_tables[agent.id], delimiter=',', fmt='%f')
            # print(f'{fileName} saved')

            # Load no apple q_table
            motion_state[agent.id]["apples"] = 0
            fileName = f'q_table{agent.id}_{motion_state[agent.id]["apples"]}.csv'
            q_tables[agent.id] = np.loadtxt(folder_path / fileName, delimiter=',')

    if (args.mode == "display") or (ep == NUM_EPISODES-1):
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

        if (args.mode == "display") or (ep == NUM_EPISODES-1):
            env.render()
            ax.set_title(f"Episode {ep+1} | Step {step+1}")
            update_labels(ax, agent_labels, apple_labels, agents, apples)

        # Update the q_table files at every step for debugging purposes
        # for a in agents:
        #     fileName = f'q_table{a.id}_{motion_state[a.id]["apples"]}.csv'
        #     np.savetxt(folder_path / fileName, q_tables[a.id], delimiter=',', fmt='%f')

    clear_apples()
    clear_labels(agent_labels, apple_labels)
    print(f"Episode {ep+1} finished.")

print("Simulation ended.")

for a in agents:
    fileName = f'q_table{a.id}_{motion_state[a.id]["apples"]}.csv'
    np.savetxt(folder_path / fileName, q_tables[a.id], delimiter=',', fmt='%f')

env.end()
