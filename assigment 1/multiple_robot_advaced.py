import irsim
import numpy as np
import random
from irsim.util.util import WrapToPi
import math

# Initialize environment
env = irsim.make('multiple_robot_advanced.yaml')
robots = env.robot_list
num_robots = len(robots)
circle_radius = 5.0

# Action parameters
V_MAX = 1.0
W_MAX = 1.0

# Learning parameters
epsilon = 0.05
alpha = 0.5
gamma = 0.9

num_states = 25
num_actions = 8
num_episodes = 4000
num_steps = 2500

# Initialize Q-tables for each robot
Q_tables = [np.zeros((num_states, num_actions)) for _ in range(num_robots)]

# ------------------ FUNCTIONS ------------------

def take_action(action):
    actions = [
        np.array([V_MAX, W_MAX * 0.2]),
        np.array([V_MAX, W_MAX]),
        np.array([V_MAX, -W_MAX * 0.2]),
        np.array([V_MAX, -W_MAX]),
        np.array([V_MAX * 0.3, 0.0]),
        np.array([V_MAX, W_MAX]),
        np.array([V_MAX * 0.3, W_MAX]),
        np.array([V_MAX * 0.3, -W_MAX])
    ]
    return actions[action] if 0 <= action < len(actions) else np.array([0.0, 0.0])

def calculate_R(robot):
    current_x, current_y, theta = robot.state
    goal_x, goal_y = robot.goal[0], robot.goal[1]

    dx = current_x - goal_x
    dy = current_y - goal_y
    dist = np.sqrt(dx**2 + dy**2)
    radial_error = dist - circle_radius

    angle_to_goal = np.arctan2(dy, dx)
    desired_heading = WrapToPi(angle_to_goal + np.pi / 2)
    heading_error = WrapToPi(theta - desired_heading)

    dist_ok = abs(radial_error) < 0.2
    heading_ok = abs(heading_error) < 0.3

    reward = 0.0
    if heading_ok and not dist_ok:
        reward = -6.0 * abs(radial_error)
    elif dist_ok and not heading_ok:
        reward = -3.0 * abs(heading_error)
    elif not dist_ok and not heading_ok:
        reward = -6.0 * abs(radial_error) - 3.0 * abs(heading_error)
    else:
        reward = 5.0

    return reward

def get_state(robot):
    current_x, current_y, theta = robot.state
    goal_x, goal_y = robot.goal[0], robot.goal[1]

    dist = np.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
    radial_error = dist - circle_radius

    if radial_error < -0.4:
        radial_state = 0
    elif radial_error < -0.1:
        radial_state = 1
    elif radial_error < 0.1:
        radial_state = 2
    elif radial_error < 0.4:
        radial_state = 3
    else:
        radial_state = 4

    angle_to_goal = np.arctan2((goal_y - current_y), (goal_x - current_x))
    tangential_angle = angle_to_goal + np.pi / 2
    angle_error = WrapToPi(theta - tangential_angle)

    if angle_error < -0.6:
        angular_state = 0
    elif angle_error < -0.2:
        angular_state = 1
    elif angle_error < 0.2:
        angular_state = 2
    elif angle_error < 0.6:
        angular_state = 3
    else:
        angular_state = 4

    return radial_state * 5 + angular_state

def detect_obstacle(robot):
    for obs in env.obstacle_list:
        dist = np.linalg.norm(robot.state[:2] - obs.position)
        if dist < 2.0:
            return True
    return False

def obstacle_avoidance_action(robot):
    # Turn right and slow down
    return np.array([V_MAX * 0.3, -W_MAX])

def circle_following_action(state, Q_table):
    if random.random() < epsilon:
        return take_action(random.randint(0, num_actions - 1))
    else:
        return take_action(np.argmax(Q_table[state, :]))

# ------------------ TRAINING LOOP ------------------

for episode in range(num_episodes):
    states = [get_state(robot) for robot in robots]

    for step in range(num_steps):
        actions = []

        for i, robot in enumerate(robots):
            if detect_obstacle(robot):
                actions.append(obstacle_avoidance_action(robot))
            else:
                actions.append(circle_following_action(states[i], Q_tables[i]))

        env.step(actions)

        for i, robot in enumerate(robots):
            reward = calculate_R(robot)
            next_state = get_state(robot)

            if not detect_obstacle(robot):
                a = np.argmax(Q_tables[i][states[i], :])
                Q_tables[i][states[i], a] += alpha * (
                    reward + gamma * np.max(Q_tables[i][next_state]) - Q_tables[i][states[i], a]
                )

            states[i] = next_state

        env.render()
        if env.done(): break 

env.end()
