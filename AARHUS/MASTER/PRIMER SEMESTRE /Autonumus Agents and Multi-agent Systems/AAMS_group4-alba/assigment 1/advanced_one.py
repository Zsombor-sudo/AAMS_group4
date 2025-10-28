
import irsim
import numpy as np
import random
from irsim.util.util import relative_position, WrapToPi
import math
import matplotlib.pyplot as plt

env = irsim.make('advanced_one.yaml') # initialize the environment with the configuration file

robot = env.robot_list[0] 
robot_goal = robot.goal
circle_radius = 5.0 

# Action parameters
V_MAX = 1.0  # Max linear speed (m/s)
W_MAX = 1.0  # Max angular speed (rad/s)

# Variables 
epsilon = 0.05
alpha = 0.5
gamma = 0.9

num_states = 25
num_actions = 8

num_episodes = 2000
num_steps = 2500

#FUNCTIONS

def take_action(action):

    if action == 0:
        return np.array([V_MAX, W_MAX * 0.2])      # small trun left 
    elif action == 1:
        return np.array([V_MAX, W_MAX])            # trun left 
    elif action == 2:
        return np.array([V_MAX, -W_MAX * 0.2])    # small trun rigth 
    elif action == 3:
        return np.array([V_MAX, -W_MAX])          # trun rigth 
    elif action == 4:
        return np.array([V_MAX * 0.3, 0.0])       # slow forward
    elif action == 5:
        return np.array([V_MAX* 0.3, W_MAX])            # slow trun left 
    elif action == 6:
        return np.array([V_MAX* 0.3, -W_MAX])            # slow trun rigth
    elif action == 7:
        return np.array([V_MAX, W_MAX])            # trun left 
    else:
        return np.array([0.0, 0.0])               # stay


def calculate_R():
    current_x, current_y, theta = robot.state
    goal_x, goal_y = robot.goal[0], robot.goal[1]

    # --- Cálculo de errores ---
    dx = current_x - goal_x
    dy = current_y - goal_y
    dist = np.sqrt(dx**2 + dy**2)
    radial_error = dist - circle_radius

    angle_to_goal = np.arctan2(dy, dx)
    desired_heading = WrapToPi(angle_to_goal + np.pi / 2)
    heading_error = WrapToPi(theta - desired_heading)

    # --- Umbrales de precisión ---
    dist_ok = abs(radial_error) < 0.2
    heading_ok = abs(heading_error) < 0.3

    # --- Recompensa condicional ---
    if heading_ok and not dist_ok:
        # Orientación correcta, pero distancia incorrecta → penaliza distancia
        reward = -6.0 * abs(radial_error)
    elif dist_ok and not heading_ok:
        # Distancia correcta, pero orientación incorrecta → penaliza orientación
        reward = -3.0 * abs(heading_error)
    elif not dist_ok and not heading_ok:
        # Ambos incorrectos → penaliza ambos
        reward = -6.0 * abs(radial_error) - 3.0 * abs(heading_error)
    else:
        # Ambos correctos → recompensa positiva
        reward = 5.0

    return reward



def get_state():
    
    current_x, current_y, theta = robot.state  
    goal_x, goal_y = robot.goal[0], robot.goal[1] 

    # --- Radial error computation ---
   
    dist = np.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
    radial_error = dist - circle_radius

    # Discretize the radial error into 5 states (0–4)
    if radial_error < -0.4:      # Robot closer than desired
        radial_state = 0
    elif radial_error < -0.1:    # Robot is slightly closer than desired
        radial_state = 1
    elif radial_error < 0.1:     # Robot is within acceptable range
        radial_state = 2
    elif radial_error < 0.4:     # Robot is slightly farther than desired
        radial_state = 3
    else:                        # Robot is much farther than desired
        radial_state = 4

    # --- Angular error computation ---
    angle_to_goal = np.arctan2((goal_y - current_y), (goal_x - current_x))
    
    # Desired tangential direction is perpendicular to the goal direction
    tangential_angle = angle_to_goal + np.pi/2  
    
    angle_error = WrapToPi(theta - tangential_angle)

    # Discretize the angular error into 5 states (0–4)
    if angle_error < -0.6:       # Facing much too far left
        angular_state = 0
    elif angle_error < -0.2:     # Slightly left of tangent
        angular_state = 1
    elif angle_error < 0.2:      # Aligned with tangent
        angular_state = 2
    elif angle_error < 0.6:      # Slightly right of tangent
        angular_state = 3
    else:                        # Facing much too far right
        angular_state = 4

    # Combine radial and angular states into a single state ID
    # There are 5 radial × 5 angular = 25 possible states
    return radial_state * 5 + angular_state

# --------------- ADVANCED ------------------

# TWO LAYERS FUNCTION

# L2: Obstacle Avoidance, has the highest priority
def detect_obstacle():

    # Check if any obstacle is within a threshold distance
    for obs in env.obstacle_list:
        dist = np.linalg.norm(robot.state[:2] - obs.position)
        if dist < 2.0:  # threshold distance
            return True
    return False

def obstacle_avoidance_action():
    # Simple reactive behavior: turn right or slow down
    return np.array([V_MAX * 0.3, -W_MAX])



# L1: Circle Following, is the default behavior
def circle_following_action(state):

    if random.random() < epsilon:
        return take_action(random.randint(0, num_actions - 1))
    else:
        return take_action(np.argmax(Q_s_a[state, :]))
    
    

#ALGORITME 

#init Q(S,A) 
Q_s_a = np.zeros((num_states, num_actions))

for episode in range(num_episodes):

    #inisalize S
    state = get_state() 

    for step in range(num_steps):

        if detect_obstacle():
            action = obstacle_avoidance_action()
        else:
            action = circle_following_action(state)
        
        env.step(action)
        
        #TAKE ACTIOON
        #env.step(take_action(action))
        #env.step(np.array([1.0, 0.2]))
        
        #Observe R, S'
        reward = calculate_R()
        next_state = get_state() 

        #Get Q(S,A)
        if not detect_obstacle():
            
            action = np.argmax(Q_s_a[state, :])

            Q_s_a[state, action] =  Q_s_a[state, action] + alpha * (reward + gamma * np.max(Q_s_a[next_state]) - Q_s_a[state, action])

        #act S
        state = next_state
        env.render() # render the environment
        if env.done(): break # check if the simulation is done

env.end()

