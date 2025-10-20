from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi
import numpy as np
from pathlib import Path
import random
import math
from Metrics import Metrics
#default radius on 1
metrics = Metrics(10)

@register_behavior("diff", "RL_circle")
def beh_diff_dash(
    ego_object: any, external_objects: list[any], **kwargs: any
) -> np.ndarray:

    state = ego_object.state
    goal = ego_object.goal
    _, max_vel = ego_object.get_vel_range()
    circle_radius = kwargs.get("circle_radius", 5)
    alpha = kwargs.get("alpha", 0.1) # Learning rate
    gamma = kwargs.get("gamma", 0.6) # Discount factor
    epsilon = kwargs.get("epsilon", 0.1) # Exploration/Exploitation trade off
    agent_id = ego_object.id

    if goal is None:
        if world_param.count % 10 == 0:
            env_param.logger.warning(
                "Goal is currently None. This dash behavior is waiting for goal configuration"
            )

        return np.zeros((2, 1))

    return RL_circle(state, goal, max_vel, circle_radius, alpha, gamma, epsilon, agent_id)


def RL_circle(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    circle_radius: float = 5,
    alpha: float = 0.1,
    gamma: float = 0.6,
    epsilon: float = 0.1,
    agent_id: int = 10,
) -> np.ndarray:

    ### Source: ###
    # https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ #

    metrics.setCircleRadius(circle_radius)
    ## Load tables:
    file_path = Path('q_table.csv') # When executing the file while in the Assignemnt 1 folder
    # file_path = Path('Assignment 1/q_table.csv') # When executing the file while in the Assignemnt 1 parent folder

    if not hasattr(RL_circle, "q_table"):
        # Create/Load Q table:
        # If it doesn't exist, create a Q table
        if not file_path.exists():
            RL_circle.q_table = np.zeros([500, 10]) # 500 states (distance to center in 0.1 increments) and 10 actions (angle to center between 0.785 and 2.355 in 0.157 increments)
        else: # Otherwise load the Q table:
            RL_circle.q_table = np.loadtxt(file_path, delimiter=',')
        
        # Create old values table:
        RL_circle.old_values = np.zeros([10, 2], dtype=int)

        # File save counter:
        RL_circle.counter = 0

    ## Training:
    # In my RL environment the distance is the state
    
    distance, radian = relative_position(state, goal) # Get distance and angle to circle center
    distance_float = distance
    distance = int(round(distance, 1) * 10) # Round the distance to nearest single decimal and multiply by 10 to get the current state

    # Calculate new Q value:
    if (RL_circle.old_values[agent_id, 0] != 0):
        ### Tip: Try to use https://www.geogebra.org/classic?lang=en to visualise the reward curve
        circle_radius = circle_radius * 10
        if distance > circle_radius:
            reward = 2*(-distance + circle_radius)
        if distance < circle_radius:
            reward = 2*(distance - circle_radius)
        if distance == circle_radius:
            reward = 0

        old_value = RL_circle.q_table[RL_circle.old_values[agent_id, 0], RL_circle.old_values[agent_id, 1]]
        next_max = np.max(RL_circle.q_table[distance])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        RL_circle.q_table[RL_circle.old_values[agent_id, 0], RL_circle.old_values[agent_id, 1]] = new_value
        
        # Save the Q table every once in a while
        if RL_circle.counter >= 50:
            RL_circle.counter = 0
            # Save the Q table:
            np.savetxt(file_path, RL_circle.q_table, delimiter=',', fmt='%f')
        RL_circle.counter += 1

    # Decide next action:
    if random.uniform(0,1) < epsilon:
        action = random.randrange(RL_circle.q_table.shape[1]) # Explore action space
    else:
        action = np.argmax(RL_circle.q_table[distance]) # Exploit learned values

    RL_circle.old_values[agent_id, 0] = distance
    RL_circle.old_values[agent_id, 1] = int(action)

    ## Set next step:
    # Calculate angular:
    radian += (action / RL_circle.q_table.shape[1]) * np.pi/2 + np.pi/4
    diff_radian = WrapToPi(radian - state[2, 0])
    angular = max_vel[1, 0] * np.tanh(1 * diff_radian)

    # Always move forward
    linear = max_vel[0, 0] * (np.cos(diff_radian) + 1) / 2  

    
    metrics.update(agent_id, distance_float)
    metrics.updateSpeed(agent_id, linear)
    metrics.updateAngular(agent_id, angular)
    metrics.print(agent_id)
    return np.array([[linear], [angular]]) 