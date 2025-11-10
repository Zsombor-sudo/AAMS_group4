from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi
#import irism
import numpy as np
from pathlib import Path
import random
import math
#default radius on 1
#metrics = Metrics(10)

@register_behavior("diff", "basic_circle")
def beh_diff_dash(
    ego_object: any, external_objects: list[any], **kwargs: any
) -> np.ndarray:
    """
    Behavior function for differential drive robot using dash-to-goal behavior.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: Additional keyword arguments:
            - angle_tolerance (float): Allowable angular deviation, default 0.1.

    Returns:
        np.array: Velocity [linear, angular] (2x1) for differential drive.
    """

    agent_id = ego_object.id
    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    circle_radius = kwargs.get("circle_radius", 5)
    correction_multiplier = kwargs.get("correction_multiplier", 1)

    if goal is None:
        if world_param.count % 10 == 0:
            env_param.logger.warning(
                "Goal is currently None. This dash behavior is waiting for goal configuration"
            )

        return np.zeros((2, 1))

    return basic_circle(state, goal, max_vel, goal_threshold, angle_tolerance, circle_radius, correction_multiplier, agent_id)


def basic_circle(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.1,
    angle_tolerance: float = 0.2,
    circle_radius: float = 5,
    correction_multiplier: float = 1,
    agent_id: int = 10
) -> np.ndarray:
    """
    Calculate the differential drive velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y] (2x1).
        max_vel (np.array): Maximum velocity [linear, angular] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).
        angle_tolerance (float): Allowable angular deviation (default 0.2).

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    distance, radian = relative_position(state, goal)

    # if distance < goal_threshold:
    #     return np.zeros((2, 1))

    # Attempt at integral error corection
    # distanceCorrection = 0
    # distanceCorrection += (circle_radius - distance) * correction_multiplier
    
    distanceCorrection = (circle_radius - distance) * correction_multiplier
    radian += np.pi/2 + distanceCorrection     # Attempt to change the desired angle to 90 degrees to the angle between the goal and the object

    diff_radian = WrapToPi(radian - state[2, 0])
    linear = max_vel[0, 0] * np.cos(diff_radian)

    # if abs(diff_radian) < angle_tolerance:
    #     angular = 0
    # else:
    #     angular = max_vel[1, 0] * np.sign(diff_radian)
    angular = max_vel[1, 0] * np.sign(diff_radian)

    
    return np.array([[linear], [angular]])

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
    # distance = int(round(distance)) # Round the distance to nearest integer to get the current state
    
    # print(distance)

    # Calculate new Q value:
    if (RL_circle.old_values[agent_id, 0] != 0):
        ### Tip: Try to use https://www.geogebra.org/classic?lang=en to visualise the reward curve
        # reward = -abs(circle_radius*10 - distance) * 100
        # reward = -abs(circle_radius - distance) * 100
        # reward = (math.pow(np.finfo(np.float32).eps, math.pow(-(distance - circle_radius), 2))) * 100  
        # reward = 5 * (1 / (1 + 0.1 * math.pow(distance - circle_radius, 2))) - 0.5
        # reward = 50 * (1 / (1 + 0.01 * math.pow(distance - circle_radius*10, 2))) - 5
        # reward = -0.5 * math.pow(distance - circle_radius, 2)
        # reward = -0.025 * math.pow(distance - circle_radius*10, 2)
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
    # angular = max_vel[1, 0] * np.sign(diff_radian)
    # angular = np.clip(1 * diff_radian, -max_vel[1, 0], max_vel[1, 0])
    angular = max_vel[1, 0] * np.tanh(1 * diff_radian)

    # Always move forward at a set speed
    # linear = max_vel[0, 0] * np.cos(diff_radian)
    linear = max_vel[0, 0] * (np.cos(diff_radian) + 1) / 2  
    return np.array([[linear], [angular]]) 

@register_behavior("diff", "follow_agent")
def beh_diff_dash(
    ego_object: any, external_objects: list[any], **kwargs: any
) -> np.ndarray:
    """
    Behavior function for differential drive robot using dash-to-goal behavior.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: Additional keyword arguments:
            - angle_tolerance (float): Allowable angular deviation, default 0.1.

    Returns:
        np.array: Velocity [linear, angular] (2x1) for differential drive.
    """

    agent_id = ego_object.id
    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()

    if goal is None:
        if world_param.count % 10 == 0:
            env_param.logger.warning(
                "Goal is currently None. This dash behavior is waiting for goal configuration"
            )

        return np.zeros((2, 1))

    return basic_circle(state, goal, max_vel, goal_threshold, agent_id)


def follow_agent(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.1,
    agent_id: int = 10
) -> np.ndarray:
    """
    Calculate the differential drive velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y] (2x1).
        max_vel (np.array): Maximum velocity [linear, angular] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).
        angle_tolerance (float): Allowable angular deviation (default 0.2).

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    distance, radian = relative_position(state, goal)

    #env.robot_list
    
    return np.array([[1], [0]])