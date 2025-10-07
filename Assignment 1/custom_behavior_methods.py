from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import math
from Metrics import Metrics

#default radius on 1
metrics = Metrics(1)
#errLabelObj = plt.gcf()
#errLabelText = [errLabelObj.text(0.15, 0.75, "Placeholder ", ha="left", va="top"),
#                errLabelObj.text(0.15, 0.80, "Placeholder ", ha="left", va="top"),
#                errLabelObj.text(0.15, 0.85, "Placeholder ", ha="left", va="top"),
#                errLabelObj.text(0.15, 0.90, "Placeholder ", ha="left", va="top")]

#explaination label
#plt.gcf().text(0.15,0.95,"Current error [m] | accumulated error [m] | average error per iteration [m/iter]", ha="left", va="top")

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

    return basic_circle(state, goal, max_vel, goal_threshold, angle_tolerance, circle_radius, correction_multiplier)


def basic_circle(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.1,
    angle_tolerance: float = 0.2,
    circle_radius: float = 5,
    correction_multiplier: float = 1,
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

    robotid = ego_object.id
    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    circle_radius = kwargs.get("circle_radius", 5)
    metrics.setCircleRadius(circle_radius)
    alpha = kwargs.get("alpha", 0.1) # Learning rate
    gamma = kwargs.get("gamma", 0.6) # Discount factor
    epsilon = kwargs.get("epsilon", 0.1) # Exploration/Exploitation trade off

    if goal is None:
        if world_param.count % 10 == 0:
            env_param.logger.warning(
                "Goal is currently None. This dash behavior is waiting for goal configuration"
            )

        return np.zeros((2, 1))

    return RL_circle(state, goal, max_vel, goal_threshold, angle_tolerance, circle_radius, alpha, gamma, epsilon, robotid)


def RL_circle(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.1,
    angle_tolerance: float = 0.2,
    circle_radius: float = 5,
    alpha: float = 0.1,
    gamma: float = 0.6,
    epsilon: float = 0.1,
    robotid: int = 0
) -> np.ndarray:

    
    ### Source: ###
    # https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ #
    
    ## Create/Load Q table:
    # file_path = Path('q_table.csv') # When executing the file while in the Assignemnt 1 folder
    file_path = Path('Assignment 1/q_table.csv') # When executing the file while in the Assignemnt 1 parent folder

    # If it doesn't exist, create a Q table
    if not file_path.exists():
        # q_table = np.zeros([500, 314]) # 500 states (distance to center in 0.1 increments) and 314 actions (angle to center between 0 and 3.14 in 0.01 increments)
        # q_table = np.zeros([500, 2]) # 500 states (distance to center in 0.1 increments) and 2 actions (going away from center and opposite)
        q_table = np.zeros([50, 10]) # 50 states (distance to center in 1 increments) and 10 actions (angle to center between 0 and 3.14 in 0.314 increments)
    else: # Otherwise load the Q table:
        q_table = np.loadtxt(file_path, delimiter=',')

    ## Training:
    # In my RL environment the distance is the state
    
    distance, radian = relative_position(state, goal) # Get distance and angle to circle center
    # distance = int(round(distance, 1) * 10) # Round the distance to nearest single decimal and multiply by 10 to get the current state
    
    #float value of distance
    distance_float = distance 
    distance = int(round(distance)) # Round the distance to nearest integer to get the current state

    # If first step, don't calculate Q value
    if not hasattr(RL_circle, "old_distance"):
        RL_circle.old_distance = distance
        
        #init error variables
        #metrics.initRobot(circle_radius)
        #RL_circle.accDistance = 0
        #RL_circle.iterations = 1
    else: # Calculate new Q value:
        ### Tip: Try to use https://www.geogebra.org/classic?lang=en to visualise the reward curve
        # reward = -abs(circle_radius*10 - distance) * 100
        # reward = -abs(circle_radius - distance) * 100
        # reward = (math.pow(np.finfo(np.float32).eps, math.pow(-(distance - circle_radius), 2))) * 100  
        
        #updating error
        metrics.update(robotid, distance_float)
        #RL_circle.accDistance += abs(distance_float - circle_radius)
        #RL_circle.iterations += 1
        
        reward = 2 * (1 / (1 + 5 * math.pow(distance - circle_radius, 2)))

        old_value = q_table[RL_circle.old_distance, RL_circle.old_action]
        next_max = np.max(q_table[distance])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[RL_circle.old_distance, RL_circle.old_action] = new_value
        
        # Save the Q table:
        np.savetxt(file_path, q_table, delimiter=',', fmt='%f')

    #printing error
    metrics.print(robotid)
    #errLabelText[robotid].set_text(f" {abs(distance - circle_radius)}  ||  {round(RL_circle.accDistance, 2)}  ||  {round(RL_circle.accDistance/RL_circle.iterations, 2)}")
    
    # Decide next action:
    if random.uniform(0,1) < epsilon:
        action = random.randrange(q_table.shape[1]) # Explore action space
    else:
        action = np.argmax(q_table[distance]) # Exploit learned values

    RL_circle.old_action = int(action)
    RL_circle.old_distance = distance

    ## Set next step:
    # Calculate angular:
    radian += (action / q_table.shape[1]) * np.pi
    diff_radian = WrapToPi(radian - state[2, 0])
    angular = max_vel[1, 0] * np.sign(diff_radian)

    # Always move forward at a set speed
    linear = max_vel[0, 0]

    metrics.updateSpeed(robotid, linear)
    metrics.updateAngular(robotid, angular)
    return np.array([[linear], [angular]]) 