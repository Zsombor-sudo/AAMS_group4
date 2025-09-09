from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi
import numpy as np

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

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    circle_radius = kwargs.get("circle_radius", 5)
    alpha = kwargs.get("alpha", 5) # Learning rate
    gamma = kwargs.get("gamma", 5) # Discount factor
    epsilon = kwargs.get("epsilon", 5) # Exploration/Exploitation trade off

    if goal is None:
        if world_param.count % 10 == 0:
            env_param.logger.warning(
                "Goal is currently None. This dash behavior is waiting for goal configuration"
            )

        return np.zeros((2, 1))

    return RL_circle(state, goal, max_vel, goal_threshold, angle_tolerance, circle_radius, alpha, gamma, epsilon)


def RL_circle(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.1,
    angle_tolerance: float = 0.2,
    circle_radius: float = 5,
    alpha: float = 5,
    gamma: float = 5,
    epsilon: float = 5,
) -> np.ndarray:

    # Always move forward at a set speed
    linear = max_vel[0, 0]

    # Set angle :
    distance, radian = relative_position(state, goal) # Get distance and angle to circle center

    angular = 0

    return np.array([[linear], [angular]])