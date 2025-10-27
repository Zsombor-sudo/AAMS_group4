from math import cos, sin
from typing import Any, Optional

import numpy as np

from irsim.config import env_param, world_param
from irsim.lib import reciprocal_vel_obs, register_behavior
from irsim.util.util import WrapToPi, omni_to_diff, relative_position


@register_behavior("diff", "follow_leader")
def beh_diff_followLeader(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:

    state = ego_object.state
    _, max_vel = ego_object.get_vel_range()
    leader = external_objects[kwargs.get("leaderID", 0)]
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)

    neighbor_list = ego_object.get_fov_detected_objects()

    behavior_vel = followLeader(state, max_vel, leader, angle_tolerance, neighbor_list)

    return behavior_vel


def followLeader(
    state: np.ndarray,
    max_vel: np.ndarray,
    leader: np.ndarray,
    angle_tolerance: float = 0.2,
    neighbor_list: Optional[list[Any]] = None,
) -> np.ndarray:
    distance, radian = relative_position(state, leader.state)

    diff_radian = WrapToPi(radian - state[2, 0])
    linear = (max_vel[0, 0] * np.cos(diff_radian)) / 2

    if abs(diff_radian) < angle_tolerance:
        angular = 0
    else:
        angular = max_vel[1, 0] * np.sign(diff_radian)

    if neighbor_list is not None:
        for neighbor in neighbor_list:
            distance, radian = relative_position(state, neighbor.state)
            diff_radian = WrapToPi(radian - state[2, 0])

            # linear += (max_vel[0, 0] * np.sin(diff_radian)) / (2 * len(neighbor_list))
            linear += (max_vel[0, 0] * -np.cos(diff_radian)) / (2 * len(neighbor_list))

    return np.array([[linear], [angular]])