from math import cos, sin
from typing import Any, Optional

import numpy as np

from irsim.config import env_param, world_param
from irsim.lib import reciprocal_vel_obs, register_behavior
from irsim.util.util import WrapToPi, omni_to_diff, relative_position


@register_behavior("diff", "follow_leader")
def beh_diff_followLeader(
    ego_object: any, external_objects: list[any], **kwargs: any
) -> np.ndarray:

    state = ego_object.state
    _, max_vel = ego_object.get_vel_range()
    leader = external_objects[kwargs.get("leaderID", 0)]

    behavior_vel = followLeader(state, max_vel, leader)

    return behavior_vel


def followLeader(
    state: np.ndarray,
    max_vel: np.ndarray,
    leader: np.ndarray,
) -> np.ndarray:
    distance, radian = relative_position(state, leader.state)

    diff_radian = WrapToPi(radian - state[2, 0])
    linear = max_vel[0, 0] * np.cos(diff_radian)

    angular = max_vel[1, 0] * np.sign(diff_radian)

    return np.array([[linear], [angular]])