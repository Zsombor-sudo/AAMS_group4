import math
from typing import Dict, List, Tuple

import numpy as np
from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi

# -----------------------
# Tunables / Hyperparams
# -----------------------

R_TARGET: float = 3.0          # desired orbit radius around goal
S_DIR: int = +1                # +1 for CCW, -1 for CW

ALPHA: float = 0.5             # Q-learning step size
GAMMA: float = 0.9             # discount factor
EPSILON: float = 0.2           # epsilon-greedy exploration

# Discretization bins
R_BINS: np.ndarray   = np.array([-1e9, -0.2, 0.2, 1e9])               # radial error: inside / on / outside
ANG_BINS: np.ndarray = np.array([-math.pi, -math.pi/4, math.pi/4, math.pi])  # heading error: left / aligned / right

N_STATES: int = 9
N_ACTIONS: int = 4

# Global Q-table and last step cache (shared across robots/episodes)
Q_TABLE: np.ndarray = np.zeros((N_STATES, N_ACTIONS), dtype=float)
_LAST: Dict[str, int] | None = None


def build_actions(max_v: float, max_w: float) -> List[Tuple[float, float]]:
    """
    Build a small discrete action set:
      0: fast forward + turn left
      1: fast forward straight
      2: fast forward + turn right
      3: slow forward straight
    """
    v_fast = 0.8 * float(max_v)
    v_slow = 0.4 * float(max_v)
    w      = 0.8 * float(max_w)
    return [(v_fast, +w), (v_fast, 0.0), (v_fast, -w), (v_slow, 0.0)]


def pick_action(q_row: np.ndarray, epsilon: float = EPSILON) -> int:
    """ε-greedy action selection."""
    if np.random.rand() < epsilon:
        return np.random.randint(q_row.shape[0])
    return int(np.argmax(q_row))


def state_index_from(state_col: np.ndarray, goal_col: np.ndarray) -> Tuple[float, float, int]:
    """
    state_col: (3,1) column vector [x;y;theta]
    goal_col : (2,1) or (3,1) column vector
    Returns (e_r, e_psi, s_idx)
    """
    # IMPORTANT: relative_position returns (distance, radian)
    distance, radian = relative_position(state_col, goal_col)
    theta = float(state_col[2, 0])  # get heading from state col vector

    psi   = float(radian) + S_DIR * math.pi / 2.0  # tangential heading
    e_r   = float(distance) - R_TARGET
    e_psi = WrapToPi(float(psi) - theta)

    i_r   = int(np.digitize([e_r],   R_BINS)[0] - 1)
    i_ang = int(np.digitize([e_psi], ANG_BINS)[0] - 1)
    s_idx = i_r * 3 + i_ang  # 0..8
    return e_r, e_psi, s_idx


def reward(e_r: float, e_psi: float, v_cmd: float) -> float:
    """
    Shaped reward:
      - penalize radial error |e_r|
      - penalize heading error |e_psi|
      + bonus for tangential progress along desired direction
    """
    alpha = 2.0    # weight radial error
    beta  = 1.0    # weight heading error
    lam   = 0.4    # weight tangential progress
    return float(-alpha * abs(e_r) - beta * abs(e_psi) + lam * (v_cmd * math.cos(e_psi)))


# -----------------------
# Registered Behavior
# -----------------------

@register_behavior("diff", "docircle")
def beh_docircle(ego_object, **kwargs) -> np.ndarray:
    """
    Q-learning based circle-following behavior for a differential drive robot.
    """
    global _LAST, Q_TABLE

    # Use irsim’s column vectors directly
    state_col = ego_object.state              # (3,1)
    goal_col  = ego_object.goal               # (2,1) or (3,1)  <-- DO NOT reshape/flatten

    # Velocity range -> extract scalars
    min_vel_col, max_vel_col = ego_object.get_vel_range()  # each is (2,1)
    min_v, min_w = float(min_vel_col[0, 0]), float(min_vel_col[1, 0])
    max_v, max_w = float(max_vel_col[0, 0]), float(max_vel_col[1, 0])

    # Cache action set per robot (built from scalars)
    if not hasattr(ego_object, "_docircle_actions"):
        ego_object._docircle_actions = build_actions(max_v, max_w)
    actions = ego_object._docircle_actions

    # Current discrete state
    e_r, e_psi, s_idx = state_index_from(state_col, goal_col)

    # Q-learning update from previous transition
    if _LAST is not None:
        s_prev  = _LAST["s"]
        a_prev  = _LAST["a"]
        er_prev = _LAST["er"]
        epsi_prev = _LAST["epsi"]
        v_prev, _ = actions[a_prev]

        r = reward(er_prev, epsi_prev, float(v_prev))
        td_target = r + GAMMA * float(np.max(Q_TABLE[s_idx]))
        td_error  = td_target - Q_TABLE[s_prev, a_prev]
        Q_TABLE[s_prev, a_prev] += ALPHA * td_error

    # Choose action for current state (ε-greedy)
    a_chosen = pick_action(Q_TABLE[s_idx])

    # Command selected action (clipped to limits)
    v_cmd, w_cmd = actions[a_chosen]
    v_cmd = float(np.clip(v_cmd, min_v, max_v))
    w_cmd = float(np.clip(w_cmd, min_w, max_w))

    # Cache for next step's update
    _LAST = {"s": s_idx, "a": a_chosen, "er": e_r, "epsi": e_psi}

    # Return a (2,1) column vector as irsim expects
    return np.array([[v_cmd], [w_cmd]], dtype=float)
