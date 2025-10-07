import math
from typing import Tuple
import os
import numpy as np
from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi
from Metrics import Metrics

metrics = Metrics(1)

R_TARGET: float = 3.0
S_DIR: int = +1
R_OBS: int = 1

ALPHA: float = 0.5
GAMMA: float = 0.9
EPSILON: float = 0.2

R_BINS: np.ndarray   = np.array([-1e9, -0.2, 0.2, 1e9])                     # inside / on / outside
ANG_BINS: np.ndarray = np.array([-math.pi, -math.pi/4, math.pi/4, math.pi])  # left / aligned / right

N_STATES: int = 9
N_ACTIONS: int = 4

Q_CSV_PATH: str = os.environ.get("IRSIM_QTABLE_CSV", "q_table.csv")
TRAINING: bool  = os.environ.get("IRSIM_TRAIN", "1") != "0"     
EVAL_EPS: float = float(os.environ.get("IRSIM_EVAL_EPS", "0.0")) 
SAVE_EVERY: int = int(os.environ.get("IRSIM_SAVE_EVERY", "50"))

def load_q_table(path: str) -> np.ndarray:
    if os.path.exists(path):
        arr = np.loadtxt(path, delimiter=",")
        arr = np.asarray(arr, dtype=float)
        return arr.reshape((N_STATES, N_ACTIONS))
    else:
        return np.zeros((N_STATES, N_ACTIONS), dtype=float)

def save_q_table(q: np.ndarray, path: str):
    np.savetxt(path, q.reshape(N_STATES, N_ACTIONS), delimiter=",", fmt="%.8f")

Q_TABLE: np.ndarray = load_q_table(Q_CSV_PATH)

def build_actions(max_v: float, max_w: float):
    """
    0: fast forward + turn left
    1: fast forward straight
    2: fast forward + turn right
    3: slow forward straight
    """
    v_fast = 0.8 * float(max_v)
    v_slow = 0.4 * float(max_v)
    w      = 0.8 * float(max_w)
    return [(v_fast, +w), (v_fast, 0.0), (v_fast, -w), (v_slow, 0.0)]


def pick_action(q_row: np.ndarray, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(q_row.shape[0])
    return int(np.argmax(q_row))


def state_index_from(state_col: np.ndarray, goal_col: np.ndarray) -> Tuple[float, float, int]:
    distance, radian = relative_position(state_col, goal_col)
    theta = float(state_col[2, 0])  

    psi   = float(radian) + S_DIR * math.pi / 2.0 
    e_r   = float(distance) - R_TARGET
    e_psi = WrapToPi(float(psi) - theta)

    i_r   = int(np.digitize([e_r],   R_BINS)[0] - 1)
    i_ang = int(np.digitize([e_psi], ANG_BINS)[0] - 1)
    s_idx = i_r * 3 + i_ang 
    return e_r, e_psi, s_idx


def shaped_reward(e_r: float, e_psi: float, v_cmd: float) -> float:
    alpha = 2.0    # weight radial error
    beta  = 1.0    # w heading error
    tan  = 0.4    # w tangential(?)
    return float(-alpha * abs(e_r) - beta * abs(e_psi) + tan * (v_cmd * math.cos(e_psi)))

@register_behavior("diff", "docircle")
def beh_docircle(ego_object, **kwargs) -> np.ndarray:
    global _LAST, Q_TABLE

    state_col = ego_object.state
    robotid = ego_object.id              
    goal_col  = ego_object.goal  

    min_vel_col, max_vel_col = ego_object.get_vel_range()  
    min_v, min_w = float(min_vel_col[0, 0]), float(min_vel_col[1, 0])
    max_v, max_w = float(max_vel_col[0, 0]), float(max_vel_col[1, 0])

    if not hasattr(ego_object, "_docircle_actions"):
        ego_object._docircle_actions = build_actions(max_v, max_w)
    actions = ego_object._docircle_actions

    e_r, e_psi, s_idx = state_index_from(state_col, goal_col)

    last = getattr(ego_object, "_docircle_last", None)
    if TRAINING and last is not None:
        s_prev  = int(last["s"])
        a_prev  = int(last["a"])
        er_prev = float(last["er"])
        ep_prev = float(last["epsi"])
        v_prev  = float(last["v"])

        r = shaped_reward(er_prev, ep_prev, v_prev)
        td_target = r + GAMMA * float(np.max(Q_TABLE[s_idx]))
        td_error  = td_target - Q_TABLE[s_prev, a_prev]
        Q_TABLE[s_prev, a_prev] += ALPHA * td_error
        
        metrics.update(robotid, distance=e_r + R_TARGET)

        metrics.updateSpeed(robotid, v_cmd)

        metrics.updateAngular(robotid, e_psi)
        
        metrics.print(robotid)
        save_q_table(Q_TABLE, Q_CSV_PATH)

        
    epsilon_now = EPSILON if TRAINING else EVAL_EPS
    the_chosen = pick_action(Q_TABLE[s_idx], epsilon=epsilon_now)        
        
    v_cmd, w_cmd = actions[the_chosen]
    v_cmd = float(np.clip(v_cmd, min_v, max_v))
    w_cmd = float(np.clip(w_cmd, min_w, max_w))

    metrics.update(robotid, distance=e_r + R_TARGET)
    metrics.updateSpeed(robotid, v_cmd)
    metrics.updateAngular(robotid, e_psi)
    metrics.print(robotid)

    ego_object._docircle_last = {"s": s_idx, "a": the_chosen, "er": e_r, "epsi": e_psi, "v": v_cmd}
    return np.array([[v_cmd], [w_cmd]], dtype=float)
    






    