import math
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi, dist_hypot

from AAMS_group4.custom_behavior import *

# Collision parameters
SAFETY_DISTANCE: float = 0.8  # How close is too close

# Expand state space to include collision information
N_COLLISION_STATES: int = 2  # safe / threat
N_STATES_ENHANCED: int = N_STATES * N_COLLISION_STATES  # 9 * 2 = 18

# Add "stop" action
N_ACTIONS_ENHANCED: int = 5  # 4 original + 1 stop

# Global Q-table with enhanced size
Q_TABLE_ENHANCED: np.ndarray = None

def load_enhanced_q_table(path: str) -> np.ndarray:
    if os.path.exists(path):
        try:
            arr = np.loadtxt(path, delimiter=",")
            arr = np.asarray(arr, dtype=float)
            if arr.size == N_STATES_ENHANCED * N_ACTIONS_ENHANCED:
                return arr.reshape((N_STATES_ENHANCED, N_ACTIONS_ENHANCED))
        except:
            pass
    return np.zeros((N_STATES_ENHANCED, N_ACTIONS_ENHANCED), dtype=float)

def save_enhanced_q_table(q: np.ndarray, path: str):
    np.savetxt(path, q.reshape(N_STATES_ENHANCED, N_ACTIONS_ENHANCED), delimiter=",", fmt="%.8f") 

def get_other_robots_states(ego_object, **kwargs) -> List[np.ndarray]:
    """
    Get the states of other robots in the environment.
    """
    external_objects = kwargs.get("external_objects", [])
    other_robots_states = []

    # print("Current ego state:", ego_object.state, ego_object.id)
    for obj in external_objects:
        if obj.role == "robot" and obj is not ego_object:
            # print("Added other robot state:", obj.state, obj.id)
            other_robots_states.append(obj.state)
    return other_robots_states

def detect_collision_threat(ego_state: np.ndarray, other_robots_states: List[np.ndarray]) -> Tuple[bool, float, bool]:
    """
    Detect if there's a collision threat
    Returns: (threat_detected, min_distance, crashed)
    """
    ego_x, ego_y = float(ego_state[0, 0]), float(ego_state[1, 0])
    min_distance = float('inf')
    
    for other_state in other_robots_states:
        other_x, other_y = float(other_state[0, 0]), float(other_state[1, 0])
        distance = dist_hypot(ego_x, ego_y, other_x, other_y)
        min_distance = min(min_distance, distance)
    
    crashed = min_distance < 0.3  # Collision threshold
    threat = min_distance < SAFETY_DISTANCE
    return threat, min_distance, crashed

def collision_aware_build_actions(max_v: float, max_w: float):
    """
    Add 'stop' action
    0-3: original actions
    4: stop
    """
    original_actions = build_actions(max_v, max_w)
    original_actions.append((0.0, 0.0))  # Stop action
    return original_actions

def collision_aware_state_index(state_col: np.ndarray, goal_col: np.ndarray, collision_threat: bool) -> Tuple[float, float, int]:
    """Enhanced state that includes collision information"""
    e_r, e_psi, base_state = state_index_from(state_col, goal_col)
    
    enhanced_state = base_state
    if collision_threat:
        enhanced_state += N_STATES  # Shift to collision states if threat detected
    
    return e_r, e_psi, enhanced_state

def collision_aware_reward(e_r: float, e_psi: float, v_cmd: float, collision_threat: bool, min_distance: float, crashed: bool) -> float:
    """Enhanced reward with collision penalty"""
    base_reward = shaped_reward(e_r, e_psi, v_cmd)

    if crashed:
        return -100.0  # Heavy penalty for collision
    
     # Penalty for stopping when no collision threat
    if abs(v_cmd) < 0.1 and not collision_threat:
        base_reward -= 2.0  # Penalty for unnecessary stopping
    
    if collision_threat:
        proximity_factor = max(0.1, min_distance / 1.0)  # Normalize by max safe distance
        collision_penalty = 10.0 / proximity_factor  # Heavy penalty for collision threat

        if abs(v_cmd) < 0.1:  # Robot stopped
            collision_penalty = collision_penalty / 2.0  # Mild penalty for stopping
        base_reward -= collision_penalty
    
    # Reward for maintaining distance
    if min_distance < float('inf'):
        distance_reward = (min_distance - SAFETY_DISTANCE) * 2.0  # Reward for being further away
        base_reward += distance_reward
    
    return base_reward

@register_behavior("diff", "docircle_collision_avoid")
def beh_docircle_collision_avoid(ego_object, **kwargs) -> np.ndarray:
    """Enhanced behavior with collision avoidance learning"""
    global Q_TABLE_ENHANCED
    
    # Initialize Q-table if needed
    if Q_TABLE_ENHANCED is None:
        Q_TABLE_ENHANCED = load_enhanced_q_table(Q_CSV_PATH)
    
    state_col = ego_object.state
    goal_col = ego_object.goal
    
    # Get states of other robots
    other_robots_states = get_other_robots_states(ego_object, **kwargs)
    
    min_vel_col, max_vel_col = ego_object.get_vel_range()
    min_v, min_w = float(min_vel_col[0, 0]), float(min_vel_col[1, 0])
    max_v, max_w = float(max_vel_col[0, 0]), float(max_vel_col[1, 0])
    
    if not hasattr(ego_object, "_collision_actions"):
        ego_object._collision_actions = collision_aware_build_actions(max_v, max_w)
    actions = ego_object._collision_actions
    
    # Detect collision threat
    is_threat, min_distance, crashed = detect_collision_threat(state_col, other_robots_states)

    # Collision-aware state index
    e_r, e_psi, s_idx = collision_aware_state_index(state_col, goal_col, is_threat)

    # Q-learning with added collision info
    last = getattr(ego_object, "_collision_last", None)
    if TRAINING and last is not None:
        s_prev = int(last["s"])
        a_prev = int(last["a"])
        er_prev = float(last["er"])
        ep_prev = float(last["epsi"])
        v_prev = float(last["v"])
        collision_prev = bool(last["collision"])
        min_dist_prev = float(last["min_dist"])
        
        r = collision_aware_reward(er_prev, ep_prev, v_prev, collision_prev, min_dist_prev, crashed)
        td_target = r + GAMMA * float(np.max(Q_TABLE_ENHANCED[s_idx]))
        td_error = td_target - Q_TABLE_ENHANCED[s_prev, a_prev]
        Q_TABLE_ENHANCED[s_prev, a_prev] += ALPHA * td_error
        
        save_enhanced_q_table(Q_TABLE_ENHANCED, Q_CSV_PATH)
    
    # Action selection
    epsilon_now = EPSILON if TRAINING else EVAL_EPS
    the_chosen = pick_action(Q_TABLE_ENHANCED[s_idx], epsilon=epsilon_now)
    
    v_cmd, w_cmd = actions[the_chosen]
    v_cmd = float(np.clip(v_cmd, min_v, max_v))
    w_cmd = float(np.clip(w_cmd, min_w, max_w))
    
    # Save state for next update
    ego_object._collision_last = {
        "s": s_idx, 
        "a": the_chosen, 
        "er": e_r, 
        "epsi": e_psi, 
        "v": v_cmd,
        "collision": is_threat,
        "min_dist": min_distance
    }
    
    return np.array([[v_cmd], [w_cmd]], dtype=float)