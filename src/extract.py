"""
Pull one training example (agents, map points, traffic lights, ego target)
out of a Waymax scenario at decision time t.

FIELD NAMES ARE BEST-EFFORT based on Waymax's public datatypes and may need
adjusting to your installed version -- check scenario.log_trajectory,
scenario.roadgraph_points, and your traffic-light field name (this file
assumes `scenario.log_traffic_light`; some versions expose it differently)
before trusting this end to end. Each function is small and independent
specifically so a mismatch is easy to isolate and fix.
"""

import numpy as np
from src.transforms import transform_positions, transform_velocities, transform_yaw

MAX_AGENTS = 32
MAX_MAP_POINTS = 200
MAX_LIGHTS = 16


def _pad(arr, max_n):
    """Pad first axis of arr with zeros up to max_n; return (padded, mask)."""
    n = arr.shape[0]
    mask = np.zeros(max_n, dtype=bool)
    if n == 0:
        pad_shape = (max_n,) + arr.shape[1:]
        return np.zeros(pad_shape, dtype=arr.dtype), mask
    mask[:min(n, max_n)] = True
    if n >= max_n:
        return arr[:max_n], mask
    pad_shape = (max_n - n,) + arr.shape[1:]
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0), mask


def get_ego_frame(scenario, ego_idx, t):
    """Returns (ego_xy, ego_yaw, ego_vxvy) at time t -- the frame anchor."""
    traj = scenario.log_trajectory
    ego_xy = np.array([traj.x[ego_idx, t], traj.y[ego_idx, t]])
    ego_yaw = float(traj.yaw[ego_idx, t])
    ego_vxvy = np.array([traj.vel_x[ego_idx, t], traj.vel_y[ego_idx, t]])
    return ego_xy, ego_yaw, ego_vxvy


def extract_agents(scenario, ego_idx, t, hist_len=10):
    """Non-ego agents, history window [t-hist_len+1, t], ego frame.
    Returns hist_feats (MAX_AGENTS, hist_len, 6), class_ids (MAX_AGENTS,), mask (MAX_AGENTS,).
    Vectorized across agents (no per-agent Python loop) -- this was the main
    slow part of extraction before."""
    traj = scenario.log_trajectory
    ego_xy, ego_yaw, ego_vxvy = get_ego_frame(scenario, ego_idx, t)
    t0 = t - hist_len + 1
    if t0 < 0:
        return None  # not enough history at this t

    valid_window = np.array(traj.valid[:, t0:t + 1])   # (num_objects, hist_len)
    full_valid = valid_window.all(axis=1)               # (num_objects,)
    full_valid[ego_idx] = False                          # exclude ego
    idx = np.where(full_valid)[0]

    if len(idx) == 0:
        return {
            "hist": np.zeros((MAX_AGENTS, hist_len, 6), dtype=np.float32),
            "class": np.zeros(MAX_AGENTS, dtype=np.int64),
            "mask": np.zeros(MAX_AGENTS, dtype=bool),
        }

    xy_raw = np.stack([np.array(traj.x)[idx, t0:t + 1], np.array(traj.y)[idx, t0:t + 1]], axis=-1)
    xy = transform_positions(xy_raw, ego_xy, ego_yaw)
    yaw = transform_yaw(np.array(traj.yaw)[idx, t0:t + 1], ego_yaw)
    v_raw = np.stack([np.array(traj.vel_x)[idx, t0:t + 1], np.array(traj.vel_y)[idx, t0:t + 1]], axis=-1)
    v_rel = transform_velocities(v_raw, ego_vxvy, ego_yaw)
    feats = np.concatenate([xy, np.cos(yaw)[..., None], np.sin(yaw)[..., None], v_rel], axis=-1).astype(np.float32)

    # ADAPT: object_metadata.object_types field name/encoding may differ
    cls = np.array(scenario.object_metadata.object_types)[idx].astype(np.int64)

    hist, mask = _pad(feats, MAX_AGENTS)
    cls, _ = _pad(cls, MAX_AGENTS)
    return {"hist": hist, "class": cls, "mask": mask}


def extract_map(scenario, ego_idx, t):
    """Static roadgraph points (lane markings, stop signs, crosswalks, ...), ego frame.
    Returns xy (MAX_MAP_POINTS, 2), type_ids (MAX_MAP_POINTS,), mask (MAX_MAP_POINTS,)."""
    ego_xy, ego_yaw, _ = get_ego_frame(scenario, ego_idx, t)
    rg = scenario.roadgraph_points
    valid = np.array(rg.valid)
    xy_raw = np.stack([np.array(rg.x)[valid], np.array(rg.y)[valid]], axis=-1)
    types = np.array(rg.types)[valid].astype(np.int64)

    # keep only nearby points to bound compute -- crude radius filter pre-padding
    xy_local = transform_positions(xy_raw, ego_xy, ego_yaw)
    dist = np.linalg.norm(xy_local, axis=-1)
    keep = dist <= 80.0  # meters, generous radius
    xy_local, types = xy_local[keep], types[keep]

    xy_local, mask = _pad(xy_local.astype(np.float32), MAX_MAP_POINTS)
    types, _ = _pad(types, MAX_MAP_POINTS)
    return {"xy": xy_local, "type": types, "mask": mask}


def extract_traffic_lights(scenario, ego_idx, t, hist_len=10):
    """Traffic light stop-line positions + state over history, ego frame.
    ADAPT: field name/shape for your Waymax version -- assumed scenario.log_traffic_light
    with .x, .y, .state, .valid shaped (num_lights, num_timesteps).
    Vectorized across lights (no per-light Python loop)."""
    ego_xy, ego_yaw, _ = get_ego_frame(scenario, ego_idx, t)
    t0 = t - hist_len + 1
    empty = {
        "hist_xy": np.zeros((MAX_LIGHTS, hist_len, 2), dtype=np.float32),
        "state": np.zeros((MAX_LIGHTS, hist_len), dtype=np.int64),
        "mask": np.zeros(MAX_LIGHTS, dtype=bool),
    }
    if t0 < 0 or not hasattr(scenario, "log_traffic_light"):
        return empty

    tl = scenario.log_traffic_light
    valid_window = np.array(tl.valid[:, t0:t + 1])
    full_valid = valid_window.all(axis=1)
    idx = np.where(full_valid)[0]
    if len(idx) == 0:
        return empty

    xy_raw = np.stack([np.array(tl.x)[idx, t0:t + 1], np.array(tl.y)[idx, t0:t + 1]], axis=-1)
    xy = transform_positions(xy_raw, ego_xy, ego_yaw).astype(np.float32)
    state = np.array(tl.state)[idx, t0:t + 1].astype(np.int64)

    hist_xy, mask = _pad(xy, MAX_LIGHTS)
    state, _ = _pad(state, MAX_LIGHTS)
    return {"hist_xy": hist_xy, "state": state, "mask": mask}


def extract_ego_target(scenario, ego_idx, t, future_len=30):
    """Ego's own future, as ABSOLUTE positions in its own frame at t -- this must
    match Stage1Model's output space, since the model cumsums its decoder's raw
    per-step offsets internally before returning `pred`. Returns (future_len, 2)
    absolute positions, or None."""
    traj = scenario.log_trajectory
    if t + future_len >= traj.x.shape[1] or not np.all(traj.valid[ego_idx, t:t + future_len + 1]):
        return None
    ego_xy, ego_yaw, _ = get_ego_frame(scenario, ego_idx, t)
    fut_raw = np.stack([traj.x[ego_idx, t + 1:t + future_len + 1],
                         traj.y[ego_idx, t + 1:t + future_len + 1]], axis=-1)
    fut_local = transform_positions(fut_raw, ego_xy, ego_yaw)
    return fut_local.astype(np.float32)


def get_ego_vec(scenario, ego_idx, t):
    """Ego's own current conditioning: [cos(yaw), sin(yaw), speed] -- yaw is always
    0 relative to itself so this is really just speed, kept as 3-dim for headroom."""
    traj = scenario.log_trajectory
    vx, vy = traj.vel_x[ego_idx, t], traj.vel_y[ego_idx, t]
    speed = float(np.hypot(vx, vy))
    return np.array([1.0, 0.0, speed], dtype=np.float32)


def extract_example(scenario, ego_idx, t, hist_len=10, future_len=30):
    """One full training example, or None if history/future isn't available at t."""
    target = extract_ego_target(scenario, ego_idx, t, future_len)
    agents = extract_agents(scenario, ego_idx, t, hist_len)
    if target is None or agents is None:
        return None
    return {
        "agent_hist": agents["hist"], "agent_class": agents["class"], "agent_mask": agents["mask"],
        **{f"map_{k}": v for k, v in extract_map(scenario, ego_idx, t).items()},
        **{f"light_{k}": v for k, v in extract_traffic_lights(scenario, ego_idx, t, hist_len).items()},
        "ego_vec": get_ego_vec(scenario, ego_idx, t),
        "target": target,
    }
