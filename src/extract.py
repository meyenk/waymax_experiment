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
from src import features as F
from src.features import pad as _pad  # kept importable under the old name
from src.transforms import transform_positions


def get_ego_frame(scenario, ego_idx, t):
    """Returns (ego_xy, ego_yaw, ego_vxvy) at time t -- the frame anchor.
    Single-instant reference, used where there's no history to time-match
    (the map branch, the ego's own future target space)."""
    traj = scenario.log_trajectory
    ego_xy = np.array([traj.x[ego_idx, t], traj.y[ego_idx, t]])
    ego_yaw = float(traj.yaw[ego_idx, t])
    ego_vxvy = np.array([traj.vel_x[ego_idx, t], traj.vel_y[ego_idx, t]])
    return ego_xy, ego_yaw, ego_vxvy


def get_ego_frame_window(scenario, ego_idx, t0, t):
    """Returns (ego_xy, ego_yaw, ego_vxvy) over the window [t0, t] inclusive
    -- (hist_len, 2), (hist_len,), (hist_len, 2) -- the ego's own state at
    EACH individual historical instant, not just at t (Part B, task 2: the
    time-matched / tau convention). Frames where the ego itself is invalid
    are filled from its neighbours (features.ego_window)."""
    traj = scenario.log_trajectory
    ego_xy = np.stack([traj.x[ego_idx, t0:t + 1], traj.y[ego_idx, t0:t + 1]], axis=-1)
    ego_yaw = np.array(traj.yaw[ego_idx, t0:t + 1])
    ego_vxvy = np.stack([traj.vel_x[ego_idx, t0:t + 1], traj.vel_y[ego_idx, t0:t + 1]], axis=-1)
    return F.ego_window(ego_xy, ego_yaw, ego_vxvy, np.array(traj.valid[ego_idx, t0:t + 1]))


def select_agents(scenario, ego_idx, t):
    """Object indices the model looks at, at decision instant t: valid at t,
    not the ego, within features.AGENT_RADIUS of the ego, nearest-first,
    capped at features.MAX_AGENTS. Exposed separately so the visualization
    can show exactly what was (and wasn't) picked."""
    traj = scenario.log_trajectory
    valid_now = np.array(traj.valid[:, t]).copy()
    valid_now[ego_idx] = False
    cand = np.where(valid_now)[0]
    ego_xy = np.array([traj.x[ego_idx, t], traj.y[ego_idx, t]])
    xy_now = np.stack([np.array(traj.x)[cand, t], np.array(traj.y)[cand, t]], axis=-1)
    dist = np.linalg.norm(xy_now - ego_xy, axis=-1)
    return cand[F.nearest_within(dist, F.AGENT_RADIUS, F.MAX_AGENTS)]


def extract_agents(scenario, ego_idx, t, hist_len=10):
    """Non-ego agents in the region of interest at t (see select_agents),
    history window [t-hist_len+1, t] with tracker dropouts filled in and
    flagged (features.py). Each historical instant is expressed relative to
    the ego's own state at that same instant (Part B, task 2: time-matched).
    Returns hist_feats (MAX_AGENTS, hist_len, AGENT_FEAT_DIM), class_ids (MAX_AGENTS,), mask (MAX_AGENTS,)."""
    traj = scenario.log_trajectory
    t0 = t - hist_len + 1
    if t0 < 0:
        return None  # not enough history at this t
    ego_xy, ego_yaw, ego_vxvy = get_ego_frame_window(scenario, ego_idx, t0, t)

    idx = select_agents(scenario, ego_idx, t)
    if len(idx) == 0:
        return {
            "hist": np.zeros((F.MAX_AGENTS, hist_len, F.AGENT_FEAT_DIM), dtype=np.float32),
            "class": np.zeros(F.MAX_AGENTS, dtype=np.int64),
            "mask": np.zeros(F.MAX_AGENTS, dtype=bool),
        }

    w = slice(t0, t + 1)
    observed = np.array(traj.valid)[idx, w]
    raw = np.stack([np.array(traj.x)[idx, w], np.array(traj.y)[idx, w], np.array(traj.yaw)[idx, w],
                    np.array(traj.vel_x)[idx, w], np.array(traj.vel_y)[idx, w]], axis=-1)
    filled = F.fill_gaps(raw, observed, angle_channels=(2,))
    feats = F.agent_features(filled[..., :2], filled[..., 2], filled[..., 3:5], observed,
                             ego_xy, ego_yaw, ego_vxvy)

    # ADAPT: object_metadata.object_types field name/encoding may differ
    cls = np.array(scenario.object_metadata.object_types)[idx].astype(np.int64)

    hist, mask = F.pad(feats, F.MAX_AGENTS)
    cls, _ = F.pad(cls, F.MAX_AGENTS)
    return {"hist": hist, "class": cls, "mask": mask}


def extract_map(scenario, ego_idx, t):
    """Static roadgraph points (lane markings, stop signs, crosswalks, ...),
    ego frame at t: everything within features.MAP_RADIUS, nearest-first,
    capped at features.MAX_MAP_POINTS (see features.select_map).
    Returns xy (MAX_MAP_POINTS, 2), type_ids, mask, and n_in_radius (how many
    were in range before the cap -- popped by collate, used for stats)."""
    ego_xy, ego_yaw, _ = get_ego_frame(scenario, ego_idx, t)
    rg = scenario.roadgraph_points
    valid = np.array(rg.valid)
    xy_raw = np.stack([np.array(rg.x)[valid], np.array(rg.y)[valid]], axis=-1)
    types = np.array(rg.types)[valid].astype(np.int64)

    xy_local, types, n_in_radius = F.select_map(xy_raw, types, ego_xy, ego_yaw)
    xy_local, mask = F.pad(xy_local, F.MAX_MAP_POINTS)
    types, _ = F.pad(types, F.MAX_MAP_POINTS)
    return {"xy": xy_local, "type": types, "mask": mask, "n_in_radius": n_in_radius}


def extract_traffic_lights(scenario, ego_idx, t, hist_len=10):
    """Traffic light stop-line positions + state over history, time-matched
    (tau) ego frame like extract_agents -- position only,
    lights have no yaw/velocity feature. A light is used if it's valid at t
    and within features.MAP_RADIUS, nearest-first; frames where it wasn't
    observed get state 0 (UNKNOWN) and its current position.
    ADAPT: field name/shape for your Waymax version -- assumed scenario.log_traffic_light
    with .x, .y, .state, .valid shaped (num_lights, num_timesteps)."""
    t0 = t - hist_len + 1
    empty = {
        "hist_xy": np.zeros((F.MAX_LIGHTS, hist_len, 2), dtype=np.float32),
        "state": np.zeros((F.MAX_LIGHTS, hist_len), dtype=np.int64),
        "mask": np.zeros(F.MAX_LIGHTS, dtype=bool),
    }
    if t0 < 0 or not hasattr(scenario, "log_traffic_light"):
        return empty
    ego_xy, ego_yaw, _ = get_ego_frame_window(scenario, ego_idx, t0, t)

    tl = scenario.log_traffic_light
    valid_now = np.array(tl.valid[:, t])
    cand = np.where(valid_now)[0]
    if len(cand) == 0:
        return empty
    xy_now = np.stack([np.array(tl.x)[cand, t], np.array(tl.y)[cand, t]], axis=-1)
    dist = np.linalg.norm(xy_now - ego_xy[-1], axis=-1)
    idx = cand[F.nearest_within(dist, F.MAP_RADIUS, F.MAX_LIGHTS)]
    if len(idx) == 0:
        return empty

    xy_world, state = F.light_window_world(tl, idx, t0, t)
    xy = transform_positions(xy_world, ego_xy, ego_yaw)

    hist_xy, mask = F.pad(xy.astype(np.float32), F.MAX_LIGHTS)
    state, _ = F.pad(state, F.MAX_LIGHTS)
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
