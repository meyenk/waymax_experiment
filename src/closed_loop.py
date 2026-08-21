"""
Closed-loop runner: encoder -> MPC -> dynamics (ego) + IDM (other agents),
repeated every real timestep.

From the first step onward, BOTH the ego and the other agents run our own
simulation, not the log:
  - ego: bicycle model, driven by the MPC's chosen control each step.
  - other agents: IDM (same-lane car-following only -- no lane changes, see
    mpc.py), driven by whatever's actually ahead of them each step, which
    may now be our diverging ego.

KNOWN LIMITATION addressed in Part B, task 1: agents currently move in a
straight line at a fixed heading (see mpc.py's bicycle_free) instead of
following their logged path's actual curvature.

KNOWN LIMITATION addressed in Part B, task 2: the agent-history and
light-history features fed to the model are all anchored to the ego's pose
at the CURRENT decision instant, not to the ego's pose at each individual
historical timestep. See build_model_batch below.

Only two things still come from the log every step, because they're
exogenous (not affected by any agent's actions):
  - the static map (read once, kept in world frame, re-transformed into the
    current ego frame every step)
  - traffic light state (re-read from the log by real time index every step
    -- Waymax logs the full light sequence for the whole scene regardless of
    what any agent does)

FIELD NAMES on the scenario object mirror extract.py's assumptions (same
ADAPT caveats apply -- check against your installed Waymax version).

This module does NOT depend on Waymax/JAX itself -- it only reads plain
array fields off whatever scenario object is passed in, so it can be
exercised against a synthetic fake scenario for testing (see
tests/test_closed_loop.py) before running against the real thing.
"""
import numpy as np
import torch

from src.transforms import transform_positions, transform_velocities, transform_yaw
from src.dynamics import bicycle_step
from src.mpc import run_mpc_step, rollout_other_agents

MAX_AGENTS = 32
MAX_MAP_POINTS = 200
MAX_LIGHTS = 16


def _pad(arr, max_n):
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


# ---------------------------------------------------------------------------
# One-time initialization from the log
# ---------------------------------------------------------------------------

def init_other_agents(scenario, ego_idx, t0, hist_len):
    """Fixed set of non-ego agents valid at t0 (agents that appear/disappear
    later in the log are not tracked -- known simplification. Part B, task 1
    changes how agents ENTER the sim over time; see that spec).
    Returns: world_states0 (N,4) [x,y,yaw,speed] at t0, class_ids (N,),
    hist_world (N, hist_len, 4) for the buffer, built from the log over
    [t0-hist_len+1, t0] (missing history at scene start is a real edge case
    -- caller should pick t0 >= hist_len-1)."""
    traj = scenario.log_trajectory
    valid_t0 = np.array(traj.valid[:, t0]).copy()
    valid_t0[ego_idx] = False
    idx = np.where(valid_t0)[0]

    x = np.array(traj.x)[idx, t0]
    y = np.array(traj.y)[idx, t0]
    yaw = np.array(traj.yaw)[idx, t0]
    vx = np.array(traj.vel_x)[idx, t0]
    vy = np.array(traj.vel_y)[idx, t0]
    speed = np.hypot(vx, vy)
    world_states0 = np.stack([x, y, yaw, speed], axis=-1)

    cls = np.array(scenario.object_metadata.object_types)[idx].astype(np.int64)

    t_start = t0 - hist_len + 1
    hx = np.array(traj.x)[idx, t_start:t0 + 1]
    hy = np.array(traj.y)[idx, t_start:t0 + 1]
    hyaw = np.array(traj.yaw)[idx, t_start:t0 + 1]
    hvx = np.array(traj.vel_x)[idx, t_start:t0 + 1]
    hvy = np.array(traj.vel_y)[idx, t_start:t0 + 1]
    hspeed = np.hypot(hvx, hvy)
    hist_world = np.stack([hx, hy, hyaw, hspeed], axis=-1)  # (N, hist_len, 4)

    return world_states0, cls, hist_world, idx


def init_map_points(scenario, radius_center_xy, cache_radius=200.0):
    """Roadgraph points in world frame, cached once. cache_radius should be
    generous enough to cover the whole rollout, not just the current step
    (unlike the per-step 80m extraction radius applied later)."""
    rg = scenario.roadgraph_points
    valid = np.array(rg.valid)
    xy = np.stack([np.array(rg.x)[valid], np.array(rg.y)[valid]], axis=-1)
    types = np.array(rg.types)[valid].astype(np.int64)
    dist = np.linalg.norm(xy - radius_center_xy, axis=-1)
    keep = dist <= cache_radius
    return xy[keep], types[keep]


def get_light_window_world(scenario, t, hist_len):
    """Mirrors extract_traffic_lights but returns raw WORLD-frame history
    (no ego transform) -- transform happens fresh every step using our own
    current ego pose instead of the logged one."""
    empty = (np.zeros((0, hist_len, 2), dtype=np.float32), np.zeros((0, hist_len), dtype=np.int64))
    t0 = t - hist_len + 1
    if t0 < 0 or not hasattr(scenario, "log_traffic_light"):
        return empty
    tl = scenario.log_traffic_light
    valid_window = np.array(tl.valid[:, t0:t + 1])
    full_valid = valid_window.all(axis=1)
    idx = np.where(full_valid)[0]
    if len(idx) == 0:
        return empty
    xy = np.stack([np.array(tl.x)[idx, t0:t + 1], np.array(tl.y)[idx, t0:t + 1]], axis=-1).astype(np.float32)
    state = np.array(tl.state)[idx, t0:t + 1].astype(np.int64)
    return xy, state


# ---------------------------------------------------------------------------
# Per-step: build model input in the CURRENT (simulated) ego frame
# ---------------------------------------------------------------------------

def build_model_batch(agent_hist_world, agent_cls, map_xy_world, map_types,
                       light_hist_world_xy, light_state, ego_world_xy, ego_yaw,
                       ego_speed, map_radius=80.0):
    """All *_world inputs are plain numpy in world frame; returns a batch
    dict of torch tensors (batch size 1) in the model's expected ego-local
    frame, built fresh from our own current ego pose.

    NOTE (Part B, task 2 will change this function): every historical
    timestep in agent_hist_world/light_hist_world_xy is currently transformed
    using this SAME (ego_world_xy, ego_yaw, ego_speed) triple -- i.e. the
    ego's pose at the CURRENT decision instant, not at each individual
    historical timestep. See Part B, task 2 spec below for the fix."""
    # agents: (N, hist_len, 4) world -> (N, hist_len, 6) ego-local feats
    N, hist_len, _ = agent_hist_world.shape
    xy_local = transform_positions(agent_hist_world[..., :2], ego_world_xy, ego_yaw)
    yaw_local = transform_yaw(agent_hist_world[..., 2], ego_yaw)
    # velocity as (vx, vy) isn't tracked in our (x,y,yaw,speed) state; approximate
    # from heading * speed, consistent with how we drive agents in mpc.py.
    vx_world = agent_hist_world[..., 3] * np.cos(agent_hist_world[..., 2])
    vy_world = agent_hist_world[..., 3] * np.sin(agent_hist_world[..., 2])
    # ego's velocity in WORLD frame is needed for transform_velocities' relative
    # subtraction to be meaningful; reconstruct it from ego_world_xy's yaw/speed
    ego_vxvy_world = np.array([ego_speed * np.cos(ego_yaw), ego_speed * np.sin(ego_yaw)])
    v_rel = transform_velocities(np.stack([vx_world, vy_world], axis=-1), ego_vxvy_world, ego_yaw)
    agent_feats = np.concatenate(
        [xy_local, np.cos(yaw_local)[..., None], np.sin(yaw_local)[..., None], v_rel],
        axis=-1).astype(np.float32)
    agent_hist, agent_mask = _pad(agent_feats, MAX_AGENTS)
    agent_class, _ = _pad(agent_cls, MAX_AGENTS)

    # map: world -> ego-local, radius filter, pad
    map_local = transform_positions(map_xy_world, ego_world_xy, ego_yaw)
    dist = np.linalg.norm(map_local, axis=-1)
    keep = dist <= map_radius
    map_xy_kept, map_types_kept = map_local[keep].astype(np.float32), map_types[keep]
    map_xy, map_mask = _pad(map_xy_kept, MAX_MAP_POINTS)
    map_type, _ = _pad(map_types_kept, MAX_MAP_POINTS)

    # lights: world -> ego-local, pad
    if light_hist_world_xy.shape[0] > 0:
        light_local = transform_positions(light_hist_world_xy, ego_world_xy, ego_yaw).astype(np.float32)
    else:
        light_local = light_hist_world_xy.astype(np.float32)
    light_hist_xy, light_mask = _pad(light_local, MAX_LIGHTS)
    light_state_p, _ = _pad(light_state, MAX_LIGHTS)

    def t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype).unsqueeze(0)

    return {
        "agent_hist": t(agent_hist), "agent_class": t(agent_class, torch.int64), "agent_mask": t(agent_mask, torch.bool),
        "map_xy": t(map_xy), "map_type": t(map_type, torch.int64), "map_mask": t(map_mask, torch.bool),
        "light_hist_xy": t(light_hist_xy), "light_state": t(light_state_p, torch.int64), "light_mask": t(light_mask, torch.bool),
        "ego_vec": t(np.array([1.0, 0.0, ego_speed], dtype=np.float32)),
    }


# ---------------------------------------------------------------------------
# The closed loop itself
# ---------------------------------------------------------------------------

def run_closed_loop(scenario, model, ego_idx, t0, max_steps=20, dt=0.1,
                     hist_len=10, future_len=30, n_candidates=32,
                     r_min=1.0, r_max=10.0, reactive=True, device="cpu"):
    """Returns a dict with per-step logs: ego_states (list of (4,) world),
    agent_states (list of (N,4) world), selection_modes (list of str)."""
    model.eval()

    traj = scenario.log_trajectory
    ego_xy0 = np.array([traj.x[ego_idx, t0], traj.y[ego_idx, t0]])
    ego_yaw0 = float(traj.yaw[ego_idx, t0])
    ego_v0 = float(np.hypot(traj.vel_x[ego_idx, t0], traj.vel_y[ego_idx, t0]))
    ego_state = np.array([ego_xy0[0], ego_xy0[1], ego_yaw0, ego_v0])

    other_states, agent_cls, agent_hist_buf, agent_idx = init_other_agents(scenario, ego_idx, t0, hist_len)
    map_xy_world, map_types = init_map_points(scenario, ego_xy0)

    log = {"ego_states": [ego_state.copy()], "agent_states": [other_states.copy()],
           "selection_modes": [], "ref_trajectories": []}

    scene_len = traj.x.shape[1]
    for step in range(max_steps):
        t = t0 + step
        if t >= scene_len:
            break

        light_hist_xy, light_state = get_light_window_world(scenario, t, hist_len)
        batch = build_model_batch(
            agent_hist_buf, agent_cls, map_xy_world, map_types,
            light_hist_xy, light_state, ego_state[:2], ego_state[2], ego_state[3])
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            pred, _ = model(batch)
        ref_trajectory_xy = pred[0].cpu().numpy()  # (future_len, 2), ego-local

        other_xy_local = transform_positions(other_states[:, :2], ego_state[:2], ego_state[2])
        other_yaw_local = transform_yaw(other_states[:, 2], ego_state[2])
        other_local = np.stack([other_xy_local[:, 0], other_xy_local[:, 1],
                                 other_yaw_local, other_states[:, 3]], axis=-1)

        mpc_out = run_mpc_step(
            np.array([0.0, 0.0, 0.0, ego_state[3]]), ref_trajectory_xy, other_local, dt,
            n_candidates=n_candidates, r_min=r_min, r_max=r_max, reactive=reactive)

        accel, steer = mpc_out["first_control"]
        new_ego_state = bicycle_step(ego_state, (accel, steer), dt)

        if other_states.shape[0] > 0:
            two_step_ego = np.stack([ego_state, new_ego_state])
            new_other_states = rollout_other_agents(
                other_states, two_step_ego, horizon=1, dt=dt, reactive=reactive)[:, 1, :]
        else:
            new_other_states = other_states

        agent_hist_buf = np.concatenate([agent_hist_buf[:, 1:, :], new_other_states[:, None, :]], axis=1) \
            if other_states.shape[0] > 0 else agent_hist_buf
        ego_state, other_states = new_ego_state, new_other_states

        log["ego_states"].append(ego_state.copy())
        log["agent_states"].append(other_states.copy())
        log["selection_modes"].append(mpc_out["selection_mode"])
        log["ref_trajectories"].append(ref_trajectory_xy)

    log["agent_idx"] = agent_idx
    return log
