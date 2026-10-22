"""
Closed-loop runner: encoder -> MPC -> dynamics (ego) + IDM (other agents),
repeated every real timestep.

From the first step onward, BOTH the ego and the other agents run our own
simulation, not the log:
  - ego: bicycle model, driven by the MPC's chosen control each step.
  - other agents: IDM (same-lane car-following only -- no lane changes, see
    mpc.py), driven by whatever's actually ahead of them each step, which
    may now be our diverging ego.

Part B, task 1 (done): other agents path-follow their logged path's actual
curvature via a cached arc-length parameterization (see mpc.py's AgentPath),
instead of moving in a straight line at a fixed heading. IDM only ever
controls how fast an agent moves.

Part B, task 2 (done): the agent-history and light-history features fed to
the model are time-matched (tau) -- each historical instant is expressed
relative to the ego's own pose at THAT instant, not the ego's pose at the
current decision instant t. See build_model_batch below. This changes the
model's input feature values (not their dimensionality), so the existing
checkpoints (stage1_weights.pth, best_weights_v2.pth) are stale against this
code and a retrain is required before using them again.

Tracker dropouts (meeting-prep branch): real WOMD validity masks have holes
-- 95% of scenarios have at least one object that vanishes for a few frames
and comes back (occlusion), which Task 1's one-contiguous-span assumption
rejected outright. Now gaps of up to MAX_BRIDGE_GAP frames are bridged (the
object is treated as present throughout, its logged path linearly
interpolated across the hole); a longer gap splits the object into separate
spans -- it leaves the sim at the gap and re-enters at the next span, through
the same entry/exit logic as any other agent. See build_agent_paths.

What the MODEL sees each step goes through the same features.py selection
as training (ROI radius, nearest-first, dropout filling + observed flag),
built from our own simulated states. Every active agent is still SIMULATED
regardless of distance (it may come close later), but only those within
MPC_AGENT_RADIUS go into the MPC's candidate rollouts -- the leader search is
O(N^2) per candidate per step, and agents beyond that can't interact with the
ego within the 3s horizon anyway.

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

from src import features as F
from src.features import pad as _pad  # kept importable under the old name
from src.transforms import transform_positions, transform_yaw
from src.dynamics import bicycle_step
from src.mpc import run_mpc_step, rollout_other_agents, AgentPath

MAX_BRIDGE_GAP = 10       # frames (1s at 10Hz): longest dropout treated as "still there"
MPC_AGENT_RADIUS = 50.0   # m: agents considered in the MPC's candidate rollouts


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
    -- caller should pick t0 >= hist_len-1).

    Kept as-is (t0-only snapshot) for callers that don't need path-following
    (e.g. standalone use). run_closed_loop itself now uses
    build_agent_paths + ActiveAgentSet below, which scans the whole scene
    once and tracks agents entering/leaving over time (Part B, task 1)."""
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


# ---------------------------------------------------------------------------
# Part B, task 1 -- whole-scene scan + path-following active agent set.
# ---------------------------------------------------------------------------

def build_agent_paths(scenario, ego_idx, max_gap=MAX_BRIDGE_GAP):
    """Whole-scene scan, done once at load time: for every non-ego object,
    bridge tracker dropouts of up to max_gap frames (features.bridge_short_gaps),
    linearly interpolate its logged state across them (features.fill_gaps),
    then split what's left into contiguous spans -- one per stretch of the
    scene the object is actually present for. Each span gets its own cached
    AgentPath (arc-length parameterized, Part B task 1).

    Before this branch, any gap at all raised a ValueError (the original
    Task 1 spec asked for that rather than guessing), which made 95% of real
    scenarios unusable in the closed loop.

    Returns dict:
      spans {obj_idx: [(t_first, t_last, AgentPath), ...]} in time order,
      paths {obj_idx: AgentPath} (first span; kept for older callers),
      first_valid / last_valid {obj_idx: int} (over all spans),
      cls {obj_idx: int}, present (num_objects, T) bridged validity,
      observed (num_objects, T) raw log validity, and the filled world-frame
      state arrays x, y, yaw, vel_x, vel_y (num_objects, T).
    Objects that are never valid are skipped."""
    traj = scenario.log_trajectory
    observed = np.array(traj.valid, dtype=bool)  # (num_objects, T)
    raw = np.stack([np.array(traj.x), np.array(traj.y), np.array(traj.yaw),
                    np.array(traj.vel_x), np.array(traj.vel_y)], axis=-1).astype(float)
    filled = F.fill_gaps(raw, observed, angle_channels=(2,))
    present = F.bridge_short_gaps(observed, max_gap)
    obj_types = np.array(scenario.object_metadata.object_types)

    out = {"spans": {}, "paths": {}, "first_valid": {}, "last_valid": {}, "cls": {},
           "present": present, "observed": observed,
           "x": filled[..., 0], "y": filled[..., 1], "yaw": filled[..., 2],
           "vel_x": filled[..., 3], "vel_y": filled[..., 4]}
    for i in range(observed.shape[0]):
        if i == ego_idx or not observed[i].any():
            continue
        spans = []
        for t_first, t_last in F.valid_runs(present[i]):
            wp_xy = filled[i, t_first:t_last + 1, :2]
            spans.append((t_first, t_last, AgentPath(wp_xy)))
        out["spans"][i] = spans
        out["paths"][i] = spans[0][2]
        out["first_valid"][i] = spans[0][0]
        out["last_valid"][i] = spans[-1][1]
        out["cls"][i] = int(obj_types[i])
    return out


def _agent_world_state(scene_paths, obj_idx, t):
    """(x, y, yaw, speed) for object obj_idx at time t, from the gap-filled log."""
    sp = scene_paths
    speed = float(np.hypot(sp["vel_x"][obj_idx, t], sp["vel_y"][obj_idx, t]))
    return np.array([sp["x"][obj_idx, t], sp["y"][obj_idx, t], sp["yaw"][obj_idx, t], speed])


class ActiveAgentSet:
    """Tracks the non-ego agents currently in the real (simulated) closed
    loop: per-agent arc length `s` along the cached AgentPath of its current
    span (Part B, task 1), current world (x, y, yaw, speed), object class,
    and a rolling hist_len-step history buffer (+ per-frame observed flags)
    for the model.

    An agent is in the set while the scene says it's present (bridged
    validity, so a short tracker dropout doesn't remove it) and leaves the
    moment it isn't -- fallback case 2, distinct from "ran past the end of
    the known path", which is fallback case 1, handled inside
    bicycle_free/rollout_other_agents via path.exhausted. After a long gap
    it re-enters at its next span, starting at s=0 on that span's path."""

    def __init__(self, scene_paths, hist_len):
        self.scene_paths = scene_paths
        self.hist_len = hist_len
        self.order = []            # active obj_idx, fixed iteration order
        self.s = {}                # obj_idx -> float arc length
        self.state = {}            # obj_idx -> (4,) world
        self.hist = {}             # obj_idx -> (hist_len, 4) world, oldest..newest
        self.hist_obs = {}         # obj_idx -> (hist_len,) bool, real (log/sim) vs filled
        self.path = {}             # obj_idx -> AgentPath of the current span

    def sync(self, scenario, t):
        """Drop agents not present at t (fallback case 2); add any object
        present at t that isn't tracked yet -- on the first sync that picks
        up every agent already mid-span, later it's mid-scene entry or
        re-entry after a long gap. Well-defined even though the ego/other
        agents may have diverged from the log, since presence is purely a
        property of the log's own timeline."""
        present_t = self.scene_paths["present"][:, t]

        for obj_idx in list(self.order):
            if not present_t[obj_idx]:
                self.order.remove(obj_idx)
                for d in (self.s, self.state, self.hist, self.hist_obs, self.path):
                    del d[obj_idx]

        for obj_idx, spans in self.scene_paths["spans"].items():
            if obj_idx in self.s or not present_t[obj_idx]:
                continue
            span = next(sp for sp in spans if sp[0] <= t <= sp[1])
            self._add(obj_idx, t, span)

    def _add(self, obj_idx, t, span):
        """Start tracking obj_idx from real time t. Its arc length starts at
        however far along its span's path the log says it already is (0 for
        a fresh entrant, non-zero for one that's been driving since before t)."""
        t_first, _, path = span
        offset = int(np.clip(t - t_first, 0, len(path.cum_s) - 1))
        self.order.append(obj_idx)
        self.s[obj_idx] = float(path.cum_s[offset])
        self.path[obj_idx] = path
        self.state[obj_idx] = _agent_world_state(self.scene_paths, obj_idx, t)
        self.hist[obj_idx], self.hist_obs[obj_idx] = self._log_history(obj_idx, t, t_first)

    def _log_history(self, obj_idx, t, t_first):
        """(hist_len, 4) world history over [t-hist_len+1, t] read from the
        gap-filled log -- correct here because an agent is only ever added at
        the moment we start simulating it, so nothing before that has
        diverged. Window entries earlier than the span's first timestep repeat
        its first state (there's nothing real to put there) and, like bridged
        frames, are flagged unobserved -- same convention as training."""
        taus = range(t - self.hist_len + 1, t + 1)
        rows = [_agent_world_state(self.scene_paths, obj_idx, max(tau, t_first)) for tau in taus]
        obs = np.array([tau >= t_first and tau >= 0 and bool(self.scene_paths["observed"][obj_idx, tau])
                        for tau in taus])
        return np.stack(rows, axis=0), obs

    def is_empty(self):
        return len(self.order) == 0

    def states_array(self):
        return np.stack([self.state[i] for i in self.order], axis=0) if self.order \
            else np.zeros((0, 4))

    def hist_array(self):
        return np.stack([self.hist[i] for i in self.order], axis=0) if self.order \
            else np.zeros((0, self.hist_len, 4))

    def hist_obs_array(self):
        return np.stack([self.hist_obs[i] for i in self.order], axis=0) if self.order \
            else np.zeros((0, self.hist_len), dtype=bool)

    def cls_array(self):
        return np.array([self.scene_paths["cls"][i] for i in self.order], dtype=np.int64)

    def paths_list(self):
        return [self.path[i] for i in self.order]

    def s_array(self):
        return np.array([self.s[i] for i in self.order], dtype=float)

    def advance(self, ego_state, new_ego_state, dt, reactive):
        """Step every active agent forward exactly one real timestep, path-
        following via its cached AgentPath (Part B, task 1), and update the
        rolling history buffers (simulated frames count as observed)."""
        if self.is_empty():
            return
        two_step_ego = np.stack([ego_state, new_ego_state])
        new_states, new_s = rollout_other_agents(
            self.states_array(), two_step_ego, horizon=1, dt=dt, reactive=reactive,
            paths=self.paths_list(), s0=self.s_array())
        for k, obj_idx in enumerate(self.order):
            self.state[obj_idx] = new_states[k, 1, :]
            self.s[obj_idx] = new_s[k]
            self.hist[obj_idx] = np.concatenate(
                [self.hist[obj_idx][1:], self.state[obj_idx][None, :]], axis=0)
            self.hist_obs[obj_idx] = np.concatenate([self.hist_obs[obj_idx][1:], [True]])


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


def get_light_window_world(scenario, t, hist_len, ego_xy=None):
    """Mirrors extract_traffic_lights but returns raw WORLD-frame history
    (no ego transform) -- transform happens fresh every step using our own
    current ego pose instead of the logged one. Same selection as training:
    valid at t, within features.MAP_RADIUS of ego_xy (our simulated ego),
    nearest-first, unobserved frames -> state 0 (features.light_window_world)."""
    empty = (np.zeros((0, hist_len, 2), dtype=np.float32), np.zeros((0, hist_len), dtype=np.int64))
    t0 = t - hist_len + 1
    if t0 < 0 or not hasattr(scenario, "log_traffic_light"):
        return empty
    tl = scenario.log_traffic_light
    cand = np.where(np.array(tl.valid[:, t]))[0]
    if len(cand) == 0:
        return empty
    if ego_xy is not None:
        xy_now = np.stack([np.array(tl.x)[cand, t], np.array(tl.y)[cand, t]], axis=-1)
        cand = cand[F.nearest_within(np.linalg.norm(xy_now - ego_xy, axis=-1), F.MAP_RADIUS, F.MAX_LIGHTS)]
        if len(cand) == 0:
            return empty
    return F.light_window_world(tl, cand, t0, t)


# ---------------------------------------------------------------------------
# Per-step: build model input in the CURRENT (simulated) ego frame
# ---------------------------------------------------------------------------

def build_model_batch(agent_hist_world, agent_cls, map_xy_world, map_types,
                       light_hist_world_xy, light_state,
                       ego_hist_xy, ego_hist_yaw, ego_hist_vxvy,
                       agent_hist_obs=None, time_matched=True, return_selection=False):
    """All *_world inputs are plain numpy in world frame; returns a batch
    dict of torch tensors (batch size 1) in the model's expected ego-local
    frame, built through the SAME features.py selection as training.

    agent_hist_world: (N, hist_len, 4) [x, y, yaw, speed] for every active
    agent; agent_hist_obs: (N, hist_len) bool observed flags (None = all
    observed). Selection: within features.AGENT_RADIUS of the ego now,
    nearest-first, capped at features.MAX_AGENTS.

    ego_hist_xy/ego_hist_yaw/ego_hist_vxvy: (hist_len,2)/(hist_len,)/(hist_len,2)
    -- the ego's own state at EACH individual historical instant tau (Part B,
    task 2). The caller is responsible for the tau<t0 fallback (use the log
    there, since the ego hadn't diverged from it yet) -- see run_closed_loop.
    time_matched=False reproduces the pre-Task-2 features (must match how the
    loaded checkpoint was trained).

    return_selection=True also returns a dict with the chosen agent rows
    ("agent_rows", indices into agent_hist_world) and map stats, for logging
    and visualization."""
    ego_world_xy = ego_hist_xy[-1]
    ego_yaw = float(ego_hist_yaw[-1])
    ego_speed = float(np.hypot(ego_hist_vxvy[-1, 0], ego_hist_vxvy[-1, 1]))
    N, hist_len, _ = agent_hist_world.shape
    if agent_hist_obs is None:
        agent_hist_obs = np.ones((N, hist_len), dtype=bool)

    # agents: ROI selection on current positions, then shared feature builder.
    # velocity as (vx, vy) isn't tracked in our (x,y,yaw,speed) state; approximate
    # from heading * speed, consistent with how we drive agents in mpc.py.
    if N > 0:
        dist = np.linalg.norm(agent_hist_world[:, -1, :2] - ego_world_xy, axis=-1)
        rows = F.nearest_within(dist, F.AGENT_RADIUS, F.MAX_AGENTS)
    else:
        rows = np.zeros(0, dtype=int)
    sel = agent_hist_world[rows]
    vxvy = np.stack([sel[..., 3] * np.cos(sel[..., 2]), sel[..., 3] * np.sin(sel[..., 2])], axis=-1)
    agent_feats = F.agent_features(sel[..., :2], sel[..., 2], vxvy, agent_hist_obs[rows],
                                   ego_hist_xy, ego_hist_yaw, ego_hist_vxvy, time_matched=time_matched)
    agent_hist, agent_mask = F.pad(agent_feats, F.MAX_AGENTS)
    agent_class, _ = F.pad(np.asarray(agent_cls, dtype=np.int64)[rows], F.MAX_AGENTS)

    # map: ego frame at the CURRENT instant only -- single static snapshot,
    # task 2 doesn't apply here.
    map_xy_kept, map_types_kept, n_map_in_radius = F.select_map(map_xy_world, map_types, ego_world_xy, ego_yaw)
    map_xy, map_mask = F.pad(map_xy_kept, F.MAX_MAP_POINTS)
    map_type, _ = F.pad(map_types_kept, F.MAX_MAP_POINTS)

    # lights: same anchoring convention as agents (position only), pad
    if light_hist_world_xy.shape[0] > 0:
        if time_matched:
            light_local = transform_positions(light_hist_world_xy, ego_hist_xy, ego_hist_yaw)
        else:
            light_local = transform_positions(light_hist_world_xy, ego_world_xy, ego_yaw)
        light_local = light_local.astype(np.float32)
    else:
        light_local = light_hist_world_xy.astype(np.float32)
    light_hist_xy, light_mask = F.pad(light_local, F.MAX_LIGHTS)
    light_state_p, _ = F.pad(light_state, F.MAX_LIGHTS)

    def t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype).unsqueeze(0)

    batch = {
        "agent_hist": t(agent_hist), "agent_class": t(agent_class, torch.int64), "agent_mask": t(agent_mask, torch.bool),
        "map_xy": t(map_xy), "map_type": t(map_type, torch.int64), "map_mask": t(map_mask, torch.bool),
        "light_hist_xy": t(light_hist_xy), "light_state": t(light_state_p, torch.int64), "light_mask": t(light_mask, torch.bool),
        "ego_vec": t(np.array([1.0, 0.0, ego_speed], dtype=np.float32)),
    }
    if return_selection:
        return batch, {"agent_rows": rows, "n_map_in_radius": n_map_in_radius,
                       "n_map_used": int(map_mask.sum())}
    return batch


# ---------------------------------------------------------------------------
# The closed loop itself
# ---------------------------------------------------------------------------

def run_closed_loop(scenario, model, ego_idx, t0, max_steps=20, dt=0.1,
                     hist_len=10, future_len=30, n_candidates=32,
                     r_min=1.0, r_max=10.0, reactive=True, device="cpu",
                     time_matched=True, cost_kwargs=None):
    """Returns a dict with per-step logs: ego_states (list of (4,) world),
    agent_states (list of (N,4) world, every active agent), agent_ids (list
    of (N,) object indices matching agent_states row-for-row),
    model_agent_ids (list of object indices the MODEL saw that step),
    mpc_agent_ids (list of object indices in the MPC rollouts that step),
    selection_modes, ref_trajectories.

    Other agents (Part B, task 1 + dropout handling): the whole scene is
    scanned once up front (build_agent_paths) to find every object's spans
    and cache its logged path per span. From then on, an ActiveAgentSet
    tracks which agents are currently in the sim, path-following via arc
    length so each agent's shape always matches its logged curve while IDM
    only ever controls its speed.

    Ego history (Part B, task 2): ego_hist_buf is seeded from the log for
    the very first window (gaps filled, features.ego_window), then rolled
    forward with our own simulated ego state every step."""
    model.eval()

    traj = scenario.log_trajectory
    ego_xy0 = np.array([traj.x[ego_idx, t0], traj.y[ego_idx, t0]])
    ego_yaw0 = float(traj.yaw[ego_idx, t0])
    ego_v0 = float(np.hypot(traj.vel_x[ego_idx, t0], traj.vel_y[ego_idx, t0]))
    ego_state = np.array([ego_xy0[0], ego_xy0[1], ego_yaw0, ego_v0])

    w = slice(t0 - hist_len + 1, t0 + 1)
    hxy, hyaw, hvxvy = F.ego_window(
        np.stack([np.array(traj.x[ego_idx, w]), np.array(traj.y[ego_idx, w])], axis=-1),
        np.array(traj.yaw[ego_idx, w]),
        np.stack([np.array(traj.vel_x[ego_idx, w]), np.array(traj.vel_y[ego_idx, w])], axis=-1),
        np.array(traj.valid[ego_idx, w]))
    ego_hist_buf = np.concatenate([hxy, hyaw[:, None], np.hypot(hvxvy[:, 0], hvxvy[:, 1])[:, None]],
                                  axis=-1)  # (hist_len, 4) -- from the log; correct pre-t0

    scene_paths = build_agent_paths(scenario, ego_idx)
    agents = ActiveAgentSet(scene_paths, hist_len)
    agents.sync(scenario, t0)
    map_xy_world, map_types = init_map_points(scenario, ego_xy0)

    log = {"ego_states": [ego_state.copy()], "agent_states": [agents.states_array()],
           "agent_ids": [np.array(agents.order)], "model_agent_ids": [], "mpc_agent_ids": [],
           "selection_modes": [], "ref_trajectories": [], "n_map_in_radius": []}

    scene_len = traj.x.shape[1]
    for step in range(max_steps):
        t = t0 + step
        if t >= scene_len:
            break

        ego_hist_xy = ego_hist_buf[:, :2]
        ego_hist_yaw = ego_hist_buf[:, 2]
        ego_hist_vxvy = np.stack(
            [ego_hist_buf[:, 3] * np.cos(ego_hist_yaw), ego_hist_buf[:, 3] * np.sin(ego_hist_yaw)], axis=-1)

        order = np.array(agents.order, dtype=int)
        light_hist_xy, light_state = get_light_window_world(scenario, t, hist_len, ego_xy=ego_state[:2])
        batch, sel = build_model_batch(
            agents.hist_array(), agents.cls_array(), map_xy_world, map_types,
            light_hist_xy, light_state, ego_hist_xy, ego_hist_yaw, ego_hist_vxvy,
            agent_hist_obs=agents.hist_obs_array(), time_matched=time_matched, return_selection=True)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            pred, _ = model(batch)
        ref_trajectory_xy = pred[0].cpu().numpy()  # (future_len, 2), ego-local

        other_states = agents.states_array()
        near = np.linalg.norm(other_states[:, :2] - ego_state[:2], axis=-1) <= MPC_AGENT_RADIUS \
            if len(order) else np.zeros(0, dtype=bool)
        other_near = other_states[near]
        other_xy_local = transform_positions(other_near[:, :2], ego_state[:2], ego_state[2])
        other_yaw_local = transform_yaw(other_near[:, 2], ego_state[2])
        other_local = np.stack([other_xy_local[:, 0], other_xy_local[:, 1],
                                 other_yaw_local, other_near[:, 3]], axis=-1)

        mpc_out = run_mpc_step(
            np.array([0.0, 0.0, 0.0, ego_state[3]]), ref_trajectory_xy, other_local, dt,
            n_candidates=n_candidates, r_min=r_min, r_max=r_max, reactive=reactive,
            cost_kwargs=cost_kwargs)

        accel, steer = mpc_out["first_control"]
        new_ego_state = bicycle_step(ego_state, (accel, steer), dt)

        agents.advance(ego_state, new_ego_state, dt, reactive)
        ego_state = new_ego_state
        ego_hist_buf = np.concatenate([ego_hist_buf[1:], ego_state[None, :]], axis=0)

        log["model_agent_ids"].append(order[sel["agent_rows"]] if len(order) else order)
        log["mpc_agent_ids"].append(order[near])
        log["n_map_in_radius"].append(sel["n_map_in_radius"])
        log["selection_modes"].append(mpc_out["selection_mode"])
        log["ref_trajectories"].append(ref_trajectory_xy)

        # agents entering (or re-entering after a long gap) at the NEXT real
        # timestep are picked up here so their history buffer/state exist
        # before the next step's model batch is built.
        if t + 1 < scene_len:
            agents.sync(scenario, t + 1)
        log["ego_states"].append(ego_state.copy())
        log["agent_states"].append(agents.states_array())
        log["agent_ids"].append(np.array(agents.order))

    log["agent_idx"] = np.array(agents.order)
    return log
