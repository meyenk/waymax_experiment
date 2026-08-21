"""
Simple closed-loop MPC over the kinematic bicycle model.

Pipeline for one planning step:
  reference trajectory (from encoder, ego-local, yaw=0)
    -> infer_reference_controls (dynamics.py)
    -> sample_candidate_controls: perturb around reference, hard-constrained
       to a cone that grows linearly from r_min at the first future step to
       r_max at the last (1m -> 10m by default)
    -> rollout_bicycle each candidate (ego) + rollout_other_agents via IDM
    -> compute_cost per candidate: collision + progress + comfort
       (Waymax-metric-style categories, NOT an L2-to-reference term -- the
       cone already bounds how far a candidate can stray from the reference,
       cost decides which of the *allowed* candidates is best)
    -> run_mpc_step returns the best candidate's full control sequence; only
       its first control is actually executed (receding horizon), then the
       whole thing repeats from the new real state.

No Waymax/JAX dependency here -- everything operates on plain numpy arrays
so this can be unit-tested standalone before wiring to a real scenario.
"""
import numpy as np

from src.dynamics import rollout_bicycle, infer_reference_controls, WHEELBASE


# ---------------------------------------------------------------------------
# IDM -- same-lane car-following only (does not handle lane changes / merges
# / intersections -- named plainly, not hidden, per earlier discussion).
#
# KNOWN LIMITATION (see Part B, task 1): rollout_other_agents / bicycle_free
# currently hold each agent's yaw FIXED for the entire rollout -- an agent
# only ever moves in a straight line at whatever heading it had when the
# rollout started. It does NOT follow the curvature of its logged path. This
# is the exact thing Part B, task 1 fixes.
# ---------------------------------------------------------------------------

IDM_DEFAULTS = dict(v0=15.0, T=1.5, a_max=1.5, b_comf=2.0, s0=2.0, delta=4.0)


# ---------------------------------------------------------------------------
# Part B, task 1 -- cached logged path per agent, arc-length parameterized.
# IDM only ever controls *how fast* an agent moves; the *shape* of its path
# always matches the log, via lookup(s) below. Pure numpy, no Waymax
# dependency, so it stays unit-testable standalone like the rest of this file.
# ---------------------------------------------------------------------------

class AgentPath:
    """Cached (x, y) waypoints for one non-ego agent across its own valid
    span in the log, parameterized by cumulative arc length `s` (0 at the
    agent's first valid timestep)."""

    def __init__(self, waypoints_xy):
        waypoints_xy = np.asarray(waypoints_xy, dtype=float)
        if waypoints_xy.ndim != 2 or waypoints_xy.shape[1] != 2 or waypoints_xy.shape[0] < 1:
            raise ValueError("waypoints_xy must be (M, 2) with M >= 1")
        self.waypoints_xy = waypoints_xy
        if waypoints_xy.shape[0] == 1:
            self.cum_s = np.array([0.0])
        else:
            seg_len = np.hypot(np.diff(waypoints_xy[:, 0]), np.diff(waypoints_xy[:, 1]))
            self.cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.total_length = float(self.cum_s[-1])

    def lookup(self, s):
        """Returns (x, y, yaw) at arc length s. s is clamped to
        [0, total_length] -- callers that need the "ran past the end of the
        known path" behavior (fallback case 1, see bicycle_free) must check
        that themselves before calling this with an out-of-range s."""
        if self.waypoints_xy.shape[0] == 1:
            return self.waypoints_xy[0, 0], self.waypoints_xy[0, 1], 0.0
        s_clamped = float(np.clip(s, 0.0, self.total_length))
        i = int(np.clip(np.searchsorted(self.cum_s, s_clamped, side="right") - 1,
                         0, len(self.cum_s) - 2))
        s0, s1 = self.cum_s[i], self.cum_s[i + 1]
        p0, p1 = self.waypoints_xy[i], self.waypoints_xy[i + 1]
        frac = 0.0 if s1 <= s0 else (s_clamped - s0) / (s1 - s0)
        xy = p0 + frac * (p1 - p0)
        dx, dy = p1 - p0
        yaw = np.arctan2(dy, dx)
        return float(xy[0]), float(xy[1]), float(yaw)

    def exhausted(self, s):
        return s > self.total_length


def idm_accel(v, v_lead, gap, params=None):
    """Standard IDM formula. gap: bumper-to-bumper distance to leader (m).
    If there's no leader, call with gap=large (e.g. 1e6) and v_lead=v0."""
    p = {**IDM_DEFAULTS, **(params or {})}
    gap = max(gap, 1e-3)
    dv = v - v_lead
    s_star = p["s0"] + max(
        0.0, v * p["T"] + (v * dv) / (2 * np.sqrt(p["a_max"] * p["b_comf"]))
    )
    return p["a_max"] * (1 - (v / p["v0"]) ** p["delta"] - (s_star / gap) ** 2)


def _find_leader(idx, positions, yaws, corridor_half_width=1.5):
    """positions: (M, 2) all objects (other agents + ego) at this timestep.
    yaws: (M,). Returns (leader_gap, leader_j) for object idx, or (None, None)
    if nothing qualifies as a leader. "Ahead" = positive projection along the
    agent's own heading; "same lane" = lateral offset within corridor_half_width."""
    x, y = positions[idx]
    yaw = yaws[idx]
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    best_gap, best_j = None, None
    for j in range(positions.shape[0]):
        if j == idx:
            continue
        rel = positions[j] - positions[idx]
        along = rel @ fwd
        lateral = rel @ left
        if along > 0.5 and abs(lateral) <= corridor_half_width:
            if best_gap is None or along < best_gap:
                best_gap, best_j = along, j
    return best_gap, best_j


def rollout_other_agents(agent_states, ego_states, horizon, dt, reactive=True, params=None,
                          paths=None, s0=None):
    """agent_states: (N, 4) array of (x, y, yaw, v) for N other agents at t=0.
    ego_states: (horizon+1, 4) the ego's already-decided rollout for this
    candidate (agents react to it, per the professor's point about replanning
    invalidating a fixed logged future).
    IDM only ever controls each agent's speed; how its heading evolves
    depends on `paths` (Part B, task 1):
      paths=None (default, matches Part A exactly): every agent keeps
        constant yaw/heading (no lane-change modeling) for the whole
        rollout -- unchanged legacy behavior, and the return value is just
        the (N, horizon+1, 4) states array as before.
      paths=list of length N (AgentPath or None per agent), s0=(N,) initial
        arc length: agents with a real AgentPath entry follow that path's
        curvature instead of a frozen heading; agents whose path entry is
        None fall back to the frozen-heading behavior for that agent only.
        Returns (states, s_final) in this mode -- s_final is the (N,)
        updated arc length, needed by the caller to continue next step.
    reactive=False: agents hold constant velocity (used for on/off comparison).
    """
    N = agent_states.shape[0]
    states = np.zeros((N, horizon + 1, 4))
    states[:, 0, :] = agent_states
    s_cur = np.array(s0, dtype=float).copy() if s0 is not None else None

    def _step(i, t, accel):
        path_i = paths[i] if paths is not None else None
        if path_i is None:
            states[i, t + 1] = bicycle_free(states[i, t], dt, accel=accel)
        else:
            new_state, s_cur[i] = bicycle_free(states[i, t], dt, accel=accel, path=path_i, s=s_cur[i])
            states[i, t + 1] = new_state

    for t in range(horizon):
        if not reactive:
            for i in range(N):
                _step(i, t, accel=0.0)
            continue

        # gather all objects (agents + ego) at this timestep for leader search
        positions = np.vstack([states[:, t, :2], ego_states[t, :2]])
        yaws = np.concatenate([states[:, t, 2], [ego_states[t, 2]]])
        speeds = np.concatenate([states[:, t, 3], [ego_states[t, 3]]])

        for i in range(N):
            gap, j = _find_leader(i, positions, yaws)
            v = states[i, t, 3]
            if j is None:
                accel = idm_accel(v, IDM_DEFAULTS["v0"], gap=1e6)
            else:
                accel = idm_accel(v, speeds[j], gap=gap, params=params)
            _step(i, t, accel)

    if paths is not None:
        return states, s_cur
    return states


def bicycle_free(state, dt, accel=0.0, path=None, s=None):
    """Move a non-ego agent forward one step.
    path=None (default, matches Part A exactly): move in a straight line at
      the frozen heading stored in `state` -- no steering input. See KNOWN
      LIMITATION note above -- Part B, task 1 replaces this assumption with
      path-following when a path is given.
    path=an AgentPath, s=current arc length: IDM/accel only controls `v`;
      (x, y, yaw) are re-derived from the new arc length via path.lookup, so
      the agent's shape always matches its logged curve. Returns
      (new_state, new_s) in this mode.
      Two fallback cases (do not conflate): this function only ever handles
      "path exhausted, agent still logged as valid" (hold the last known
      heading and continue straight from the last waypoint) -- "agent's
      valid flag goes false" is a scene-timeline fact the caller (closed_loop.py)
      must handle by dropping the agent from the active set entirely, not by
      calling this function at all for that agent.
    """
    x, y, yaw, v = state
    if path is None:
        x_next = x + v * np.cos(yaw) * dt
        y_next = y + v * np.sin(yaw) * dt
        v_next = max(v + accel * dt, 0.0)
        return np.array([x_next, y_next, yaw, v_next])

    if s is None:
        raise ValueError("bicycle_free: s is required when path is given")
    v_next = max(v + accel * dt, 0.0)
    s_next = s + v * dt
    if path.exhausted(s_next):
        last_x, last_y, last_yaw = path.lookup(path.total_length)
        overrun = s_next - path.total_length
        x_next = last_x + overrun * np.cos(last_yaw)
        y_next = last_y + overrun * np.sin(last_yaw)
        return np.array([x_next, y_next, last_yaw, v_next]), s_next
    x_next, y_next, yaw_next = path.lookup(s_next)
    return np.array([x_next, y_next, yaw_next, v_next]), s_next


# ---------------------------------------------------------------------------
# Cone-constrained candidate sampling
# ---------------------------------------------------------------------------

def cone_radius(step_idx, horizon, r_min=1.0, r_max=10.0):
    """step_idx: 1..horizon (first future step to last). Linear interpolation."""
    if horizon <= 1:
        return r_max
    frac = (step_idx - 1) / (horizon - 1)
    return r_min + (r_max - r_min) * frac


def sample_candidate_controls(state0, ref_controls, dt, n_candidates=32,
                               accel_std=1.0, steer_std=0.15,
                               r_min=1.0, r_max=10.0, max_draws_per_slot=12):
    """Returns (n_candidates, horizon, 2) control sequences. Rejection
    sampling, not shrinking: draw fresh noise, roll it out, keep it only if
    every resulting position stays within the cone around the reference
    trajectory at that timestep; otherwise redraw. This preserves the actual
    sampled distribution instead of squashing rejected draws toward the
    reference. candidates[0] is always the raw reference itself (trivially
    valid -- zero distance from itself at every step), giving MPPI-style
    methods a safe nominal to fall back on. If a slot exhausts its draw
    budget without finding a valid sample, it reuses the reference for that
    slot rather than returning a distorted one."""
    horizon = ref_controls.shape[0]
    ref_states = rollout_bicycle(state0, ref_controls, dt)
    radii = np.array([cone_radius(t, horizon, r_min, r_max) for t in range(1, horizon + 1)])

    candidates = np.zeros((n_candidates, horizon, 2))
    candidates[0] = ref_controls
    for k in range(1, n_candidates):
        found = False
        for _ in range(max_draws_per_slot):
            noise = np.stack([
                np.random.normal(0, accel_std, horizon),
                np.random.normal(0, steer_std, horizon),
            ], axis=-1)
            cand_controls = ref_controls + noise
            cand_states = rollout_bicycle(state0, cand_controls, dt)
            dists = np.hypot(cand_states[1:, 0] - ref_states[1:, 0],
                              cand_states[1:, 1] - ref_states[1:, 1])
            if np.all(dists <= radii):
                candidates[k] = cand_controls
                found = True
                break
        if not found:
            candidates[k] = ref_controls
    return candidates


# ---------------------------------------------------------------------------
# Cost -- Waymax-metric-style categories (collision / progress / comfort),
# deliberately NOT an L2-to-reference term.
# ---------------------------------------------------------------------------

def compute_cost(ego_states, other_agent_rollouts, dt,
                  collision_radius=2.5, w_collision=50.0, w_progress=1.0, w_comfort=0.05):
    """ego_states: (horizon+1, 4). other_agent_rollouts: (N, horizon+1, 4)."""
    horizon = ego_states.shape[0] - 1

    # collision: soft penalty once inside collision_radius, scaled by how deep
    collision_cost = 0.0
    for t in range(1, horizon + 1):
        ex, ey = ego_states[t, :2]
        for i in range(other_agent_rollouts.shape[0]):
            ax, ay = other_agent_rollouts[i, t, :2]
            d = np.hypot(ex - ax, ey - ay)
            if d < collision_radius:
                collision_cost += (collision_radius - d) ** 2

    # progress: reward forward travel -> cost is negative net displacement
    # KNOWN LIMITATION: this is straight-line net displacement in ANY
    # direction, not projected onto the ego's original heading -- a
    # candidate that drifts sideways scores the same "progress" as one that
    # goes straight ahead. Not addressed in this task; flagged for awareness.
    progress = np.hypot(ego_states[-1, 0] - ego_states[0, 0],
                         ego_states[-1, 1] - ego_states[0, 1])
    progress_cost = -progress

    # comfort: penalize speed changes and heading-rate changes step to step
    # KNOWN LIMITATION: dyaw is not wrapped to [-pi, pi] -- only matters for
    # near-180-degree turns within one horizon, unlikely at these
    # timescales/speeds, but a latent edge case. Not addressed here.
    dv = np.diff(ego_states[:, 3])
    dyaw = np.diff(ego_states[:, 2])
    comfort_cost = np.sum(dv ** 2) + np.sum(dyaw ** 2)

    return (w_collision * collision_cost
            + w_progress * progress_cost
            + w_comfort * comfort_cost)


# ---------------------------------------------------------------------------
# One full MPC step
# ---------------------------------------------------------------------------

def run_mpc_step(ego_state0, ref_trajectory_xy, other_agent_states, dt,
                  n_candidates=32, r_min=1.0, r_max=10.0, reactive=True,
                  lambda_temp=5.0, cost_kwargs=None):
    """ego_state0: (4,) real ego state (x, y, yaw, v) in world frame.
    ref_trajectory_xy: (horizon, 2) encoder output, ego-local frame (yaw=0).
    NOTE: caller must rotate/translate ref_trajectory_xy into the same frame
    as ego_state0 before calling this (world frame), OR call with
    ego_state0=(0,0,0,v) if operating purely in ego-local frame for this step.
    other_agent_states: (N, 4) in the same frame as ego_state0.

    Selection: MPPI-style soft blend (exp(-cost/lambda_temp) weighted average
    of all candidates' controls) vs. plain argmin over individual candidates
    -- whichever scores lower under the real cost function wins. This gets
    MPPI's smoothing when it doesn't hurt, and falls back to a single
    physically-consistent candidate when blending across different maneuvers
    (e.g. pass-left vs pass-right) would produce something worse than either.

    KNOWN LIMITATION: no warm-starting between real steps -- each call
    re-infers the reference and samples fresh candidates from scratch, with
    nothing carried over from the previous real step's chosen sequence. Not
    addressed in this task.

    Returns dict: best_controls (horizon,2), best_states (horizon+1,4),
    first_control (2,) -- the only one actually executed before replanning,
    selection_mode -- "mppi_blend" or "argmin_fallback", for visibility.
    """
    v0 = ego_state0[3]
    ref_controls = infer_reference_controls(ref_trajectory_xy, v0, dt)
    horizon = ref_controls.shape[0]

    candidates = sample_candidate_controls(
        ego_state0, ref_controls, dt, n_candidates=n_candidates,
        r_min=r_min, r_max=r_max)

    costs = np.zeros(candidates.shape[0])
    rollouts = []
    for k in range(candidates.shape[0]):
        ego_states = rollout_bicycle(ego_state0, candidates[k], dt)
        other_rollouts = rollout_other_agents(
            other_agent_states, ego_states, horizon, dt, reactive=reactive)
        costs[k] = compute_cost(ego_states, other_rollouts, dt, **(cost_kwargs or {}))
        rollouts.append(ego_states)

    best_idx = int(np.argmin(costs))
    best_individual_controls = candidates[best_idx]
    best_individual_states = rollouts[best_idx]
    best_individual_cost = costs[best_idx]

    weights = np.exp(-(costs - costs.min()) / lambda_temp)
    weights /= weights.sum()
    blended_controls = np.tensordot(weights, candidates, axes=(0, 0))
    blended_states = rollout_bicycle(ego_state0, blended_controls, dt)
    blended_other = rollout_other_agents(
        other_agent_states, blended_states, horizon, dt, reactive=reactive)
    blended_cost = compute_cost(blended_states, blended_other, dt, **(cost_kwargs or {}))

    if blended_cost <= best_individual_cost:
        chosen_controls, chosen_states, chosen_cost = blended_controls, blended_states, blended_cost
        mode = "mppi_blend"
    else:
        chosen_controls, chosen_states, chosen_cost = (
            best_individual_controls, best_individual_states, best_individual_cost)
        mode = "argmin_fallback"

    return {
        "best_controls": chosen_controls,
        "best_states": chosen_states,
        "first_control": chosen_controls[0],
        "best_cost": chosen_cost,
        "selection_mode": mode,
    }
