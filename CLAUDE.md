# CLAUDE.md — Waymax MPC Phase 2

## Status (as of 2026-08-29)

This file was the working spec for the session that implemented everything
below; the original spec (unmodified) follows this section as a record of
what was asked. Current state:

- **Part A (foundation)** -- ✅ done, commit `b04c44e`. All five files
  created verbatim, both test modules pass.
- **Task 1 (path-following IDM agents)** -- ✅ done, commit `2ba3965`.
  `AgentPath` arc-length cache + optional `path=None` param on
  `bicycle_free`/`rollout_other_agents` (default reproduces Part A's
  behavior exactly, so its tests still pass byte-for-byte unmodified).
  Whole-scene scan + `ActiveAgentSet` for entry/exit, both fallback cases,
  4 new tests, all passing.
- **Task 2 (time-matched/tau features)** -- ✅ done, commit `d1cb311`.
  Windowed `transform_positions`/`transform_velocities`/`transform_yaw`,
  updated `extract.py` + `closed_loop.py`, regression test showing tau vs t
  diverge under a synthetic sharp turn+brake. Confirmed this invalidates
  `stage1_weights.pth`/`best_weights_v2.pth` as noted below.
- **Retrain** -- ✅ done on Colab (`Av2_retrain_task2.ipynb`, adapted from
  the original training notebook to clone the repo instead of rewriting
  `src/*.py` by hand). 10 epochs, `weights/stage1_weights_tau.pth` pushed in
  commit `aca67c5`. Results: test loss=1.07, ADE=0.82m, FDE=2.07m (3s
  horizon). Val loss plateaus ~epoch 4-8 and ticks back up slightly by
  epoch 9 -- mild overfit/plateau signal, not obviously "needs more data."
  The model shows a general bias toward continuing the ego's current motion
  rather than committing to a maneuver change -- broader than the
  turn-undershoot issue named below as out of scope (visible on turns in
  the snapshots, but not turn-specific); root cause not yet isolated.
- **Closed-loop, real scenario** -- ⏳ pending. Code is ready and compatible
  with `stage1_weights_tau.pth` with no changes (Task 2 only changed
  feature values, not the model's input dimensionality), synthetic tests
  pass, but it hasn't yet been run end-to-end against a real Waymax
  scenario. Being checked next.
- **Candidate next steps, not started, in rough priority order once
  closed-loop is confirmed working:**
  1. Cheap ablation test -- mask the agent/map/light branches at inference
     and compare predictions, to check whether the model is meaningfully
     using scene context at all vs. substantially just extrapolating its
     own current speed/heading. Would clarify whether the inertia bias
     above is a context-usage problem or a commitment-under-ambiguity
     problem.
  2. If context IS being used but under-weighted: oversample and/or
     loss-reweight maneuver-heavy scenarios (large heading/speed change
     within the prediction window) rather than uniform scenario sampling.
  3. Agent-agent interaction modeling in the encoder -- explicitly deferred
     until the closed-loop planner is verified working end to end, since
     closed-loop IDM-driven agents already provide a form of interaction at
     the planning layer.

Original spec follows, unmodified.

---

Read this whole file before touching anything. It has two layers:

1. **Foundation** (Part A) — code that was already written and unit-tested
   (against synthetic data, not real Waymax) in a prior session. Create these
   files with the exact content given. Don't "improve" them without asking —
   if something looks off, flag it in a comment or ask, don't silently change
   behavior.
2. **Pending work** (Part B) — two specific changes, agreed on in design
   discussion but not yet implemented. These have precise specs, not code.
   Implement them, write tests, and check with the repo owner before
   committing if anything in the spec doesn't resolve cleanly against the
   real code.

**Explicitly out of scope for this pass** — do not touch these:
- Fixing the Stage 1 model's turn-undershooting (known, ~14% of scenarios,
  unresolved, tracked separately).
- Any hardware/deployment migration.
- The closed-loop visualization (extending `plot_world_frame_snapshots` to
  show the IDM-driven rollout) — not built yet, not part of this task.

---

## Part A — Foundation (create exactly as given)

### Repo state before this task

Already on `main`, already trained/working, do not modify:
`src/transforms.py`, `src/extract.py`, `src/model.py`, `src/dataset.py`,
`src/train.py`, `src/visualize.py`, `src/metrics.py`, two checkpoints under
`weights/`.

**Not yet in the repo** — create these new files:
`src/dynamics.py`, `src/mpc.py`, `src/closed_loop.py`,
`tests/test_mpc.py`, `tests/test_closed_loop.py`.

None of the new files import or modify `extract.py`. `closed_loop.py`
reimplements a small parallel padding helper and its own copies of
`MAX_AGENTS`/`MAX_MAP_POINTS`/`MAX_LIGHTS` instead of importing them from
`extract.py`, because `extract.py`'s functions hard-code reading the ego pose
from the log — `closed_loop.py` needs to override that with a simulated
pose. This is a known duplication risk: if `extract.py`'s constants or
feature layout ever change, `closed_loop.py`'s copy won't know. Flagged, not
fixed here — a shared-implementation refactor is a separate task.

### `src/dynamics.py`

```python
"""
Kinematic bicycle model. Exact, known physics for the ego -- no learning here.

rollout_bicycle: controls -> states, used both to score MPC candidates and to
actually execute the chosen action.

infer_reference_controls: takes the encoder's predicted trajectory (positions
only, in the ego-local frame, yaw=0 by convention) and works backward to the
(accel, steer) sequence that would approximately produce it. This becomes the
MPC's reference -- controls are frame-agnostic, so this same sequence can be
replayed starting from the ego's real world pose and produces a correctly
placed/oriented real-world reference trajectory.
"""
import numpy as np

WHEELBASE = 2.7  # meters, reasonable passenger car default


def bicycle_step(state, control, dt):
    """state: (x, y, yaw, v). control: (accel, steer). Returns next state."""
    x, y, yaw, v = state
    accel, steer = control
    x_next = x + v * np.cos(yaw) * dt
    y_next = y + v * np.sin(yaw) * dt
    yaw_next = yaw + (v / WHEELBASE) * np.tan(steer) * dt
    v_next = max(v + accel * dt, 0.0)  # no reversing in this simple version
    return np.array([x_next, y_next, yaw_next, v_next])


def rollout_bicycle(state0, controls, dt):
    """controls: (horizon, 2) array of (accel, steer).
    Returns (horizon+1, 4) states, including state0 as the first row."""
    states = [np.array(state0, dtype=float)]
    for c in controls:
        states.append(bicycle_step(states[-1], c, dt))
    return np.stack(states)


def infer_reference_controls(trajectory_xy, v0, dt):
    """trajectory_xy: (horizon, 2), ego-local frame at the current decision
    time (yaw=0 by definition of that frame). Returns (horizon, 2) controls
    (accel, steer) -- an approximate inverse, good enough as a starting
    reference for the optimizer to search around, not an exact fit.

    Verified against a self-consistent ground truth (reference generated by
    forward-simulating known controls, then recovered): the earlier version
    of this function had a real bug, a missing dt term in the steer formula's
    denominator, which produced steer values roughly (1/dt)x too small --
    fine on straight segments (steer near zero regardless) but badly wrong
    on turns, producing large compounding position error. Fixed below."""
    horizon = trajectory_xy.shape[0]
    pts = np.vstack([[0.0, 0.0], trajectory_xy])
    yaw, v = 0.0, v0
    controls = []
    for i in range(horizon):
        dx, dy = pts[i + 1] - pts[i]
        step_dist = np.hypot(dx, dy)
        heading = np.arctan2(dy, dx) if step_dist > 1e-3 else yaw
        v_next = step_dist / dt
        accel = (v_next - v) / dt
        yaw_rate = ((heading - yaw) + np.pi) % (2 * np.pi) - np.pi
        steer = np.arctan(yaw_rate * WHEELBASE / max(v * dt, 1e-3))
        controls.append([accel, steer])
        yaw, v = heading, v_next
    return np.array(controls)
```

### `src/mpc.py`

```python
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


def rollout_other_agents(agent_states, ego_states, horizon, dt, reactive=True, params=None):
    """agent_states: (N, 4) array of (x, y, yaw, v) for N other agents at t=0.
    ego_states: (horizon+1, 4) the ego's already-decided rollout for this
    candidate (agents react to it, per the professor's point about replanning
    invalidating a fixed logged future).
    Each agent keeps constant yaw/heading (no lane-change modeling) and only
    IDM-controls its speed along that heading.
    reactive=False: agents hold constant velocity (used for on/off comparison).
    Returns (N, horizon+1, 4) states including t=0.
    """
    N = agent_states.shape[0]
    states = np.zeros((N, horizon + 1, 4))
    states[:, 0, :] = agent_states

    for t in range(horizon):
        if not reactive:
            for i in range(N):
                states[i, t + 1] = bicycle_free(states[i, t], dt)
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
            states[i, t + 1] = bicycle_free(states[i, t], dt, accel=accel)

    return states


def bicycle_free(state, dt, accel=0.0):
    """Move a non-ego agent forward along its own constant heading (no
    steering input). See KNOWN LIMITATION note above -- Part B, task 1
    replaces the constant-heading assumption with path-following."""
    x, y, yaw, v = state
    x_next = x + v * np.cos(yaw) * dt
    y_next = y + v * np.sin(yaw) * dt
    v_next = max(v + accel * dt, 0.0)
    return np.array([x_next, y_next, yaw, v_next])


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
```

### `tests/test_mpc.py`

```python
import numpy as np
from src.mpc import (
    idm_accel, cone_radius, sample_candidate_controls,
    rollout_other_agents, run_mpc_step,
)
from src.dynamics import rollout_bicycle


def test_idm_braking():
    # follower closing on a much slower leader at short gap -> should brake hard
    a = idm_accel(v=20.0, v_lead=5.0, gap=8.0)
    assert a < 0, f"expected braking, got accel={a}"
    # follower well below desired speed, huge gap, no leader effect -> should accelerate
    a2 = idm_accel(v=5.0, v_lead=15.0, gap=1e6)
    assert a2 > 0, f"expected acceleration in free flow, got accel={a2}"
    print(f"idm braking ok: close-gap accel={a:.2f}, free-flow accel={a2:.2f}")


def test_cone_constraint():
    state0 = np.array([0.0, 0.0, 0.0, 10.0])
    horizon = 30
    dt = 0.1
    # straight-line reference, mild turn partway
    ref_xy = np.zeros((horizon, 2))
    for t in range(horizon):
        ref_xy[t] = [1.0 * (t + 1), 0.3 * max(0, t - 15)]
    from src.dynamics import infer_reference_controls
    ref_controls = infer_reference_controls(ref_xy, state0[3], dt)
    ref_states = rollout_bicycle(state0, ref_controls, dt)

    candidates = sample_candidate_controls(state0, ref_controls, dt, n_candidates=50,
                                            r_min=1.0, r_max=10.0)
    max_violation = 0.0
    for k in range(candidates.shape[0]):
        cand_states = rollout_bicycle(state0, candidates[k], dt)
        for t in range(1, horizon + 1):
            r = cone_radius(t, horizon, 1.0, 10.0)
            d = np.hypot(cand_states[t, 0] - ref_states[t, 0],
                         cand_states[t, 1] - ref_states[t, 1])
            max_violation = max(max_violation, d - r)
    assert max_violation <= 1e-6, f"cone constraint violated by {max_violation:.4f}m"
    print(f"cone constraint ok: max signed slack (should be <=0) = {max_violation:.4f}")


def test_reactivity_on_off():
    # professor's literal example: car behind should brake when ego (ahead) slows,
    # only if reactive=True
    dt = 0.1
    horizon = 20
    other = np.array([[-10.0, 0.0, 0.0, 15.0]])  # one agent behind ego, same heading
    # ego decelerates hard and stays slow
    ego_states = np.zeros((horizon + 1, 4))
    ego_states[:, 3] = np.maximum(15.0 - 3.0 * np.arange(horizon + 1) * dt, 2.0)
    for t in range(1, horizon + 1):
        ego_states[t, 0] = ego_states[t - 1, 0] + ego_states[t - 1, 3] * dt

    reactive_out = rollout_other_agents(other, ego_states, horizon, dt, reactive=True)
    non_reactive_out = rollout_other_agents(other, ego_states, horizon, dt, reactive=False)

    v_final_reactive = reactive_out[0, -1, 3]
    v_final_free = non_reactive_out[0, -1, 3]
    assert v_final_reactive < v_final_free, (
        f"expected reactive agent to slow more than non-reactive: "
        f"reactive={v_final_reactive:.2f}, free={v_final_free:.2f}")
    print(f"reactivity on/off ok: reactive final v={v_final_reactive:.2f}, "
          f"non-reactive final v={v_final_free:.2f}")


def test_obstacle_avoidance_sanity():
    np.random.seed(0)
    dt = 0.1
    horizon = 25
    state0 = np.array([0.0, 0.0, 0.0, 10.0])
    ref_xy = np.stack([np.linspace(1, 25, horizon), np.zeros(horizon)], axis=-1)
    # static obstacle directly on the reference path partway ahead
    obstacle = np.array([[12.0, 0.0, 0.0, 0.0]])
    collision_radius = 2.5

    # the raw reference itself passes through the obstacle (sanity check on the setup)
    from src.dynamics import infer_reference_controls
    ref_controls = infer_reference_controls(ref_xy, state0[3], dt)
    ref_states = rollout_bicycle(state0, ref_controls, dt)
    ref_min_dist = np.min(np.hypot(ref_states[1:, 0] - obstacle[0, 0],
                                    ref_states[1:, 1] - obstacle[0, 1]))
    assert ref_min_dist < collision_radius, "test setup bug: reference should clip the obstacle"

    out = run_mpc_step(state0, ref_xy, obstacle, dt, n_candidates=200, reactive=False)
    best_min_dist = np.min(np.hypot(out["best_states"][1:, 0] - obstacle[0, 0],
                                     out["best_states"][1:, 1] - obstacle[0, 1]))
    print(f"obstacle avoidance: reference min dist to obstacle={ref_min_dist:.2f}m, "
          f"chosen path min dist={best_min_dist:.2f}m (collision_radius={collision_radius}), "
          f"selection_mode={out['selection_mode']}")
    assert best_min_dist > ref_min_dist, (
        "expected the planner to choose a path further from the obstacle than the raw reference")


if __name__ == "__main__":
    test_idm_braking()
    test_cone_constraint()
    test_reactivity_on_off()
    test_obstacle_avoidance_sanity()
    print("\nall mpc tests passed")
```

Expected output when run (`python3 -m tests.test_mpc` from repo root, with an
`__init__.py` in `tests/`): all four tests pass. The obstacle-avoidance test
should print `selection_mode=argmin_fallback` — this is correct and expected,
not a bug: with a single obstacle directly ahead, blending across candidates
that avoid it from different directions is exactly the failure mode the
argmin fallback exists to catch, so seeing it trigger here is confirmation
the safety logic is working, not a sign something's wrong.

### `src/closed_loop.py`

```python
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
```

### `tests/test_closed_loop.py`

```python
"""
Sanity test for closed_loop.py against a synthetic fake scenario -- checks
shapes, frame transforms, and buffer updates run without crashing, using a
random-initialized model (pipeline mechanics only, not prediction quality,
same spirit as the original dynamics/mpc tests)."""
import types
import numpy as np
import torch

from src.model import Stage1Model
from src.closed_loop import run_closed_loop


def make_fake_scenario(num_timesteps=60, hist_len=10):
    num_objects = 3  # ego + 2 agents
    ego_idx = 0

    x = np.zeros((num_objects, num_timesteps))
    y = np.zeros((num_objects, num_timesteps))
    yaw = np.zeros((num_objects, num_timesteps))
    vel_x = np.zeros((num_objects, num_timesteps))
    vel_y = np.zeros((num_objects, num_timesteps))
    valid = np.ones((num_objects, num_timesteps), dtype=bool)

    # ego and agent 1 drive straight along +x at different speeds/lanes,
    # agent 2 is ahead of agent 1 in the same lane (leader for IDM check)
    dt = 0.1
    speeds = {0: 10.0, 1: 8.0, 2: 8.0}
    lanes_y = {0: 0.0, 1: -3.5, 2: -3.5}
    starts_x = {0: 0.0, 1: -5.0, 2: 15.0}
    for i in range(num_objects):
        for t in range(num_timesteps):
            x[i, t] = starts_x[i] + speeds[i] * dt * t
            y[i, t] = lanes_y[i]
            vel_x[i, t] = speeds[i]
            vel_y[i, t] = 0.0
            yaw[i, t] = 0.0

    log_trajectory = types.SimpleNamespace(x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, valid=valid)
    object_metadata = types.SimpleNamespace(object_types=np.array([1, 1, 1]))

    # a simple straight two-lane road, points every 2m for 100m
    rg_x = np.arange(0, 100, 2.0)
    roadgraph_points = types.SimpleNamespace(
        x=np.concatenate([rg_x, rg_x]),
        y=np.concatenate([np.zeros_like(rg_x) + 1.75, np.zeros_like(rg_x) - 1.75]),
        types=np.zeros(len(rg_x) * 2, dtype=np.int64),
        valid=np.ones(len(rg_x) * 2, dtype=bool),
    )

    # one traffic light, always green (state=1), sitting ahead at x=50
    num_lights = 1
    tl_x = np.full((num_lights, num_timesteps), 50.0)
    tl_y = np.full((num_lights, num_timesteps), 0.0)
    tl_state = np.ones((num_lights, num_timesteps), dtype=np.int64)
    tl_valid = np.ones((num_lights, num_timesteps), dtype=bool)
    log_traffic_light = types.SimpleNamespace(x=tl_x, y=tl_y, state=tl_state, valid=tl_valid)

    scenario = types.SimpleNamespace(
        log_trajectory=log_trajectory,
        object_metadata=object_metadata,
        roadgraph_points=roadgraph_points,
        log_traffic_light=log_traffic_light,
    )
    return scenario, ego_idx


def test_closed_loop_runs():
    scenario, ego_idx = make_fake_scenario()
    model = Stage1Model(hidden_dim=64, future_len=30)  # random init, mechanics only
    log = run_closed_loop(scenario, model, ego_idx, t0=10, max_steps=8, dt=0.1)

    n_steps = len(log["ego_states"])
    assert n_steps == 9, f"expected 8 steps + initial state = 9, got {n_steps}"
    for s in log["ego_states"]:
        assert s.shape == (4,)
    for a in log["agent_states"]:
        assert a.shape[-1] == 4

    ego_path = np.stack(log["ego_states"])
    total_dx = ego_path[-1, 0] - ego_path[0, 0]
    print(f"closed loop ran {n_steps - 1} steps, ego moved {total_dx:.2f}m in x, "
          f"final speed={ego_path[-1, 3]:.2f} m/s")
    print(f"selection modes used: {set(log['selection_modes'])}")
    assert total_dx > 0, "ego should have made forward progress, not stayed put or gone backward"

    ref = log["ref_trajectories"][0]
    assert ref.shape == (30, 2)
    print("all shapes ok, closed loop wiring sanity check passed")


if __name__ == "__main__":
    test_closed_loop_runs()
```

**Note**: this test uses a random-initialized `Stage1Model`, not the trained
checkpoint — it's checking that the wiring runs and produces correctly-shaped
output, not that the driving behavior is sensible. Ego forward progress will
be small/erratic since the "reference trajectory" being followed is noise.
That's expected and not a bug.

### Setup / verification steps for Part A

1. Create the five files above verbatim.
2. `touch tests/__init__.py` if not already present.
3. `pip install torch numpy --break-system-packages` (or however this repo's
   env is normally set up).
4. Run `python3 -m tests.test_mpc` from repo root — expect "all mpc tests
   passed", four print lines as described above.
5. Run `python3 -m tests.test_closed_loop` from repo root — expect "all
   shapes ok, closed loop wiring sanity check passed".
6. Do not attempt to run `closed_loop.py` against a real Waymax scenario as
   part of this setup step — that requires Waymax/JAX/GCS access this
   environment may not have, and is the repo owner's job to verify
   separately. If Waymax happens to be installed and available, a real-data
   smoke test is welcome, but isn't required to consider Part A done.

---

## Part B — Pending work (implement from spec, no existing code)

### Task 1 — IDM agents follow their logged path, not a frozen heading

**Problem.** `bicycle_free` in `mpc.py` currently holds `yaw` constant for
the entire rollout of a non-ego agent — it only ever moves in a straight
line at whatever heading it had the instant we started simulating it. If the
agent's real logged path curves (a bend, a turn at an intersection), the
simulated version silently diverges into a straight line instead of
following that curve. IDM should only ever control *how fast* an agent moves
— the *shape* of its path should always match the log.

**Design — path/velocity decomposition.**

1. **At scenario-load time** (not per-agent-entry — scan the whole scene
   once): for every object index in `scenario.log_trajectory` (not just ones
   valid at `t0`), find its valid span — first and last valid timestep. This
   repo assumes each object has one contiguous valid span (appears once,
   drives, disappears once); if you find evidence of a gappy validity mask
   in real data, stop and flag it rather than silently handling it, since
   the plan below doesn't cover that case.
2. **Cache each object's logged `(x, y)` waypoints** across its own valid
   span, once, at load time. Convert to an arc-length parameterization:
   cumulative distance traveled along consecutive waypoints, starting at 0
   at the object's first valid timestep. Store enough to look up, for any
   arc-length `s`, an interpolated `(x, y, yaw)` — yaw as the local tangent
   direction between the two nearest waypoints.
3. **Change agent state representation** from `(x, y, yaw, v)` to `(s, v)` +
   a reference to its cached path. `(x, y, yaw)` become values *derived* by
   looking up `s` in the cached path (interpolate between the two nearest
   waypoints when `s` falls between them), not stored directly.
4. **Each real simulation step**: leader-finding (`_find_leader`) runs
   exactly as it does now — nearest agent/ego ahead within the corridor —
   except it now uses each agent's derived (curving) heading instead of a
   frozen one, which should make leader detection more accurate for free
   during turns. Compute `idm_accel`, update `v`. Advance `s += v * dt`.
   Re-derive `(x, y, yaw)` from the new `s`.
5. **Ego is unaffected** — this change is scoped to `bicycle_free` /
   `rollout_other_agents` and the closed-loop's real-agent-stepping call
   only. The ego's own motion stays fully MPC-chosen, never path-following.

**Two distinct fallback cases — do not conflate them:**
- **Path exhausted, agent still logged as valid** (our IDM-driven agent
  outpaced its own logged self and ran past the end of its cached
  waypoints): hold the last known heading and continue in a straight line
  from the last waypoint. This is "ran out of known future shape," not "the
  agent left the scene."
- **Agent's `valid` flag goes false at the current real timestep** (per the
  log, this object has genuinely left the scene): drop the agent from the
  active simulation set entirely. Do not continue extrapolating it.

**New agents entering mid-scene.** Since the whole scene is scanned once at
load time, this is now well-defined: at each real step `t`, check which
object indices have a first-valid-timestep equal to the current real `t` and
aren't yet in the active agent set. Initialize them the same way agents at
`t0` are initialized today (`s=0` at their first waypoint, `v` from the
log's `vel_x`/`vel_y` at that instant). This works even though the ego has
diverged from the log, because *whether object i becomes valid at real time
t* is a fact about the log's own timeline, untouched by anything the ego or
other agents do.

**Files touched:** `src/mpc.py` (`bicycle_free`, `rollout_other_agents`, and
the state representation used throughout), `src/closed_loop.py`
(`init_other_agents` needs to become a whole-scene scan instead of a
`t0`-only snapshot; the real per-step agent-stepping call in
`run_closed_loop` needs to use the new path-following step function; new
logic needed for mid-scene agent entry).

**Testing.** Extend the synthetic fake scenario in `tests/test_closed_loop.py`
(or add a new synthetic test) with at least one agent whose logged path
actually curves (e.g. a quarter-circle), and assert the simulated agent's
position stays close to the logged path's shape even after several real
steps — this is the thing the current implementation would fail. Also test
both fallback cases explicitly (path-exhausted vs. valid-flag-false) with
separate synthetic setups, and test mid-scene agent entry.

---

### Task 2 — Time-matched (τ) relative features instead of decision-instant (t) anchoring

**Problem.** In `extract.py`'s `extract_agents` (and identically in
`extract_traffic_lights`), the ego's pose is read **once**, at the current
decision instant `t`, via `get_ego_frame`. That single `(ego_xy, ego_yaw,
ego_vxvy)` is then used to transform the *entire* `hist_len`-step historical
window — so an agent's position/velocity from, say, 9 steps in the past is
expressed relative to where the ego is **right now**, not relative to where
the ego actually was 9 steps ago. `closed_loop.py`'s `build_model_batch`
replicates this same convention for consistency with the trained model.

This is fine for position (a coordinate difference between two consecutive
historical entries cancels out the anchor point, so the shape of relative
motion across the window is preserved regardless of which instant you
anchor to). It is not fine for velocity: `v_rel(τ) = agent_v(τ) − ego_v(t)`
mixes two different instants' velocities, and decomposes as
`[agent_v(τ) − ego_v(τ)] + [ego_v(τ) − ego_v(t)]` — the second bracket is
pure contamination from the ego's own acceleration between τ and t, present
identically across every agent slot, carrying no information about the
specific agent being described.

**Two things this touches, worth naming separately so nothing gets missed:**
- **The rotation basis.** Which direction counts as "+x" when expressing any
  vector numerically. A free convention, not a physical claim by itself.
- **The reference value being subtracted.** The actual physical quantity
  being measured: how the agent's state compared to the ego's state *at the
  instant being described*.

**Decision, confirmed with repo owner:** both switch to `τ`. At each
historical instant, the agent's state is expressed relative to the ego's
state *at that same instant*, rotated into the ego's *own heading at that
same instant* — not a shared frame borrowed from `t`. This is the fully
time-matched / co-moving convention: every entry in the window is a
self-contained "what the ego would have perceived, standing where it stood,
facing where it faced, at that exact moment" — nothing about entry τ depends
on `t` except that the last entry (τ=t) happens to equal the present.

**Trade-off worth knowing about, not a reason to change the decision:**
today, every historical entry shares one rotation (the ego's current
heading), so consecutive entries' positions can be directly differenced to
recover true world-frame relative motion — the shared frame cancels out.
Under full time-matching, each entry is rotated by that instant's own
(different) heading, so consecutive entries no longer share a frame — a
GRU reading the sequence has to account for the frame itself changing
between steps, not just the content. This is a real modeling difference,
not a bug, and it's a standard convention in trajectory prediction (each
timestep expressed in its own co-moving frame) — just flagging it so it
isn't mistaken for an oversight if the retrained model behaves differently
from what the previous (shared-frame) version would have.

This was explicitly checked against "why not just use `t` everywhere, since
it's simpler and no extra compute" — rejected, because the error from using
`t` instead of `τ` scales with how much the ego's own heading/speed changed
within the window, which is smallest on straight roads and **largest during
turns** — exactly the one regime this project has a known, unresolved
problem in. Do not simplify this to `t`-only without checking with the repo
owner first.

**Formulas** (for a historical instant τ within the window, `R(·)` = 2D
rotation matrix, everything on the right-hand side evaluated at τ, nothing
borrowed from `t`):

```
pos_local(τ) = R(-yaw_ego(τ)) · (pos_world_agent(τ) − pos_world_ego(τ))
yaw_local(τ) = yaw_agent(τ) − yaw_ego(τ)        [wrapped to (−π, π]]
v_rel(τ)     = R(-yaw_ego(τ)) · (v_world_agent(τ) − v_world_ego(τ))
```

**Scope:**
- **Applies to:** the agent branch (position, yaw, velocity) and the light
  branch (position only — lights have no yaw/velocity feature). Both have a
  historical window.
- **Does not apply to:** the map branch — single static snapshot at `t`, no
  history, nothing to time-match.

**Files touched:**
- `src/transforms.py` — `transform_positions`, `transform_velocities`, and
  `transform_yaw` currently each take a single reference value (one `ego_xy`
  / `ego_vxvy` / `ego_yaw`) and broadcast it across a whole window; **all
  three** need to accept a **windowed** reference `(hist_len, ...)` and
  apply it timestep-by-timestep instead — this is a bigger change to this
  file than originally scoped (previously only `transform_velocities` looked
  like it needed this; now the rotation itself is windowed too, so
  `transform_positions` and `transform_yaw` need it as well). This is the
  one place this task reaches into code used by the already-trained model —
  treat it as a deliberate, separate step, not a drive-by edit alongside
  Task 1.
- `src/extract.py` — `extract_agents` and `extract_traffic_lights` need the
  ego's own state pulled for the *whole* history window (`traj.x/y/yaw/vel_x/vel_y[ego_idx,
  t0:t+1]`), not just at `t`, then passed to the updated `transform_*`
  functions.
- `src/closed_loop.py` — `build_model_batch` needs the same treatment. For
  `τ >= t0` this uses our own persisted simulated ego history (already being
  logged every step). For `τ < t0` (the very first window, before any
  simulation has started), there's no simulated ego history yet — fall back
  to the log for exactly that initial slice. This is not an approximation:
  the ego hadn't diverged from the log before `t0` anyway, so the log is
  correct there.

**Consequence — retrain required.** This changes the model's input feature
values (not the dimensionality — still the same 6-dim agent layout — just
the values going into it change, and now every window entry lives in its
own rotated frame rather than a shared one). The two existing checkpoints
(`stage1_weights.pth`, `best_weights_v2.pth`) become invalid against the new
feature definition and cannot be used as a warm start. A full retrain is
required after this change. This is independent of, and should not be
bundled with, the separate turn-undershoot investigation (explicitly out of
scope for this task, see top of file) — that looks like a decoder-capacity/
data-imbalance issue, not an input-feature issue, so fixing this doesn't
address it and shouldn't be assumed to.

**Testing.** Add a unit test with a synthetic scenario where the ego's
heading/speed changes noticeably within a single `hist_len` window (a sharp
synthetic turn), and assert that the resulting `v_rel`/`pos_local`/`yaw_local`
values for a stationary or straight-moving reference agent differ from what
the old (`t`-anchored) implementation would have produced — this is the
regression test that would have caught the original issue, and confirms the
fix actually changes behavior in the case that matters.

---

## Commit granularity

Recommended: one commit for Part A (all five files, once all tests pass),
then Task 1 and Task 2 as separate commits (they touch different files
except where noted, and Task 2's retrain requirement makes it worth being
able to isolate/revert independently of Task 1).
