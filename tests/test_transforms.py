"""
Part B, task 2 -- regression test for the time-matched (tau) convention vs
the old decision-instant (t) anchoring.

Position differences between consecutive window entries cancel out whichever
single anchor you pick, so position/yaw alone wouldn't have caught the
original bug. Velocity is where it actually breaks: v_rel(tau) = agent_v(tau)
- ego_v(t) mixes two different instants' velocities whenever the ego's own
heading/speed changes noticeably within the window -- exactly what a sharp
turn produces, and exactly the regime (turns) this project has a known,
unresolved problem in (see CLAUDE.md's out-of-scope note).
"""
import types
import numpy as np

from src.transforms import transform_positions, transform_velocities, transform_yaw
from src.extract import extract_agents


def test_windowed_vs_single_reference_differ_under_ego_turn():
    hist_len = 5
    # ego makes a sharp turn AND changes speed within the window (yaw ramps
    # 0 -> pi/2, speed ramps 15 -> 5 m/s, e.g. braking into a turn). Note:
    # a turn at CONSTANT speed alone wouldn't expose this bug for a
    # stationary reference agent -- rotating the ego's own forward-aligned
    # velocity into its own current heading trivially gives (speed, 0)
    # regardless of which instant's heading you pick. It's specifically the
    # ego's speed CHANGING across the window that a shared single anchor
    # cannot capture.
    ego_yaw_window = np.linspace(0.0, np.pi / 2, hist_len)
    ego_speed_window = np.linspace(15.0, 5.0, hist_len)
    ego_vxvy_window = np.stack(
        [ego_speed_window * np.cos(ego_yaw_window), ego_speed_window * np.sin(ego_yaw_window)], axis=-1)
    # ego position isn't used by transform_velocities; give it an arbitrary track
    ego_xy_window = np.stack([np.arange(hist_len) * 1.0, np.zeros(hist_len)], axis=-1)

    # a single stationary reference agent, same world state at every tau
    agent_xy_window = np.tile(np.array([50.0, 50.0]), (hist_len, 1))
    agent_vxvy_window = np.zeros((hist_len, 2))
    agent_yaw_window = np.zeros(hist_len)

    # new (tau): each entry transformed by the ego's OWN state at that instant
    v_rel_tau = transform_velocities(agent_vxvy_window, ego_vxvy_window, ego_yaw_window)
    pos_local_tau = transform_positions(agent_xy_window, ego_xy_window, ego_yaw_window)
    yaw_local_tau = transform_yaw(agent_yaw_window, ego_yaw_window)

    # old (t): every entry transformed by the ego's state at the CURRENT
    # instant only -- exactly what calling with a single (non-windowed)
    # reference reproduces, since that's the legacy call signature.
    v_rel_t = transform_velocities(agent_vxvy_window, ego_vxvy_window[-1], ego_yaw_window[-1])
    pos_local_t = transform_positions(agent_xy_window, ego_xy_window[-1], ego_yaw_window[-1])
    yaw_local_t = transform_yaw(agent_yaw_window, ego_yaw_window[-1])

    v_diff = np.abs(v_rel_tau - v_rel_t).max()
    pos_diff = np.abs(pos_local_tau - pos_local_t).max()
    yaw_diff = np.abs(yaw_local_tau - yaw_local_t).max()
    print(f"tau vs t under a sharp ego turn: max |v_rel diff|={v_diff:.2f} m/s, "
          f"max |pos_local diff|={pos_diff:.2f} m, max |yaw_local diff|={yaw_diff:.3f} rad")

    # velocity is the one this task exists to fix: for a stationary agent,
    # v_rel(t) should equal -ego_v(t) at every window entry (since agent_v=0
    # everywhere and it always subtracts the SAME ego_v(t)); v_rel(tau)
    # instead varies across the window because ego_v(tau) itself varies.
    # The two must differ substantially given how much the ego turned.
    assert v_diff > 5.0, (
        f"expected tau-anchored and t-anchored v_rel to differ substantially "
        f"under a sharp ego turn, got max diff {v_diff:.2f} m/s")
    # sanity: under the OLD (t-anchored) convention, v_rel is the same
    # constant vector (-ego_v(t)) at every tau, since the same anchor is
    # subtracted everywhere.
    assert np.allclose(v_rel_t, v_rel_t[0], atol=1e-9), (
        "old t-anchored v_rel should be identical across the whole window "
        "(same anchor subtracted at every tau) -- if this fails, the test's "
        "own legacy-equivalent computation is wrong, not transforms.py")


def make_turning_scenario(hist_len=5, num_timesteps=20):
    """ego turns sharply (yaw 0 -> pi/2) over [t-hist_len+1, t]; one stationary
    agent sits off to the side the whole time."""
    num_objects = 2
    ego_idx = 0
    t = hist_len + 2  # decision instant, comfortably past hist_len-1

    x = np.zeros((num_objects, num_timesteps))
    y = np.zeros((num_objects, num_timesteps))
    yaw = np.zeros((num_objects, num_timesteps))
    vel_x = np.zeros((num_objects, num_timesteps))
    vel_y = np.zeros((num_objects, num_timesteps))
    valid = np.ones((num_objects, num_timesteps), dtype=bool)

    t0 = t - hist_len + 1
    ego_yaw_window = np.linspace(0.0, np.pi / 2, hist_len)
    ego_speed_window = np.linspace(15.0, 5.0, hist_len)  # braking into the turn
    yaw[ego_idx, t0:t + 1] = ego_yaw_window
    vel_x[ego_idx, t0:t + 1] = ego_speed_window * np.cos(ego_yaw_window)
    vel_y[ego_idx, t0:t + 1] = ego_speed_window * np.sin(ego_yaw_window)
    # a simple consistent-enough position track (not used by the velocity
    # assertion below, but keeps the scenario internally plausible)
    for i, tt in enumerate(range(t0, t + 1)):
        x[ego_idx, tt] = i * 1.0

    # stationary agent, off to the side, valid for the whole window + margin
    x[1, :] = 50.0
    y[1, :] = 50.0

    log_trajectory = types.SimpleNamespace(x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, valid=valid)
    object_metadata = types.SimpleNamespace(object_types=np.array([1, 1]))
    scenario = types.SimpleNamespace(log_trajectory=log_trajectory, object_metadata=object_metadata)
    return scenario, ego_idx, t


def test_extract_agents_uses_time_matched_velocity():
    """Integration-level check: extract_agents' output for the stationary
    agent must vary across the history window under a sharp ego turn (tau),
    not stay frozen at one constant value (t) -- this is the regression the
    old code would have produced."""
    hist_len = 5
    scenario, ego_idx, t = make_turning_scenario(hist_len=hist_len)
    out = extract_agents(scenario, ego_idx, t, hist_len=hist_len)
    assert out is not None

    # feats: (MAX_AGENTS, hist_len, 6) = [x, y, cos_yaw, sin_yaw, vx_rel, vy_rel]
    agent_vxvy_rel = out["hist"][0, :, 4:6]  # our stationary agent is object idx 1 -> first non-ego slot
    per_step_diff = np.abs(np.diff(agent_vxvy_rel, axis=0)).max()
    print(f"extract_agents: max step-to-step change in v_rel across the window = {per_step_diff:.2f} m/s")
    assert per_step_diff > 1.0, (
        "expected a stationary agent's relative velocity to vary across the "
        "window under a sharp ego turn (time-matched/tau convention) -- a "
        "constant value across the window would mean the old (t-anchored) "
        "bug is still present")


if __name__ == "__main__":
    test_windowed_vs_single_reference_differ_under_ego_turn()
    test_extract_agents_uses_time_matched_velocity()
    print("\nall transform/time-matching tests passed")
