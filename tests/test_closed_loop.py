"""
Sanity test for closed_loop.py against a synthetic fake scenario -- checks
shapes, frame transforms, and buffer updates run without crashing, using a
random-initialized model (pipeline mechanics only, not prediction quality,
same spirit as the original dynamics/mpc tests)."""
import types
import numpy as np
import torch

from src.model import Stage1Model
from src.closed_loop import run_closed_loop, build_agent_paths, ActiveAgentSet
from src.mpc import AgentPath, rollout_other_agents


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


# ---------------------------------------------------------------------------
# Part B, task 1 -- path-following IDM agents (arc-length parameterized
# cached path, decoupled from IDM's speed control), plus the two distinct
# fallback cases and mid-scene agent entry.
# ---------------------------------------------------------------------------

def test_agent_follows_curving_logged_path():
    """The thing the old frozen-heading implementation would fail: an agent
    whose logged path is a curve (here, a quarter circle) should stay close
    to that curve's shape as it's IDM-driven forward, not drift into a
    straight line."""
    R = 20.0
    n_wp = 60
    theta = np.linspace(-np.pi / 2, 0.0, n_wp)
    wp = np.stack([R * np.cos(theta), R + R * np.sin(theta)], axis=-1)
    path = AgentPath(wp)

    dt = 0.1
    horizon = 40  # ~20m of travel at 5 m/s, well within the ~31.4m quarter-circle
    yaw0 = theta[0] + np.pi / 2  # tangent heading at the path's start
    agent_state0 = np.array([wp[0, 0], wp[0, 1], yaw0, 5.0])
    ego_states = np.zeros((horizon + 1, 4))  # irrelevant here, reactive=False

    states, s_final = rollout_other_agents(
        agent_state0[None, :], ego_states, horizon, dt, reactive=False,
        paths=[path], s0=np.array([0.0]))

    xy = states[0, :, :2]
    radial_error = np.abs(np.hypot(xy[:, 0], xy[:, 1] - R) - R)
    max_radial_error = radial_error.max()
    print(f"curving path follow: max radial error from true circle = "
          f"{max_radial_error:.3f}m (s={s_final[0]:.2f}m of {path.total_length:.2f}m)")
    assert max_radial_error < 1.0, (
        f"expected the IDM-driven agent to hug the logged quarter-circle path, "
        f"got max radial error {max_radial_error:.2f}m")

    # contrast: the pre-task-1 frozen-heading behavior (no path given) goes
    # straight and diverges from the circle -- confirms this is the regression
    # the fix actually addresses, not a test that would have passed anyway.
    states_frozen = rollout_other_agents(agent_state0[None, :], ego_states, horizon, dt, reactive=False)
    xy_frozen = states_frozen[0, :, :2]
    max_radial_error_frozen = np.abs(np.hypot(xy_frozen[:, 0], xy_frozen[:, 1] - R) - R).max()
    print(f"frozen-heading (pre-task-1) radial error for contrast = {max_radial_error_frozen:.2f}m")
    assert max_radial_error_frozen > max_radial_error, (
        "sanity check: frozen-heading extrapolation should diverge from the curve more than path-following")


def test_path_exhausted_fallback_holds_last_heading():
    """Fallback case 1: agent outpaces its own cached path (still logged as
    valid) -- should hold the last known heading and continue straight from
    the last waypoint, not freeze in place."""
    wp = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # total_length = 2.0m, heading 0
    path = AgentPath(wp)
    dt = 0.1
    horizon = 20
    agent_state0 = np.array([0.0, 0.0, 0.0, 10.0])  # fast enough to exhaust 2m almost immediately
    ego_states = np.zeros((horizon + 1, 4))

    states, s_final = rollout_other_agents(
        agent_state0[None, :], ego_states, horizon, dt, reactive=False,
        paths=[path], s0=np.array([0.0]))
    assert s_final[0] > path.total_length, "test setup bug: agent should have outpaced the path"

    xy, yaw = states[0, :, :2], states[0, :, 2]
    assert np.allclose(yaw, 0.0, atol=1e-6), f"expected frozen last-known heading of 0.0, got {yaw}"
    dx_tail = np.diff(xy[-6:, 0])
    assert np.all(dx_tail > 0), "agent should keep advancing in a straight line after path exhaustion, not freeze"
    print(f"path-exhausted fallback ok: final s={s_final[0]:.2f}m (path length {path.total_length:.2f}m), "
          f"agent continued straight at yaw={yaw[-1]:.3f}")


def test_agent_dropped_when_valid_flag_goes_false():
    """Fallback case 2: agent's valid flag goes False at the current real
    timestep (per the log, it has genuinely left the scene) -- must be
    dropped entirely, never extrapolated. Distinct from fallback case 1
    above (which keeps simulating, just in a straight line)."""
    num_timesteps, num_objects, ego_idx = 20, 2, 0
    x = np.zeros((num_objects, num_timesteps))
    y = np.zeros((num_objects, num_timesteps))
    yaw = np.zeros((num_objects, num_timesteps))
    vel_x = np.full((num_objects, num_timesteps), 5.0)
    vel_y = np.zeros((num_objects, num_timesteps))
    valid = np.ones((num_objects, num_timesteps), dtype=bool)
    valid[1, 10:] = False  # agent 1 leaves the scene for good at t=10
    for t in range(num_timesteps):
        x[1, t] = 5.0 * 0.1 * t

    scenario = types.SimpleNamespace(
        log_trajectory=types.SimpleNamespace(x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, valid=valid),
        object_metadata=types.SimpleNamespace(object_types=np.array([1, 1])),
    )

    scene_paths = build_agent_paths(scenario, ego_idx)
    assert scene_paths["first_valid"][1] == 0 and scene_paths["last_valid"][1] == 9

    agents = ActiveAgentSet(scene_paths, hist_len=5)
    agents.sync(scenario, 0)
    assert 1 in agents.order, "agent should be active while valid"
    agents.sync(scenario, 9)
    assert 1 in agents.order, "agent still valid at t=9"
    agents.sync(scenario, 10)
    assert 1 not in agents.order, "agent should be dropped the instant its valid flag goes False"
    print("valid-flag-false fallback ok: agent dropped from active set exactly at t=10")


def test_mid_scene_agent_entry():
    """New agents entering mid-scene: an object whose first-valid-timestep
    is not t0 should join the active set exactly when the log says it
    should, initialized with s=0 at its first waypoint and v from the log."""
    num_timesteps, num_objects, ego_idx = 20, 2, 0
    x = np.zeros((num_objects, num_timesteps))
    y = np.zeros((num_objects, num_timesteps))
    yaw = np.zeros((num_objects, num_timesteps))
    vel_x = np.zeros((num_objects, num_timesteps))
    vel_y = np.zeros((num_objects, num_timesteps))
    valid = np.zeros((num_objects, num_timesteps), dtype=bool)
    valid[1, 8:] = True  # agent 1 doesn't exist until t=8
    vel_x[1, :] = 6.0
    for t in range(8, num_timesteps):
        x[1, t] = 6.0 * 0.1 * (t - 8)

    scenario = types.SimpleNamespace(
        log_trajectory=types.SimpleNamespace(x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, valid=valid),
        object_metadata=types.SimpleNamespace(object_types=np.array([1, 1])),
    )

    scene_paths = build_agent_paths(scenario, ego_idx)
    assert scene_paths["first_valid"][1] == 8

    agents = ActiveAgentSet(scene_paths, hist_len=5)
    for t in range(8):
        agents.sync(scenario, t)
        assert 1 not in agents.order, f"agent shouldn't be active before its first valid timestep (t={t})"
    agents.sync(scenario, 8)
    assert 1 in agents.order, "agent should enter the active set exactly at its first-valid timestep"
    assert agents.s[1] == 0.0
    np.testing.assert_allclose(agents.state[1], np.array([x[1, 8], y[1, 8], 0.0, 6.0]))
    print("mid-scene agent entry ok: agent 1 entered active set at t=8 with s=0")


if __name__ == "__main__":
    test_closed_loop_runs()
    test_agent_follows_curving_logged_path()
    test_path_exhausted_fallback_holds_last_heading()
    test_agent_dropped_when_valid_flag_goes_false()
    test_mid_scene_agent_entry()
    print("\nall closed_loop tests passed")
