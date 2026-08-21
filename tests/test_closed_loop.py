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
