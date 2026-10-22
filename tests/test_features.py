"""
Region-of-interest selection, tracker-dropout filling and map selection
(src/features.py), checked through extract.py on small synthetic scenes --
no Waymax dependency, same spirit as the other test modules."""
import types
import numpy as np

from src import features as F
from src.extract import extract_agents, extract_map, select_agents, extract_example


def make_scene(num_timesteps=40):
    """ego at the origin driving +x at 10 m/s; four agents:
      1: 10m ahead, same lane, always valid                  -> picked
      2: 20m to the side, tracker DROPOUT for 3 frames
         in the middle of the history window                 -> picked (old rule dropped it)
      3: 120m away, always valid                             -> outside ROI, not picked
      4: 15m ahead, NOT valid at the decision instant t      -> not picked
    """
    dt, n = 0.1, 5
    T = num_timesteps
    x = np.zeros((n, T)); y = np.zeros((n, T)); yaw = np.zeros((n, T))
    vx = np.zeros((n, T)); vy = np.zeros((n, T))
    valid = np.ones((n, T), dtype=bool)
    ts = np.arange(T) * dt
    x[0] = 10.0 * ts; vx[0] = 10.0
    x[1] = 10.0 + 8.0 * ts; vx[1] = 8.0
    x[2] = 5.0 * ts; y[2] = 20.0; vx[2] = 5.0
    x[3] = 120.0
    x[4] = 15.0 + 10.0 * ts; vx[4] = 10.0
    valid[2, 14:17] = False
    valid[4, 19:] = False
    traj = types.SimpleNamespace(x=x, y=y, yaw=yaw, vel_x=vx, vel_y=vy, valid=valid)
    meta = types.SimpleNamespace(object_types=np.ones(n, dtype=np.int64),
                                 is_sdc=np.array([True, False, False, False, False]))
    # a dense straight road: points every 0.5m along y=+-2, x in [-100, 100]
    rx = np.arange(-100, 100, 0.5)
    rg = types.SimpleNamespace(x=np.concatenate([rx, rx]), y=np.concatenate([rx * 0 + 2, rx * 0 - 2]),
                               types=np.zeros(2 * len(rx), dtype=np.int64),
                               valid=np.ones(2 * len(rx), dtype=bool))
    return types.SimpleNamespace(log_trajectory=traj, object_metadata=meta, roadgraph_points=rg), 0, 19


def test_roi_selection_nearest_first():
    scenario, ego_idx, t = make_scene()
    picked = select_agents(scenario, ego_idx, t).tolist()
    assert picked == [1, 2], f"expected [1, 2] (nearest first, ROI, valid at t), got {picked}"
    print(f"roi selection ok: picked {picked} (far agent and not-valid-now agent excluded)")


def test_dropout_agent_kept_and_flagged():
    scenario, ego_idx, t = make_scene()
    hist_len = 10
    out = extract_agents(scenario, ego_idx, t, hist_len=hist_len)
    assert out["mask"][:2].all() and not out["mask"][2:].any()
    h = out["hist"][1]  # agent 2, second nearest
    obs = h[:, 6]
    # window is [10, 19]; dropout at 14..16 -> window positions 4..6
    assert np.array_equal(obs, [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]), f"observed flags wrong: {obs}"
    # filled frames must lie on the straight line between the neighbours (in the
    # time-matched ego frame both agent and ego move uniformly, so relative x is linear)
    rel_x = h[:, 0]
    assert np.allclose(np.diff(rel_x), np.diff(rel_x)[0], atol=1e-4), "interpolated frames should be linear"
    print(f"dropout agent ok: kept with observed flags {obs.astype(int).tolist()}, filled linearly")


def test_fill_gaps_edges_and_angles():
    vals = np.array([[[0.0, np.pi - 0.1], [0.0, 0.0], [2.0, -np.pi + 0.1], [0.0, 0.0]]])
    valid = np.array([[True, False, True, False]])
    out = F.fill_gaps(vals, valid, angle_channels=(1,))
    assert np.isclose(out[0, 1, 0], 1.0), "interior gap should interpolate"
    assert np.isclose(out[0, 3, 0], 2.0), "trailing gap should hold the last value"
    assert abs(abs(out[0, 1, 1]) - np.pi) < 1e-6, (
        f"heading crossing +-pi should interpolate through pi, not through 0: got {out[0, 1, 1]:.3f}")
    print("fill_gaps ok: interior interpolation, edge hold, angle wrap through pi")


def test_bridge_short_gaps_and_runs():
    valid = np.array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]], dtype=bool)
    bridged = F.bridge_short_gaps(valid, max_gap=3)
    assert bridged[0].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    assert F.valid_runs(bridged[0]) == [(0, 5), (11, 11)]
    print("bridge_short_gaps ok: 2-frame gap bridged, 5-frame gap kept, trailing gap kept")


def test_map_nearest_and_covers_radius():
    scenario, ego_idx, t = make_scene()
    m = extract_map(scenario, ego_idx, t)
    n = int(m["mask"].sum())
    d = np.linalg.norm(m["xy"][:n], axis=-1)
    assert np.all(np.diff(d) >= -1e-6), "map points should be ordered nearest-first"
    assert d.max() <= F.MAP_RADIUS + 1e-6
    assert m["n_in_radius"] == n, f"cap should not bind here: {m['n_in_radius']} in range, {n} used"
    # old behaviour would have used the first 200 in ARRAY order: x in [-100, 0)
    # on the first line, i.e. mostly BEHIND the ego and mostly outside 80m
    old_cap = F.MAX_MAP_POINTS
    try:
        F.MAX_MAP_POINTS = 50
        m50 = extract_map(scenario, ego_idx, t)
        d50 = np.linalg.norm(m50["xy"][:50], axis=-1)
        assert d50.max() < 7.0, f"with a small cap the NEAREST points should survive, max dist {d50.max():.1f}m"
    finally:
        F.MAX_MAP_POINTS = old_cap
    print(f"map ok: {n} points within {F.MAP_RADIUS:.0f}m, nearest-first; a small cap keeps the nearest")


def test_example_shapes():
    scenario, ego_idx, t = make_scene(num_timesteps=60)  # needs 30 future frames after t
    ex = extract_example(scenario, ego_idx, t)
    assert ex["agent_hist"].shape == (F.MAX_AGENTS, 10, F.AGENT_FEAT_DIM)
    assert ex["map_xy"].shape == (F.MAX_MAP_POINTS, 2)
    print("example shapes ok")


if __name__ == "__main__":
    test_roi_selection_nearest_first()
    test_dropout_agent_kept_and_flagged()
    test_fill_gaps_edges_and_angles()
    test_bridge_short_gaps_and_runs()
    test_map_nearest_and_covers_radius()
    test_example_shapes()
    print("\nall feature/selection tests passed")
