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
