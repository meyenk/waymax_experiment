"""
World-frame visualizations. Two things, matching how a real scenario looks:

1. play_scenario_clip -- the actual 9s scenario, rendered with Waymax's own
   BEV renderer (standard tutorial pattern: loop frames, collect images, play
   as video). This is ground truth, nothing of ours involved.

2. plot_world_frame_snapshots -- a handful of decision points from the SAME
   scenario, each showing the real scene (roadgraph + all agents, true world
   coordinates/scale) with the model's predicted future overlaid, brought
   back from ego-frame into world coordinates via the inverse transform.

3. plot_selection -- one decision instant, showing exactly what the model
   is fed: which agents/map points fall in the region of interest, which
   history frames were filled across a tracker dropout, what the OLD rule
   would have picked instead, and (optionally) the model's learned relevance
   weight for each item -- the project's core "what does it pay attention
   to" question, made visible. selection_report gives the same thing as
   numbers over many scenarios.

Loss curves stay here too, unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.dataset import get_window_starts, collate_examples
from src import features as F
from src.extract import extract_example, get_ego_frame, select_agents
from src.transforms import inverse_transform_positions


def pick_dynamic_scenario(scenarios, min_displacement=15.0):
    """Returns the first scenario where the ego's net displacement over the full
    9s is at least min_displacement meters. Some WOMD scenarios have a mostly
    stationary ego (e.g. waiting at a light) -- nothing wrong with the data,
    but it makes for a degenerate-looking, uninformative visualization. Picking
    a scenario where the car actually goes somewhere is just a better demo."""
    for scenario in scenarios:
        ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
        traj = scenario.log_trajectory
        valid = np.array(traj.valid[ego_idx])
        xy = np.stack([np.array(traj.x[ego_idx])[valid], np.array(traj.y[ego_idx])[valid]], axis=-1)
        if len(xy) < 2:
            continue
        net_disp = np.linalg.norm(xy[-1] - xy[0])
        if net_disp >= min_displacement:
            return scenario
    print(f"no scenario found with net displacement >= {min_displacement}m; returning first scenario anyway")
    return scenarios[0]


def is_turning_scenario(scenario, min_heading_change_deg=25.0, min_displacement=5.0):
    """Same test as pick_turning_scenario, but returns True/False for one
    scenario -- used to break metrics down by scenario type instead of just
    picking a demo."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    traj = scenario.log_trajectory
    valid = np.array(traj.valid[ego_idx])
    yaw = np.array(traj.yaw[ego_idx])[valid]
    xy = np.stack([np.array(traj.x[ego_idx])[valid], np.array(traj.y[ego_idx])[valid]], axis=-1)
    if len(yaw) < 2:
        return False
    heading_change = np.degrees(np.abs(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]))
    displacement = np.linalg.norm(xy[-1] - xy[0])
    return heading_change >= min_heading_change_deg and displacement >= min_displacement


def pick_turning_scenario(scenarios, min_heading_change_deg=25.0, min_displacement=5.0):
    """Returns the first scenario where the ego's heading changes by at least
    min_heading_change_deg over the full 9s (net, unwrapped -- handles the
    -pi/pi wraparound) AND it actually moved at least min_displacement meters.
    The displacement check matters: a near-stationary car can show spurious
    heading "change" from noisy yaw estimates alone, which isn't a real turn."""
    for scenario in scenarios:
        ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
        traj = scenario.log_trajectory
        valid = np.array(traj.valid[ego_idx])
        yaw = np.array(traj.yaw[ego_idx])[valid]
        xy = np.stack([np.array(traj.x[ego_idx])[valid], np.array(traj.y[ego_idx])[valid]], axis=-1)
        if len(yaw) < 2:
            continue
        heading_change = np.degrees(np.abs(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]))
        displacement = np.linalg.norm(xy[-1] - xy[0])
        if heading_change >= min_heading_change_deg and displacement >= min_displacement:
            return scenario
    print(f"no scenario found with heading change >= {min_heading_change_deg} deg "
          f"and displacement >= {min_displacement}m; returning first scenario anyway")
    return scenarios[0]


def plot_loss_curves(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (position space)")
    ax.legend()
    ax.set_title("Stage 1 training curves")
    plt.show()


def play_scenario_clip(scenario, fps=10):
    """Plays the real logged scenario end to end using Waymax's own renderer --
    same pattern as the tutorial (plot_simulator_state per frame + mediapy video).
    Purely a sanity view of the ground truth, no model involved."""
    from waymax import visualization
    import mediapy
    import dataclasses

    num_frames = scenario.log_trajectory.x.shape[1]
    images = []
    for t in range(num_frames):
        frame_state = dataclasses.replace(scenario, timestep=t)
        images.append(visualization.plot_simulator_state(frame_state, use_log_traj=True))
    mediapy.show_video(images, fps=fps)


def _plot_world_scene(ax, scenario, t, window=60.0):
    """Background for one panel: roadgraph points + all agents' current
    positions, real world coordinates, zoomed to `window` meters around ego."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    ego_xy, ego_yaw, _ = get_ego_frame(scenario, ego_idx, t)

    rg = scenario.roadgraph_points
    rg_valid = np.array(rg.valid)
    ax.scatter(np.array(rg.x)[rg_valid], np.array(rg.y)[rg_valid],
               c="lightgray", s=2, zorder=1)

    traj = scenario.log_trajectory
    valid_now = np.array(traj.valid[:, t])
    xy_now = np.stack([traj.x[:, t], traj.y[:, t]], axis=-1)
    ax.scatter(xy_now[valid_now, 0], xy_now[valid_now, 1], c="dimgray", s=25, zorder=3)

    # ego's own past trail, for context of how it got here
    past_valid = np.array(traj.valid[ego_idx, :t + 1])
    past_xy = np.stack([traj.x[ego_idx, :t + 1][past_valid], traj.y[ego_idx, :t + 1][past_valid]], axis=-1)
    ax.plot(past_xy[:, 0], past_xy[:, 1], c="steelblue", linewidth=1.5, zorder=4)
    ax.scatter(*ego_xy, c="blue", marker="*", s=200, zorder=5, label="ego (t)")

    ax.set_xlim(ego_xy[0] - window, ego_xy[0] + window)
    ax.set_ylim(ego_xy[1] - window, ego_xy[1] + window)
    ax.set_aspect("equal")
    return ego_xy, ego_yaw


def plot_world_frame_snapshots(model, scenario, hist_len=10, future_len=30,
                                num_snapshots=4, window=60.0, device="cpu"):
    """Grid of snapshots, real world coordinates. Each panel: the actual scene
    (roadgraph, all agents, ego's past trail) plus logged future (green) and
    predicted future (red dashed) -- predicted brought back to world frame
    via the inverse ego transform."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    starts = get_window_starts(hist_len, future_len, stride=10)
    if not starts:
        print("no valid decision points for this hist_len/future_len")
        return
    chosen_ts = starts[:: max(1, len(starts) // num_snapshots)][:num_snapshots]

    model.eval()
    traj = scenario.log_trajectory
    fig, axes = plt.subplots(1, len(chosen_ts), figsize=(5 * len(chosen_ts), 5))
    if len(chosen_ts) == 1:
        axes = [axes]

    for ax, t in zip(axes, chosen_ts):
        ex = extract_example(scenario, ego_idx, t, hist_len, future_len)
        if ex is None:
            ax.set_title(f"t={t} (no valid example)")
            continue
        batch, _ = collate_examples([ex])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            pred_local, _ = model(batch)
        pred_local = pred_local[0].cpu().numpy()

        ego_xy, ego_yaw = _plot_world_scene(ax, scenario, t, window=window)
        pred_world = inverse_transform_positions(pred_local, ego_xy, ego_yaw)

        logged_future = np.stack([traj.x[ego_idx, t + 1:t + future_len + 1],
                                   traj.y[ego_idx, t + 1:t + future_len + 1]], axis=-1)

        ax.plot(np.append(ego_xy[0], logged_future[:, 0]),
                np.append(ego_xy[1], logged_future[:, 1]), c="green", label="logged future")
        ax.plot(np.append(ego_xy[0], pred_world[:, 0]),
                np.append(ego_xy[1], pred_world[:, 1]), c="red", linestyle="--", label="predicted")
        ax.set_title(f"t={t}")

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Prediction snapshots -- real world coordinates")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# What the model is fed (region of interest / selection)
# ---------------------------------------------------------------------------

def _old_rule_agents(scenario, ego_idx, t, hist_len=10, max_agents=32):
    """The pre-branch agent rule, for comparison only: valid in EVERY frame
    of the history window, any distance, first max_agents in array order."""
    traj = scenario.log_trajectory
    full = np.array(traj.valid[:, t - hist_len + 1:t + 1]).all(axis=1)
    full[ego_idx] = False
    return np.where(full)[0][:max_agents]


def selection_report(scenarios, hist_len=10, future_len=30, stride=10):
    """Per decision instant across scenarios: how many agents the new ROI
    rule picks vs the old rule, how many of the new picks the old rule would
    have dropped (tracker dropout in the window), how many of the new picks'
    history frames are filled, and how many map points are in range vs used.
    Returns a dict of numpy arrays, one entry per (scenario, t)."""
    keys = ("agents_new", "agents_old", "agents_rescued", "agents_far_old",
            "frac_frames_filled", "map_in_radius", "map_used")
    out = {k: [] for k in keys}
    out["no_sdc"] = 0
    for scenario in scenarios:
        is_sdc = np.array(scenario.object_metadata.is_sdc)
        if not is_sdc.any():
            out["no_sdc"] += 1
            continue
        ego_idx = int(np.argmax(is_sdc))
        traj = scenario.log_trajectory
        rg = scenario.roadgraph_points
        rg_valid = np.array(rg.valid)
        rg_xy = np.stack([np.array(rg.x)[rg_valid], np.array(rg.y)[rg_valid]], axis=-1)
        for t in get_window_starts(hist_len, future_len, stride):
            new = select_agents(scenario, ego_idx, t)
            old = _old_rule_agents(scenario, ego_idx, t, hist_len)
            ego_xy = np.array([traj.x[ego_idx, t], traj.y[ego_idx, t]])
            xy_old = np.stack([np.array(traj.x)[old, t], np.array(traj.y)[old, t]], axis=-1)
            win = np.array(traj.valid)[new, t - hist_len + 1:t + 1]
            n_in = int((np.linalg.norm(rg_xy - ego_xy, axis=-1) <= F.MAP_RADIUS).sum())
            out["agents_new"].append(len(new))
            out["agents_old"].append(len(old))
            out["agents_rescued"].append(len(np.setdiff1d(new, old)))
            out["agents_far_old"].append(int((np.linalg.norm(xy_old - ego_xy, axis=-1) > F.AGENT_RADIUS).sum()))
            out["frac_frames_filled"].append(float(1 - win.mean()) if win.size else 0.0)
            out["map_in_radius"].append(n_in)
            out["map_used"].append(min(n_in, F.MAX_MAP_POINTS))
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in out.items()}


def plot_selection(scenario, t, model=None, hist_len=10, device="cpu", window=None):
    """Two panels for one decision instant t.
    Left, world frame: all roadgraph points (light grey), map points the
    model is fed (coloured), agent ROI circle (dashed) and map radius
    (dotted), every agent valid at t -- picked ones coloured with their 1s
    history trail (x = a frame filled across a tracker dropout), unpicked
    ones as grey rings. Red rings mark agents the OLD rule would have
    dropped (dropout in the window); grey crosses mark agents the old rule
    WOULD have fed the model that are outside the ROI.
    Right, ego frame: literally the model's agent input tensor -- each picked
    agent's history positions as the model sees them.
    If a model is given, picked agents and map points are coloured by the
    model's learned relevance weight (RelevancePool softmax) instead."""
    ego_idx = int(np.argmax(np.array(scenario.object_metadata.is_sdc)))
    traj = scenario.log_trajectory
    window = window or F.MAP_RADIUS + 10.0
    ex = extract_example(scenario, ego_idx, t, hist_len)
    if ex is None:
        print(f"t={t}: no valid example (needs {hist_len} frames of history and a full future)")
        return
    picked = select_agents(scenario, ego_idx, t)
    old = _old_rule_agents(scenario, ego_idx, t, hist_len)
    ego_xy, ego_yaw, _ = get_ego_frame(scenario, ego_idx, t)

    agent_w = map_w = None
    if model is not None:
        model.eval()
        batch, _ = collate_examples([ex])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            _, w = model(batch)
        agent_w = w["agent_w"][0].cpu().numpy()[:len(picked)]
        map_w = w["map_w"][0].cpu().numpy()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # --- left: world frame
    rg = scenario.roadgraph_points
    rgv = np.array(rg.valid)
    ax.scatter(np.array(rg.x)[rgv], np.array(rg.y)[rgv], c="lightgray", s=1, zorder=1)
    n_map = int(ex["map_mask"].sum())
    map_world = inverse_transform_positions(ex["map_xy"][:n_map], ego_xy, ego_yaw)
    if map_w is not None:
        sc = ax.scatter(map_world[:, 0], map_world[:, 1], c=map_w[:n_map], cmap="viridis", s=3, zorder=2)
        fig.colorbar(sc, ax=ax, fraction=0.04, label="map relevance weight")
    else:
        ax.scatter(map_world[:, 0], map_world[:, 1], c="cornflowerblue", s=2, zorder=2,
                   label=f"map points fed ({n_map})")
    for r, ls, lab in ((F.AGENT_RADIUS, "--", "agent ROI"), (F.MAP_RADIUS, ":", "map radius")):
        ax.add_patch(plt.Circle(ego_xy, r, fill=False, ls=ls, color="k", lw=1, label=f"{lab} ({r:.0f} m)"))

    valid_now = np.array(traj.valid[:, t]).copy()
    valid_now[ego_idx] = False
    others = np.setdiff1d(np.where(valid_now)[0], picked)
    ax.scatter(traj.x[others, t], traj.y[others, t], facecolors="none", edgecolors="gray",
               s=40, zorder=3, label="valid at t, not picked")
    old_far = [i for i in old if np.hypot(traj.x[i, t] - ego_xy[0], traj.y[i, t] - ego_xy[1]) > F.AGENT_RADIUS]
    if old_far:
        ax.scatter(traj.x[old_far, t], traj.y[old_far, t], marker="x", c="gray", s=40, zorder=3,
                   label="old rule fed these (outside ROI)")

    cmap = plt.get_cmap("plasma")
    wmax = agent_w.max() if agent_w is not None and len(agent_w) else 1.0
    w0 = t - hist_len + 1
    for k, i in enumerate(picked):
        color = cmap(agent_w[k] / wmax) if agent_w is not None else "darkorange"
        obs = np.array(traj.valid[i, w0:t + 1])
        hx, hy = np.array(traj.x[i, w0:t + 1]), np.array(traj.y[i, w0:t + 1])
        ax.plot(hx[obs], hy[obs], c=color, lw=1.5, zorder=4)
        if (~obs).any():  # filled frames, at the world position the fill puts them
            filled_world = F.fill_gaps(np.stack([hx, hy], axis=-1)[None], obs[None])[0]
            ax.scatter(filled_world[~obs, 0], filled_world[~obs, 1], marker="x", c=[color], s=25, zorder=5)
        ax.scatter(traj.x[i, t], traj.y[i, t], c=[color], s=45, zorder=6, edgecolors="k", linewidths=0.5)
        if i not in old:
            ax.scatter(traj.x[i, t], traj.y[i, t], facecolors="none", edgecolors="red", s=140,
                       linewidths=1.5, zorder=6)
    ax.scatter([], [], c="darkorange", s=45, label=f"picked agents ({len(picked)}), x = filled frame")
    ax.scatter([], [], facecolors="none", edgecolors="red", s=140, label="rescued (old rule dropped it)")
    ax.scatter(*ego_xy, c="blue", marker="*", s=250, zorder=7, label="ego (t)")
    ax.set_xlim(ego_xy[0] - window, ego_xy[0] + window)
    ax.set_ylim(ego_xy[1] - window, ego_xy[1] + window)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"world frame, t={t}: {len(picked)} agents picked (old rule: {len(old)}), "
                 f"{n_map} map points", fontsize=10)

    # --- right: ego frame, the agent tensor exactly as the model gets it
    ax2.scatter(ex["map_xy"][:n_map, 0], ex["map_xy"][:n_map, 1], c="lightgray", s=1)
    for k in range(len(picked)):
        h = ex["agent_hist"][k]
        color = cmap(agent_w[k] / wmax) if agent_w is not None else "darkorange"
        obs = h[:, 6] > 0.5
        ax2.plot(h[:, 0], h[:, 1], c=color, lw=1)
        ax2.scatter(h[obs, 0], h[obs, 1], c=[color], s=8)
        ax2.scatter(h[~obs, 0], h[~obs, 1], marker="x", c=[color], s=25)
    ax2.scatter(0, 0, c="blue", marker="*", s=250)
    w2 = F.AGENT_RADIUS + 10.0
    ax2.set_xlim(-w2, w2); ax2.set_ylim(-w2, w2)
    ax2.set_aspect("equal")
    ax2.set_title("ego frame: agent history tensor as fed to the model\n"
                  "(time-matched: frame tau is relative to the ego AT tau, so trails "
                  "show relative motion, not world paths)", fontsize=9)
    plt.tight_layout()
    plt.show()
