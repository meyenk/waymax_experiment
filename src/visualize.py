"""
World-frame visualizations. Two things, matching how a real scenario looks:

1. play_scenario_clip -- the actual 9s scenario, rendered with Waymax's own
   BEV renderer (standard tutorial pattern: loop frames, collect images, play
   as video). This is ground truth, nothing of ours involved.

2. plot_world_frame_snapshots -- a handful of decision points from the SAME
   scenario, each showing the real scene (roadgraph + all agents, true world
   coordinates/scale) with the model's predicted future overlaid, brought
   back from ego-frame into world coordinates via the inverse transform.

Loss curves stay here too, unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.dataset import get_window_starts, collate_examples
from src.extract import extract_example, get_ego_frame
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
