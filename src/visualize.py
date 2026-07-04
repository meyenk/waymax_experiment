"""
Visualization functions, kept separate from train.py so they're easy to find
and iterate on independently.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.dataset import get_window_starts, collate_examples
from src.extract import extract_example


def plot_loss_curves(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (position space)")
    ax.legend()
    ax.set_title("Stage 1 training curves")
    plt.show()


def plot_full_ego_trajectory(scenario):
    """Overview map: the ego's actual logged path across the whole scenario,
    in a world frame anchored at its first valid position (not ego-frame-at-t --
    this is a single global picture, not tied to any one decision point)."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    traj = scenario.log_trajectory
    valid = np.array(traj.valid[ego_idx])
    xy = np.stack([traj.x[ego_idx, valid], traj.y[ego_idx, valid]], axis=-1)
    origin = xy[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[:, 0] - origin[0], xy[:, 1] - origin[1], c="blue", marker=".", markersize=3)
    ax.scatter(0, 0, c="green", s=100, zorder=5, label="start")
    ax.scatter(xy[-1, 0] - origin[0], xy[-1, 1] - origin[1], c="red", s=100, zorder=5, label="end")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Actual logged ego trajectory (full scenario)")
    plt.show()


def plot_prediction_snapshots(model, scenario, hist_len=10, future_len=30,
                               num_snapshots=4, device="cpu"):
    """Grid of snapshots at different decision points t within one scenario.
    Each panel: ego at origin (its own frame at that t), logged future (green),
    predicted future (red dashed), nearby agents (gray). Axis limits are shared
    across panels so predictions are visually comparable frame to frame."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    starts = get_window_starts(hist_len, future_len, stride=10)
    if len(starts) == 0:
        print("no valid decision points for this hist_len/future_len")
        return
    chosen_ts = starts[:: max(1, len(starts) // num_snapshots)][:num_snapshots]

    model.eval()
    panels = []
    for t in chosen_ts:
        ex = extract_example(scenario, ego_idx, t, hist_len, future_len)
        if ex is None:
            continue
        batch, target = collate_examples([ex])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            pred, _ = model(batch)
        panels.append({
            "t": t,
            "pred": pred[0].cpu().numpy(),
            "target": target[0].numpy(),
            "agents_xy": ex["agent_hist"][:, -1, :2],
            "agents_mask": ex["agent_mask"],
        })

    if not panels:
        print("no valid snapshots found in this scenario")
        return

    # shared axis extent across all panels, for fair visual comparison
    all_pts = np.concatenate([p["pred"] for p in panels] + [p["target"] for p in panels])
    lim = np.abs(all_pts).max() * 1.15

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, p in zip(axes, panels):
        ax.scatter(0, 0, c="blue", marker="*", s=150, label="ego")
        ax.plot(*zip(*np.vstack([[0, 0], p["target"]])), c="green", label="logged future")
        ax.plot(*zip(*np.vstack([[0, 0], p["pred"]])), c="red", linestyle="--", label="predicted")
        valid = p["agents_mask"]
        ax.scatter(p["agents_xy"][valid, 0], p["agents_xy"][valid, 1], c="lightgray", s=20)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"t={p['t']}")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Prediction snapshots across the scenario")
    plt.tight_layout()
    plt.show()
