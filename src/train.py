"""
Training loop over a stream of Waymax scenarios, plus two visualizations:
loss curves (train/val/test) and a single BEV-style plot of predicted vs
logged ego trajectory for one held-out example.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.dataset import get_window_starts, val_test_split, collate_examples
from src.extract import extract_example


def build_examples_from_scenario(scenario, hist_len=10, future_len=30, stride=10):
    """All valid (ego, t) examples from one scenario, given our window stride."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    examples = []
    for t in get_window_starts(hist_len, future_len, stride):
        ex = extract_example(scenario, ego_idx, t, hist_len, future_len)
        if ex is not None:
            examples.append(ex)
    return examples


def run_epoch(model, optimizer, scenario_iter, max_scenarios, batch_size=16, train=True):
    """One pass over up to max_scenarios scenarios. Returns mean loss."""
    model.train() if train else model.eval()
    buffer, losses = [], []

    def flush(chunk):
        if not chunk:
            return None
        batch, target = collate_examples(chunk)
        if train:
            optimizer.zero_grad()
            pred, _ = model(batch)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred, _ = model(batch)
                loss = torch.mean((pred - target) ** 2)
        return loss.item()

    for i, scenario in enumerate(scenario_iter):
        if i >= max_scenarios:
            break
        buffer.extend(build_examples_from_scenario(scenario))
        while len(buffer) >= batch_size:
            chunk, buffer = buffer[:batch_size], buffer[batch_size:]
            loss = flush(chunk)
            if loss is not None:
                losses.append(loss)
    loss = flush(buffer)  # remainder
    if loss is not None:
        losses.append(loss)
    return float(np.mean(losses)) if losses else float("nan")


def plot_loss_curves(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (offset space)")
    ax.legend()
    ax.set_title("Stage 1 training curves")
    plt.show()


def plot_bev_prediction(model, scenario, hist_len=10, future_len=30, t=None):
    """One example: ego past, logged future, predicted future, nearby agents -- all
    already in the ego-centric frame at t, so this plots directly, no re-transform."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    if t is None:
        starts = get_window_starts(hist_len, future_len, stride=10)
        t = starts[len(starts) // 2]

    ex = extract_example(scenario, ego_idx, t, hist_len, future_len)
    if ex is None:
        print("no valid example at this t, try a different scenario/t")
        return

    batch, target = collate_examples([ex])
    model.eval()
    with torch.no_grad():
        pred, weights = model(batch)
    pred = pred[0].numpy()
    target = target[0].numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(0, 0, c="blue", marker="*", s=200, label="ego (t)")
    ax.plot(*zip(*np.vstack([[0, 0], np.cumsum(target, axis=0)])), c="green", label="logged future")
    ax.plot(*zip(*np.vstack([[0, 0], pred])), c="red", linestyle="--", label="predicted future")

    agent_xy_now = ex["agent_hist"][:, -1, :2]           # last observed frame, x/y only
    valid = ex["agent_mask"]
    ax.scatter(agent_xy_now[valid, 0], agent_xy_now[valid, 1], c="lightgray", label="other agents (now)")

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"BEV prediction check (t={t})")
    plt.show()
