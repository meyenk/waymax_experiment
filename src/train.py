"""
Training loop over a stream of Waymax scenarios. Visualization lives in
src/visualize.py -- kept separate so it's easy to find/iterate on independently.
"""

import numpy as np
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


def run_epoch(model, optimizer, scenario_iter, max_scenarios, batch_size=16, train=True, device="cpu"):
    """One pass over up to max_scenarios scenarios. Returns mean loss.
    Both pred (model output) and target (from collate_examples) are absolute
    positions in the ego frame at t -- same space, safe to compare directly."""
    model.train() if train else model.eval()
    buffer, losses = [], []

    def flush(chunk):
        if not chunk:
            return None
        batch, target = collate_examples(chunk)
        batch = {k: v.to(device) for k, v in batch.items()}
        target = target.to(device)
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
