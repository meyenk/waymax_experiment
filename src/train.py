"""
Training loop over a stream of Waymax scenarios. Visualization lives in
src/visualize.py -- kept separate so it's easy to find/iterate on independently.
"""

import time
import numpy as np
import torch

from src.dataset import get_window_starts, val_test_split, collate_examples
from src.extract import extract_example
from src.metrics import compute_ade_fde


def build_examples_from_scenario(scenario, hist_len=10, future_len=30, stride=10):
    """All valid (ego, t) examples from one scenario, given our window stride."""
    ego_idx = int(np.argmax(scenario.object_metadata.is_sdc))
    examples = []
    for t in get_window_starts(hist_len, future_len, stride):
        ex = extract_example(scenario, ego_idx, t, hist_len, future_len)
        if ex is not None:
            examples.append(ex)
    return examples


def run_epoch(model, optimizer, scenario_iter, max_scenarios, batch_size=16, train=True,
              device="cpu", grad_clip_norm=1.0, verbose_every=5):
    """One pass over up to max_scenarios scenarios. Returns a dict with mean
    loss, ADE, and FDE. Both pred (model output) and target (from
    collate_examples) are absolute positions in the ego frame at t -- same
    space, safe to compare directly. Prints a running update every
    verbose_every batches so a long epoch isn't silent."""
    model.train() if train else model.eval()
    buffer, losses, ades, fdes = [], [], [], []
    batch_count = 0
    start_time = time.time()
    mode = "train" if train else "val"

    def flush(chunk):
        nonlocal batch_count
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                pred, _ = model(batch)
                loss = torch.mean((pred - target) ** 2)
        ade, fde = compute_ade_fde(pred.detach(), target)
        batch_count += 1
        if verbose_every and batch_count % verbose_every == 0:
            elapsed = time.time() - start_time
            print(f"    [{mode}] batch {batch_count} | loss={loss.item():.3f} "
                  f"ade={ade:.3f} fde={fde:.3f} | {elapsed:.0f}s elapsed")
        return loss.item(), ade, fde

    for i, scenario in enumerate(scenario_iter):
        if i >= max_scenarios:
            break
        buffer.extend(build_examples_from_scenario(scenario))
        while len(buffer) >= batch_size:
            chunk, buffer = buffer[:batch_size], buffer[batch_size:]
            result = flush(chunk)
            if result is not None:
                losses.append(result[0]); ades.append(result[1]); fdes.append(result[2])
    result = flush(buffer)  # remainder
    if result is not None:
        losses.append(result[0]); ades.append(result[1]); fdes.append(result[2])

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "ade": float(np.mean(ades)) if ades else float("nan"),
        "fde": float(np.mean(fdes)) if fdes else float("nan"),
    }
