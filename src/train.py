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

# Scenarios skipped because the SDC isn't among the loaded objects. Waymax's
# dataloader truncates to the first max_num_objects objects in file order
# (womd_dataloader.py has a TODO saying exactly this), and np.argmax over an
# all-False is_sdc silently returns 0 -- i.e. a random object would be
# treated as the ego. With max_num_objects=128 (all of WOMD) this should
# stay 0; it's counted so that's verified rather than assumed.
SKIPPED_NO_SDC = 0

BRANCHES = ("agents", "map", "lights")


def get_ego_idx(scenario):
    """Index of the SDC, or None if it isn't in the loaded objects."""
    is_sdc = np.array(scenario.object_metadata.is_sdc)
    return int(np.argmax(is_sdc)) if is_sdc.any() else None


def build_examples_from_scenario(scenario, hist_len=10, future_len=30, stride=10):
    """All valid (ego, t) examples from one scenario, given our window stride."""
    global SKIPPED_NO_SDC
    ego_idx = get_ego_idx(scenario)
    if ego_idx is None:
        SKIPPED_NO_SDC += 1
        return []
    examples = []
    for t in get_window_starts(hist_len, future_len, stride):
        ex = extract_example(scenario, ego_idx, t, hist_len, future_len)
        if ex is not None:
            examples.append(ex)
    return examples


def ablate_batch(batch, ablate):
    """Hide whole encoder branches by masking every item out -- the pool then
    returns zeros for that branch, exactly as if the scene had nothing of that
    kind. ablate: iterable of names from BRANCHES. Used at eval time to check
    how much each branch actually contributes to the prediction."""
    batch = dict(batch)
    for name in ablate or ():
        key = {"agents": "agent_mask", "map": "map_mask", "lights": "light_mask"}[name]
        batch[key] = torch.zeros_like(batch[key])
    return batch


def run_epoch(model, optimizer, scenario_iter, max_scenarios, batch_size=16, train=True,
              device="cpu", grad_clip_norm=1.0, verbose_every=5, ablate=None):
    """One pass over up to max_scenarios scenarios. Returns a dict with mean
    loss, ADE, FDE (weighted by batch size, so a small remainder batch
    doesn't count as much as a full one), n_examples, and map_truncated_frac
    -- the share of examples with more map points in range than
    MAX_MAP_POINTS. Both pred (model output) and target (from
    collate_examples) are absolute positions in the ego frame at t -- same
    space, safe to compare directly. Prints a running update every
    verbose_every batches so a long epoch isn't silent."""
    model.train() if train else model.eval()
    buffer, losses, ades, fdes, sizes = [], [], [], [], []
    n_trunc = 0
    batch_count = 0
    start_time = time.time()
    mode = "train" if train else "val"

    def flush(chunk):
        nonlocal batch_count, n_trunc
        batch, target = collate_examples(chunk)
        n_in_radius = batch.pop("map_n_in_radius", None)
        if n_in_radius is not None:
            n_trunc += int((n_in_radius > batch["map_mask"].shape[1]).sum())
        batch = ablate_batch({k: v.to(device) for k, v in batch.items()}, ablate)
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
        losses.append(loss.item()); ades.append(ade); fdes.append(fde); sizes.append(len(chunk))
        batch_count += 1
        if verbose_every and batch_count % verbose_every == 0:
            elapsed = time.time() - start_time
            print(f"    [{mode}] batch {batch_count} | loss={loss.item():.3f} "
                  f"ade={ade:.3f} fde={fde:.3f} | {elapsed:.0f}s elapsed")

    for i, scenario in enumerate(scenario_iter):
        if i >= max_scenarios:
            break
        buffer.extend(build_examples_from_scenario(scenario))
        while len(buffer) >= batch_size:
            chunk, buffer = buffer[:batch_size], buffer[batch_size:]
            flush(chunk)
    if buffer:  # remainder
        flush(buffer)

    n = np.array(sizes, dtype=float)
    wmean = lambda xs: float(np.dot(xs, n) / n.sum()) if n.size else float("nan")
    return {"loss": wmean(losses), "ade": wmean(ades), "fde": wmean(fdes),
            "n_examples": int(n.sum()), "map_truncated_frac": n_trunc / max(n.sum(), 1)}
