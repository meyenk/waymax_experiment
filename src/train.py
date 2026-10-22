"""
Training loop over a stream of Waymax scenarios. Visualization lives in
src/visualize.py -- kept separate so it's easy to find/iterate on independently.

run_epoch_variants trains/evaluates several model variants on the SAME
scenarios in the SAME order -- each variant only differs in how features
are computed (e.g. time_matched=True vs False for the before/after-Task-2
comparison). Data loading is shared, so a fair A/B costs one data pass, not
two. run_epoch is the original single-model API, kept as a thin wrapper.
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


def build_examples_from_scenario(scenario, hist_len=10, future_len=30, stride=10, time_matched=True):
    """All valid (ego, t) examples from one scenario, given our window stride."""
    global SKIPPED_NO_SDC
    ego_idx = get_ego_idx(scenario)
    if ego_idx is None:
        SKIPPED_NO_SDC += 1
        return []
    examples = []
    for t in get_window_starts(hist_len, future_len, stride):
        ex = extract_example(scenario, ego_idx, t, hist_len, future_len, time_matched=time_matched)
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


def run_epoch_variants(variants, scenario_iter, max_scenarios, batch_size=16, train=True,
                        device="cpu", grad_clip_norm=1.0, verbose_every=5, ablate=None):
    """One pass over up to max_scenarios scenarios for every variant.
    variants: {name: {"model": ..., "optimizer": ... (None if not training),
    "time_matched": bool}}. Returns {name: {"loss", "ade", "fde",
    "n_examples", "map_truncated_frac"}} where map_truncated_frac is the
    share of examples with more map points in range than MAX_MAP_POINTS.
    Both pred (model output) and target (from collate_examples) are absolute
    positions in the ego frame at t -- same space, safe to compare directly."""
    for v in variants.values():
        v["model"].train() if train else v["model"].eval()
    mode = "train" if train else "val"
    buffers = {name: [] for name in variants}
    stats = {name: {"loss": [], "ade": [], "fde": [], "n": [], "trunc": 0, "count": 0}
             for name in variants}
    batch_count = 0
    start_time = time.time()

    def flush(name, chunk):
        v, st = variants[name], stats[name]
        batch, target = collate_examples(chunk)
        n_in_radius = batch.pop("map_n_in_radius", None)
        if n_in_radius is not None:
            st["trunc"] += int((n_in_radius > batch["map_mask"].shape[1]).sum())
        st["count"] += len(chunk)
        batch = {k: t.to(device) for k, t in batch.items()}
        batch = ablate_batch(batch, ablate)
        target = target.to(device)
        if train:
            v["optimizer"].zero_grad()
            pred, _ = v["model"](batch)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v["model"].parameters(), grad_clip_norm)
            v["optimizer"].step()
        else:
            with torch.no_grad():
                pred, _ = v["model"](batch)
                loss = torch.mean((pred - target) ** 2)
        ade, fde = compute_ade_fde(pred.detach(), target)
        st["loss"].append(loss.item()); st["ade"].append(ade); st["fde"].append(fde)
        st["n"].append(len(chunk))
        return loss.item(), ade, fde

    for i, scenario in enumerate(scenario_iter):
        if i >= max_scenarios:
            break
        for name, v in variants.items():
            buffers[name].extend(build_examples_from_scenario(scenario, time_matched=v["time_matched"]))
        while all(len(b) >= batch_size for b in buffers.values()):
            msgs = []
            for name in variants:
                chunk, buffers[name] = buffers[name][:batch_size], buffers[name][batch_size:]
                loss, ade, fde = flush(name, chunk)
                msgs.append(f"{name}: loss={loss:.3f} ade={ade:.3f} fde={fde:.3f}")
            batch_count += 1
            if verbose_every and batch_count % verbose_every == 0:
                print(f"    [{mode}] batch {batch_count} | " + " | ".join(msgs)
                      + f" | {time.time() - start_time:.0f}s elapsed")
    for name in variants:  # remainder
        if buffers[name]:
            flush(name, buffers[name])

    out = {}
    for name, st in stats.items():
        n = np.array(st["n"], dtype=float)
        wmean = (lambda xs: float(np.sum(np.array(xs) * n) / n.sum())) if n.size else (lambda xs: float("nan"))
        out[name] = {"loss": wmean(st["loss"]), "ade": wmean(st["ade"]), "fde": wmean(st["fde"]),
                     "n_examples": int(n.sum()),
                     "map_truncated_frac": st["trunc"] / max(st["count"], 1)}
    return out


def run_epoch(model, optimizer, scenario_iter, max_scenarios, batch_size=16, train=True,
              device="cpu", grad_clip_norm=1.0, verbose_every=5, time_matched=True, ablate=None):
    """One pass over up to max_scenarios scenarios. Returns a dict with mean
    loss, ADE, and FDE (now weighted by batch size, so a small remainder
    batch no longer counts as much as a full one). Single-model wrapper
    around run_epoch_variants, same signature as before."""
    res = run_epoch_variants(
        {"model": {"model": model, "optimizer": optimizer, "time_matched": time_matched}},
        scenario_iter, max_scenarios, batch_size=batch_size, train=train, device=device,
        grad_clip_norm=grad_clip_norm, verbose_every=verbose_every, ablate=ablate)
    return res["model"]
