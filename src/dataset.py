"""
Turns a stream of Waymax scenarios into (scenario, t) example indices with a
fixed stride, and collates extracted examples into batched torch tensors.

Split strategy: WOMD already ships separate training/validation tfrecord
directories -- use those for train/val (no manual splitting needed, and no
leakage risk). "Test" is carved out of the validation stream by alternating
scenario index, since WOMD's official test split has no labels (unusable for
us). This is a pragmatic choice, not a standard one -- worth knowing.
"""

import numpy as np
import torch

NUM_FRAMES = 91  # WOMD scenario length at 10Hz


def get_window_starts(hist_len=10, future_len=30, stride=10):
    """Valid t values within one scenario, strided to reduce redundancy
    between near-identical consecutive frames."""
    t0 = hist_len - 1
    t1 = NUM_FRAMES - future_len - 1
    return list(range(t0, t1, stride))


def val_test_split(scenario_index, val_fraction=0.5):
    """Alternate validation scenarios between val/test by index parity."""
    return "val" if (scenario_index % 2 == 0) else "test"


def collate_examples(examples):
    """List of dicts (from extract.extract_example) -> dict of batched tensors."""
    out = {}
    keys = examples[0].keys()
    for k in keys:
        arr = np.stack([ex[k] for ex in examples])
        if k in ("agent_class", "map_type", "light_state"):
            out[k] = torch.tensor(arr, dtype=torch.long)
        elif k in ("agent_mask", "map_mask", "light_mask"):
            out[k] = torch.tensor(arr, dtype=torch.bool)
        else:
            out[k] = torch.tensor(arr, dtype=torch.float32)
    target = out.pop("target")
    return out, target
