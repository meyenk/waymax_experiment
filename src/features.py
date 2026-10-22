"""
Shared scene-selection and feature-building logic, used by BOTH extract.py
(training/eval, reads the log) and closed_loop.py (reads our own simulated
states). Previously each file had its own copy of the padding helper, the
MAX_* constants and the feature layout -- a known duplication risk flagged
in CLAUDE.md. Anything that decides *what the model sees* lives here now, so
training and the closed loop can't silently drift apart.

Region of interest (ROI), decided at the current decision instant t:
  - agents: every non-ego object that is valid AT t and within AGENT_RADIUS
    of the ego at t, ranked nearest-first, capped at MAX_AGENTS. Its 1s
    history window is then filled in: frames where the tracker dropped it
    (occlusion etc.) are linearly interpolated between the nearest observed
    frames, and frames before it was first seen hold its earliest observed
    state. A 7th feature channel, `observed`, marks which frames were real
    (1) vs filled (0), so the model can tell the difference.
    Before: an agent had to be valid in ALL hist_len frames or it was
    dropped entirely, there was no distance filter or ranking, and the cap
    took whichever 32 came first in array order.
  - map points: every valid roadgraph point within MAP_RADIUS of the ego,
    ranked nearest-first, capped at MAX_MAP_POINTS (set high enough that the
    cap should rarely bind -- the training notebook measures this). Before:
    the first 200 within 80m in ARRAY order, i.e. an arbitrary subset.
  - traffic lights: valid at t and within MAP_RADIUS, nearest-first, capped
    at MAX_LIGHTS. Frames where a light wasn't observed get state 0
    (UNKNOWN in WOMD) and its current position. Before: a light had to be
    valid in every history frame.

The constants are module globals read at call time, so a notebook can do
`features.MAX_MAP_POINTS = ...` before building examples. None of them are
part of the model's weights (every branch pools over a variable-size set),
so changing them never invalidates a checkpoint -- only AGENT_FEAT_DIM does.
"""

import numpy as np

from src.transforms import transform_positions, transform_velocities, transform_yaw

AGENT_RADIUS = 50.0      # m, fixed for now (could be made speed-dependent later)
MAX_AGENTS = 32
MAP_RADIUS = 80.0        # m
MAX_MAP_POINTS = 6000    # high on purpose: cover everything in the radius, don't subsample
MAX_LIGHTS = 16
AGENT_FEAT_DIM = 7       # [x, y, cos_yaw, sin_yaw, vx_rel, vy_rel, observed]


def pad(arr, max_n):
    """Pad first axis of arr with zeros up to max_n; return (padded, mask).
    Callers pass arrays already ranked by priority, so truncation (if any)
    drops the least important items."""
    n = arr.shape[0]
    mask = np.zeros(max_n, dtype=bool)
    if n == 0:
        pad_shape = (max_n,) + arr.shape[1:]
        return np.zeros(pad_shape, dtype=arr.dtype), mask
    mask[:min(n, max_n)] = True
    if n >= max_n:
        return arr[:max_n], mask
    pad_shape = (max_n - n,) + arr.shape[1:]
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0), mask


def nearest_within(dist, radius, max_n=None):
    """Indices with dist <= radius, sorted nearest-first, optionally capped."""
    idx = np.where(dist <= radius)[0]
    idx = idx[np.argsort(dist[idx], kind="stable")]
    return idx if max_n is None else idx[:max_n]


# ---------------------------------------------------------------------------
# Gap filling (tracker dropouts)
# ---------------------------------------------------------------------------

def fill_gaps(values, valid, angle_channels=()):
    """values: (N, T, C) per-object time series, valid: (N, T) bool.
    Returns a copy where every invalid frame is linearly interpolated from
    the nearest valid frames on either side (np.interp semantics: frames
    before the first / after the last valid frame hold that edge value).
    Channels listed in angle_channels are unwrapped before interpolating and
    re-wrapped to (-pi, pi] after, so a heading crossing +-pi doesn't spin
    the wrong way round. Rows with no valid frame at all are returned as-is."""
    values = np.array(values, dtype=float, copy=True)
    valid = np.asarray(valid, dtype=bool)
    N, T, C = values.shape
    ts = np.arange(T)
    for i in range(N):
        v = valid[i]
        if v.all() or not v.any():
            continue
        for c in range(C):
            known = values[i, v, c]
            if c in angle_channels:
                known = np.unwrap(known)
            filled = np.interp(ts, ts[v], known)
            if c in angle_channels:
                filled = (filled + np.pi) % (2 * np.pi) - np.pi
            values[i, ~v, c] = filled[~v]
    return values


def valid_runs(valid_row):
    """(start, end) inclusive index pairs of each contiguous True run."""
    v = np.asarray(valid_row, dtype=bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], v, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def bridge_short_gaps(valid, max_gap):
    """valid: (N, T). Returns a copy where interior invalid stretches of at
    most max_gap frames (bounded by valid frames on BOTH sides) are marked
    valid -- these are tracker dropouts, the object didn't actually leave.
    Longer gaps, and anything before the first / after the last valid frame,
    stay invalid: the object really is gone (or not there yet)."""
    out = np.array(valid, dtype=bool, copy=True)
    for i in range(out.shape[0]):
        runs = valid_runs(out[i])
        for (_, e0), (s1, _) in zip(runs[:-1], runs[1:]):
            if s1 - e0 - 1 <= max_gap:
                out[i, e0 + 1:s1] = True
    return out


# ---------------------------------------------------------------------------
# Feature building (pure numpy, no scenario object -- both callers use this)
# ---------------------------------------------------------------------------

def agent_features(hist_xy, hist_yaw, hist_vxvy, observed,
                   ego_hist_xy, ego_hist_yaw, ego_hist_vxvy, time_matched=True):
    """All world frame. hist_*: (N, T, ...) agents, ego_hist_*: (T, ...).
    time_matched=True (Task 2 / tau): history frame tau is expressed relative
    to the ego's own state at tau. time_matched=False (pre-Task-2 / t): every
    frame uses the ego's state at the current instant (the window's last
    entry) -- kept only so the before/after-Task-2 comparison can be rerun
    on identical code. Returns (N, T, AGENT_FEAT_DIM) float32."""
    if time_matched:
        ref_xy, ref_yaw, ref_v = ego_hist_xy, ego_hist_yaw, ego_hist_vxvy
    else:
        ref_xy, ref_yaw, ref_v = ego_hist_xy[-1], float(ego_hist_yaw[-1]), ego_hist_vxvy[-1]
    xy = transform_positions(hist_xy, ref_xy, ref_yaw)
    yaw = transform_yaw(hist_yaw, ref_yaw)
    v_rel = transform_velocities(hist_vxvy, ref_v, ref_yaw)
    return np.concatenate(
        [xy, np.cos(yaw)[..., None], np.sin(yaw)[..., None], v_rel,
         np.asarray(observed, dtype=float)[..., None]], axis=-1).astype(np.float32)


def select_map(map_xy_world, map_types, ego_xy, ego_yaw):
    """Roadgraph points within MAP_RADIUS, nearest-first, in the ego frame at
    t. Returns (xy_local (n,2) float32, types (n,), n_in_radius) -- the last
    one is how many were in range BEFORE the cap, for truncation stats."""
    xy_local = transform_positions(map_xy_world, ego_xy, ego_yaw)
    dist = np.linalg.norm(xy_local, axis=-1)
    in_radius = nearest_within(dist, MAP_RADIUS)
    keep = in_radius[:MAX_MAP_POINTS]
    return xy_local[keep].astype(np.float32), map_types[keep], len(in_radius)


def ego_window(xy, yaw, vxvy, valid):
    """Ego's own (T,2)/(T,)/(T,2) history with any invalid frames filled the
    same way as agents. Before, an invalid ego frame inside the window was
    used as-is (Waymax fills those with sentinel values), which would have
    corrupted every time-matched agent feature at that frame."""
    vals = np.concatenate([xy, yaw[:, None], vxvy], axis=-1)[None]
    vals = fill_gaps(vals, np.asarray(valid, dtype=bool)[None], angle_channels=(2,))[0]
    return vals[:, :2], vals[:, 2], vals[:, 3:5]


def light_window_world(tl, idx, t0, t):
    """World-frame (n, T, 2) positions and (n, T) states for lights idx over
    [t0, t] from a log_traffic_light-like object. Unobserved frames get the
    light's current position and state 0 (UNKNOWN in WOMD)."""
    w = slice(t0, t + 1)
    valid = np.array(tl.valid)[idx, w]
    xy = np.stack([np.array(tl.x)[idx, w], np.array(tl.y)[idx, w]], axis=-1).astype(np.float32)
    state = np.array(tl.state)[idx, w].astype(np.int64)
    xy = np.where(valid[..., None], xy, xy[:, -1:, :])
    state = np.where(valid, state, 0)
    return xy, state
