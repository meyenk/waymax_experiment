"""Ego-centric frame transforms.

Every positional/velocity field in the pipeline goes through these. Two
reference conventions are supported (Part B, task 2):
  - single reference: ego_xy is (2,), ego_yaw is a scalar -- one anchor
    broadcast across every entry of xy. Used by the map branch (a single
    static snapshot at t, no history to time-match) and the ego's own future
    target space.
  - windowed reference: ego_xy is (hist_len, 2), ego_yaw is (hist_len,) --
    one anchor PER historical instant tau, applied timestep-by-timestep.
    Used by the agent and traffic-light branches' history windows, so that
    entry tau is expressed relative to the ego's own state at tau, not at
    the current decision instant t. Required for velocity in particular:
    v_rel(tau) = agent_v(tau) - ego_v(t) would otherwise mix two different
    instants' velocities."""

import numpy as np


def transform_positions(xy, ego_xy, ego_yaw):
    """World xy -> ego-centered, ego-heading-aligned frame.
    Single reference: xy: (..., 2), ego_xy: (2,), ego_yaw: scalar.
    Windowed reference: xy: (..., hist_len, 2), ego_xy: (hist_len, 2),
    ego_yaw: (hist_len,) -- each of xy's hist_len entries is rotated/
    translated by that same-index entry of ego_xy/ego_yaw."""
    ego_yaw = np.asarray(ego_yaw)
    dxy = xy - ego_xy
    if ego_yaw.ndim == 0:
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        rot = np.array([[c, -s], [s, c]])
        return dxy @ rot.T
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)  # (hist_len,)
    out_x = dxy[..., 0] * c - dxy[..., 1] * s
    out_y = dxy[..., 0] * s + dxy[..., 1] * c
    return np.stack([out_x, out_y], axis=-1)


def transform_velocities(vxvy, ego_vxvy, ego_yaw):
    """World velocity -> ego-relative, ego-heading-aligned (closing/lateral
    speed). Same single-vs-windowed reference convention as
    transform_positions -- see module docstring."""
    ego_yaw = np.asarray(ego_yaw)
    rel = vxvy - ego_vxvy
    if ego_yaw.ndim == 0:
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        rot = np.array([[c, -s], [s, c]])
        return rel @ rot.T
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    out_x = rel[..., 0] * c - rel[..., 1] * s
    out_y = rel[..., 0] * s + rel[..., 1] * c
    return np.stack([out_x, out_y], axis=-1)


def transform_yaw(yaw, ego_yaw):
    """yaw - ego_yaw, wrapped to (-pi, pi]. ego_yaw may be a scalar (single
    reference) or windowed (matching yaw's history axis) -- broadcasts
    either way, no branching needed."""
    d = yaw - ego_yaw
    return (d + np.pi) % (2 * np.pi) - np.pi


def inverse_transform_positions(local_xy, ego_xy, ego_yaw):
    """World xy <- ego-frame local xy. Exact inverse of transform_positions,
    used to bring model predictions (made in ego frame) back to world
    coordinates for plotting on top of the real scene."""
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    rot = np.array([[c, -s], [s, c]])
    return local_xy @ rot.T + ego_xy
