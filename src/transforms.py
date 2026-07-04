"""Ego-centric frame transforms. Every positional/velocity field in the
pipeline goes through these, anchored at the ego's state at time t."""

import numpy as np


def transform_positions(xy, ego_xy, ego_yaw):
    """World xy -> ego-centered, ego-heading-aligned frame. xy: (..., 2)."""
    dxy = xy - ego_xy
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    rot = np.array([[c, -s], [s, c]])
    return dxy @ rot.T


def transform_velocities(vxvy, ego_vxvy, ego_yaw):
    """World velocity -> ego-relative, ego-heading-aligned (closing/lateral speed)."""
    rel = vxvy - ego_vxvy
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    rot = np.array([[c, -s], [s, c]])
    return rel @ rot.T


def transform_yaw(yaw, ego_yaw):
    return yaw - ego_yaw


def inverse_transform_positions(local_xy, ego_xy, ego_yaw):
    """World xy <- ego-frame local xy. Exact inverse of transform_positions,
    used to bring model predictions (made in ego frame) back to world
    coordinates for plotting on top of the real scene."""
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    rot = np.array([[c, -s], [s, c]])
    return local_xy @ rot.T + ego_xy
