"""Optional frontier viewpoint goal generation.

ROS 2/Nav2 adaptation inspired by exploration systems that choose a safe
viewpoint to observe a frontier instead of navigating directly to the frontier
centroid.

Not enabled by default.
"""

from __future__ import annotations


def generate_viewpoint_candidates(*args, **kwargs):
    raise NotImplementedError("Frontier viewpoint generation is not implemented yet.")
