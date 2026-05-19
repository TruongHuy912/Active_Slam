"""RRT frontier detection helpers.

Ported from aslam_rosbot/src/global_rrt_detector.cpp,
aslam_rosbot/src/local_rrt_detector.cpp, and aslam_rosbot/src/functions.cpp.

The original ROS 1 nodes publish PointStamped messages on /detected_points.
This module keeps the same algorithmic core ROS-independent: random sampling,
Nearest, Steer, and ObstacleFree over an OccupancyGrid-like array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Optional, Sequence

from bumperbot_active_slam.entropy_utils import GridMeta, Point2D, world_to_map


@dataclass
class RrtFrontierDetector:
    """Combined global/local RRT frontier detector.

    Global RRT keeps its tree between calls. Local RRT resets its tree to the
    robot pose whenever a frontier is detected, matching local_rrt_detector.cpp.
    """

    eta: float = 0.5
    samples_per_cycle: int = 80
    seed: Optional[int] = None
    global_tree: list[Point2D] = field(default_factory=list)
    local_tree: list[Point2D] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self.global_tree.clear()
        self.local_tree.clear()

    def detect(
        self,
        data: Sequence[int],
        meta: GridMeta,
        robot_xy: Point2D,
        *,
        include_global: bool = True,
        include_local: bool = True,
    ) -> list[Point2D]:
        """Return raw RRT frontier points in world coordinates.

        ObstacleFree semantics are ported from aslam_rosbot/src/functions.cpp:
        1 means free, 0 means occupied, -1 means first unknown cell on the ray.
        """

        if meta.width <= 0 or meta.height <= 0 or meta.resolution <= 0.0:
            return []

        if include_global and not self.global_tree:
            self.global_tree.append(robot_xy)
        if include_local and not self.local_tree:
            self.local_tree.append(robot_xy)

        frontiers: list[Point2D] = []
        for _ in range(max(1, self.samples_per_cycle)):
            sample = self._sample_in_map(meta)
            if include_global:
                frontier = self._extend_tree(data, meta, self.global_tree, sample, reset_to=None)
                if frontier is not None:
                    frontiers.append(frontier)
            if include_local:
                frontier = self._extend_tree(data, meta, self.local_tree, sample, reset_to=robot_xy)
                if frontier is not None:
                    frontiers.append(frontier)
        return frontiers

    def _extend_tree(
        self,
        data: Sequence[int],
        meta: GridMeta,
        tree: list[Point2D],
        sample: Point2D,
        *,
        reset_to: Optional[Point2D],
    ) -> Optional[Point2D]:
        nearest = nearest_point(tree, sample)
        xnew = steer(nearest, sample, self.eta)
        if world_to_map(xnew[0], xnew[1], meta) is None:
            return None

        status, checked_point = obstacle_free(nearest, xnew, data, meta)
        if status == -1:
            if reset_to is not None:
                tree.clear()
                tree.append(reset_to)
            return checked_point
        if status == 1:
            tree.append(checked_point)
        return None

    def _sample_in_map(self, meta: GridMeta) -> Point2D:
        x = self._rng.uniform(meta.origin_x, meta.origin_x + meta.width * meta.resolution)
        y = self._rng.uniform(meta.origin_y, meta.origin_y + meta.height * meta.resolution)
        return x, y


def nearest_point(points: Sequence[Point2D], target: Point2D) -> Point2D:
    """Ported from aslam_rosbot/src/functions.cpp::Nearest()."""

    if not points:
        return target
    return min(points, key=lambda point: norm(point, target))


def steer(x_nearest: Point2D, x_rand: Point2D, eta: float) -> Point2D:
    """Ported from aslam_rosbot/src/functions.cpp::Steer()."""

    distance = norm(x_nearest, x_rand)
    if distance <= eta or distance <= 1.0e-12:
        return x_rand
    scale = eta / distance
    return (
        x_nearest[0] + (x_rand[0] - x_nearest[0]) * scale,
        x_nearest[1] + (x_rand[1] - x_nearest[1]) * scale,
    )


def obstacle_free(
    xnear: Point2D,
    xnew: Point2D,
    data: Sequence[int],
    meta: GridMeta,
) -> tuple[int, Point2D]:
    """Ported from aslam_rosbot/src/functions.cpp::ObstacleFree().

    Returns ``(status, point)`` where status is ``1`` for free, ``0`` for
    occupied, and ``-1`` for frontier/unknown. The point is the last checked
    point, matching the C++ code's mutation of xnew to ``xi``.
    """

    step = max(meta.resolution * 0.2, 1.0e-6)
    steps = max(1, int(math.ceil(norm(xnew, xnear) / step)))
    xi = xnear
    saw_obstacle = False
    saw_unknown = False

    for _ in range(steps):
        xi = steer(xi, xnew, step)
        value = grid_value(data, meta, xi)
        if value == 100:
            saw_obstacle = True
        if value == -1:
            saw_unknown = True
            break

    # Match the original C++ assignment order: an obstacle seen anywhere on
    # the segment rejects the sample even if an unknown cell was also reached.
    if saw_obstacle:
        return 0, xi
    if saw_unknown:
        return -1, xi
    return 1, xi


def grid_value(data: Sequence[int], meta: GridMeta, point: Point2D) -> int:
    cell = world_to_map(point[0], point[1], meta)
    if cell is None:
        return 100
    mx, my = cell
    return int(data[my * meta.width + mx])


def norm(a: Point2D, b: Point2D) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
