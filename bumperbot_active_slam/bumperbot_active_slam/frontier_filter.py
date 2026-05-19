"""Frontier filtering helpers.

Ported from aslam_rosbot/scripts/filter.py.
Ported from aslam_rosbot/scripts/functions.py::informationGain().
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from nav_msgs.msg import OccupancyGrid, Path

from bumperbot_active_slam.entropy_utils import (
    Cell,
    GridMeta,
    Point2D,
    bresenham,
    grid_meta_from_map_info,
    map_to_world,
    world_to_map,
)


def information_gain(map_msg: OccupancyGrid, meta: GridMeta, point: Point2D, radius: float) -> float:
    """Ported from aslam_rosbot/scripts/functions.py::informationGain()."""

    center = world_to_map(point[0], point[1], meta)
    if center is None or meta.resolution <= 0.0:
        return 0.0

    radius_cells = int(radius // meta.resolution)
    info_gain = 0
    cx, cy = center
    for my in range(cy - radius_cells, cy + radius_cells + 1):
        for mx in range(cx - radius_cells, cx + radius_cells + 1):
            if not (0 <= mx < meta.width and 0 <= my < meta.height):
                continue
            wx, wy = map_to_world(mx, my, meta, center=False)
            if math.hypot(point[0] - wx, point[1] - wy) > radius:
                continue
            if int(map_msg.data[my * meta.width + mx]) == -1:
                info_gain += 1
    return info_gain * (meta.resolution ** 2)


def costmap_value_at(
    costmap: Optional[OccupancyGrid],
    global_frame: str,
    xy: Point2D,
) -> tuple[Optional[int], str]:
    """Return costmap value at a world point using filter.py threshold semantics."""

    if costmap is None:
        return None, "waiting_costmap"
    if costmap.header.frame_id and costmap.header.frame_id != global_frame:
        return None, "costmap_frame_mismatch"
    meta = grid_meta_from_map_info(costmap.info)
    cell = world_to_map(xy[0], xy[1], meta)
    if cell is None:
        return None, "costmap_rejected"
    x, y = cell
    return int(costmap.data[y * meta.width + x]), ""


def is_acceptable_cost(cost: int, threshold: int, allow_unknown_goal: bool) -> bool:
    """Ported from filter.py costmap_clearing_threshold logic."""

    if cost < 0:
        return allow_unknown_goal
    return cost < threshold


def path_has_blocked_costmap_cell(
    path: Path,
    costmap: Optional[OccupancyGrid],
    global_frame: str,
    threshold: int,
    allow_unknown_goal: bool,
) -> bool:
    if costmap is None:
        return True
    if costmap.header.frame_id and costmap.header.frame_id != global_frame:
        return True

    meta = grid_meta_from_map_info(costmap.info)
    previous_cell: Optional[Cell] = None
    for pose in path.poses:
        xy = (float(pose.pose.position.x), float(pose.pose.position.y))
        cell = world_to_map(xy[0], xy[1], meta)
        if cell is None:
            return True
        segment = [cell] if previous_cell is None else bresenham(previous_cell, cell)
        for sx, sy in segment:
            if not (0 <= sx < meta.width and 0 <= sy < meta.height):
                return True
            cost = int(costmap.data[sy * meta.width + sx])
            if not is_acceptable_cost(cost, threshold, allow_unknown_goal):
                return True
        previous_cell = cell
    return False


def path_has_occupied_cell(data: Sequence[int], meta: GridMeta, path_cells: list[Cell]) -> bool:
    for x, y in path_cells:
        if not (0 <= x < meta.width and 0 <= y < meta.height):
            return True
        if int(data[y * meta.width + x]) >= 100:
            return True
    return False


def is_free_cell(data: Sequence[int], meta: GridMeta, cell: Cell) -> bool:
    x, y = cell
    if not (0 <= x < meta.width and 0 <= y < meta.height):
        return False
    return int(data[y * meta.width + x]) == 0


def find_nearest_free_cell(
    data: Sequence[int],
    meta: GridMeta,
    start_cell: Cell,
    search_radius_cells: int,
) -> Optional[Cell]:
    if is_free_cell(data, meta, start_cell):
        return start_cell

    sx, sy = start_cell
    max_radius = max(0, search_radius_cells)
    best_cell: Optional[Cell] = None
    best_dist_sq: Optional[int] = None

    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                cell = (sx + dx, sy + dy)
                if not is_free_cell(data, meta, cell):
                    continue
                dist_sq = dx * dx + dy * dy
                if best_dist_sq is None or dist_sq < best_dist_sq:
                    best_cell = cell
                    best_dist_sq = dist_sq
        if best_cell is not None:
            return best_cell
    return None


class FrontierPointFilter:
    """filter.py-style raw frontier buffer, clustering, and filtering.

    Ported from aslam_rosbot/scripts/filter.py::frontiersCallBack() and
    aslam_rosbot/scripts/filter.py::node(). The original used
    sklearn.cluster.MeanShift(bandwidth=1.5). To avoid adding a runtime
    dependency, this class uses a deterministic radius-merge approximation with
    the same bandwidth parameter.
    """

    def __init__(
        self,
        *,
        timeout_sec: float = 2.0,
        merge_radius: float = 1.5,
        min_cluster_size: int = 1,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.merge_radius = merge_radius
        self.min_cluster_size = max(1, min_cluster_size)
        self._points: list[tuple[Point2D, float]] = []

    def add_points(self, points: Sequence[Point2D], stamp_sec: float) -> None:
        """Store raw frontiers and refresh duplicate timestamps."""

        for point in points:
            self._upsert((float(point[0]), float(point[1])), stamp_sec)

    def prune(self, now_sec: float) -> None:
        self._points = [entry for entry in self._points if abs(now_sec - entry[1]) <= self.timeout_sec]

    def raw_points(self, now_sec: float) -> list[Point2D]:
        self.prune(now_sec)
        return [point for point, _stamp in self._points]

    def clustered_points(
        self,
        now_sec: float,
        *,
        mode: str = "radius_merge",
        meanshift_bandwidth: float = 1.5,
        meanshift_max_iterations: int = 20,
        meanshift_tolerance: float = 0.05,
        meanshift_min_cluster_size: int = 1,
    ) -> list[Point2D]:
        """Return clustered centroids for buffered raw frontiers.

        ``radius_merge`` preserves the stable ROS 2 baseline.
        ``mean_shift_like`` is a lightweight, dependency-free approximation of
        aslam_rosbot/scripts/filter.py MeanShift(bandwidth=1.5).
        """

        points = self.raw_points(now_sec)
        if len(points) <= 1:
            return points
        if mode == "mean_shift_like":
            return mean_shift_like_points(
                points,
                bandwidth=meanshift_bandwidth,
                max_iterations=meanshift_max_iterations,
                tolerance=meanshift_tolerance,
                min_cluster_size=meanshift_min_cluster_size,
            )
        return radius_merge_points(points, self.merge_radius, self.min_cluster_size)

    def filtered_points(
        self,
        now_sec: float,
        map_msg: OccupancyGrid,
        meta: GridMeta,
        costmap: Optional[OccupancyGrid],
        global_frame: str,
        *,
        costmap_threshold: int = 70,
        allow_unknown_goal: bool = False,
        info_radius: float = 1.0,
        information_threshold: float = 0.45,
    ) -> tuple[list[Point2D], dict[str, int]]:
        """Apply filter.py costmap and information-gain rejection logic."""

        accepted: list[Point2D] = []
        rejected: dict[str, int] = {}
        for point in self.clustered_points(now_sec):
            cost, reason = costmap_value_at(costmap, global_frame, point)
            if cost is None or not is_acceptable_cost(cost, costmap_threshold, allow_unknown_goal):
                rejected[reason or "costmap_rejected"] = rejected.get(reason or "costmap_rejected", 0) + 1
                continue
            gain = information_gain(map_msg, meta, point, info_radius)
            if gain < information_threshold:
                rejected["low_information_gain"] = rejected.get("low_information_gain", 0) + 1
                continue
            accepted.append(point)
        return accepted, rejected

    def _upsert(self, point: Point2D, stamp_sec: float) -> None:
        for index, (stored, _stamp) in enumerate(self._points):
            if math.hypot(stored[0] - point[0], stored[1] - point[1]) <= 1.0e-9:
                self._points[index] = (stored, stamp_sec)
                return
        self._points.append((point, stamp_sec))


def radius_merge_points(points: Sequence[Point2D], merge_radius: float, min_cluster_size: int = 1) -> list[Point2D]:
    """Stable radius-merge clustering used by the ROS 2 baseline."""

    if len(points) <= 1:
        return list(points)
    remaining = set(range(len(points)))
    centroids: list[Point2D] = []
    while remaining:
        seed = min(remaining)
        queue = [seed]
        remaining.remove(seed)
        members: list[int] = []
        while queue:
            idx = queue.pop()
            members.append(idx)
            px, py = points[idx]
            close = [
                other for other in list(remaining)
                if math.hypot(points[other][0] - px, points[other][1] - py) <= merge_radius
            ]
            for other in close:
                remaining.remove(other)
                queue.append(other)
        if len(members) >= max(1, min_cluster_size):
            centroids.append(mean_point([points[idx] for idx in members]))
    return centroids


def mean_shift_like_points(
    points: Sequence[Point2D],
    *,
    bandwidth: float = 1.5,
    max_iterations: int = 20,
    tolerance: float = 0.05,
    min_cluster_size: int = 1,
) -> list[Point2D]:
    """Dependency-free approximation of filter.py MeanShift(bandwidth=1.5)."""

    if len(points) <= 1:
        return list(points)
    radius = max(1.0e-6, float(bandwidth))
    shifted: list[Point2D] = []
    for point in points:
        center = (float(point[0]), float(point[1]))
        for _ in range(max(1, int(max_iterations))):
            neighbors = [candidate for candidate in points if point_distance(center, candidate) <= radius]
            if not neighbors:
                break
            new_center = mean_point(neighbors)
            if point_distance(center, new_center) <= max(0.0, float(tolerance)):
                center = new_center
                break
            center = new_center
        shifted.append(center)

    centers: list[tuple[Point2D, int]] = []
    merge_distance = max(max(0.0, float(tolerance)), radius * 0.5)
    for center in shifted:
        for index, (existing, count) in enumerate(centers):
            if point_distance(center, existing) <= merge_distance:
                total = float(count + 1)
                merged = (
                    (existing[0] * count + center[0]) / total,
                    (existing[1] * count + center[1]) / total,
                )
                centers[index] = (merged, count + 1)
                break
        else:
            centers.append((center, 1))

    min_size = max(1, int(min_cluster_size))
    return [center for center, count in centers if count >= min_size]


def mean_point(points: Sequence[Point2D]) -> Point2D:
    count = float(len(points))
    if count <= 0.0:
        raise ValueError("Cannot compute mean of an empty point set")
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
    )


def point_distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clusters_from_points(points: Sequence[Point2D], meta: GridMeta) -> list[object]:
    """Build FrontierCluster objects from centroid points without ROS imports."""

    from bumperbot_active_slam.frontier_detector import FrontierCluster

    clusters = []
    for point in points:
        cell = world_to_map(point[0], point[1], meta)
        cells = tuple([cell]) if cell is not None else tuple()
        clusters.append(FrontierCluster(cells=cells, centroid=(float(point[0]), float(point[1]))))
    return clusters
