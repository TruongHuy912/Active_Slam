"""ASLAM utility core ported from aslam_rosbot.

Ported from:
- aslam_rosbot/scripts/functions.py::compute_entropy()
- aslam_rosbot/scripts/functions.py::cellInformation()
- aslam_rosbot/scripts/functions.py::count_digits_before_decimal()
- aslam_rosbot/scripts/controller_graphD.py::hallucinateGraph()
- aslam_rosbot/scripts/weighted_pose_graph_class.py::compute_anchored_L()

ROS 2/Nav2 adaptation: planner paths come from ComputePathToPose instead of
aslam_rosbot/scripts/functions.py::robot.makePlan(). Bumper-Bot currently has no
live g2o pose-graph source, so the spanning-tree term is computed from a
hallucinated path-only graph. This preserves the ASLAM utility formula while
making the missing pose graph explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from geometry_msgs.msg import PoseStamped

from bumperbot_active_slam.entropy_utils import (
    Cell,
    GridMeta,
    Point2D,
    bresenham,
    cell_entropy,
    euclidean_distance,
    map_to_world,
    occupancy_probability,
    path_occupancy_values,
    world_to_map,
)


@dataclass(frozen=True)
class AslamUtility:
    utility: float
    spann: float
    inv_entropy: float
    eta: float
    gamma: float
    entropy: float
    distance: float
    path_cells: list[Cell]
    pose_graph_fallback: bool


def compute_entropy(
    data: Sequence[int],
    meta: GridMeta,
    robot_xy: Point2D,
    frontier_xy: Point2D,
    *,
    unknown_probability: float = 0.1,
    known_probability: float = 0.45,
) -> tuple[float, float, list[Cell]]:
    """Ported from aslam_rosbot/scripts/functions.py::compute_entropy()."""

    start = world_to_map(robot_xy[0], robot_xy[1], meta)
    end = world_to_map(frontier_xy[0], frontier_xy[1], meta)
    if start is None or end is None:
        return 1.0, 0.0, []

    path_cells = bresenham(start, end)
    values = path_occupancy_values(data, path_cells, meta)
    entropy = 0.0
    for value in values:
        probability = occupancy_probability(
            value,
            unknown_probability=unknown_probability,
            known_probability=known_probability,
        )
        entropy += cell_entropy(probability)
    if values:
        entropy /= float(len(values))
    inv_entropy = max(0.0, min(1.0, 1.0 - entropy))
    return entropy, inv_entropy, path_cells


def cell_information(data: Sequence[int], meta: GridMeta, point: Point2D, radius: float) -> tuple[float, float]:
    """Ported from aslam_rosbot/scripts/functions.py::cellInformation()."""

    center = world_to_map(point[0], point[1], meta)
    if center is None or meta.resolution <= 0.0:
        return 0.0, 0.0
    radius_cells = int(radius // meta.resolution)
    cells_total = 0
    cells_unknown = 0
    cells_occupied = 0
    cx, cy = center
    for my in range(cy - radius_cells, cy + radius_cells + 1):
        for mx in range(cx - radius_cells, cx + radius_cells + 1):
            if not (0 <= mx < meta.width and 0 <= my < meta.height):
                continue
            xy = map_to_world(mx, my, meta, center=False)
            if euclidean_distance(point, xy) > radius:
                continue
            cells_total += 1
            value = int(data[my * meta.width + mx])
            if value == -1:
                cells_unknown += 1
            elif value == 100:
                cells_occupied += 1
    if cells_total == 0:
        return 0.0, 0.0
    return cells_unknown / float(cells_total), cells_occupied / float(cells_total)


def compute_aslam_utility(
    data: Sequence[int],
    meta: GridMeta,
    robot_xy: Point2D,
    frontier_xy: Point2D,
    planner_poses: Sequence[PoseStamped],
    *,
    info_radius: float = 1.0,
    lambda_decay: float = 0.6,
    unknown_probability: float = 0.1,
    known_probability: float = 0.45,
) -> AslamUtility:
    """Compute controller_graphD.py style utility.

    Original final formula:
    ``utility = spann + (inv_entropy * eta) + gamma``.
    """

    entropy, inv_entropy, path_cells = compute_entropy(
        data,
        meta,
        robot_xy,
        frontier_xy,
        unknown_probability=unknown_probability,
        known_probability=known_probability,
    )
    distance = euclidean_distance(robot_xy, frontier_xy)
    gamma = math.exp(-lambda_decay * distance)
    spann = hallucinated_spanning_tree_score(data, meta, planner_poses, info_radius=info_radius)
    eta = 10.0 ** float(count_digits_before_decimal(spann)) if spann > 0.0 else 1.0
    utility = spann + (inv_entropy * eta) + gamma
    return AslamUtility(
        utility=utility,
        spann=spann,
        inv_entropy=inv_entropy,
        eta=eta,
        gamma=gamma,
        entropy=entropy,
        distance=distance,
        path_cells=path_cells,
        pose_graph_fallback=True,
    )


def hallucinated_spanning_tree_score(
    data: Sequence[int],
    meta: GridMeta,
    planner_poses: Sequence[PoseStamped],
    *,
    info_radius: float,
    plan_point_threshold: int = 100,
) -> float:
    """Path-only adaptation of controller_graphD.py::hallucinateGraph().

    Without a live g2o graph, build a simple chain graph over sampled planner
    poses. Edge weights use ``1 + normalized_unknown_region_info`` as in the
    original odometry FIM scaling. The Matrix Tree theorem is still applied via
    an anchored weighted Laplacian.
    """

    poses = list(planner_poses)
    if len(poses) <= 1:
        return 0.0

    sample_count = max(1, int(math.ceil(len(poses) / float(max(1, plan_point_threshold)))))
    step = max(1, len(poses) // sample_count)
    indices = list(range(step, len(poses), step))
    if not indices or indices[-1] != len(poses) - 1:
        indices.append(len(poses) - 1)

    # Include the current graph start node and sampled hallucinated nodes.
    n_nodes = len(indices) + 1
    edges: list[tuple[int, int, float]] = []
    previous_id = 0
    for edge_id, pose_index in enumerate(indices, start=1):
        pose = poses[pose_index].pose.position
        point = (float(pose.x), float(pose.y))
        unknown_gain, lc_gain = cell_information(data, meta, point, info_radius)
        weight = max(1.0e-9, 1.0 + unknown_gain)
        # Small LC-inspired boost when occupied structure is visible, matching
        # the intent of controller_graphD.py without inventing extra topology.
        if lc_gain > 0.03 and edge_id > 1:
            weight += 0.1 * lc_gain
        edges.append((previous_id, edge_id, weight))
        previous_id = edge_id

    return spanning_tree_score(n_nodes, edges)


def spanning_tree_score(n_nodes: int, edges: Sequence[tuple[int, int, float]]) -> float:
    """Port of weighted_pose_graph_class anchored-Laplacian logdet score."""

    if n_nodes <= 1 or not edges:
        return 0.0
    lap = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    for u, v, weight in edges:
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            continue
        w = max(0.0, float(weight))
        lap[u][u] += w
        lap[v][v] += w
        lap[u][v] -= w
        lap[v][u] -= w

    anchored = [row[1:] for row in lap[1:]]
    logdet = log_determinant_spd(anchored)
    if logdet is None:
        return 0.0
    return (float(n_nodes) ** (1.0 / float(n_nodes))) * math.exp(logdet / float(n_nodes))


def log_determinant_spd(matrix: list[list[float]]) -> float | None:
    """Numerically small Gaussian-elimination logdet for positive matrices."""

    n = len(matrix)
    if n == 0:
        return None
    a = [row[:] for row in matrix]
    logdet = 0.0
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        pivot = a[pivot_row][i]
        if abs(pivot) <= 1.0e-12:
            return None
        if pivot_row != i:
            a[i], a[pivot_row] = a[pivot_row], a[i]
        pivot = a[i][i]
        if pivot <= 0.0:
            return None
        logdet += math.log(pivot)
        for r in range(i + 1, n):
            factor = a[r][i] / pivot
            if factor == 0.0:
                continue
            for c in range(i, n):
                a[r][c] -= factor * a[i][c]
    return logdet


def count_digits_before_decimal(number: float) -> int:
    """Ported from aslam_rosbot/scripts/functions.py::count_digits_before_decimal()."""

    if not math.isfinite(number) or number <= 0.0:
        return 0
    return len(str(number).split(".", maxsplit=1)[0])
