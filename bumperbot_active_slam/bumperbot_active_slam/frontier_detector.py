"""ROS-independent frontier detection for occupancy grids."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from bumperbot_active_slam.entropy_utils import Cell, GridMeta, Point2D, map_to_world


@dataclass(frozen=True)
class FrontierCluster:
    """A connected frontier component and its world-frame centroid."""

    cells: tuple[Cell, ...]
    centroid: Point2D

    @property
    def size(self) -> int:
        return len(self.cells)


def detect_frontier_cells(
    data: Sequence[int],
    meta: GridMeta,
    *,
    connectivity: int = 8,
    free_threshold: int = 0,
    unknown_value: int = -1,
) -> list[Cell]:
    """Find free cells adjacent to unknown cells."""

    _validate_grid(data, meta)
    offsets = neighbor_offsets(connectivity)
    frontiers: list[Cell] = []

    for my in range(meta.height):
        row = my * meta.width
        for mx in range(meta.width):
            index = row + mx
            if not _is_free(data[index], free_threshold):
                continue
            if any(
                _is_unknown_neighbor(data, meta, mx + dx, my + dy, unknown_value)
                for dx, dy in offsets
            ):
                frontiers.append((mx, my))

    return frontiers


def cluster_frontiers(
    frontier_cells: Iterable[Cell],
    meta: GridMeta,
    *,
    connectivity: int = 8,
    min_cluster_size: int = 5,
) -> list[FrontierCluster]:
    """Group frontier cells into connected components with BFS."""

    frontier_set = set(frontier_cells)
    visited: set[Cell] = set()
    offsets = neighbor_offsets(connectivity)
    clusters: list[FrontierCluster] = []

    for seed in sorted(frontier_set):
        if seed in visited:
            continue

        queue: deque[Cell] = deque([seed])
        visited.add(seed)
        component: list[Cell] = []

        while queue:
            cell = queue.popleft()
            component.append(cell)
            mx, my = cell
            for dx, dy in offsets:
                neighbor = (mx + dx, my + dy)
                if neighbor in frontier_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component) >= min_cluster_size:
            clusters.append(
                FrontierCluster(
                    cells=tuple(component),
                    centroid=frontier_centroid(component, meta),
                )
            )

    return clusters


def frontier_centroid(cells: Iterable[Cell], meta: GridMeta) -> Point2D:
    """Compute the world-frame centroid of a frontier cluster."""

    world_points = [map_to_world(mx, my, meta) for mx, my in cells]
    if not world_points:
        raise ValueError("Cannot compute centroid of an empty frontier cluster")

    count = float(len(world_points))
    x = sum(point[0] for point in world_points) / count
    y = sum(point[1] for point in world_points) / count
    return x, y


def detect_frontiers(
    data: Sequence[int],
    meta: GridMeta,
    *,
    connectivity: int = 8,
    min_cluster_size: int = 5,
    free_threshold: int = 0,
    unknown_value: int = -1,
) -> list[FrontierCluster]:
    """Detect, cluster, and summarize frontiers in one call."""

    cells = detect_frontier_cells(
        data,
        meta,
        connectivity=connectivity,
        free_threshold=free_threshold,
        unknown_value=unknown_value,
    )
    return cluster_frontiers(
        cells,
        meta,
        connectivity=connectivity,
        min_cluster_size=min_cluster_size,
    )


def detect_frontier_clusters(
    data: Sequence[int],
    width: int,
    height: int,
    resolution: float = 1.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    min_cluster_size: int = 5,
    connectivity: int = 8,
    free_threshold: int = 20,
) -> list[FrontierCluster]:
    """Detect frontier clusters from raw occupancy-grid values.

    This is the convenience public API for tests and non-ROS callers. It builds
    the internal :class:`GridMeta` and returns :class:`FrontierCluster` objects
    with ``cells``, ``centroid``, and ``size``.
    """

    meta = GridMeta(
        width=int(width),
        height=int(height),
        resolution=float(resolution),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
    )
    return detect_frontiers(
        data,
        meta,
        connectivity=connectivity,
        min_cluster_size=min_cluster_size,
        free_threshold=free_threshold,
    )


def neighbor_offsets(connectivity: int = 8) -> tuple[Cell, ...]:
    """Return 4- or 8-connected neighbor offsets."""

    if connectivity == 4:
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        return (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    raise ValueError("connectivity must be 4 or 8")


def _validate_grid(data: Sequence[int], meta: GridMeta) -> None:
    expected = meta.width * meta.height
    if len(data) != expected:
        raise ValueError(f"Occupancy data length {len(data)} does not match grid size {expected}")


def _is_free(value: int, free_threshold: int) -> bool:
    return 0 <= int(value) <= free_threshold


def _is_unknown_neighbor(
    data: Sequence[int],
    meta: GridMeta,
    mx: int,
    my: int,
    unknown_value: int,
) -> bool:
    if not (0 <= mx < meta.width and 0 <= my < meta.height):
        return False
    return int(data[my * meta.width + mx]) == unknown_value
