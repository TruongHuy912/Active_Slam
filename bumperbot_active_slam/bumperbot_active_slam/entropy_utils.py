"""ROS-independent grid, ray tracing, and entropy helpers.

The helpers in this module operate on plain Python values. A ROS 2 node can
adapt ``nav_msgs/msg/OccupancyGrid.info`` into :class:`GridMeta` without this
module importing any ROS package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

Cell = Tuple[int, int]
Point2D = Tuple[float, float]


@dataclass(frozen=True)
class GridMeta:
    """Minimal occupancy-grid metadata needed by the pure algorithm."""

    width: int
    height: int
    resolution: float
    origin_x: float = 0.0
    origin_y: float = 0.0


def grid_meta_from_map_info(map_info: object) -> GridMeta:
    """Create :class:`GridMeta` from a ROS-like MapMetaData object.

    This stays ROS-independent by using duck typing instead of importing
    ``nav_msgs``.
    """

    return GridMeta(
        width=int(getattr(map_info, "width")),
        height=int(getattr(map_info, "height")),
        resolution=float(getattr(map_info, "resolution")),
        origin_x=float(getattr(getattr(map_info, "origin").position, "x")),
        origin_y=float(getattr(getattr(map_info, "origin").position, "y")),
    )


def world_to_map(x: float, y: float, meta: GridMeta) -> Optional[Cell]:
    """Convert world coordinates to a map cell.

    Returns ``None`` when the coordinate falls outside the grid.
    """

    if meta.resolution <= 0.0:
        raise ValueError("Grid resolution must be positive")

    mx = int(math.floor((x - meta.origin_x) / meta.resolution))
    my = int(math.floor((y - meta.origin_y) / meta.resolution))
    if 0 <= mx < meta.width and 0 <= my < meta.height:
        return mx, my
    return None


def map_to_world(mx: int, my: int, meta: GridMeta, *, center: bool = True) -> Point2D:
    """Convert a map cell to world coordinates."""

    offset = 0.5 if center else 0.0
    x = meta.origin_x + (float(mx) + offset) * meta.resolution
    y = meta.origin_y + (float(my) + offset) * meta.resolution
    return x, y


def bresenham(*args: Union[int, Cell]) -> list[Cell]:
    """Return inclusive Bresenham cells as ``list[(x, y)]``.

    Supported call styles:

    - ``bresenham((x0, y0), (x1, y1))``
    - ``bresenham(x0, y0, x1, y1)``
    """

    start, end = _parse_bresenham_args(args)

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    err = dx - dy

    cells: list[Cell] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x0 += step_x
        if err2 < dx:
            err += dx
            y0 += step_y
    return cells


def _parse_bresenham_args(args: tuple[Union[int, Cell], ...]) -> tuple[Cell, Cell]:
    if len(args) == 2:
        start_arg, end_arg = args
        try:
            start = tuple(start_arg)  # type: ignore[arg-type]
            end = tuple(end_arg)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("bresenham(start, end) expects two (x, y) cells") from exc
        if len(start) != 2 or len(end) != 2:
            raise TypeError("bresenham(start, end) expects two (x, y) cells")
        return (int(start[0]), int(start[1])), (int(end[0]), int(end[1]))

    if len(args) == 4:
        x0, y0, x1, y1 = args
        return (int(x0), int(y0)), (int(x1), int(y1))

    raise TypeError("bresenham expects either (start, end) or (x0, y0, x1, y1)")


def occupancy_probability(
    occupancy_value: int,
    *,
    unknown_probability: float = 0.1,
    known_probability: float = 0.45,
) -> float:
    """Map an OccupancyGrid value to the probability used by path entropy."""

    if occupancy_value == -1:
        return _clamp_probability(unknown_probability)
    return _clamp_probability(known_probability)


def cell_entropy(probability: float) -> float:
    """Compute binary entropy ``H(p)`` in bits."""

    p = _clamp_probability(probability)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def compute_path_entropy(
    occupancy_values: Iterable[int],
    *,
    unknown_probability: float = 0.1,
    known_probability: float = 0.45,
) -> tuple[float, float]:
    """Return ``(total_entropy, normalized_entropy)`` for path cells."""

    total = 0.0
    count = 0
    for value in occupancy_values:
        probability = occupancy_probability(
            int(value),
            unknown_probability=unknown_probability,
            known_probability=known_probability,
        )
        total += cell_entropy(probability)
        count += 1

    normalized = total / float(count) if count else 0.0
    return total, normalized


def distance_decay(distance: float, lambda_decay: float = 0.6) -> float:
    """Compute the exponential distance penalty/reward term."""

    if distance < 0.0:
        raise ValueError("Distance must be non-negative")
    if lambda_decay < 0.0:
        raise ValueError("lambda_decay must be non-negative")
    return math.exp(-lambda_decay * distance)


def path_occupancy_values(
    data: Sequence[int],
    cells: Iterable[Cell],
    meta: GridMeta,
    *,
    outside_value: int = 100,
) -> list[int]:
    """Read occupancy values for cells, using ``outside_value`` off-map."""

    values: list[int] = []
    for mx, my in cells:
        if 0 <= mx < meta.width and 0 <= my < meta.height:
            values.append(int(data[my * meta.width + mx]))
        else:
            values.append(outside_value)
    return values


def euclidean_distance(a: Point2D, b: Point2D) -> float:
    """Return Euclidean distance between two 2D world points."""

    return math.hypot(b[0] - a[0], b[1] - a[1])


def _clamp_probability(probability: float) -> float:
    epsilon = 1.0e-12
    return min(1.0 - epsilon, max(epsilon, float(probability)))
