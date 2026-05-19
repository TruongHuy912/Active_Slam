"""Shared data types for bumperbot_active_slam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nav_msgs.msg import Path
from rclpy.time import Time

from bumperbot_active_slam.entropy_utils import Cell, Point2D
from bumperbot_active_slam.frontier_detector import FrontierCluster


@dataclass(frozen=True)
class CandidateGoal:
    """Navigation goal derived from one frontier candidate."""

    xy: Point2D
    cell: Cell
    map_path_cells: list[Cell]
    planner_path: Path
    planner_path_length: float
    costmap_cost: int
    used_offset: bool


@dataclass(frozen=True)
class ScoredFrontier:
    """Scoring result for one frontier candidate."""

    cluster: FrontierCluster
    path_cells: list[Cell]
    utility: float
    entropy_reward: float
    distance_reward: float
    distance: float
    centroid_cell: Cell
    nav_goal_xy: Point2D
    nav_goal_cell: Cell
    nav_path_cells: list[Cell]
    planner_path: Path
    planner_path_length: float
    information_gain: float
    costmap_cost: int
    used_offset_goal: bool


@dataclass
class BlacklistedGoal:
    """Temporarily ignored goal region."""

    xy: Point2D
    expires_at: Time


@dataclass
class VisitedGoal:
    """Successfully reached goal or frontier region."""

    xy: Point2D
    expires_at: Time


@dataclass
class PlannerCacheEntry:
    """Short-lived planner validation result for a candidate goal."""

    xy: Point2D
    expires_at: Time
    path: Optional[Path]
    reason: str = ""
