"""Exploration bookkeeping for Active SLAM.

Visited goals, blacklist, planner cache, and settling are ROS 2/Nav2 adaptations
or legacy safety logic; they are not present verbatim in aslam_rosbot.
"""

from __future__ import annotations

from typing import Optional

from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.time import Time

from bumperbot_active_slam.active_slam_types import BlacklistedGoal, PlannerCacheEntry, VisitedGoal
from bumperbot_active_slam.entropy_utils import Point2D, euclidean_distance


class ExplorationState:
    def __init__(
        self,
        *,
        blacklist_radius: float,
        blacklist_timeout_sec: float,
        visited_radius: float,
        visited_timeout_sec: float,
        planner_cache_radius: float,
        planner_cache_ttl_sec: float,
        planner_validation_period_sec: float,
        post_goal_settle_time_sec: float,
    ) -> None:
        self.blacklist_radius = blacklist_radius
        self.blacklist_timeout_sec = blacklist_timeout_sec
        self.visited_radius = visited_radius
        self.visited_timeout_sec = visited_timeout_sec
        self.planner_cache_radius = planner_cache_radius
        self.planner_cache_ttl_sec = planner_cache_ttl_sec
        self.planner_validation_period_sec = planner_validation_period_sec
        self.post_goal_settle_time_sec = post_goal_settle_time_sec

        self.blacklist: list[BlacklistedGoal] = []
        self.visited_goals: list[VisitedGoal] = []
        self.planner_cache: list[PlannerCacheEntry] = []
        self.planner_request_active = False
        self.last_planner_validation_time: Optional[Time] = None
        self.settling_until: Optional[Time] = None
        self.goal_retries: dict[tuple[int, int], int] = {}

    def prune(self, now: Time) -> None:
        self.blacklist = [entry for entry in self.blacklist if entry.expires_at > now]
        self.visited_goals = [entry for entry in self.visited_goals if entry.expires_at > now]
        self.planner_cache = [entry for entry in self.planner_cache if entry.expires_at > now]

    def add_to_blacklist(self, now: Time, xy: Point2D, timeout_sec: Optional[float] = None) -> None:
        timeout = self.blacklist_timeout_sec if timeout_sec is None else timeout_sec
        self.blacklist.append(BlacklistedGoal(xy=xy, expires_at=now + Duration(seconds=timeout)))

    def is_blacklisted(self, xy: Point2D) -> bool:
        return any(euclidean_distance(xy, entry.xy) <= self.blacklist_radius for entry in self.blacklist)

    def add_to_visited(self, now: Time, xy: Point2D) -> None:
        expires_at = now + Duration(seconds=max(0.1, self.visited_timeout_sec))
        self.visited_goals = [
            entry for entry in self.visited_goals
            if euclidean_distance(xy, entry.xy) > self.visited_radius
        ]
        self.visited_goals.append(VisitedGoal(xy=xy, expires_at=expires_at))

    def is_visited(self, xy: Point2D) -> bool:
        return any(euclidean_distance(xy, entry.xy) <= self.visited_radius for entry in self.visited_goals)

    def get_cached_planner_path(self, goal_xy: Point2D) -> Optional[Path]:
        for entry in self.planner_cache:
            if entry.path is not None and euclidean_distance(goal_xy, entry.xy) <= self.planner_cache_radius:
                return entry.path
        return None

    def get_cached_planner_failure(self, goal_xy: Point2D) -> Optional[str]:
        for entry in self.planner_cache:
            if entry.path is None and euclidean_distance(goal_xy, entry.xy) <= self.planner_cache_radius:
                return entry.reason or "planner_cached_failed"
        return None

    def cache_planner_result(self, now: Time, goal_xy: Point2D, path: Optional[Path], reason: str) -> None:
        expires_at = now + Duration(seconds=max(0.1, self.planner_cache_ttl_sec))
        self.planner_cache = [
            entry for entry in self.planner_cache
            if euclidean_distance(goal_xy, entry.xy) > self.planner_cache_radius
        ]
        self.planner_cache.append(PlannerCacheEntry(xy=goal_xy, expires_at=expires_at, path=path, reason=reason))

    def planner_validation_ready(self, now: Time) -> tuple[bool, str]:
        if self.last_planner_validation_time is not None:
            elapsed = (now - self.last_planner_validation_time).nanoseconds / 1.0e9
            if elapsed < self.planner_validation_period_sec:
                return False, "planner_validation_period"
        if self.planner_request_active:
            return False, "planner_request_active"
        return True, ""

    def start_planner_request(self, now: Time) -> None:
        self.last_planner_validation_time = now
        self.planner_request_active = True

    def finish_planner_request(self) -> None:
        self.planner_request_active = False

    def start_settling(self, now: Time) -> None:
        self.settling_until = now + Duration(seconds=self.post_goal_settle_time_sec)

    def is_settling(self, now: Time) -> bool:
        if self.settling_until is None:
            return False
        if now < self.settling_until:
            return True
        self.settling_until = None
        return False
