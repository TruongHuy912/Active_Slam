"""Temporary suppression of repeatedly unreachable frontier regions.

ROS 2/Nav2 adaptation inspired by goal blacklisting and suppression in
frontier exploration stacks. This is not part of the original ASLAM paper core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bumperbot_active_slam.entropy_utils import Point2D, euclidean_distance


@dataclass
class SuppressedFrontierRegion:
    """A frontier region with repeated failures and a finite expiry time."""

    xy: Point2D
    failure_count: int
    last_reason: str
    expires_at_sec: float


class FrontierSuppressionManager:
    """Track temporary suppression for frontier regions that repeatedly fail.

    A region is considered suppressed only after ``failure_count`` reaches
    ``max_failures``. Expired regions are pruned automatically by public methods.
    """

    def __init__(
        self,
        radius: float,
        timeout_sec: float,
        max_failures: int,
        max_regions: int,
    ) -> None:
        self.radius = max(0.0, float(radius))
        self.timeout_sec = max(0.0, float(timeout_sec))
        self.max_failures = max(1, int(max_failures))
        self.max_regions = max(1, int(max_regions))
        self._regions: list[SuppressedFrontierRegion] = []

    def prune_expired(self, now_sec: float) -> None:
        self._regions = [region for region in self._regions if region.expires_at_sec > now_sec]

    def is_suppressed(self, xy: Point2D, now_sec: float) -> bool:
        self.prune_expired(now_sec)
        region = self._find_region(xy)
        return region is not None and region.failure_count >= self.max_failures

    def record_failure(self, xy: Point2D, reason: str, now_sec: float) -> bool:
        """Record one failed frontier attempt.

        Returns True when the region has just reached the suppression threshold.
        """

        self.prune_expired(now_sec)
        region = self._find_region(xy)
        if region is None:
            region = SuppressedFrontierRegion(
                xy=(float(xy[0]), float(xy[1])),
                failure_count=0,
                last_reason=str(reason),
                expires_at_sec=now_sec + self.timeout_sec,
            )
            self._regions.append(region)
        was_suppressed = region.failure_count >= self.max_failures
        region.failure_count += 1
        region.last_reason = str(reason)
        region.expires_at_sec = now_sec + self.timeout_sec
        self._trim_regions(now_sec)
        return not was_suppressed and region.failure_count >= self.max_failures

    def clear_near(self, xy: Point2D) -> None:
        self._regions = [region for region in self._regions if euclidean_distance(region.xy, xy) > self.radius]

    def summary(self, now_sec: float) -> dict[str, Any]:
        active = self.active_regions(now_sec)
        reasons: dict[str, int] = {}
        for region in active:
            reasons[region.last_reason] = reasons.get(region.last_reason, 0) + 1
        return {
            "regions": len(active),
            "tracked_regions": len(self._regions),
            "top_reasons": dict(sorted(reasons.items(), key=lambda item: item[1], reverse=True)),
        }

    def active_regions(self, now_sec: float) -> list[SuppressedFrontierRegion]:
        self.prune_expired(now_sec)
        return [region for region in self._regions if region.failure_count >= self.max_failures]

    def _find_region(self, xy: Point2D) -> SuppressedFrontierRegion | None:
        for region in self._regions:
            if euclidean_distance(region.xy, xy) <= self.radius:
                return region
        return None

    def _trim_regions(self, now_sec: float) -> None:
        self.prune_expired(now_sec)
        while len(self._regions) > self.max_regions:
            oldest_index = min(range(len(self._regions)), key=lambda idx: self._regions[idx].expires_at_sec)
            del self._regions[oldest_index]
