"""ROS 2/Nav2 adapters for planner and navigation actions.

ROS 2/Nav2 adaptation of aslam_rosbot/scripts/functions.py::robot.makePlan().
ROS 2/Nav2 adaptation of aslam_rosbot/scripts/functions.py::robot.sendGoal().
"""

from __future__ import annotations

import math
import threading
from typing import Callable, Optional, Sequence

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from bumperbot_active_slam.entropy_utils import Cell, GridMeta, Point2D, bresenham, euclidean_distance, world_to_map


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def make_pose_stamped(frame_id: str, stamp, xy: Point2D, *, yaw: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = xy[0]
    pose.pose.position.y = xy[1]
    pose.pose.position.z = 0.0
    pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = yaw_to_quaternion(yaw)
    return pose


def path_cells_from_poses(poses: Sequence[PoseStamped], meta: GridMeta) -> list[Cell]:
    cells: list[Cell] = []
    previous: Optional[Cell] = None
    for pose in poses:
        cell = world_to_map(pose.pose.position.x, pose.pose.position.y, meta)
        if cell is None:
            continue
        segment = [cell] if previous is None else bresenham(previous, cell)
        for item in segment:
            if not cells or cells[-1] != item:
                cells.append(item)
        previous = cell
    return cells


def path_length(poses: Sequence[PoseStamped]) -> float:
    total = 0.0
    previous: Optional[Point2D] = None
    for pose in poses:
        current = (float(pose.pose.position.x), float(pose.pose.position.y))
        if previous is not None:
            total += euclidean_distance(previous, current)
        previous = current
    return total


def goal_status_name(status: int) -> str:
    from action_msgs.msg import GoalStatus

    names = {
        GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
        GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
        GoalStatus.STATUS_EXECUTING: "EXECUTING",
        GoalStatus.STATUS_CANCELING: "CANCELING",
        GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
        GoalStatus.STATUS_CANCELED: "CANCELED",
        GoalStatus.STATUS_ABORTED: "ABORTED",
    }
    return names.get(status, f"STATUS_{status}")


class Nav2ActionAdapter:
    """Small wrapper around Nav2 action clients."""

    def __init__(
        self,
        node,
        callback_group: ReentrantCallbackGroup,
        *,
        global_frame: str,
        planner_action_name: str,
        navigate_action_name: str,
        planner_id: str,
    ) -> None:
        self.node = node
        self.global_frame = global_frame
        self.planner_id = planner_id
        self.path_client = ActionClient(node, ComputePathToPose, planner_action_name, callback_group=callback_group)
        self.nav_client = ActionClient(node, NavigateToPose, navigate_action_name, callback_group=callback_group)

    def request_path_blocking(
        self,
        robot_xy: Point2D,
        goal_xy: Point2D,
        *,
        timeout_sec: float,
        log_throttled: Callable[..., None],
    ) -> tuple[Optional[Path], str]:
        now_msg = self.node.get_clock().now().to_msg()
        goal_msg = ComputePathToPose.Goal()
        goal_msg.use_start = True
        goal_msg.planner_id = str(self.planner_id)
        goal_msg.start = make_pose_stamped(self.global_frame, now_msg, robot_xy, yaw=0.0)
        yaw = math.atan2(goal_xy[1] - robot_xy[1], goal_xy[0] - robot_xy[0])
        goal_msg.goal = make_pose_stamped(self.global_frame, now_msg, goal_xy, yaw=yaw)

        done = threading.Event()
        result_box: dict[str, Optional[Path]] = {"path": None}
        reason_box = {"reason": "planner_failed"}

        def on_result(result_future) -> None:
            try:
                result = result_future.result().result
                result_box["path"] = result.path
                reason_box["reason"] = ""
            except Exception as exc:  # pragma: no cover - defensive ROS callback guard
                reason_box["reason"] = "planner_result_error"
                log_throttled("planner_result_error", f"ComputePathToPose result failed: {exc}", level="warn")
                result_box["path"] = None
            finally:
                done.set()

        def on_goal_response(send_future) -> None:
            try:
                goal_handle = send_future.result()
            except Exception as exc:  # pragma: no cover - defensive ROS callback guard
                reason_box["reason"] = "planner_goal_error"
                log_throttled("planner_goal_error", f"ComputePathToPose goal failed: {exc}", level="warn")
                done.set()
                return
            if not goal_handle.accepted:
                reason_box["reason"] = "planner_rejected"
                done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        try:
            self.path_client.send_goal_async(goal_msg).add_done_callback(on_goal_response)
        except Exception as exc:  # pragma: no cover - defensive ROS callback guard
            log_throttled("planner_send_error", f"ComputePathToPose send failed: {exc}", level="warn")
            return None, "planner_send_error"

        if not done.wait(timeout=max(0.1, timeout_sec)):
            log_throttled(
                "planner_timeout",
                "ComputePathToPose timed out for a frontier candidate; caching this candidate as planner_failed briefly.",
                level="warn",
                period_sec=10.0,
            )
            return None, "planner_timeout"
        return result_box["path"], reason_box["reason"]

    def wait_for_planner(self, timeout_sec: float = 0.0) -> bool:
        return self.path_client.wait_for_server(timeout_sec=timeout_sec)

    def wait_for_navigation(self, timeout_sec: float = 0.0) -> bool:
        return self.nav_client.wait_for_server(timeout_sec=timeout_sec)

    def send_navigation_goal(self, goal_msg: NavigateToPose.Goal, feedback_callback):
        return self.nav_client.send_goal_async(goal_msg, feedback_callback=feedback_callback)
