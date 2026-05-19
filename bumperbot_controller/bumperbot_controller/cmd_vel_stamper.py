#!/usr/bin/env python3
"""Bridge Nav2 Twist commands to the Bumper-Bot TwistStamped controller input."""

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node


class CmdVelStamper(Node):
    def __init__(self):
        super().__init__("cmd_vel_stamper")
        self.declare_parameter("input_topic", "/cmd_vel")
        self.declare_parameter("output_topic", "/bumperbot_controller/cmd_vel")
        self.declare_parameter("frame_id", "base_link")

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.cmd_sub = self.create_subscription(
            Twist,
            self.input_topic,
            self.cmd_vel_callback,
            10,
        )
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.output_topic,
            10,
        )
        self.get_logger().info(
            f"Stamping Twist commands from {self.input_topic} to {self.output_topic} "
            f"with frame_id={self.frame_id}"
        )

    def cmd_vel_callback(self, msg):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.frame_id
        stamped.twist = msg
        self.cmd_pub.publish(stamped)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelStamper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
