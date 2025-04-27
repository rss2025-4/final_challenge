import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from .detector import Detector

"""
High level node to control other nodes
"""

class StatesNode(Node):
    def __init__(self):
        super().__init__("states_node")

        self.pose_sub = self.create_subscription(Pose, "/pf/pose/odom", self.pf_pose_callback, 10)

        

    # Control stopping and use of different nodes
    # Keep track of: location, path, current speed and steering, whether pure pursuit on
    # handle current location
     


def main(args=None):
    rclpy.init(args=args)
    detector = StatesNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()