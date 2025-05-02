import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker

class SimplePointFollower(Node):
    def __init__(self):
        super().__init__("simple_point_follower")
        
        # Parameters
        self.speed = 1.0  # Fixed forward speed
        self.max_steer = np.pi / 4  # Maximum steering angle
        
        # Publishers/Subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, 
            '/vesc/high_level/input/nav_0', 
            1
        )
        self.marker_sub = self.create_subscription(
            Marker, 
            '/destination_marker', 
            self.marker_callback, 
            1
        )
        
    def marker_callback(self, msg):
        # Since car is at (0,0), angle to goal is just atan2(y,x) of goal point
        angle_to_goal = np.arctan2(
            msg.pose.position.y,  # y coordinate of marker
            msg.pose.position.x   # x coordinate of marker
        )
        
        # Limit steering angle
        steering_angle = np.clip(angle_to_goal, -self.max_steer, self.max_steer)
        
        # Create and publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        # if abs(steering_angle) < 0.1:
        drive_msg.drive.steering_angle = steering_angle -0.07
        # else:
        # drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimplePointFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()