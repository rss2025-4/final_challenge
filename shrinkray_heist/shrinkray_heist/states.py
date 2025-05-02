import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import Image
from .detector import Detector
from tf_transformations import euler_from_quaternion

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String
"""
High level node to control other nodes
"""

class StatesNode(Node):
    def __init__(self):
        super().__init__("states_node")

        # Subscribers
        self.start_pose = self.create_subscription(PoseWithCovarianceStamped, '/initial_pose', self.start_pose_cb, 10)
        self.end_pose = self.create_subscription(PoseWithCovarianceStamped, '/goal_pose', self.end_pose_cb, 10)
        self.locations = self.create_subscription(PoseWithCovarianceStamped, '/locations', self.locations_pose_cb, 10)
        self.current_goal_pose = None

        # self.trajectory = LineTrajectory("/followed_trajectory")
        self.np_trajectory = None
        # self.trajectory_current = self.create_subscription(PoseArray, '/trajectory/current', self.trajectory_cb, 10)

        self.current_pose = self.create_subscription(Pose, '/pf/pose/odom', self.current_pose_cb, 10)

        # self.traffic_light_check = self.create_subscription(String, '/traffic_light_checked', self.traffic_light_handled_cb, 10)
        self.shrink_ray_detected = self.create_subscription(String, '/shrink_ray_detected', self.shrink_ray_detected_cb, 10)

        # Publishers
        # self.detector_state_pub = self.create_publisher(String, '/detector_states')
        self.at_shrinkray_loc_pub = self.create_publisher(String, '/at_shrinkray_loc', 1)
        # self.check_trafficlight_pub = self.create_publisher(String, '/traffic_light_check', 10)
    
    # def trajectory_cb(self, msg: PoseArray):
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        
        self.np_trajectory = np.array(self.trajectory.points)

    # Shrink Ray Handling 
    def current_pose_cb(self, odometry_msg: Pose):
        current_point = np.array([odometry_msg.position.x, odometry_msg.position.y])
        # z_rotation = euler_from_quaternion(
        #         [
        #             odometry_msg.pose.pose.orientation.x,
        #             odometry_msg.pose.pose.orientation.y,
        #             odometry_msg.pose.pose.orientation.z,
        #             odometry_msg.pose.pose.orientation.w,
        #         ]
        #     )[2]
        # current_pose = np.array([odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y, z_rotation])
        current_goal_point = np.array([self.current_goal_pose.position.x, self.current_goal_pose.position.y])
        # check if reached goal point
        # last_point = #np.array(trajectory[-1][:2])
        if np.linalg.norm(current_point - current_goal_point) < 0.5:
            self.get_logger().info("Goal reached")
            # self.stop = True

        if (current location is at shrink ray location 1) and (location1 is not handled):
            # self.detector_state_pub.publish(String('shrinkray_detect'))
            self.at_shrinkray_loc_pub.publish(String('True'))

    # def shrink_ray_detected_cb(self, msg: String):
    #     if msg.data == 'True':
    #       self.detector_state_pub.publish(String('shrinkray_stop'))
    #       location1 is not handled
          # self.at_shrinkray_loc_pub.publish(String('False')) # stop detecting for shrink ray, already done 
    
    # # Traffic Light Handling
    # def check_trafficlight(self, msg: PoseWithCovarianceStamped):
    #     self.check_trafficlight_pub.publish(String('True'))
    
    # def traffic_light_handled_cb(self, msg: String):
    #     if msg.data == 'True':
    #       self.check_trafficlight_pub.publish(String('False')) # stop detecting traffic light but for 5 seconds? until traffic light out of view
   
   


        

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