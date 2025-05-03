import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import Image
# from .detector import Detector
# from tf_transformations import euler_from_quaternion

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray, Point, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String, Int32
from .definitions import  Target, TripSegment #States
from typing import List, Tuple


"""
High level node to control other nodes
"""

class StatesNode(Node):
    def __init__(self):
        super().__init__("states_node")
        
        # Parameters
        self.debug = False 

        # Subscribers
        self.start_pose = self.create_subscription(PoseWithCovarianceStamped, '/initial_pose', self.start_pose_cb, 10)
        self.points_sub = self.create_subscription(PoseArray, '/shell_points', self.points_cb, 5)
        self.current_pose = self.create_subscription(Pose, '/pf/pose/odom', self.current_pose_cb, 10)
        self.traj_sub = self.create_subscription(PoseArray, '/trajectory/current', self.trajectory_cb, 10)
        self.ray_sub = self.create_subscription(PoseWithCovarianceStamped, '/shrink_ray_loc', self.ray_cb, 10)
        self.traffic_light_sub = self.create_subscription(PoseWithCovarianceStamped, '/traffic_light', self.traffic_cb, 10)
        
        # Class Attributes
        self.start = None
        self.trip_segment = TripSegment.RAY_LOC1
        self.goal_points: List[Tuple[float, float]] = []
        self.current_point: Tuple[float, float] | None = None
        
        # Publishers 
        self.state_pub = self.create_publisher(Int32, '/toggle_state', 1)
        
        self.points_pub = self.create_publisher(PoseArray, '/planned_pts', 1) # publish the points we want to plan a path 
        
        self.get_logger().info('State Node Initialized with State: "%s"' % self.trip_segment)

        # ? are we getting an end pose from the tas?
        # self.end_pose = self.create_subscription(PoseWithCovarianceStamped, '/goal_pose', self.end_pose_cb, 10)
        # self.locations = self.create_subscription(PoseWithCovarianceStamped, '/locations', self.locations_pose_cb, 10)
        
        
        
        
        

        # self.trajectory = LineTrajectory("/followed_trajectory")
        # self.np_trajectory = None
        # self.trajectory_current = self.create_subscription(PoseArray, '/trajectory/current', self.trajectory_cb, 10)

        

        # self.traffic_light_check = self.create_subscription(String, '/traffic_light_checked', self.traffic_light_handled_cb, 10)
        #self.shrink_ray_detected = self.create_subscription(String, '/shrink_ray_detected', self.shrink_ray_detected_cb, 10)

        # Publishers
        # self.detector_state_pub = self.create_publisher(String, '/detector_states')
        #self.at_shrinkray_loc_pub = self.create_publisher(String, '/at_shrinkray_loc', 1)
        # self.check_trafficlight_pub = self.create_publisher(String, '/traffic_light_check', 10)
    def start_pose_cb(self, msg: PoseWithCovarianceStamped):
        pass
    def trajectory_cb(self, msg: PoseArray):
        self.get_logger().info("Received trajectory")
        pass 
        # self.np_trajectory = np.array(self.trajectory.points)

    
    def points_cb(self, msg: PoseArray):
        self.get_logger().info("Received basement points")
        
        # iterate through the poses in the PoseArray
        for pose in msg.poses:
            x, y = pose.position.x, pose.position.y
            self.get_logger().info(f"Received point: {x}, {y}")
            self.goal_points.append((x,y))
        if self.debug: 
            self.get_logger().info(f"Received points: {self.goal_points}")
        start_point = (self.start.position.x, self.start.position.y)
        
        # sort the goal points based on distance from start point
        sorted_checkpoints = sorted(
            self.goal_points,
            key=lambda checkpoint: ((checkpoint[0] - start_point[0]) ** 2 + (checkpoint[1] - start_point[1]) ** 2) ** 0.5
        )
        self.goal_points = sorted_checkpoints
        self.trip_segment = TripSegment.RAY_LOC1
        
        # !! not sure if this is the right place to do this
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        
        self.control_node(target=Target.PLANNER)
        self.request_path()
        
    def control_node(self, target: Target):
        self.get_logger().info(f"Controlling node: {target}")
        msg = Int32()
        msg.data = target
        self.state_pub.publish(msg)
        
        
    def request_path(self, **kwargs):
        if self.trip_segment == TripSegment.RAY_LOC1:
            self.get_logger().info("Requesting path for RAY_LOC1")
            start_point = (self.start.position.x, self.start.position.y)
            end_point = self.goal_points[0]
        elif self.trip_segment == TripSegment.RAY_LOC2:
            self.get_logger().info("Requesting path for RAY_LOC2")
            start_point = self.current_point
            end_point = self.goal_points[1]
        elif self.trip_segment == TripSegment.RAY_OBJ1 or TripSegment.RAY_OBJ2:
            self.get_logger().info("Requesting path for shrink ray object")
            start_point = self.current_point
            
            end_point = kwargs.get('shrink_ray_loc', None)
            if end_point is None:
                self.get_logger().warn("No shrink ray location provided")
                return
        elif self.trip_segment == TripSegment.START:
            start_point = self.current_point
            end_point =  (self.start.position.x, self.start.position.y)
        else:
            self.get_logger().warn("Invalid trip segment")
        array = []
        for x, y in [start_point, end_point]:
            array.append(Pose(position=Point(x=x, y=y, z=0.0)))
        # Publish the points we want to plan a path
        self.points_pub.publish_pts(array)
        
    def publish_pts(self, array):
        # Publish PoseArray
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.poses = array
        self.points_pub.publish(pose_array)

        # Print to Command Line
        points_str = '\n'+'\n'.join([f"({p.position.x},{p.position.y})" for p in array])
        self.get_logger().info(f"Published 2 points: {points_str}")
        
        return 
        
            
            
    '''
    EGGOS IDEAS
    1) the state machine does not need to continously check if the robot is at the goal point, that 
    is done by the pure pursuit node.
    2) the state machine can listen to the pure pursuit node and when it is at the goal point,
    it can publish a message to the pure pursuit node to stop. AND 
    depending on other states like if we are at the shrink ray location, 
    it can publish a message to the shrink ray node to start detecting.
    3) the state machine can also listen to the shrink ray node and when it is done detecting,
    it can publish a message to the pure pursuit node to start again.
    
    NOTE: the pure pursuit should not be moving still even if started again 
    4) the state machine can then publish to the planner node to start planning a new path, 
    HACK: depending on the location of the shrink ray object, given by the shrink ray node? 
    
    since the state machine intercepts the trajectory message, it will know when the robot 
    is moving or not. 
     
    
    '''
            
            
            
            
    # Shrink Ray Handling 
    def current_pose_cb(self, odometry_msg: Pose):
        self.current_point = (odometry_msg.position.x, odometry_msg.position.y)
    
    
    
    def ray_cb(self, msg: PoseWithCovarianceStamped):
        self.get_logger().info("Received shrink ray location")
        pass 
    
    def traffic_cb(self, msg: PoseWithCovarianceStamped):
        self.get_logger().info("Received traffic light location")
        pass

        # z_rotation = euler_from_quaternion(
        #         [
        #             odometry_msg.pose.pose.orientation.x,
        #             odometry_msg.pose.pose.orientation.y,
        #             odometry_msg.pose.pose.orientation.z,
        #             odometry_msg.pose.pose.orientation.w,
        #         ]
        #     )[2]
        # current_pose = np.array([odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y, z_rotation])
        # current_goal_point = np.array([self.current_goal_pose.position.x, self.current_goal_pose.position.y])
        # # check if reached goal point
        # # last_point = #np.array(trajectory[-1][:2])
        # if np.linalg.norm(current_point - current_goal_point) < 0.5:
        #     self.get_logger().info("Goal reached")
        #     # self.stop = True

        # if (current location is at shrink ray location 1) and (location1 is not handled):
        #     # self.detector_state_pub.publish(String('shrinkray_detect'))
        #     self.at_shrinkray_loc_pub.publish(String('True'))

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