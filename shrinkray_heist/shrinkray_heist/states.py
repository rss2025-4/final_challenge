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
from .definitions import  Target, TripSegment, ObjectDetected, State, TrafficSimulation, Direction 
from typing import List, Tuple
from .helper import visualize_pose
from visualization_msgs.msg import Marker
from .utils import LineTrajectory


"""
High level node to control other nodes
"""
# TODO make sure all the state msgs i access using Enum.Object.value

class StatesNode(Node):
    def __init__(self):
        super().__init__("states_node")
        
        # Parameters
        self.debug = False
        self.stopping_distance = 0.75 # meters for the robot to stop at the traffic light
       

        # Subscribers
        self.start_pose = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.start_pose_cb, 5)
        self.points_sub = self.create_subscription(PoseArray, '/shrinkray_part', self.points_cb, 5)
        self.current_pose = self.create_subscription(Odometry, '/pf/pose/odom', self.current_pose_cb, 10)
        self.traj_sub = self.create_subscription(PoseArray, '/trajectory/current', self.trajectory_cb, 10)
        
        self.traffic_light_sub = self.create_subscription(Float32, '/traffic_light', self.traffic_cb, 10)
        self.detection_sub = self.create_subscription(Int32, '/detected_obj', self.detection_cb, 5)
        
        # pursuit should tell us when it is done / arrived at goal
        self.pursuit_sub = self.create_subscription(Int32, '/pursuit_state', self.reached_goal_cb, 5)
        
       
        # Class Attributes
        self.start = None
        self.timer = None
        self.trip_segment = TripSegment.RAY_LOC1
        self.goal_points: List[Tuple[float, float]] = []
        self.current_point: Tuple[float, float] | None = None
        self.state = State.IDLE
        self.at_stopping_point = False
        self.traffic_state = TrafficSimulation.NO_TRAFFIC
        self.direction = Direction.WAY_THERE
        
        
        # Publishers 
        self.state_pub = self.create_publisher(Int32, '/toggle_state', 5)
        self.points_pub = self.create_publisher(PoseArray, '/planned_pts', 1) # publish the points we want to plan a path 
        self.start_pub = self.create_publisher(Marker, '/start_pose', 1)
        self.get_logger().info('State Node Initialized with State: "%s"' % self.trip_segment)
        # self.traffic_stop_drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/high_level/input/nav_0", 10)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
    
    
    def start_pose_cb(self, pose):
        self.start = pose.pose.pose
        self.trajectory.addPoint((self.start.position.x, self.start.position.y))
        
        self.get_logger().info("Got start pose")
        if self.debug:
            self.get_logger().info(f"Start pose: {self.start}")
        visualize_pose(pose=self.start, publisher=self.start_pub, color=(1.0, 0.0, 0.0, 1.0), id=0)
        if len(self.goal_points) > 0:
            self.get_logger().info("StatesNode: Starting trip")
            self.start_trip()
        else:
            self.get_logger().warn("StatesNode: Waypoints not received yet, cannot start trip. Run the basement point publisher node?")
            return
        
    def trajectory_cb(self, msg: PoseArray):
        self.get_logger().info("StatesNode: Received trajectory, turning on traffic light detector")
        self.state = State.FOLLOWING
        self.control_node(target=Target.DETECTOR_TRAFFIC_LIGHT) # turn on traffic light detection since we are following
            
        
        
       
    '''
        called when the robot has reached the goal point
        this is where we need to check if we are at the shrink ray location
    '''
    def reached_goal_cb(self, msg: Int32):
        self.get_logger().info("StatesNode: We have arrived at current goal point")
        self.control_node(target=Target.DETECTOR_TRAFFIC_LIGHT) # turn off traffic light detection 
        if not self.trip_segment == TripSegment.END:
            self.control_node(target=Target.FOLLOWER) # turn OFF PURE PURSUIT
            self.control_node(target=Target.DETECTOR_SHRINK_RAY) # start detecting
            self.state = State.DETECTING
        if self.trip_segment == TripSegment.RAY_LOC1:
            self.get_logger().info("StatesNode: Reached RAY_LOC1")
            
            
        elif self.trip_segment == TripSegment.RAY_LOC2:
            self.get_logger().info("StatesNode: Reached RAY_LOC2")
            
            
        elif self.trip_segment == TripSegment.END and self.direction == Direction.WAY_THERE:
            self.direction = Direction.WAY_BACK
            self.control_node(target=Target.FOLLOWER) # turn OFF PURE PURSUIT
            self.get_logger().info("StatesNode: Reached END")
            
    def resume_planning_cb(self):
        self.get_logger().info("StatesNode: 5-second wait over. Resuming path planning.")
        self.control_node(target=Target.PLANNER) # Assuming you have a planner control
        self.control_node(target=Target.FOLLOWER) # Start pure pursuit again
        self.state = State.PLANNING # Update your state
        
        # Represents the transition from travelling to the shrink ray location to next location
        # TripSegment.RAY_LOC1 ==> TripSegment.RAY_LOC2 or TripSegment.RAY_LOC2 ==> TripSegment.END
        if self.trip_segment == TripSegment.RAY_LOC1:
            self.trip_segment = TripSegment.RAY_LOC2
        elif self.trip_segment == TripSegment.RAY_LOC2:
            self.trip_segment = TripSegment.END
        
        self.timer.cancel()     # It's good practice to cancel the timer once its task is done
        self.timer = None       # Optionally set self.timer to None to avoid dangling references
        
        # TODO make sure state transitions make sense from this point
        self.request_path()     # Request a new path after the timer is done
        
    def detection_cb(self, msg: Int32):
        self.get_logger().info("StatesNode: Received detection")
        if msg.data == ObjectDetected.SHRINK_RAY.value and self.state == State.DETECTING:
            self.get_logger().info("StatesNode: Detected shrink ray")
            self.control_node(target=Target.DETECTOR_SHRINK_RAY) # can stop detecting now 
            
            self.state = State.WAITING
            timer_period = 5.0  # in seconds
            self.get_logger().info("StatesNode: Starting timer for 5 seconds")
            self.timer = self.create_timer(timer_period, self.resume_planning_cb)
            
            
        elif msg.data == ObjectDetected.TRAFFIC_LIGHT_RED.value and self.state == State.FOLLOWING and self.traffic_state == TrafficSimulation.NO_TRAFFIC:
            self.get_logger().info("StatesNode: Detected red traffic light, transition into traffic simulation")
            
            self.traffic_state = TrafficSimulation.INCOMING_TRAFFIC
            
        elif msg.data == ObjectDetected.TRAFFIC_LIGHT_RED.value and self.state == State.FOLLOWING and self.traffic_state == TrafficSimulation.INCOMING_TRAFFIC:
            self.get_logger().info("StatesNode: Detected RED traffic light, not at stopping point")
            
        elif msg.data == ObjectDetected.TRAFFIC_LIGHT_RED.value and self.state == State.WAITING and self.traffic_state == TrafficSimulation.ONGOING_TRAFFIC:
            self.get_logger().info("StatesNode: Detected RED traffic light, waiting at stopping point")
            
        elif msg.data == ObjectDetected.TRAFFIC_LIGHT_GREEN.value and self.state == State.WAITING:
            self.get_logger().info("StatesNode: Detected green traffic light : resuming")
            self.control_node(target=Target.FOLLOWER) # start the pure pursuit again
            self.traffic_state = TrafficSimulation.HANDLED_TRAFFIC
            self.state = State.FOLLOWING
            self.control_node(target=Target.DETECTOR_TRAFFIC_LIGHT_GREEN) # turn off the detector 
            self.at_stopping_point = False
        elif msg.data == ObjectDetected.TRAFFIC_LIGHT_GREEN.value and self.state == State.FOLLOWING:
            self.get_logger().info("StatesNode: Detected green traffic light : continue")
        else:
            self.get_logger().warn(f"StatesNode: Detected traffic light but state combination is not valid/ supported: {ObjectDetected(msg.data)}, {self.state}, {self.traffic_state}")
            
    def traffic_cb(self, msg: Float32):
        self.get_logger().info("StatesNode: Received traffic light location")
        
        # This transition holds as long as we are only in traffic when the light is red
        if 0.5 <= msg.data <= 1.0 and self.traffic_state == TrafficSimulation.INCOMING_TRAFFIC and not self.at_stopping_point:
            self.get_logger().info("StatesNode: traffic_cb: Stopping at traffic light")
            # if we detected the traffic light, we need to stop the pure pursuit
            self.control_node(target=Target.FOLLOWER)
            self.at_stopping_point = True
            self.state = State.WAITING
            self.traffic_state = TrafficSimulation.ONGOING_TRAFFIC
            self.control_node(target=Target.DETECTOR_TRAFFIC_LIGHT_GREEN) # turn off the detector
        
    def points_cb(self, msg: PoseArray):
        self.get_logger().info("StatesNode: Received basement points")
        
        # iterate through the poses in the PoseArray
        for pose in msg.poses:
            x, y = pose.position.x, pose.position.y
            self.get_logger().info(f"StatesNode: Received point: {x}, {y}")
            self.goal_points.append((x,y))
        if self.debug: 
            self.get_logger().info(f"StatesNode: Received points: {self.goal_points}")
        
         # !! not sure if this is the right place to do this
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        if self.start:
            self.get_logger().info("StatesNode: Starting trip")
            self.start_trip()
        else:
            self.get_logger().warn("StatesNode: Start pose not received yet, cannot start trip")
            return
    
    def start_trip(self):
        start_point = (self.start.position.x, self.start.position.y)
        
        # sort the goal points based on distance from start point
        # sorted_checkpoints = sorted(
        #     self.goal_points,
        #     key=lambda checkpoint: ((checkpoint[0] - start_point[0]) ** 2 + (checkpoint[1] - start_point[1]) ** 2) ** 0.5
        # )
        self.trip_segment = TripSegment.RAY_LOC1
        
        
        self.control_node(target=Target.PLANNER)
        
        self.request_path()
        
    def control_node(self, target: Target):
        self.get_logger().info(f"StatesNode: Controlling node: {target}")
        msg = Int32()
        # self.get_logger().info(f"StatesNode: Controlling node type: {type(target)}")
        msg.data = target.value
        self.state_pub.publish(msg)
        
        
    def request_path(self, **kwargs):
        if self.trip_segment == TripSegment.RAY_LOC1:
            self.get_logger().info("StatesNode: Requesting path for RAY_LOC1")
            start_point = (self.start.position.x, self.start.position.y)
            end_point = self.goal_points[0]
        elif self.trip_segment == TripSegment.RAY_LOC2:
            self.get_logger().info("StatesNode: Requesting path for RAY_LOC2")
            start_point = self.current_point
            end_point = self.goal_points[1]
        elif self.trip_segment == TripSegment.START:
            start_point = self.current_point
            end_point =  (self.start.position.x, self.start.position.y)
        else:
            self.get_logger().warn("Invalid trip segment")
            return
        # array = []
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        for x, y in [start_point, end_point]:
            # pose = Pose()
            pose_array.poses.append(Pose(position=Point(x=x, y=y, z=0.0)))
        
        # Publish the points we want to plan a path
        self.points_pub.publish(pose_array)

        self.state = State.PLANNING
        
    def publish_pts(self, array):
        # Publish PoseArray
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.poses = array
        self.points_pub.publish(pose_array)

        # Print to Command Line
        points_str = '\n'+'\n'.join([f"({p.position.x},{p.position.y})" for p in array])
        self.get_logger().info(f"StatesNode: Published 2 points: {points_str}")
        
        return 
        
            
            
    def current_pose_cb(self, odometry_msg: Pose):
        # self.current_point = (odometry_msg.position.x, odometry_msg.position.y)
        self.current_point = (odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y)
   

     


def main(args=None):
    rclpy.init(args=args)
    detector = StatesNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
    
         
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
            