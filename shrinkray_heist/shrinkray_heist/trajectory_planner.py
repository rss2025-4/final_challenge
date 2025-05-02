from math import atan2, cos, sin
from typing import List, Tuple
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, PointStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from visualization_msgs.msg import Marker



from .a_star import AStar
from .eval_utils import InjectedConfig
from .helper import grid_to_world, world_to_grid
from .utils import LineTrajectory


class PathPlan(Node):
    """Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self, extra_eval_configs=InjectedConfig()):
        super().__init__("trajectory_planner", **extra_eval_configs.extra_node_kwargs())

        # how much to dilate the map, higher value = safer, less agressive path finding
        self.safety_buffer = 0.6
        self.obstacle_buffer = 0.1

        # NOTE toggle this to True to enable debug mode (i.e. add additional logging)
        self.debug = False

        # NOTE toggle this to True to enable realtime path planning (i.e. plan a new path every time the car moves or continuous callbacks)

        self.declare_parameter("final_challenge", True)
        self.declare_parameter("is_sim", True)
        self.declare_parameter("debug", False)
        self.declare_parameter("odom_topic", "default")
        self.declare_parameter("map_topic", "default")
        self.declare_parameter("initial_pose_topic", "default")
        
        self.final_challenge = self.get_parameter("final_challenge").get_parameter_value().bool_value
        self.is_sim = self.get_parameter("is_sim").get_parameter_value().bool_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter("initial_pose_topic").get_parameter_value().string_value

        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_sub = self.create_subscription(
            msg_type=OccupancyGrid, topic=self.map_topic, callback=self.map_cb, qos_profile=map_qos
        )

        self.map_pub = self.create_publisher(OccupancyGrid, "/dilated_map", 
                                             qos_profile=map_qos)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            qos_profile=extra_eval_configs.goal_and_pose_qos(),
        )
        #  for the final challenge, we can pick two points on the map as shrink ray locaitons
        self.click_sub = self.create_subscription(PointStamped, "/clicked_point", self.clicked_pose, 10)
        
        

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            qos_profile=QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # self.pose_est_sub = self.create_subscription(Odometry, "pf/pose/odom", self.odom_cb, 10)
        self.goal = None
        self.start = None
        self.map = None
        self.R = self.C = None
        self.dilated_map = None

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        # added pose estimates
        self.start_pub = self.create_publisher(Marker, "/start", 1)
        self.goal_pub = self.create_publisher(Marker, "/goal", 1)
        if self.final_challenge:
            self.shrinkray_pts_pub = self.create_publisher(Marker, "/ray_pts", 2)
            self.data_points: List[Tuple[float, float]] = []
            self.count = 0
            self.trip_segment = 0

    def map_cb(self, msg):

        map = msg.data
        self.get_logger().info(f"ORIGINAL Map data length: {msg.info.height}x {msg.info.width}")
        self.map = np.array(map).reshape((msg.info.height, msg.info.width))
        self.R = msg.info.height
        self.C = msg.info.width
        self.get_logger().info(f"Unique values: {np.unique(map)}")
        self.get_logger().info("Got map")
        self.map_info = msg.info
        

        self.dilated_map = self.dilate_occupancy_map(
            occupancy_data=msg.data,
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            dilation_radius_meters=self.safety_buffer,
            occupancy_threshold=50,
        )
        # self.map *= 100
        self.convert_map_to_publisher_format()

    def convert_map_to_publisher_format(self):
        # if self.debug:
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_info

        self.dilated_map *= 100
        if self.debug:
            self.get_logger().info(f"Unique values: {np.unique(self.dilated_map)}")
        msg.data = self.dilated_map.flatten().tolist()  # Flatten the 2D array to 1D
        self.map_pub.publish(msg)
        self.get_logger().info(f"Dilated map dimensions: {msg.info.height}x {msg.info.width}")
        self.get_logger().info(f"Map data length: {(len(msg.data))}")
        # self.get_logger().info(f"Map data: {}")
        self.get_logger().info(f"Published dilated map")

    def dilate_occupancy_map(
        self, occupancy_data, width, height, resolution, dilation_radius_meters, occupancy_threshold=50
    ):
        """
        Preprocess the occupancy grid by performing morphological dilation to "inflate" obstacles.

        Any cell with a value greater than occupancy_threshold or with a value of -1 (unknown)
        is considered an obstacle.

        Args:
            occupancy_data: The flat occupancy grid data (list or numpy array).
            width: Grid width.
            height: Grid height.
            resolution: Map resolution (meters per cell).
            dilation_radius_meters: Desired safety buffer in meters.
            occupancy_threshold: Cells with a value greater than this threshold are obstacles.

        Returns:
            A dilated binary numpy array where obstacles (including their safety margin) are marked as 1.
        """
        # Reshape occupancy data to a 2D grid.
        grid = np.array(occupancy_data).reshape((height, width))
        # Create a binary map: mark as obstacle if occupancy > threshold OR if occupancy == -1 (unknown).
        binary_map = np.uint8((grid > occupancy_threshold) | (grid == -1))

        # Calculate the number of cells required to cover the dilation radius.
        # Here, we force the effective half-width to be at least ceil(dilation_radius_meters/resolution)
        safety_margin_cells = int(np.ceil(dilation_radius_meters / resolution))
        # Kernel size is set such that half the kernel is at least safety_margin_cells.
        kernel_size = 2 * safety_margin_cells - 1
        # Create an elliptical (disk) structuring element.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Perform the dilation.
        dilated = cv2.dilate(binary_map, kernel, iterations=1)

        self.get_logger().info(
            f"Performed dilation with kernel size {kernel_size} (cells); effective margin ≈ {(kernel_size-1)/2 * resolution:.2f} m"
        )
        return dilated
    def dilate_obstacles(self, occupancy_data, width, height, resolution, dilation_radius_meters, occupancy_threshold=0):
        """
        Preprocess the occupancy grid by performing morphological dilation to "inflate" obstacles.

        Any cell with a value greater than occupancy_threshold or with a value of -1 (unknown)
        is considered an obstacle.

        Args:
            occupancy_data: The flat occupancy grid data (list or numpy array).
            width: Grid width.
            height: Grid height.
            resolution: Map resolution (meters per cell).
            dilation_radius_meters: Desired safety buffer in meters.
            occupancy_threshold: Cells with a value greater than this threshold are obstacles.

        Returns:
            A dilated binary numpy array where obstacles (including their safety margin) are marked as 1.
        """
        
        grid = np.array(occupancy_data).reshape((height, width))
        # Create a binary map: mark as obstacle if occupancy > threshold OR if occupancy == -1 (unknown).
        binary_map = np.uint8((grid > occupancy_threshold) | (grid == -1))

        # Calculate the number of cells required to cover the dilation radius.
        # Here, we force the effective half-width to be at least ceil(dilation_radius_meters/resolution)
        safety_margin_cells = int(np.ceil(dilation_radius_meters / resolution))
        # Kernel size is set such that half the kernel is at least safety_margin_cells.
        kernel_size = 2 * safety_margin_cells - 1
        # Create an elliptical (disk) structuring element.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Perform the dilation.
        dilated = cv2.dilate(binary_map, kernel, iterations=1)

        self.get_logger().info(
            f"Performed dilation with kernel size {kernel_size} (cells); effective margin ≈ {(kernel_size-1)/2 * resolution:.2f} m"
        )
        return dilated

    def is_collision_free(self, x1, y1, x2, y2, num_points=20):
        """
        Check if the straight-line path between (x1, y1) and (x2, y2) is collision free,
        using the preprocessed, dilated map.
        """
        if self.dilated_map is None:
            self.get_logger().warn("No dilated map available for collision check")
            return False

        try:
            pts = np.linspace([x1, y1], [x2, y2], num=num_points)
            for pt in pts:
                cell = world_to_grid(pt[0], pt[1], self.map_info, self.get_logger(), debug=self.debug)
                if cell is None:
                    return False
                u, v = cell
                if u < 0 or u >= self.map_info.width or v < 0 or v >= self.map_info.height:
                    return False
                if self.dilated_map[v, u] != 0:
                    return False
            return True
        except Exception as e:
            self.get_logger().error(f"Collision check error: {str(e)}")
            return False
    def clicked_pose(self, msg):
        self.count += 1
        point = Point()
        point.x = msg.point.x
        point.y = msg.point.y
        self.data_points.append((point.x, point.y))
        self.mark_pt(self.shrinkray_pts_pub, (0.0, 1.0, 0.0), self.data_points)
        if self.count > 3:
            self.get_logger().info("Finding path to nearest checkpoint")
            start_point = (self.start.position.x, self.start.position.y)
            sorted_checkpoints = sorted(
                self.data_points,
                key=lambda checkpoint: ((checkpoint[0] - start_point[0]) ** 2 + (checkpoint[1] - start_point[1]) ** 2) ** 0.5
            )
            self.data_points = sorted_checkpoints
            self.get_logger().info(f"Waypoints: {self.waypoints}")
            self.plan_trip()
            
    def arrived(self):
        self.trip_segment += 1
    '''
    obstacles: list of tuples (x, y) representing the world coordinates of obstacles
    This function should be called when the robot detects an obstacle in its path.
    It will update the map and replan the path to avoid the obstacles.
    NOTE: 
    '''
    def obstacle_handler(self, obstacles):
        # Convert world coordinates to grid coordinates
        obstacle_coords = [
            world_to_grid(obstacle[0], obstacle[1], self.map_info, self.get_logger(), debug=self.debug)
            for obstacle in obstacles
        ]
        # Mark the obstacles on the map
        for x,y in obstacle_coords:
            if 0 <= x < self.map_info.width and 0 <= y < self.map_info.height:
                self.map[y,x] = 100
                if self.debug:
                    self.get_logger().info(f"Obstacle at grid coordinates: ({x}, {y}, map coordinates, )")
        # Dilate the map to account for the safety buffer
        self.dilated_map = self.dilate_occupancy_map(
            occupancy_data=self.map.flatten().tolist(),
            width=self.map_info.width,
            height=self.map_info.height,
            resolution=self.map_info.resolution,
            dilation_radius_meters= self.obstacle_buffer ,
            occupancy_threshold=50
        )
        
            
        # Replan the path
        self.plan_trip()
                
        
    def plan_trip(self):
        # the start and end points depend on what part of the trip we are on
        if self.trip_segment == 0:
            start_point = (self.start.position.x, self.start.position.y)
            end_point = self.data_points[0]
        elif self.trip_segment == 1:
            start_point = self.data_points[0]
            end_point = self.data_points[1]
        elif self.trip_segment == 2:
            start_point = self.data_points[1]
            end_point = (self.start.position.x, self.start.position.y)
    
       
        start_coords = world_to_grid(start_point[0], start_point[1], self.map_info, self.get_logger(), debug=self.debug)
        end_coords = world_to_grid(end_point[0], end_point[1], self.map_info, self.get_logger(), debug=self.debug)
        self.get_logger().info(f"Planning path... (in grid) start coords: {start_coords} to end coords: {end_coords}")
        self.plan_path(start_coords, end_coords, self.dilated_map)
            
    def smooth_path(self, path):
        """
        Simplify a path by repeatedly trying to shortcut between non-consecutive points.

        Args:
        path: List of [x, y] points (in world coordinates) representing the path.

        Returns:
        A simplified, smoothed path.
        """
        if not path:
            return []

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Try to jump as far forward as possible in the path.
            for j in range(len(path) - 1, i, -1):
                if self.is_collision_free(smoothed[-1][0], smoothed[-1][1], path[j][0], path[j][1]):
                    # We can connect directly from point i to point j, skip the ones in between.
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                # If no shortcut is found (should not happen if i+1 is always collision free), advance by one.
                i += 1
                smoothed.append(path[i])

        return smoothed

    def pose_cb(self, pose):
        if not hasattr(self, "map"):
            self.get_logger().warn("Waiting for map...")
            return
        if not hasattr(self, "goal"):
            self.get_logger().warn("No goal set yet...")
        self.start = pose.pose.pose
        self.trajectory.addPoint((self.start.position.x, self.start.position.y))
        
        self.get_logger().info("Got start pose")
        if self.debug:
            self.get_logger().info(f"Start pose: {self.start}")
        self.visualize_pose(pose=self.start, publisher=self.start_pub, color=(1.0, 0.0, 0.0, 1.0), id=0)
        self.count += 1

    def goal_cb(self, msg):
        if not hasattr(self, "map"):
            self.get_logger().warn("Waiting for map...")
            return

        if self.start == None:
            self.get_logger().warn("No start pose sent...")
            return
            # return
        self.goal = msg.pose
        self.get_logger().info("Got goal pose")

        if self.debug:
            self.get_logger().info(f"Goal pose: {self.goal}")
        self.visualize_pose(pose=self.goal, publisher=self.goal_pub, color=(0.0, 1.0, 0.0, 1.0), id=1)
       
        start_point = (self.start.position.x, self.start.position.y)
        start_coords = world_to_grid(self.start.position.x, self.start.position.y, self.map_info, self.get_logger(), debug=self.debug)
        end_point = (self.goal.position.x, self.goal.position.y)
        end_coords = world_to_grid(self.goal.position.x, self.goal.position.y, self.map_info, self.get_logger(), debug=self.debug)
        self.get_logger().info(f"End points (in world) start: {start_point} to {end_point}")

        
        self.get_logger().info(f"Planning path... (in grid) start coords: {start_coords} to end coords: {end_coords}")

        self.plan_path(start_coords, end_coords, self.dilated_map)
        
    # def odom_cb(self, msg):
    #     current_pose = msg.pose.pose
    #     if not self.start:
    #         self.get_logger().warn("Waiting for start pose...")
    #         return
    #     if not self.goal:
    #         self.get_logger().warn("Waiting for goal pose...")
    #         return
    #     position = (current_pose.position.x, current_pose.position.y)
    #     if not self.published:
    #         self.get_logger().info(f"Current pose: {position}")
    #         start_point = (self.start.position.x, self.start.position.y)
    #         end_point = (self.goal.position.x, self.goal.position.y)
    #         self.plan_path(start_point, end_point, self.map)
    #         self.get_logger().info(f"Planning path...")
    #         self.published = True
    #     else:
    #         return 

    def plan_path(self, start_point, end_point, map):
        try:
            return self.plan_path_(start_point, end_point, map)
        except Exception as e:
            self.get_logger().info(f"plan_path: failed \n {e}")
            print(e)

    def plan_path_(self, start_point, end_point, map):
        a_star = AStar(map, self.get_logger(), debug=self.debug)
        path = a_star.a_star(start=start_point, goal=end_point)
        if path is None:
            self.get_logger().error("plan_path: No path found")
            return
        if self.debug:
            self.get_logger().info(f"Path found: {path}\n\n")
        path = [grid_to_world(p[0], p[1], self.map_info, self.get_logger(), debug=self.debug) for p in path]
        if self.debug:
            self.get_logger().info(f"Path found in world coordinates: {path}\n\n")

        # Create PoseArray for visualization
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()

        # Clear existing trajectory
        self.trajectory.clear()
        # path = self.smooth_path(path)

        # Convert path to poses and add to trajectory
        for i in range(len(path)):
            # Create pose
            pose = Pose()
            pose.position.x = path[i][0]
            pose.position.y = path[i][1]
            pose.position.z = 0.0

            # Calculate orientation for all except last point
            if i < len(path) - 1:
                dx = path[i + 1][0] - path[i][0]
                dy = path[i + 1][1] - path[i][1]
                yaw = atan2(dy, dx)
                pose.orientation.z = sin(yaw / 2)
                pose.orientation.w = cos(yaw / 2)
            else:
                # Use previous orientation for last point
                pose.orientation.z = pose_array.poses[-1].orientation.z if pose_array.poses else 0.0
                pose.orientation.w = pose_array.poses[-1].orientation.w if pose_array.poses else 1.0

            pose_array.poses.append(pose)
            # Add to trajectory using position only
            self.trajectory.addPoint([pose.position.x, pose.position.y])

        # Publish results
        self.get_logger().info("Publishing trajectory...")
        self.traj_pub.publish(pose_array)
        self.trajectory.publish_viz()
        self.get_logger().info("Path published successfully")

        # self.traj_pub.publish(self.trajectory.toPoseArray())
        # self.trajectory.publish_viz()
    def mark_pt(self, subscriber, color_tup, data):
        msg_data = self.tuple_to_point(data)

        mark_pt = Marker()
        mark_pt.header.frame_id = "/map"
        mark_pt.header.stamp = self.get_clock().now().to_msg()
        mark_pt.type = mark_pt.SPHERE_LIST
        mark_pt.action = mark_pt.ADD
        mark_pt.scale.x = 0.5
        mark_pt.scale.y = 0.5
        mark_pt.scale.z = 0.5
        mark_pt.color.a = 1.0
        mark_pt.color.r = color_tup[0]
        mark_pt.color.g = color_tup[1]
        mark_pt.color.b = color_tup[2]
        mark_pt.points = msg_data
        subscriber.publish(mark_pt)
    def visualize_pose(self, pose, publisher, color, id):
        pt = Marker()
        pt.type = Marker.SPHERE
        pt.id = id
        pt.action = Marker.ADD
        pt.header.frame_id = "/map"
        if self.debug:
            self.get_logger().info(f"Visualizing pose: {pose} {type(pose)}")
        pt.pose = pose
        pt.scale.x = pt.scale.y = pt.scale.z = 0.2
        pt.color.r, pt.color.g, pt.color.b, pt.color.a = color
        publisher.publish(pt)
        if self.debug:
            self.get_logger().info(f"Published pose visualization")

    
def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
