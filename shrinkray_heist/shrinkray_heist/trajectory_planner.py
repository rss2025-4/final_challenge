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
    """Listens for path points published by the state node and uses it to plan a path
    """

    def __init__(self, extra_eval_configs=InjectedConfig()):
        super().__init__("trajectory_planner", **extra_eval_configs.extra_node_kwargs())

       
       
        self.declare_parameter("safety_buffer", 0.3)
        self.declare_parameter("odom_topic", "default")
        self.declare_parameter("map_topic", "default")
        self.declare_parameter("path_topic", "default")

        # # how much to dilate the map, higher value = safer, less agressive path finding
        self.safety_buffer = self.get_parameter("safety_buffer").get_parameter_value().double_value
        
        # Other parameters
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value

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

        self.path_sub = self.create_subscription(
            PoseArray,
            self.path_topic,
            self.path_cb,
            qos_profile=extra_eval_configs.goal_and_pose_qos(),
        )
        

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

         # TODO: if needed, implement dilation specifically for obstacles
        self.obstacle_buffer = 0.1

        # NOTE toggle this to True to enable debug mode (i.e. add additional logging)
        self.debug = False
        self.external_map = True
        
        self.map = None
        self.dilated_map = None

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        
    
    def init_map(self):
        
        if self.external_map:
            
            # image = cv2.imread("flipped_occupany_map_conservative.png", cv2.IMREAD_GRAYSCALE)
            image = cv2.imread("dilated_occupancy_map_pop.png", cv2.IMREAD_GRAYSCALE)
            self.get_logger().info("dilated_occupancy_map_pop")
            newmap = np.array(image).astype(np.uint8)
            newmap = np.flipud(newmap)
            shift = lambda x: x / 255
            newmap = shift(newmap)
            newmap = np.logical_not(newmap)
            
            self.dilated_map = newmap.astype(np.int8)
            
            
            # newmap = newmap.reshape((self.map_info.height, self.map_info.width))
            self.get_logger().info(f"PathPlan: Unique values for NEW loaded map: {np.unique(newmap)}")
            self.get_logger().info(f"PathPlan: NEW loaded dimensions: np.shape {newmap.shape}")
            self.convert_map_to_publisher_format()
            
            
        
        
        

    def map_cb(self, msg):
        
        map = msg.data
        self.get_logger().info(f"PathPlan: ORIGINAL Map data length: {msg.info.height}x {msg.info.width}")
        self.map = np.array(map).reshape((msg.info.height, msg.info.width))
        
        self.get_logger().info(f"PathPlan: Unique values: {np.unique(map)}")
        self.get_logger().info(f"PathPlan: ORIGINAL DILATED MAP data type {self.map.dtype}")
        self.get_logger().info("PathPlan: Got map")
        self.map_info = msg.info
        
        # grid = np.array(msg.).reshape((height, width))
        # # Create a binary map: mark as obstacle if occupancy > threshold OR if occupancy == -1 (unknown).
        # binary_map = np.uint8((grid > 50) | (grid == -1))
        # # map is predilated 
        # self.dilated_map = binary_map 
        if self.external_map:
            self.init_map()
            dilated_map = self.dilate_occupancy_map(
                occupancy_data=msg.data,
                width=msg.info.width,
                height=msg.info.height,
                resolution=msg.info.resolution,
                dilation_radius_meters=self.safety_buffer,
                occupancy_threshold=50,
                )
            
            
                
        # pass
        else:
            self.dilated_map = self.dilate_occupancy_map(
                occupancy_data=msg.data,
                width=msg.info.width,
                height=msg.info.height,
                resolution=msg.info.resolution,
                dilation_radius_meters=self.safety_buffer,
                occupancy_threshold=50,
            )
            self.get_logger().info(f"PathPlan: ORIGINAL DILATED MAP dimensions: np.shape {self.dilated_map.shape}")
            self.get_logger().info(f"PathPlan: ORIGINAL DILATED MAP unique values: np.shape {np.unique(self.dilated_map)}")
            self.get_logger().info(f"PathPlan: ORIGINAL DILATED MAP data type {self.dilated_map.dtype}")
            self.convert_map_to_publisher_format()
        self.save_map_as_img("old_occupancy_map.png",  width=msg.info.width,
                height=msg.info.height)
        # self.map *= 100
        
        
    
    def save_map_as_img(self, filename,width, height):
        """
        Save the map as an image file.
        """
        
        if self.dilated_map is None:
            self.get_logger().warn("No dilated map available to save")
            return
        self.get_logger().info(f"PathPlan: Saving map as {filename}, these are the unique values: {np.unique(self.dilated_map)}")
        grid = self.dilated_map.reshape((height, width))
        # Convert the map to a format suitable for saving as an image
        binary_map = np.uint8(np.logical_not(grid))
        

        
        img = (binary_map * 255).astype(np.uint8)

        # Save the image
        cv2.imwrite(filename, img)
        self.get_logger().info(f"Map saved as {filename}")
        

    def convert_map_to_publisher_format(self):
        # if self.debug:
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_info

        self.dilated_map *= 100
        if self.debug:
            self.get_logger().info(f"PathPlan: Unique values: {np.unique(self.dilated_map)}")
        msg.data = self.dilated_map.flatten().tolist()  # Flatten the 2D array to 1D
        self.map_pub.publish(msg)
        self.get_logger().info(f"PathPlan: Dilated map dimensions: {msg.info.height}x {msg.info.width}")
        self.get_logger().info(f"PathPlan: Map data length: {(len(msg.data))}")
        # self.get_logger().info(f"PathPlan: Map data: {}")
        self.get_logger().info(f"PathPlan: Published dilated map")

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
    
            
    
    '''
    obstacles: list of tuples (x, y) representing the world coordinates of obstacles
    This function should be called when the robot detects an obstacle in its path.
    It will update the map and replan the path to avoid the obstacles.
    NOTE: not implemented yet
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
                    self.get_logger().info(f"PathPlan: Obstacle at grid coordinates: ({x}, {y}, map coordinates, )")
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
    
    '''
    Args: 
        msg: PoseArray message containing the start and end pose of a trip 
    
    This function will plan a trip from the start pose to the end pose using the A* algorithm.
    '''            
        
    def path_cb(self,msg):
        # the start and end points depend on what part of the trip we are on
        coords = []
        for pose in msg.poses:
            x, y = pose.position.x, pose.position.y
            coords.append( world_to_grid(x, y, self.map_info, self.get_logger(), debug=self.debug))
            
        if len(coords) != 2:
            self.get_logger().error("PathPlan: path_cb: Invalid path message (not 2 points)")
            return
        self.get_logger().info(f"PathPlan: path_cb: Planning path... (in grid) start coords: {coords[0]} to end coords: {coords[1]}")
        self.plan_path(coords[0],coords[1], self.dilated_map)
     
        
    # TODO: implement smooth path        
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

    

    def plan_path(self, start_point, end_point, map):
        try:
            return self.plan_path_(start_point, end_point, map)
        except Exception as e:
            self.get_logger().info(f"PathPlan: plan_path: failed \n {e}")
            print(e)

    def plan_path_(self, start_point, end_point, map):
        a_star = AStar(map, self.get_logger(), debug=self.debug)
        path = a_star.a_star(start=start_point, goal=end_point)
        if path is None:
            self.get_logger().error("PathPlan: plan_path: No path found")
            return
        if self.debug:
            self.get_logger().info(f"PathPlan: Path found: {path}\n\n")
        path = [grid_to_world(p[0], p[1], self.map_info, self.get_logger(), debug=self.debug) for p in path]
        if self.debug:
            self.get_logger().info(f"PathPlan: Path found in world coordinates: {path}\n\n")

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
        self.get_logger().info("PathPlan: Publishing trajectory...")
        self.traj_pub.publish(pose_array)
        self.trajectory.publish_viz()
        self.get_logger().info("PathPlan: Path published successfully")

       
   
    def visualize_pose(self, pose, publisher, color, id):
        pt = Marker()
        pt.type = Marker.SPHERE
        pt.id = id
        pt.action = Marker.ADD
        pt.header.frame_id = "/map"
        if self.debug:
            self.get_logger().info(f"PathPlan: Visualizing pose: {pose} {type(pose)}")
        pt.pose = pose
        pt.scale.x = pt.scale.y = pt.scale.z = 0.2
        pt.color.r, pt.color.g, pt.color.b, pt.color.a = color
        publisher.publish(pt)
        if self.debug:
            self.get_logger().info(f"PathPlan: Published pose visualization")

    
def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
