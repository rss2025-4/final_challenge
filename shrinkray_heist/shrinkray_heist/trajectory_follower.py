import numpy as np
import rclpy
import time

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseArray, PoseWithCovarianceStamped, PoseStamped, Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String, Int32
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

from .utils import LineTrajectory

from .definitions import Drive
from shrinkray_heist.definitions import  Target, TripSegment, ObjectDetected, State

class PurePursuit(Node):
    """Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed."""

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter("odom_topic", "default")
        self.declare_parameter("drive_topic", "default")

        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value

        # self.odom_topic = "/pf/pose/odom"
        # self.odom_topic = "/odom"

        self.get_logger().info('odom topic: "%s"' % self.odom_topic)
        self.get_logger().info('drive topic: "%s"' % self.drive_topic)

        # print(f"odom_topic: {self.odom_topic}")
        # print(f"drive_topic: {self.drive_topic}")

        self.speed = 0.3  # ADJUST SPEED m/s#
        self.lookahead = 0.8 # ADJUST LOOKAHEAD m -- NEEDS TO BE TUNED IN REAL LIFE 
        self.steering_angle = 0.0
        # FOR VARIABLE LOOKAHEAD (MAYBE NOT NEEDED FOR FINAL RACE THOUGH)
        # self.max_speed = self.speed + 1.0

        # self.max_lookahead = np.clip(2.0 * self.speed, 0.75, 0.9)  # Adjust max gain
        # self.min_lookahead = np.clip(1.0 * self.speed, 0.6, 0.8)  # Adjust min gain
        # self.lookahead = self.min_lookahead

        self.wheelbase_length = 0.33
        self.max_steer = np.pi / 4

        self.initialized_traj = False
        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray, "/trajectory/current", self.trajectory_callback, 1)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)
        self.back_drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/safety", 1)
        self.lookahead_publisher = self.create_publisher(Marker, "/lookahead_point", 1)
        self.nearest_dist_pub = self.create_publisher(Float32, "/nearest_dist", 10)

        self.min_dist = 0.0

        # for sim
        # self.alert_subscriber = self.create_subscription(String, "/alert", self.alert_callback, 10)
        # self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 10)

        
        # for publishing to state node
        self.purepursuit_state_pub = self.create_publisher(Int32, "/pursuit_state", 10)

        # listen for state machine state
        self.state_sub = self.create_subscription(Int32, "/toggle_state", self.state_cb, 10)
        self.purepursuit_on = True # default is on
        # self.stop = False # default is going to drive
        self.goal_reached = False
        self.dist_to_last_point = 0.0

        # for back up and aligning with goal
        self.initiate_back_up = False
        self.backup_start_time = None
        self.backup_state = 0


        self.alert_sub = self.create_subscription(String, "/alert", self.stopalert_cb, 10)

        # TODO add topic
        # for orientation of goal pose (listen from states)
        self.goal_sub = self.create_subscription(Pose, "/curr_goal_pose", self.curr_goal_pose_cb, 10)
        self.curr_goal_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        
        self.get_logger().info("Pure Pursuit Initialized")


    def state_cb(self, msg):
        # pass # for testing
        if msg.data == Target.FOLLOWER.value:
            self.get_logger().info("Follower: Received Target.FOLLOWER")
            self.purepursuit_on = not self.purepursuit_on

            if self.purepursuit_on:
                self.get_logger().info("Follower: Pure Pursuit Activated")
                # self.stop = False
            else:
                self.get_logger().info("Follower: Pure Pursuit Deactivated")
        else:
            self.get_logger().info("Follower: NOT Target.follower")

    def curr_goal_pose_cb(self, msg):
        z_rotation = euler_from_quaternion(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )[2]
        self.curr_goal_pose = np.array([msg.position.x, msg.position.y, z_rotation])
        self.get_logger().info('Current goal pose "%s"' % self.curr_goal_pose) 
           
    def stopalert_cb(self, msg):
        if self.purepursuit_on and msg.data == "STOP":
            # back up 
            # self.get_logger().info("Follower: Received stop alert msg")

            # Declare msg
            drive_msg = AckermannDriveStamped()

            curr_time = self.get_clock().now()

            # assign values
            drive_msg.header.stamp = curr_time.to_msg()
            drive_msg.header.frame_id = "base_link"
            drive_msg.drive.steering_angle = -self.steering_angle  # rad
            drive_msg.drive.steering_angle_velocity = 0.0  # rad/s
            drive_msg.drive.speed = -0.3  # m/s
            drive_msg.drive.acceleration = 0.0  # m/s^2
            drive_msg.drive.jerk = 0.0  # m/s^3

            self.back_drive_pub.publish(drive_msg)

    # def pose_callback(self, odometry_msg):
    #     "For sim testing with safety controller"
    #     self.get_logger().info("Initial pose callback")
        # self.stop = False

    # def alert_callback(self, alert_msg):
    #     """For sim, but can use as for debug, testing with safety controller"""
    #     if alert_msg.data == "STOP":
    #         self.stop = True
    #         self.get_logger().info('Got alert msg: "%s"' % alert_msg.data)
        # else:
        #     self.stop = False

    def nearest_point(self, p, trajectory):
        """
        Finds the nearest point on the trajectory to the current pose using vectorized numpy operations.
        """
        trajectory = np.asarray(trajectory)
        p = np.asarray(p)

        # Get all segment start and end points
        p1s = trajectory[:-1]  # all but last
        p2s = trajectory[1:]  # all but first

        segment_vectors = p2s - p1s
        p_vectors = p - p1s  # broadcasted

        # Project point p onto each segment vector
        segment_lengths_squared = np.sum(segment_vectors**2, axis=1)

        # batch dot product: for each segment i, we want t_i = (p-p1_i) . (p2_i - p1_i) / |p2_i - p1_i|^2
        # np.einsum('ij,ij->i', A, B) = np.sum(A * B, axis=1) but more efficient?
        t = np.einsum("ij,ij->i", p_vectors, segment_vectors) / segment_lengths_squared
        t = np.clip(t, 0.0, 1.0)  # Clip to within segment

        # Get projected points (use broadcasting)
        # for each segment, we want proj_i = p1_i + t_i * (p2_i - p1_i)
        projections = p1s + (t[:, np.newaxis] * segment_vectors)

        # Compute distances from p to each projection
        dists = np.linalg.norm(projections - p, axis=1)

        # Find the closest projection
        min_index = np.argmin(dists)
        nearest_point = projections[min_index]
        index_p1 = min_index
        index_p2 = min_index + 1

        # publish minimum distance for safety controller
        self.min_dist = dists[min_index]

        return nearest_point, [trajectory[index_p1], trajectory[index_p2]], index_p1, index_p2

    def circle_segment_intersection(self, p, p1, p2, index_p1, index_p2):
        """
        Finds the first intersection between a lookahead circle centered at p and
        a line segment on the trajectory, starting from index_p1.
        """
        r = self.lookahead
        p = np.asarray(p)
        last_index = index_p1
        trajectory = self.trajectory.points

        # all intersections
        all_intersections = []

        for i in range(index_p1, len(trajectory) - 1):
            p1 = np.array(trajectory[i][:2])
            p2 = np.array(trajectory[i + 1][:2])
            d = p2 - p1
            f = p1 - p

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - r**2
            discriminant = b**2 - 4 * a * c

            if discriminant < 0:
                continue  # No intersection

            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            valid_points = []

            for t in (t1, t2):
                if 0 <= t <= 1:
                    intersect = p1 + t * d
                    valid_points.append(intersect)

            if not valid_points:
                continue

            # Choose the intersection point closest to p2 (helps with moving forward)
            # checks each intersection
            # goal_point = min(valid_points, key=lambda pt: np.linalg.norm(pt - p2))
            if len(valid_points) == 2:
                dist1 = np.linalg.norm(valid_points[0] - p2)
                dist2 = np.linalg.norm(valid_points[1] - p2)

                if dist1 < dist2:
                    goal_point = valid_points[0]
                else:
                    goal_point = valid_points[1]

            elif len(valid_points) == 1:
                goal_point = valid_points[0]

            all_intersections.append(valid_points)
            if len(all_intersections) > 2:
                self.get_logger().info("More than 2 intersections found '%s'" % all_intersections)
            # Make sure the goal point is closer to p2 than the current robot position
            if np.linalg.norm(goal_point - p2) < np.linalg.norm(p - p2):
                self.goal_point = goal_point
                last_index = i
                break
            else:
                last_index = i + 1
        else:
            # No valid intersection found; fallback to last index
            self.goal_point = trajectory[last_index][:2]

        return self.goal_point
    def send_drive_command(self, speed, steering_angle):
        drive_msg = AckermannDriveStamped()
        curr_time = self.get_clock().now()
        drive_msg.header.stamp = curr_time.to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)
    
    def drivetoorientation(self, current_pose):
        
        heading = current_pose[2] 
        # if self.dist_to_last_point < 0.75: # less than 0.75 m to last point, want to orient with goal pose
        angle_to_lookahead = self.curr_goal_pose[2] 
        self.get_logger().info("initiate back up")
        
        # Step 1: Back up at an angle (steer left while reversing)
        # back up at an angle that will 
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self.backup_start_time
        heading_error = normalize_angle(angle_to_lookahead - heading)
        backup_steering = 0.4 if heading_error > 0 else -0.4
        
        if abs(heading_error) < 0.075:
            self.send_drive_command(-0.3, 0.0)
            time.sleep(1.0)
            self.get_logger().info("Goal reached")
            self.get_logger().info("Goal reached")

            # Tell state node that goal is reached
            msg = Int32()
            msg.data = Drive.GOAL_REACHED.value
            self.purepursuit_state_pub.publish(msg)
            
            self.goal_reached = True
            
            # reset variables
            self.initiate_back_up = False
            self.backup_state = 0
            self.backup_start_time = None
        
        if self.backup_state == 0:
            self.send_drive_command(-0.3, backup_steering)  # Reverse at an angle
            if elapsed > 1.0:
                self.backup_state = 1
                self.backup_start_time = now
        elif self.backup_state == 1:
            self.send_drive_command(-0.3, -backup_steering)
            if elapsed > 1.0:
                self.backup_state = 2
                self.backup_start_time = now
        elif self.backup_state == 2:    
            self.send_drive_command(0.3, backup_steering)
            if elapsed > 1.0:
                self.backup_state = 3
                self.backup_start_time = now
        elif self.backup_state == 3:
            self.send_drive_command(0.0, 0.0)
            if elapsed > 1.0:
                self.backup_state = 0
    def drive(self, current_pose, lookahead_point):

        # Pure pursuit (note angles in map frame)
        # Get the heading of the vehicle
        heading = current_pose[2]  # np.arctan2(current_pose[1], current_pose[0]) #
        # self.get_logger().info('heading "%s"' % heading)

        # Get the angle to the lookahead point
        angle_to_lookahead = np.arctan2(lookahead_point[1] - current_pose[1], lookahead_point[0] - current_pose[0])
        if self.dist_to_last_point < 0.75: # less than 0.75 m to last point, want to orient with goal pose
            # try to orient to goal 
            angle_to_lookahead = angle_to_lookahead-self.curr_goal_pose[2] # fake angle to lookahead, actualy 
            
        # self.get_logger().info('angle_to_lookahead "%s"' % angle_to_lookahead)

        # Calculate the curvature, (the curve the car follows to get to the lookahead point)
        # curvature = 1/R where R = lookahead / (2 * sin(eta))
        # eta is the angle between the vehicle heading and the line from current point to lookahead point
        eta = angle_to_lookahead - heading  # in radians

        # normalize between [-pi, pi]
        eta = np.arctan2(np.sin(eta), np.cos(eta))

        # self.get_logger().info('eta "%s"' % eta)


        ##### FOR VARIABLE LOOKAHEAD
        ## adjust lookahead distance based on current curvature
        ## increase lookahead distance with low curvature
        ## decrease lookahead distance with high curvature
        ## max sharp turn (90 degrees?)
        ## small eta (straight), larger factor,

        # curvature_factor = np.clip(1.0 - abs(eta) / (np.pi), 0.3, 1.0)

        # lookahead_gain = curvature_factor

        # self.lookahead = np.clip(lookahead_gain * self.max_lookahead, self.min_lookahead, self.max_lookahead)
        #####

        # self.get_logger().info('lookahead "%s"' % self.lookahead)
        # Compute curvature with updated lookahead
        curvature = 2 * np.sin(eta) / self.lookahead
        # self.get_logger().info('curvature "%s"' % curvature)

        # Calculate the steering angle
        self.steering_angle = np.arctan2(curvature * self.wheelbase_length, 1)
        # self.get_logger().info('steering angle "%s"' % steering_angle)

        # Restrict steering_angle (see if needed?)
        # steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        # self.get_logger().info('steering angle "%s"' % np.rad2deg(steering_angle))

        speed = self.speed

        # Declare msg
        drive_msg = AckermannDriveStamped()

        curr_time = self.get_clock().now()

        # assign values
        drive_msg.header.stamp = curr_time.to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.steering_angle  # rad
        drive_msg.drive.steering_angle_velocity = 0.0  # rad/s
        drive_msg.drive.speed = speed  # m/s
        drive_msg.drive.acceleration = 0.0  # m/s^2
        drive_msg.drive.jerk = 0.0  # m/s^3

        if self.purepursuit_on: #only publish drive msg if pure pursuit is on 
            # if not self.stop:  
            self.drive_pub.publish(drive_msg)
                # self.get_logger().info('Published drive msg: "%s"' % drive_msg.drive.speed)
        else: # stop (when goal is reached)
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
                # self.get_logger().info('Published stop drive msg: "%s"' % drive_msg.drive.speed)

    def pose_callback(self, odometry_msg):
        if self.initialized_traj:
            # self.get_logger().info('Pose callback')
            # Get the current pose of the vehicle
            current_point = np.array([odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y])
            # self.get_logger().info('Current point "%s"' % current_point)

            z_rotation = euler_from_quaternion(
                [
                    odometry_msg.pose.pose.orientation.x,
                    odometry_msg.pose.pose.orientation.y,
                    odometry_msg.pose.pose.orientation.z,
                    odometry_msg.pose.pose.orientation.w,
                ]
            )[2]
            current_pose = np.array([odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y, z_rotation])
            trajectory = np.array(self.trajectory.points)

            # check if reached last point
            last_point = np.array(trajectory[-1][:2])
            self.dist_to_last_point = np.linalg.norm(current_point - last_point)
            if self.dist_to_last_point < 0.25: # larger tolerance?
                # # try to orient to goal 
                # self.curr_heading = z_rotation # rotation, not quat
                # self.goal_heading = self.curr_goal_pose[2] # rotation, not quat

                if self.purepursuit_on and not self.goal_reached:
                    self.initiate_back_up = True
                    
                    # # NO BACK AND FORTH
                    # self.get_logger().info("Goal reached")

                    # # Tell state node that goal is reached
                    # msg = Int32()
                    # msg.data = Drive.GOAL_REACHED.value
                    # self.purepursuit_state_pub.publish(msg)
                    
                    # self.goal_reached = True

            # self.get_logger().info('np trajectory "%s"' % trajectory)
            nearest_point, [p1_nearest, p2_nearest], index_p1, index_p2 = self.nearest_point(current_point, trajectory)

            min_dist_msg = Float32()
            min_dist_msg.data = float(self.min_dist)
            # self.get_logger().info("publiished min dist '%s'" % self.min_dist)
            self.nearest_dist_pub.publish(min_dist_msg)

            lookahead_point = self.circle_segment_intersection(
                current_point, p1_nearest, p2_nearest, index_p1, index_p2
            )

            self.visualize_lookahead(lookahead_point)
            
            # NO ALIGNING
            # self.drive(current_pose, lookahead_point) # ORIGINAL

            # FOR ALIGNING WITH GOAL
            if self.initiate_back_up:
                if self.backup_start_time is None:
                    self.backup_start_time = self.get_clock().now().nanoseconds / 1e9
                self.drivetoorientation(current_pose)
            else:
                self.drive(current_pose, lookahead_point)

            # self.get_logger().info('Pose callback')

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True
        self.goal_reached = False
        # self.get_logger().info('Trajectory "%s"' % self.trajectory.points)

    def visualize_lookahead(self, point):
        """
        Visualize lookahead point
        """
        lookahead_marker = Marker()
        lookahead_marker.type = Marker.POINTS

        lookahead_marker.header.frame_id = "/map"

        lookahead_marker.scale.x = 0.5
        lookahead_marker.scale.y = 0.5
        lookahead_marker.color.a = 1.0
        lookahead_marker.color.r, lookahead_marker.color.b, lookahead_marker.color.g = 0.0, 1.0, 0.0

        # for particle in self.particles:
        p = Point()
        p.x = point[0]
        p.y = point[1]
        lookahead_marker.points.append(p)
        self.lookahead_publisher.publish(lookahead_marker)

    


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
