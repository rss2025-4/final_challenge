#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
# from vs_msgs.msg import ConeLocation, ConeLocationPixel
from geometry_msgs.msg import Point

#The following collection of pixel locations and corresponding relative
#ground plane locations are used to compute our homography matrix

# PTS_IMAGE_PLANE units are in pixels
# see README.md for coordinate frame description

######################################################
# ## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
# PTS_IMAGE_PLANE = [[639.0, 200.0],
#                    [336.0, 177.0],
#                    [515.0, 176.0],
#                    [88.0, 199.0],] # dummy points
# ######################################################

# # PTS_GROUND_PLANE units are in inches
# # car looks along positive x axis with positive y axis to left

# ######################################################
# ## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE in inches
# PTS_GROUND_PLANE = [[50.25, -52],
#                     [99.25, 0],
#                     [99.25, -52],
#                     [56, 42]] # dummy points
# PTS_IMAGE_PLANE = [[348.0, 189.0],
#                    [508.0, 182.0],
#                    [161.0, 213.0],
#                    [610.0, 253.0],
#                    [350.0, 216.0], 
#                    [164.0, 197.0]] # dummy points
# ######################################################

# # PTS_GROUND_PLANE units are in inches
# # car looks along positive x axis with positive y axis to left

# ######################################################
# ## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
# PTS_GROUND_PLANE = [[66.5, 0],
#                     [99, -44],
#                     [43, 24],
#                     [26, -20],
#                     [41,0],
#                     [45, 30]] # dummy points
######################################################
PTS_IMAGE_PLANE = [[329.0,188.0],
                    [521.0,223.0],
                    [93.0, 269.0],
                    [265.0,192.0],
                    [47.0,228.0]
                   ] # dummy points
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[69.0,0.0,],
                    [37.5, -21.0],
                    [25.0, 18.0],
                    [70.75,13.0],
                    [38.0,30.0]
                    ] # dummy points

METERS_PER_INCH = 0.0254


class HomographyTransformer(Node):
    def __init__(self):
        super().__init__("homography_transformer")

        # self.cone_pub = self.create_publisher(ConeLocation, "/relative_cone", 10)
        # self.marker_pub = self.create_publisher(Marker, "/cone_marker", 1)
        # self.cone_px_sub = self.create_subscription(ConeLocationPixel, "/relative_cone_px", self.cone_detection_callback, 1)

        # TESTING - gets mouse clicked point in rqt image view
        # self.mouse_px_sub = self.create_subscription(Point, "/zed/zed_node/rgb/image_rect_color_mouse_left", self.mouse_detection_callback, 1)
        
        # test_timer_period = 1 / 20  # seconds, 60 Hz
        # self.test_timer = self.create_timer(test_timer_period, self.test_pub_callback)

        # self.test_pub = self.create_publisher(ConeLocation, "/cone_marker_2", 1)
        
        if not len(PTS_GROUND_PLANE) == len(PTS_IMAGE_PLANE):
            rclpy.logerr("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")

        #Initialize data into a homography matrix

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        self.get_logger().info("Homography Transformer Initialized")
    
    # def test_pub_callback(self):
    #     relative_xy_msg = ConeLocation()
    #     relative_xy_msg.x_pos = 1.0
    #     relative_xy_msg.y_pos = 1.0

    #     self.test_pub.publish(relative_xy_msg)

    def cone_detection_callback(self, msg):
        self.get_logger().info("in Homography Transformer Cone Detection Callback")
        #Extract information from message
        u = msg.u
        v = msg.v

        #Call to main function
        x, y = self.transformUvToXy(u, v)
        print("u,v in cone callback: ", u,v)
        print("x,y in cone callback: ", x,y)

        #Publish relative xy position of object in real world
        relative_xy_msg = ConeLocation()
        relative_xy_msg.x_pos = x
        relative_xy_msg.y_pos = y

        self.cone_pub.publish(relative_xy_msg)

        # RVIZ
        self.draw_marker(x, y, "/cone_marker")

    def mouse_detection_callback(self, msg):
        self.get_logger().info("in Homography Transformer Mouse Detection Callback")
        #Extract information from message
        u = msg.x
        v = msg.y
        # print("u,v in mouse callback: ", u,v)

        #Call to main function
        x, y = self.transformUvToXy(u, v)

        print("x,y in cone callback: ", x,y)

        #Publish relative xy position of object in real world
        relative_xy_msg = ConeLocation()
        relative_xy_msg.x_pos = x
        relative_xy_msg.y_pos = y

        self.cone_pub.publish(relative_xy_msg)

        # RVIZ
        # self.draw_marker(x, y, "/cone_marker")


    def transformUvToXy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y

    def draw_marker(self, cone_x, cone_y, message_frame):
        """
        Publish a marker to represent the cone in rviz.
        (Call this function if you want)
        """
        marker = Marker()
        marker.header.frame_id = message_frame
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = .5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = cone_x
        marker.pose.position.y = cone_y
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    homography_transformer = HomographyTransformer()
    rclpy.spin(homography_transformer)
    rclpy.shutdown()

if __name__ == "__main__":
    main()