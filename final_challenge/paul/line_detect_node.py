import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
import cv2
import numpy as np
  ### Pairs of parallel lines
  ### Be ready for skips
# Import the processing function from your existing module
from line_finder import process_frame


PTS_IMAGE_PLANE = [[339.0, 181.0],
                   [553.0, 196.0],
                   [216.0, 177.0],
                   [234.0, 261.0],
                   [590.0, 210.0]] # dummy points
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[67, 0],
                    [45, -28],
                    [73, 27.5],
                    [22, 9.5],
                    [38,-29]] # dummy points
######################################################

METERS_PER_INCH = 0.0254


class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.bridge = CvBridge()
        # Subscribe to raw camera images
        self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10
        )
        # Publish processed lane overlay
        self.pub = self.create_publisher(
            Image,
            '/lane_detector/image',
            10
        )
        self.get_logger().info('LaneDetectorNode initialized')
        self.marker_pub = self.create_publisher(Marker, "/destination_marker", 1)
        # self.last_left = None
        # self.last_right = None
        # self.left_counter = 0
        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])
        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)


    def image_callback(self, msg: Image):
        
        # Convert to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        debug_img = frame.copy()
        # Run your detection
        lines, intersection = process_frame(frame)
        # left_segs, right_segs, left, right = self.cluster_and_fit(frame, lines)

        # for ln in (left, right):
        #     if ln is not None:
        #         x1,y1,x2,y2 = ln
        #         cv2.line(debug_img, (x1,y1), (x2,y2), (0,255,0), 5)


        # # out = process_frame(frame)
        # # # Convert back to ROS Image and publish
        out_msg = self.bridge.cv2_to_imgmsg(lines, 'bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)
        if intersection is not None:
            u, v = intersection
            x, y = self.transformUvToXy(u,v)
            marker = Marker()
            marker.header.frame_id = '/destination_marker'
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.scale.x = .2
            marker.scale.y = .2
            marker.scale.z = .2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = .5
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = x
            marker.pose.position.y = y
            self.marker_pub.publish(marker)


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

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
