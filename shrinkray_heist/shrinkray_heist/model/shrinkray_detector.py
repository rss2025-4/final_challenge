import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
import os
import numpy as np

from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

from .detector import Detector #model.
from shrinkray_heist.definitions import  Target, TripSegment, ObjectDetected, State
from shrinkray_heist.homography_transformer import HomographyTransformer


class ShrinkRayDetector(Node):
    def __init__(self):
        super().__init__("shrinkray_detector")
        # self.detector = Detector() # ON REAL RACECAR
        self.detector = Detector(yolo_dir='/home/racecar/models', from_tensor_rt=False)
        self.detector.set_threshold(0.5)
        # self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.yolo_pub = self.create_publisher(Image, "/banana_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.img_cb, 1)
        self.bridge = CvBridge()
        

        # listen for state machine state
        self.state_sub = self.create_subscription(Int32, "/toggle_state", self.state_cb, 1)
        self.detector_on = True # should be False by default

        # for movement to shrink ray
        self.shrinkray_bbox = [0, 0, 0, 0]
        self.homography_transformer = HomographyTransformer()
        self.wheelbase_length = 0.33
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.get_logger().info("Shrink Ray Detector Initialized")
        
    def state_cb(self, msg):
        pass # for testing
        if msg.data == Target.DETECTOR:
            self.get_logger().info("Shrink Ray Detector Activated")
            self.detector_on = True
        else:
            self.get_logger().info("Shrink Ray Detector Deactivated")
            self.detector_on = False
        
        
    def get_relative_position(self, bbox):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.
        """
        # Get bottom center pixel coordinates of the bounding box
        x1, y1, x2, y2 = bbox
        v = y2 
        u = (x1+x2)/2

        # apply homography to u,v to get x,y in the world frame
        x, y = self.homography_transformer.transformUvToXy(u, v)
        return x,y

    def controller(self, x,y):
        # Get euclidean distance to the shrink ray
        L1 = np.linalg.norm(x,y)

        if L1 < 1.0: 
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
            self.get_logger().info("Arrived at shrink ray location, stopping")
            return
        
        # Too far, go closer to shrink ray
        eta = np.arctan2(y,x) # rad angle between the vehicle heading and the line from current point to lookahead point
        
        R = L1/(2*np.sin(eta))
        curvature = 1/R # curvature = 1/R where R = lookahead / (2 * sin(eta))

        # Calculate the steering angle
        steering_angle = np.arctan2(curvature * self.wheelbase_length, 1)
        
        # Send to the controller
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = 0.5 # set speed
        self.get_logger().info(f"Steering angle: {steering_angle}")
        self.drive_pub.publish(drive_msg)
        self.get_logger().info("Published steering angle")

        
    def img_cb(self, img_msg):
        if not self.detector_on:
            return

        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        try:
            results = self.detector.predict(image)
            predictions = results["predictions"]
            original_image = results["original_image"]
            
            out = self.detector.draw_box(original_image, predictions, draw_all=True)

            # Save PIL Image to file
            for bbox, label in predictions:
                # self.get_logger().info(f"Detected {label} at {bbox}")
                if label == 'traffic light':
                    # Save the image with the bounding box to directory
                    save_path = f"{os.path.dirname(__file__)}/shrinkray_detected.png"
                    out.save(save_path)
                    self.get_logger().info(f"Saved shrinkray image to {save_path}!")

                    # Get the bounding box coordinates 
                    self.shrinkray_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]] # x1, y1, x2, y2 = shrinkray_bbox
                    self.get_logger().info(f"Shrinkray bounding box: {self.shrinkray_bbox}")

                    rel_x,rel_y = self.get_relative_position(self.shrinkray_bbox)
                    
                    self.get_logger().info(f"Shrinkray relative position: {rel_x}, {rel_y}")
            
            # Convert PIL Image to OpenCV (np array)
            out_np = np.array(out)

            # Convert OpenCV to ROS Image message
            out_msg = self.bridge.cv2_to_imgmsg(out_np, "rgb8")
            self.yolo_pub.publish(out_msg)

            self.get_logger().info("Published detected image to /banana_img")
            
        except:
            pass
            
def main(args=None):
    rclpy.init(args=args)
    shrinkray_detector = ShrinkRayDetector()
    rclpy.spin(shrinkray_detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
