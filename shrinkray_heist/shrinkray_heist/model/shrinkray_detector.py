import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
import os
import numpy as np

from sensor_msgs.msg import Image
from .detector import Detector #model.

class ShrinkRayDetector(Node):
    def __init__(self):
        super().__init__("shrinkray_detector")
        # self.detector = Detector() # ON REAL RACECAR
        self.detector = Detector(yolo_dir='/home/racecar/models', from_tensor_rt=False)
        # self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.publisher = self.create_publisher(Image, "/banana_img", 10)
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
        self.bridge = CvBridge()

        self.get_logger().info("Shrink Ray Detector Initialized")

    def callback(self, img_msg):
        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        try:
            self.detector.set_threshold(0.5)
                
            results = self.detector.predict(image)
            predictions = results["predictions"]
            original_image = results["original_image"]
                
            out = self.detector.draw_box(original_image, predictions, draw_all=True)

            # Save PIL Image to file
            for bbox, label in predictions:
                # self.get_logger().info(f"Detected {label} at {bbox}")
                if label == 'banana':
                    save_path = f"{os.path.dirname(__file__)}/shrinkray_detected.png"
                    out.save(save_path)
                    self.get_logger().info(f"Saved shrinkray image to {save_path}!")
            
            # Convert PIL Image to OpenCV (np array)
            out_np = np.array(out)

            # Convert OpenCV to ROS Image message
            out_msg = self.bridge.cv2_to_imgmsg(out_np, "rgb8")
            self.publisher.publish(out_msg)

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
