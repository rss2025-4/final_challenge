import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from model.detector import Detector

class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.detector = Detector()
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.publisher = self.create_publisher(Image, "/banana_img", 10)
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
        self.bridge = CvBridge()

        self.get_logger().info("Detector Initialized")

    def callback(self, img_msg):
        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        try:
            
            
            model = Detector()
            model.set_threshold(0.5)
            
            
                
            results = model.predict(image)
            predictions = results["predictions"]
            original_image = results["original_image"]
                
            out = model.draw_box(original_image, predictions, draw_all=True)
            out_msg = self.bridge.cv2_to_imgmsg(out, "bgr8")
            self.publisher.publish(out_msg)
            self.get_logger().info("Published callback")
            
            
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    detector = DetectorNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
