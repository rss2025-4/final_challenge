import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
import os
import numpy as np

from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

from .detector import Detector #model.
from shrinkray_heist.definitions import  Target, TripSegment, ObjectDetected, State
from shrinkray_heist.homography_transformer import HomographyTransformer

import time
from datetime import datetime 

class DetectionNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        # self.detector = Detector() # ON REAL RACECAR
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S') 
        self.get_logger().info(f"curr date t{curr_datetime}")
        self.detector = Detector(yolo_dir='/home/racecar/models', from_tensor_rt=False)
        self.detector.set_threshold(0.3)
        self.yolo_pub = self.create_publisher(Image, "/yolo_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.img_cb, 1)
        self.bridge = CvBridge()
        

        # listen for state machine state
        self.state_sub = self.create_subscription(Int32, "/toggle_state", self.state_cb, 10)
        self.trafficlight_detector_on = False # should be False by default
        self.trafficlight_detector_green_on = False
        self.shrinkray_detector_on = False # should be False by default
        self.obj_detected_pub = self.create_publisher(Int32, "/detected_obj", 10)
        self.trafficlight_dist_pub = self.create_publisher(Float32, "/traffic_light", 10)
        
        # for movement to shrink ray
        self.shrinkray_bbox = [0, 0, 0, 0]
        self.homography_transformer = HomographyTransformer() # instantiate homography transformer to get transform
        self.wheelbase_length = 0.33 # in meters
        self.count = 0
        self.shrinkray_count = 1
        
        # self.drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/high_level/input/nav_0", 10)

        self.get_logger().info("Shrink Ray Detector Initialized")
        
        # for scanning for shrinkray
        self.scan_sequence_started = False
        self.shrinkray_active_start_time = 0
        self.sequence_start_time = 0
        self.scan_step = 0
        self.scan_speed = 0.3
        self.scan_angle = 0.5

        self.debug = True
    def state_cb(self, msg):
        pass # for testing

        if msg.data == Target.DETECTOR_TRAFFIC_LIGHT.value:
            self.trafficlight_detector_on = not self.trafficlight_detector_on
            
            if self.trafficlight_detector_on:
                self.get_logger().info("Detector: Traffic Light Red Detector Activated")
            else:
                self.get_logger().info("Detector: Traffic Light Red Detector Deactivated")
        
        elif msg.data == Target.DETECTOR_TRAFFIC_LIGHT_GREEN.value:
            self.trafficlight_detector_green_on = not self.trafficlight_detector_green_on
            if self.trafficlight_detector_green_on:
                self.get_logger().info("Detector: Traffic Light Green Detector Activated")
            else:
                self.get_logger().info("Detector: Traffic Light Green Detector Deactivated")
        
        elif msg.data == Target.DETECTOR_SHRINK_RAY.value:
            self.shrinkray_detector_on = not self.shrinkray_detector_on
            
            if self.shrinkray_detector_on:
                self.get_logger().info("Detector: Shrink Ray Detector Activated")
                
                self.shrinkray_active_start_time = self.get_clock().now().nanoseconds / 1e9
                self.scan_step = 0
                self.scan_sequence_started = False
            else:
                self.get_logger().info("Detector: Shrink Ray Detector Deactivated")
        
        
    def get_relative_position(self, bbox):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.
        """
        # Get bottom center pixel coordinates of the bounding box
        x1, y1, x2, y2 = bbox # upper left and lower right corners of the bounding box
        v = y2 # bottom of bounding box
        u = (x1+x2)/2 # middle of bounding box

        # apply homography to u,v to get x,y in the world frame
        x, y = self.homography_transformer.transformUvToXy(u, v) # x,y in meters
        return x,y

    def controller(self, x,y):
        self.get_logger().info("in controller")
        # Get euclidean distance to the shrink ray
        L1 = np.sqrt(x**2 + y**2) # in meters
        self.get_logger().info(f"Distance to shrink ray: {L1}")

        if L1 < 1.0: # in meters
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
            self.get_logger().info("Arrived at shrink ray location, stopping")

            # send stop state
            return L1
        
        # Too far, go closer to shrink ray
        eta = np.arctan2(y,x) # rad angle between the vehicle heading and the line from current point to lookahead point
        
        R = L1/(2*np.sin(eta))
        curvature = 1/R # curvature = 1/R where R = lookahead / (2 * sin(eta))

        # Calculate the steering angle
        steering_angle = np.arctan2(curvature * self.wheelbase_length, 1)
        
        # Send to the controller
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle # rad
        drive_msg.drive.speed = 0.3 # m/s set speed
        self.get_logger().info(f"Steering angle: {steering_angle}")
        self.drive_pub.publish(drive_msg)
        self.get_logger().info("Published steering angle")
        return L1
    
    def scan_sequence(self):
        if not self.scan_sequence_started:
            return

        # self.scan_sequence_started = True

        def send_drive(speed, angle, duration):
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = speed
            drive_msg.drive.speed = angle

            end_time = self.get_clock().now().nanoseconds + int(duration * 1e9)
            
            while self.get_clock().now().nanoseconds < end_time:
                self.drive_pub.publish(drive_msg)
                self.get_logger().info(f"Scan Sequence: Driving with speed: {speed}, angle: {angle}")
                time.sleep(0.1)
            
            # Stop the robot briefly
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0
            self.drive_pub.publish(msg)
            self.get_logger().info(f"Scan Sequence: Driving with speed: {speed}, angle: {angle}")
            time.sleep(0.2)

        # Repeat the pattern N times
        repetitions = 3
        for _ in range(repetitions):
            # Step 1: Small backup
            send_drive(-0.2, 0.0, 0.5)

            # Step 2: Move forward
            send_drive(0.2, 0.0, 0.5)

            # Step 3: Turn slightly while moving forward
            send_drive(0.2, 0.25, 0.6)

        self.get_logger().info("Repeated scan maneuver completed.")

    def execute_sequence(self):
        
        now = self.get_clock().now().nanoseconds / 1e9  # current time in seconds
        elapsed = now - self.sequence_start_time

        # Duration for each step in seconds
        durations = [3.0, 2.0, 3.0, 1.0 , 3.0, 2.0, 3.0, 2.0]
                # steps 0, 1, 2, 3, 4
        msg = AckermannDriveStamped()
        self.scan_speed = 0.3
        self.scan_angle = 0.5

        if self.scan_step == 0:
            # Step 0: Back up left
            msg.drive.speed = -self.scan_speed
            msg.drive.steering_angle = -self.scan_angle
            self.get_logger().info(f"Step: {self.scan_step}, Scan Sequence: Moving with speed: {msg.drive.speed}, angle: {msg.drive.steering_angle}")
        elif self.scan_step == 1: # pause!
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0 #self.angle
            self.get_logger().info(f"Step: {self.scan_step}, Scan Sequence: Moving with speed: {msg.drive.speed}, angle: {msg.drive.steering_angle}")

        elif self.scan_step == 2:
            # Step 2: Move forward right
            msg.drive.speed = self.scan_speed
            msg.drive.steering_angle = -self.scan_angle #self.angle
            self.get_logger().info(f"Step: {self.scan_step}, Scan Sequence: Moving with speed: {msg.drive.speed}, angle: {msg.drive.steering_angle}")
        elif self.scan_step == 3: # pause!
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0 #self.angle
            self.get_logger().info(f"Step: {self.scan_step}, Scan Sequence: Moving with speed: {msg.drive.speed}, angle: {msg.drive.steering_angle}")

        elif self.scan_step == 4:
            # Step 4: Back up right
            msg.drive.speed = -self.scan_speed
            msg.drive.steering_angle = self.scan_angle
            self.get_logger().info(f"Step: {self.scan_step}, Scan Sequence: Moving with speed: {msg.drive.speed}, angle: {msg.drive.steering_angle}")
        elif self.scan_step == 5: # pause!
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0 #self.angle
        elif self.scan_step == 6:
            # Step 1: Move forward left
            msg.drive.speed = self.scan_speed
            msg.drive.steering_angle = self.scan_angle #self.angle
            self.get_logger().info(f"Step: {self.scan_step}, Scan Sequence: Moving with speed: {msg.drive.speed}, angle: {msg.drive.steering_angle}")
        elif self.scan_step == 7: # pause!
            # Step 2: Turn slightly while moving forward
            msg.drive.speed = 0.0 #-0.3
            msg.drive.steering_angle = 0.0 #-self.angle
            self.get_logger().info(f"Step: {self.scan_step}, Scan Sequence: Moving with speed: {msg.drive.speed}, angle: {msg.drive.steering_angle}")

        self.drive_pub.publish(msg)

        if elapsed >= durations[self.scan_step] and self.scan_step < (len(durations)-1):
            # Move to the next step after the duration
            self.scan_step += 1
            self.sequence_start_time = now
            return "looking"

        if self.scan_step == len(durations)-1:
            self.get_logger().info("Sequence complete, uh no banana then?")
            return "no_banana"
        

    def img_cb(self, img_msg):
        if not self.trafficlight_detector_on and not self.shrinkray_detector_on:
            return

        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        ## image = image[200:400, :]
        ## output = np.zeros_like(image)
        # output[120:330, :] = image[120:240, :]
        ## image = output

        # # Blacken the top third
        # output = image.copy()
        # height = output.shape[0]
        # output[:height // 4, :] = 0 
        # image = output
        # image_rgb = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        if self.count == 0:
            save_path = os.path.join(os.path.dirname(__file__), f"ros_imagetocv2TEST.png")
            cv2.imwrite(save_path, image)
            
        # # 
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(save_path, rgb_image)
        # self.get_logger().info(f"CV BRIDGE Image: {image.shape}")

        # Record how long shrink ray has been on for 
        now = self.get_clock().now().nanoseconds / 1e9
        shrinkray_active_duration = now - self.shrinkray_active_start_time
        if self.shrinkray_detector_on and not self.trafficlight_detector_on and (shrinkray_active_duration > 10) and (shrinkray_active_duration < 30 ): # if no banana detected for 5 seconds, start scan sequence
            self.get_logger().info("No banana detected for 5 seconds, in scan sequence")
            
            if not self.scan_sequence_started: # so that start time set once
                self.sequence_start_time = self.get_clock().now().nanoseconds / 1e9
                self.scan_sequence_started = True
                self.get_logger().info("Scan sequence started")

            scan_state = self.execute_sequence() # "looking", but if at last scan_step, then "no_banana"
            if scan_state == "no_banana":
                
                self.get_logger().info("Scan sequence complete, no banana detected")
        
        # Time out, if no banana detected for 60 seconds, move onto next goal
        if self.shrinkray_detector_on and not self.trafficlight_detector_on and (shrinkray_active_duration > 30):
            # for states, pretend we found the shrink ray in order to move on
            obj_detected_msg = Int32()
            obj_detected_msg.data = ObjectDetected.SHRINK_RAY.value
            self.obj_detected_pub.publish(obj_detected_msg) 
            self.get_logger().info(f"No shrink ray object, move on. Published at /detected_obj")

        try:
            results = self.detector.predict(image) #image
            
            predictions = results["predictions"]
            original_image = results["original_image"]
            # self.get_logger().info(f"original_image: {original_image.shape}")
            out = self.detector.draw_box(original_image, predictions, draw_all=True)

            # Save PIL Image to file
            for bbox, label in predictions:
                # self.get_logger().info(f"Detected {label} at {bbox}")

                if self.trafficlight_detector_on and not self.shrinkray_detector_on:
                    # publish traffic light distance 
                    if label == 'traffic light':
                        # Get the bounding box coordinates 
                        #double bounding box since traffic light twice height than actual (due to stud)
                        
                        # if not self.trafficlight_detector_green_on: # save last red bounding box
                        #     self.red_trafficlight_bbox = bbox #[bbox[0], bbox[1], bbox[2], bbox[3]] # x1, y1, x2, y2 = trafficlight_bbox
                        # elif self.trafficlight_detector_green_on: # if detecting green light, use the last red light bbox
                        #     bbox = self.red_trafficlight_bbox
                        #     self.get_logger().info(f"Using last red light bbox: {self.red_trafficlight_bbox}")
                            
                        bottom = bbox[3]+(bbox[3]-bbox[1]) #y2 + (y2-y1)
                        trafficlight_bbox = [bbox[0], bbox[1], bbox[2], bottom] # double the height since traffic light has stud underneath
                        # self.get_logger().info(f"Traffic light bounding box: {trafficlight_bbox}")
                        
                        traffic_color = self.get_traffic_light_color(image, bbox)
                        self.get_logger().info(f"Traffic light color: {traffic_color}")

                        
                        obj_detected_msg = Int32()
                        if traffic_color == "red":
                            obj_detected_msg.data = ObjectDetected.TRAFFIC_LIGHT_RED.value
                            
                            rel_x,rel_y = self.get_relative_position(trafficlight_bbox)
                            # self.get_logger().info(f"Traffic light relative position: {rel_x}, {rel_y}")
                            
                            # dist = np.sqrt(rel_x**2 + rel_y**2) # in meters
                            dist_msg = Float32()
                            dist_msg.data = rel_x # dist
                            self.trafficlight_dist_pub.publish(dist_msg)
                            self.get_logger().info(f"Published traffic light x dist msg /traffic_light: {rel_x}")
                            
                            self.obj_detected_pub.publish(obj_detected_msg) 
                            self.get_logger().info("Published traffic light object detected msg /detected_obj")

                        elif self.trafficlight_detector_green_on and traffic_color != "red": #traffic_color == "green":
                            obj_detected_msg.data = ObjectDetected.TRAFFIC_LIGHT_GREEN.value
                        
                            self.obj_detected_pub.publish(obj_detected_msg) 
                            self.get_logger().info("Published traffic light object detected msg /detected_obj")
                        # self.get_logger().info(f"Traffic light relative position: {rel_x}, {rel_y}")
                
                elif self.shrinkray_detector_on and not self.trafficlight_detector_on:
                    self.get_logger().info(f"Detected {label} at {bbox}")
                    
                        # self.shrinkray_active_start_time = 0
                    if label == 'banana':
                        # Get the bounding box coordinates 
                        self.shrinkray_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]] # x1, y1, x2, y2 = shrinkray_bbox
                        # self.get_logger().info(f"Shrinkray bounding box: {self.shrinkray_bbox}")
                        rel_x,rel_y = self.get_relative_position(self.shrinkray_bbox)
                        
                        self.get_logger().info(f"Shrinkray relative position: {rel_x}, {rel_y}")
                        dist_to_shrinkray = self.controller(rel_x,rel_y)
                        self.get_logger().info(f"Distance to shrinkray: {dist_to_shrinkray}")
                        
                        if dist_to_shrinkray < 1.0: # in meters
                            self.get_logger().info("Arrived at shrink ray location, stopping")
                            # Save the image with the bounding box to directory
                            #get time
                            curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S') 

                            save_path = f"{os.path.dirname(__file__)}/shrinkray_detected_{self.shrinkray_count}_{curr_datetime}.png"
                            self.shrinkray_count += 1

                            out.save(save_path)
                            self.get_logger().info(f"Saved shrinkray image to {save_path}!")
                            
                            obj_detected_msg = Int32()
                            obj_detected_msg.data = ObjectDetected.SHRINK_RAY.value
                            self.obj_detected_pub.publish(obj_detected_msg) 
                            self.get_logger().info(f"Published shrink ray object detected msg /detected_obj")
                            # self.shrinkray_detector_on = False # turn off shrink ray detector
                        
                        
            # Convert PIL Image to OpenCV (np array)
            out_np = np.array(out)
            
            if self.count == 0:
                save_path = os.path.join(os.path.dirname(__file__), f"detect_output_cv2img.png")
                cv2.imwrite(save_path, out_np)
                # cv2.imwrite(save_path, cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))

                self.count += 1
            

            # Convert OpenCV to ROS Image message
            # out_msg = self.bridge.cv2_to_imgmsg(out_np, "bgr8")
            out_msg = self.bridge.cv2_to_imgmsg(out_np, "rgb8")
            self.yolo_pub.publish(out_msg)

            # self.get_logger().info("Published detected image to /yolo_img")
            
        except:
            pass

    def get_traffic_light_color(self, image, bbox):
        """
        Given image in the form of a BGR CV2 image and a bounding box, 
        return the traffic light of the object in the bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Copy image and blacken it
        image = np.array(image) # bgr
        output = np.zeros_like(image)
        
        output[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        
        image = output

        # Convert from BGR to HSV
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        bgr_from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        save_path = os.path.join(os.path.dirname(__file__), f"traffic_light_hsv.png") #{traffic_color}
        cv2.imwrite(save_path, hsv)
        save_path = os.path.join(os.path.dirname(__file__), f"traffic_light_hsv_toBGR.png") #{traffic_color}
        cv2.imwrite(save_path, bgr_from_hsv)
        
        def get_centroid(mask):
            # Find moments of the mask to calculate centroid
            moments = cv2.moments(mask)

            # Calculate centroid (cx, cy) from moments
            cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
            cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
            
            return cx, cy
                
        # Define red ranges and create two masks
        lower_red1 = np.array([0, 50, 150])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([160, 50, 150])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)
        red_count = cv2.countNonZero(red_mask)
        red_cx, red_cy = get_centroid(red_mask)
        
        def upper_lower_half(cy, y1, y2):
            # Check if the centroid is in the upper third or lower third of the bounding box
            bbox_height = y2 - y1
            upper_half_limit = y1 + bbox_height // 2
            lower_half_limit = y1 + 2 * bbox_height // 2

            if cy < upper_half_limit:
                return 0 # upper
            else: 
                return 1 # lower
        
        
        red_where = upper_lower_half(red_cy, y1, y2) # 0 if upper, 1 if lower


        if red_count > 3 and red_where == 0:
                traffic_color = "red"
                total_mask = red_mask
        else:
            return "not"
        
        if self.debug:
            save_path = os.path.join(os.path.dirname(__file__), f"traffic_light_{traffic_color}_mask.png")
            cv2.imwrite(save_path, total_mask)

            # Step 5: Apply mask to the original image
            color_filtered = cv2.bitwise_and(image, image, mask=total_mask)
            
            save_path = os.path.join(os.path.dirname(__file__), f"traffic_light_{traffic_color}_filtered.png")
            cv2.imwrite(save_path, color_filtered)
            

            cv2.rectangle(color_filtered, (x1, y1), (x2, y2), (255, 255, 255), 2)
            save_path = os.path.join(os.path.dirname(__file__), f"traffic_light_{traffic_color}_filtered_bbox.png")
            cv2.imwrite(save_path, color_filtered)

        return traffic_color

def main(args=None):
    rclpy.init(args=args)
    detection_node = DetectionNode()
    rclpy.spin(detection_node)
    rclpy.shutdown()

if __name__=="__main__":
    main()
