import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, PoseArray

class BasementPointPublisher(Node):
    '''
    Node that publishes a list of "shell" points
    Subscribes to the "Publish Point" topic when you click on the map in RViz
    After 2 points have been chosen, it publishes the 2 points as a PoseArray and resets the array
    '''

    def __init__(self):
        super().__init__("BasementPointPub")
        self.publisher = self.create_publisher(PoseArray, "/shrinkray_part", 1)
        self.subscriber = self.create_subscription(PointStamped, "/clicked_point", self.callback, 1)
        self.goal_subscriber = self.create_subscription(PoseStamped, "/goal_pose", self.goal_pose_callback, 1)

        self.array = []

        self.get_logger().info("Point Publisher Initialized")

    def callback(self, point_msg: PointStamped):
        x,y = point_msg.point.x, point_msg.point.y
        self.get_logger().info(f"Received point: {x}, {y}")
        self.array.append(Pose(position=Point(x=x, y=y, z=0.0)))
        
        if len(self.array) == 2:
            self.publish()
    
    def goal_pose_callback(self, pose_msg: PoseStamped):
        x,y = pose_msg.pose.position.x, pose_msg.pose.position.y

        self.get_logger().info(f"Received goal pose point: {x}, {y}")
        self.array.append(Pose(position=Point(x=x, y=y, z=0.0), orientation=pose_msg.pose.orientation))
        
        if len(self.array) == 2:
            self.publish()

    def publish(self):
        # Publish PoseArray
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.poses = self.array
        self.publisher.publish(pose_array)

        # Print to Command Line
        points_str = '\n'+'\n'.join([f"({p.position.x},{p.position.y})" for p in self.array])
        self.get_logger().info(f"Published 2 points: {points_str}")

        # Reset Array
        self.array = []
    

def main(args=None):
    rclpy.init(args=args)
    node = BasementPointPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()