import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image

from TauLidarCommon.frame import FrameType
from TauLidarCamera.camera import Camera
import cv2
import numpy as np
from cv_bridge import CvBridge


class LidarPublisher(Node):
    def __init__(self):
        super().__init__('tau_lidar_publisher')
        self.publisher_depth_ = self.create_publisher(Image, 'Tau/depth', 10)
        self.publisher_gray_ = self.create_publisher(Image, 'Tau/greyscale', 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.im_list = []
        self.bridge = CvBridge()

        port = None
        self.camera = None

        ports = Camera.scan()                      ## Scan for available Tau Camera devices

        if len(ports) > 0:
            port = ports[0]
        else:
            port = serialPort

        if port is not None:
            Camera.setRange(0, 4500)                   ## points in the distance range to be colored

            self.camera = Camera.open(port)             ## Open the first available Tau Camera
            self.camera.setModulationChannel(0)             ## autoChannelEnabled: 0, channel: 0
            self.camera.setIntegrationTime3d(0, 1000)       ## set integration time 0: 1000
            self.camera.setMinimalAmplitude(0, 10)          ## set minimal amplitude 0: 80

            cameraInfo = self.camera.info()
        
        self.cv_image = None 
        self.bridge = CvBridge()
       
    def timer_callback(self):
        
        frame = self.camera.readFrame(FrameType.DISTANCE_GRAYSCALE)

        if frame:
            mat_depth_rgb = np.frombuffer(frame.data_depth_rgb, dtype=np.uint16, count=-1, offset=0).reshape(frame.height, frame.width, 3)
            mat_depth_rgb = mat_depth_rgb.astype(np.uint8)

            mat_grayscale = np.frombuffer(frame.data_grayscale, dtype=np.uint16, count=-1, offset=0).reshape(frame.height, frame.width)
            mat_grayscale = mat_grayscale.astype(np.uint8)

        self.publisher_depth_.publish(self.bridge.cv2_to_imgmsg(np.array(mat_depth_rgb), "bgr8"))
        self.publisher_gray_.publish(self.bridge.cv2_to_imgmsg(np.array(mat_grayscale), "mono8"))
        self.get_logger().info('Publishing an image')


def main(args=None):
    rclpy.init(args=args)

    node = LidarPublisher()

    rclpy.spin(node)

    # Destroy the timer attached to the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_timer(timer)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
