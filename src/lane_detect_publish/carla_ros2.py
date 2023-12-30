import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import torch
from PIL import Image as PILImage
from torchvision import transforms
import numpy as np
from unet.unet_model import UNet
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import math

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/carla/ego_vehicle/rgb_front', 10)
        self.bridge = CvBridge()
        self.model = UNet(n_channels=3, n_classes=2, bilinear=False)
        self.model.load_state_dict(torch.load("./model/model_epoch300.pth"))
        self.model.eval()
        self.camera_info = {
            "x": 2.0, "y": 0.0, "z": 1.5,
            "pitch": 35.0, "fov": 90.0,
            "width": 800, "height": 600
        }

    def process_image(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        input_image = PILImage.fromarray(cv_image).convert('RGB')

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(input_image).unsqueeze(0)

    def save_output_as_image(self, output, filename):
        output_np = output.cpu().detach().numpy().squeeze(0)
        rgb_output = np.zeros((output_np.shape[0], output_np.shape[1], 3), dtype=np.uint8)

        rgb_output[output_np == 1] = [255, 0, 0]
        plt.imsave(filename, rgb_output)

    def image_callback(self, msg):
        input_image = self.process_image(msg)
        input_image = input_image.to(next(self.model.parameters()).device)

        input_image = input_image * 255

        with torch.no_grad():
            output = self.model(input_image)

        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        # Save and Display Output Image
        output_image_path = './output_car.png'
        self.save_output_as_image(predictions, output_image_path)
        plt.imshow(predictions.cpu().squeeze(), cmap='gray')
        plt.colorbar()
        plt.show()

        self.publish_lane_points(predictions)

    def publish_lane_points(self, predictions):
        lane_points = self.calculate_lane_points(predictions)

        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1)
        ]

        lane_msg = PointCloud2()
        lane_msg.header = Header()
        lane_msg.header.stamp = self.get_clock().now().to_msg()
        lane_msg.header.frame_id = "ego_vehicle/rgb_front"

        lane_msg.height = 1
        lane_msg.width = len(lane_points)
        lane_msg.fields = fields
        lane_msg.is_dense = True
        lane_msg.is_bigendian = False
        lane_msg.point_step = 12  
        lane_msg.row_step = lane_msg.point_step * len(lane_points)
        lane_msg.data = np.asarray(lane_points, dtype=np.float32).tobytes()

        print("start\n")

        self.publisher.publish(lane_msg)

    def calculate_lane_points(self, predictions):
        lane_points = []
        for y in range(predictions.shape[1]):
            for x in range(predictions.shape[2]):
                if predictions[0, y, x] == 1: 
                    world_x, world_y, world_z = self.pixel_to_world(x, y)
                    # world_x, world_y, world_z = self.camera_info["x"], self.camera_info["y"], self.camera_info["z"]
                    lane_points.append((world_x, world_y, world_z))
        return lane_points

    def pixel_to_world(self, px, py):
        cam_x, cam_y, cam_z = self.camera_info["x"], self.camera_info["y"], self.camera_info["z"]
        pitch = math.radians(self.camera_info["pitch"])
        fov = math.radians(self.camera_info["fov"])
        width, height = self.camera_info["width"], self.camera_info["height"]
        nx = (px / width - 0.5) * 2
        ny = (0.5 - py / height) * 2  
        nz = 1 / math.tan(fov / 2)

        world_y = cam_z / math.tan(pitch + math.atan(ny))
        world_x = world_y * nx / nz + cam_x
        world_z = 1

        # a = world_x
        # b = world_y
        # world_x = world_y * (-1)
        # world_y = a 
        # world_z = b

        return (-world_x + 2.0), (world_y - 1.5), world_z

def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)
    lane_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
