import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch
from PIL import Image as PILImage
from torchvision import transforms
import numpy as np
from unet.unet_model import UNet
import matplotlib.pyplot as plt
from cv_bridge import CvBridge

image_size = (400, 300)

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.model = UNet(n_channels=3, n_classes=2, bilinear=False)
        self.model.load_state_dict(torch.load("./model/model.pth"))
        self.model.eval()

    def process_image(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        input_image = PILImage.fromarray(cv_image).convert('RGB')

        width, height = input_image.size

        input_image = input_image.crop((0, height // 2, width, height))

        transform = transforms.Compose([
            transforms.Resize(image_size),  
            transforms.ToTensor(),
            # Add normalization if used during training
        ])
        return transform(input_image).unsqueeze(0)

    def save_output_as_image(self, output, filename):
        output_np = output.cpu().detach().numpy().squeeze(0)
        rgb_output = np.zeros((output_np.shape[0], output_np.shape[1], 3), dtype=np.uint8)
        
        # rgb_output = rgb_output * 255

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
        
        output_image_path = './output_car.png'
        self.save_output_as_image(predictions, output_image_path)

        # Display the output image
        plt.imshow(predictions.cpu().squeeze(), cmap='gray')
        plt.colorbar()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)
    lane_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
