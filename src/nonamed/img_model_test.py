import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet

# Ensure this matches the size used during training
image_size = (512, 256)

def load_image(image_path, size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),  # Ensure this matches the size used during training
        transforms.ToTensor(),
        # Add normalization if used during training
    ])
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

# Modify this function if the color mapping is different
def save_output_as_image(output, filename):
    # Assuming output is a tensor with shape [1, H, W] after argmax
    output_np = output.cpu().detach().numpy().squeeze(0)  # Now it has shape [H, W]

    # Create an RGB image with 3 channels
    rgb_output = np.zeros((output_np.shape[0], output_np.shape[1], 3), dtype=np.uint8)

    # Assuming class '1' is for lanes and class '0' is for background
    # Only lanes are colored red, background remains black
    rgb_output[output_np == 1] = [255, 0, 0]  # Color lanes red

    plt.imsave(filename, rgb_output)

def main():
    # Load model, ensure this matches the parameters used during training
    model = UNet(n_channels=3, n_classes=2, bilinear=False)

    # Load checkpoint, ensure the path is correct
    checkpoint_path = './checkpoints/checkpoint_epoch20.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    # Move model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 이미지 처리
    image_path = './carla.png'  # 이미지 파일 경로
    image = load_image(image_path, image_size)
    image = image.to(device)
    print(image)

    image = image * 255

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Since we have more than one class, we apply softmax (not sigmoid) and take argmax to find the most likely class
    probabilities = torch.softmax(output, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    # 결과 저장
    output_image_path = './output_image.png'
    save_output_as_image(predictions, output_image_path)

    # Show the output
    # We can only use plt.imshow on a 2D array, so we ensure predictions is 2D
    plt.imshow(predictions.cpu().squeeze(), cmap='gray')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()