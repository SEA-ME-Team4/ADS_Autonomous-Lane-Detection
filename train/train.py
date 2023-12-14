import argparse
import logging
import os
import cv2
import json
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import matplotlib.pyplot as plt

class TUSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path="/content/drive/MyDrive/LaneDetect_UNet/data/TUSimple/train_set", train=True, size=(512, 256)):
        self._dataset_path = dataset_path
        self._mode = "train" if train else "eval"
        self._image_size = size # w, h


        if self._mode == "train":
            label_files = [
                os.path.join(self._dataset_path, f"label_data_{suffix}.json")
                for suffix in ("0313", "0531")
            ]
        elif self._mode == "eval":
            label_files = [
                os.path.join(self._dataset_path, f"label_data_{suffix}.json")
                for suffix in ("0601",)
            ]

        self._data = []

        for label_file in label_files:
            self._process_label_file(label_file)

    def __getitem__(self, idx):
      image_path = os.path.join(self._dataset_path, self._data[idx][0])
      image = cv2.imread(image_path)
      h, w, c = image.shape

      # 이미지가 그레이스케일인 경우만 RGB로 변환
      if c == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

      image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_LINEAR)

      lanes = self._data[idx][1]

      segmentation_image = self._draw(h, w, lanes, "segmentation")
      instance_image = self._draw(h, w, lanes, "instance")

      # NumPy 배열을 텐서로 변환
      image = torch.from_numpy(image).float().permute((2, 0, 1))
      segmentation_image = torch.from_numpy(segmentation_image).to(torch.int64)
      instance_image = np.expand_dims(instance_image, axis=2)  # 채널 차원 추가
      instance_image = torch.from_numpy(instance_image).float().permute((2, 0, 1))

      return image, segmentation_image, instance_image # 1 x H x W [[0, 1], [2, 0]]
    
    def __len__(self):
        return len(self._data)

    def _draw(self, h, w, lanes, image_type):
        image = np.zeros((h, w), dtype=np.uint8)
        for i, lane in enumerate(lanes):
            color = 1 if image_type == "segmentation" else i + 1
            cv2.polylines(image, [lane], False, color, 10)

        image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_NEAREST)

        return image

    def _process_label_file(self, file_path):
        with open(file_path) as f:
            for line in f:
                info = json.loads(line)
                image = info["raw_file"]
                lanes = info["lanes"]
                h_samples = info["h_samples"]
                lanes_coords = []
                for lane in lanes:
                    x = np.array([lane]).T
                    y = np.array([h_samples]).T
                    xy = np.hstack((x, y))
                    idx = np.where(xy[:, 0] > 0)
                    lane_coords = xy[idx]
                    lanes_coords.append(lane_coords)
                self._data.append((image, lanes_coords))

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')

def save_images(epoch, images, masks, predictions, folder="saved_images", n_images=5):
    """ 이미지와 마스크, 예측을 저장합니다. """
    os.makedirs(folder, exist_ok=True)
    for i in range(min(n_images, images.size(0))):
        plt.figure(figsize=(10, 10))
        
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title("Image")

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i].cpu(), cmap='gray')
        plt.title("True Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i].cpu(), cmap='gray')
        plt.title("Predicted Mask")

        plt.savefig(f"{folder}/epoch_{epoch}_image_{i}.png")
        plt.close()

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    log_file_path = 'train_log.txt'
    with open(log_file_path, 'w') as log_file:
      log_file.write('Training log\n')

    checkpoint_dir = '/content/drive/MyDrive/LaneDetect_UNet/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = TUSimpleDataset(dataset_path="/content/drive/MyDrive/LaneDetect_UNet/data/TUSimple/train_set", train=True)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, segmentation_masks, _ = batch

                #assert images.shape[1] == model.n_channels, \
                #    f'Network has been defined with {model.n_channels} input channels, ' \
                #    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                segmentation_masks = segmentation_masks.to(device=device, dtype=torch.long)

                #true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), segmentation_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), segmentation_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, segmentation_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(segmentation_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                with open(log_file_path, 'a') as log_file:
                  log_file.write(f'Epoch {epoch}, Step {global_step}, Loss: {loss.item()}\n')

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))
                        
                        with open(log_file_path, 'a') as log_file:
                          log_file.write(f'Validation Dice score: {val_score}\n')

        with torch.no_grad():
          images, true_masks, _ = next(iter(val_loader))
          images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
          true_masks = true_masks.to(device=device, dtype=torch.long)

          if model.n_classes == 1:
              predictions = torch.sigmoid(model(images))
          else:
              predictions = torch.argmax(F.softmax(model(images), dim=1), dim=1)

          save_images(epoch, images, true_masks, predictions)

        if save_checkpoint:
          torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth'))
          logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

