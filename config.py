import torch
import albumentations as transform
import cv2
from albumentations.pytorch import ToTensorV2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100

# Data augmentation for images
train_transforms = transform.Compose(
    [
        transform.Resize(width=96, height=96),
        transform.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        transform.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
        transform.OneOf([
            transform.GaussNoise(p=0.8),
            transform.CLAHE(p=0.8),
            transform.ImageCompression(p=0.8),
            transform.RandomGamma(p=0.8),
            transform.Posterize(p=0.8),
            transform.Blur(p=0.8),
        ], p=1.0),
        transform.OneOf([
            transform.GaussNoise(p=0.8),
            transform.CLAHE(p=0.8),
            transform.ImageCompression(p=0.8),
            transform.RandomGamma(p=0.8),
            transform.Posterize(p=0.8),
            transform.Blur(p=0.8),
        ], p=1.0),
        transform.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        transform.Normalize(
            mean=[0.4897, 0.4897, 0.4897],
            std=[0.2330, 0.2330, 0.2330],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),

    ], keypoint_params=transform.KeypointParams(format="xy", remove_invisible=False),
)
