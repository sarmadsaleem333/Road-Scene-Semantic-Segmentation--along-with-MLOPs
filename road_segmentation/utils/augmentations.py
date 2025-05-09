

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentation():
    return A.Compose([
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_val_augmentation():
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0),
        ToTensorV2()
    ])