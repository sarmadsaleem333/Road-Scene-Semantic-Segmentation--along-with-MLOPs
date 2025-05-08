import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
from cityscapesscripts.helpers.labels import id2label

class CityscapesDataset(Dataset):
    def __init__(self, image_folder, label_folder,augmentation=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.augmentation = augmentation

        # Sort to keep image-label pairing aligned
        self.image_paths = sorted([os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('.png')])
        self.label_paths = sorted([os.path.join(label_folder, x) for x in os.listdir(label_folder) if x.endswith('gtFine_labelIds.png')])

        # Optional check
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image=Image.open(image_path).convert('RGB')
        label=Image.open(label_path)
        label=np.array(label)
        label=self.remap_to_train_id(label)


        if self.augmentation:
            augmented = self.augmentation(image=np.array(image), mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        return image, label

    ## this is a funtion to remap the label id to train id because in cutyscaors dataset there are 34 classses, someof which are not useful 
    ## so by converting them into 19 using Github original repo code of cityscapes dataset
    # cityscapesscripts library
    
    def remap_to_train_id(self, label_np):
        label_train = np.full_like(label_np, 255) 
        
        for id, label in id2label.items():
            if label.trainId != -1:  
                label_train[label_np == id] = label.trainId
        
        label_train[label_train == -1] = 255  

        return label_train





import albumentations as A
from albumentations.pytorch import ToTensorV2

train_augmentation = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0),
    ToTensorV2()

])

val_augmentation= A.Compose([

    A.Resize(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0),
    ToTensorV2()
])

train_input_dir="/data/Cityscapes/Input/leftImg8bit/train_final"
train_label_dir="/data/Cityscapes/Output/train_final"
val_input_dir="/data/Cityscapes/Input/leftImg8bit/val_final"
val_label_dir="/data/Cityscapes/Output/val_final"
test_input_dir="/data/Cityscapes/Input/leftImg8bit/test_final"
test_label_dir="/data/Cityscapes/Output/test_final"


train_dataset = CityscapesDataset(
    image_folder=train_input_dir,
    label_folder=train_label_dir,
    augmentation=train_augmentation
)
val_dataset = CityscapesDataset(
    image_folder=val_input_dir,
    label_folder=val_label_dir,
    augmentation=val_augmentation
)
test_dataset = CityscapesDataset(
    image_folder=test_input_dir,
    label_folder=test_label_dir,
    augmentation=val_augmentation
)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(   
    
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )