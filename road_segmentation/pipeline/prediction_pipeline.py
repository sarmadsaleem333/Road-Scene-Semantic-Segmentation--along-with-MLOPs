import os
import sys
import torch
import boto3
import io
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from road_segmentation.models.DeepVLab3 import DeepLabV3Plus
from road_segmentation.exception import RoadSegmentationException
from road_segmentation.logger import logging
from road_segmentation.entity.config_entity import PredictionConfig
from road_segmentation.constants import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    S3_BUCKET_NAME,
    S3_MODEL_KEY
)


class RoadSegmentationImage:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def preprocess(self):
        try:
            image = np.array(Image.open(self.image_path).convert("RGB"))

            transform = A.Compose([
                A.Resize(height=512, width=512),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255.0),
                ToTensorV2()
            ])

            augmented = transform(image=image)
            tensor = augmented['image'].unsqueeze(0)
            return tensor, image
        except Exception as e:
            raise RoadSegmentationException(e, sys)


class RoadSegmentationPredictor:
    def __init__(self, num_classes: int = 19):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           

            config = PredictionConfig(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                aws_default_region=AWS_DEFAULT_REGION,
                bucket_name=S3_BUCKET_NAME,
                s3_model_key=S3_MODEL_KEY
            )

            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                region_name=config.aws_default_region
            )

            model_bytes = self._download_model_from_s3(config.bucket_name, config.s3_model_key)
            self.model = self._load_model_from_bytes(model_bytes, num_classes)
            self.model.eval()
        except Exception as e:
            raise RoadSegmentationException(e, sys)

    def _download_model_from_s3(self, bucket_name: str, s3_key: str) -> bytes:
        try:
            logging.info(f"Downloading model from s3://{bucket_name}/{s3_key}")
            model_object = self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            return model_object['Body'].read()
        except Exception as e:
            raise RoadSegmentationException(e, sys)

    def _load_model_from_bytes(self, model_bytes: bytes, num_classes: int) -> torch.nn.Module:
        try:
            model = DeepLabV3Plus(num_classes=num_classes)
            model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=self.device))
            return model.to(self.device)
        except Exception as e:
            raise RoadSegmentationException(e, sys)

    def predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        try:
            image_tensor = image_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(image_tensor)
                return output.argmax(dim=1).squeeze().cpu().numpy()
        except Exception as e:
            raise RoadSegmentationException(e, sys)


def get_colormap():
    base_colors = plt.cm.tab20b(np.linspace(0, 1, 20))
    return ListedColormap(base_colors[:19])


def visualize_prediction(original_img_np: np.ndarray, predicted_mask: np.ndarray):
    cmap = get_colormap()

    original_np = original_img_np.astype(np.float32) / 255.0

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(original_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(predicted_mask, cmap=cmap, vmin=0, vmax=18)
    axs[1].set_title("Predicted Segmentation Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
