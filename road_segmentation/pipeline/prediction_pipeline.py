import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from road_segmentation.models.DeepVLab3 import DeepLabV3Plus
from road_segmentation.logger import logging
from road_segmentation.exception import RoadSegmentationException
from road_segmentation.constants import MODEL_PATH, NUM_CLASSES

class RoadSegmentationPredictor:
    def __init__(self, model_path=MODEL_PATH, num_classes=NUM_CLASSES):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepLabV3Plus(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 1024)),  # Resize as needed
            transforms.ToTensor(),
        ])

    def predict(self, image_path: str, save_path: str = None):
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            if save_path:
                colored_mask = self.decode_segmap(prediction)
                colored_mask.save(save_path)
                logging.info(f"Prediction saved to {save_path}")
            
            return prediction

        except Exception as e:
            logging.error("Prediction failed.", exc_info=True)
            raise RoadSegmentationException(e, sys)

    def decode_segmap(self, label_mask):
        # Simple color mapping (for 19 classes)
        label_colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
        rgb_mask = label_colors[label_mask]
        return Image.fromarray(rgb_mask)

