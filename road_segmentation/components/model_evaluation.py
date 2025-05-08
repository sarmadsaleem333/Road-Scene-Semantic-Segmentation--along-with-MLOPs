# components/model_evaluation.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.model import get_deeplabv3_model
from entity.config_entity import ModelEvaluationConfig
from entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from exception import RoadSegmentationException
from road_segmentation.logger import logging

class ModelEvaluation:
    def __init__(self, 
                 config: ModelEvaluationConfig, 
                 model_trainer_artifact: ModelTrainerArtifact, 
                 data_artifact: DataTransformationArtifact):
        self.config = config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_artifact = data_artifact

    def evaluate(self, model, test_loader, device):
        model.eval()
        total_pixels = 0
        correct_pixels = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)['out']
                preds = outputs.argmax(dim=1)
                correct_pixels += (preds == labels).sum().item()
                total_pixels += torch.numel(labels)

        pixel_accuracy = correct_pixels / total_pixels
        return pixel_accuracy

    def initiate(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_deeplabv3_model(num_classes=self.config.num_classes)
            model.load_state_dict(torch.load(self.model_trainer_artifact.model_path, map_location=device))
            model.to(device)

            pixel_acc = self.evaluate(model, self.data_artifact.test_loader, device)

            logging.info(f"Model Pixel Accuracy on test set: {pixel_acc:.4f}")

            return ModelEvaluationArtifact(pixel_accuracy=pixel_acc)

        except Exception as e:
            raise RoadSegmentationException(e)
