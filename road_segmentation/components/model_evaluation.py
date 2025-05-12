import torch
import torch.nn as nn
import numpy as np
import sys
from tqdm import tqdm
from road_segmentation.logger import logging
from road_segmentation.exception import RoadSegmentationException
from road_segmentation.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from road_segmentation.entity.config_entity import ModelEvaluationConfig
from road_segmentation.models.DeepVLab3 import DeepLabV3Plus

class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig, data_artifact: DataTransformationArtifact, model_artifact: ModelTrainerArtifact):
        self.config = config
        self.data_artifact = data_artifact
        self.model_artifact = model_artifact

    def evaluate(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DeepLabV3Plus(num_classes=self.config.num_classes).to(device)
            model.load_state_dict(torch.load(self.model_artifact.model_path, map_location=device))
            model.eval()

            val_loader = self.data_artifact.val_loader
            criterion = nn.CrossEntropyLoss(ignore_index=255)

            val_loss = 0
            total_correct = 0
            total_pixels = 0
            confusion = np.zeros((self.config.num_classes, self.config.num_classes), dtype=np.int64)

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Evaluating"):
                    images, labels = images.to(device), labels.to(device).long()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    valid_mask = labels != 255
                    correct = (preds == labels) & valid_mask
                    total_correct += correct.sum().item()
                    total_pixels += valid_mask.sum().item()

                    labels_flat = labels.view(-1).cpu().numpy()
                    preds_flat = preds.view(-1).cpu().numpy()
                    valid_mask_flat = labels_flat != 255
                    valid_labels = labels_flat[valid_mask_flat]
                    valid_preds = preds_flat[valid_mask_flat]
                    np.add.at(confusion, (valid_labels, valid_preds), 1)

            avg_loss = val_loss / len(val_loader)
            pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0

            iou_per_class = []
            for c in range(self.config.num_classes):
                TP = confusion[c, c]
                FP = confusion[:, c].sum() - TP
                FN = confusion[c, :].sum() - TP
                total = TP + FP + FN
                iou_c = TP / total if total > 0 else 0.0
                iou_per_class.append(iou_c)
            mIoU = np.mean(iou_per_class)

            logging.info(f"Validation Loss: {avg_loss:.4f} | Pixel Accuracy: {pixel_acc:.4f} | mIoU: {mIoU:.4f}")

            return ModelEvaluationArtifact(
                model_path=self.model_artifact.model_path,
                pixel_accuracy=pixel_acc,
                mean_iou=mIoU
            )

        except Exception as e:
            logging.error("Error during model evaluation.", exc_info=True)
            raise RoadSegmentationException(e, sys)
