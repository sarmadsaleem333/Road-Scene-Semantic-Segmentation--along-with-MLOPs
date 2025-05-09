
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from road_segmentation.models.DeepVLab3 import DeepLabV3Plus
from road_segmentation.entity.config_entity import ModelTrainerConfig
from road_segmentation.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from road_segmentation.exception import RoadSegmentationException
from road_segmentation.logger import logging

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_artifact: DataTransformationArtifact):
        self.config = config
        self.data_artifact = data_artifact

    def train(self, model, train_loader, val_loader, device):
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        best_val_loss = float("inf")

        for epoch in range(self.config.epochs):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

            val_loss = self.validate(model, val_loader, device, criterion)
            logging.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.config.model_path)
                logging.info(f"Best model saved at {self.config.model_path}")

    def validate(self, model, val_loader, device, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def initiate(self) -> ModelTrainerArtifact:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DeepLabV3Plus(num_classes=self.config.num_classes).to(device)
            self.train(
                model=model,
                train_loader=self.data_artifact.train_loader,
                val_loader=self.data_artifact.val_loader,
                device=device
            )

            return ModelTrainerArtifact(
                model_path=self.config.model_path
            )
        except Exception as e:
            raise RoadSegmentationException(e)
