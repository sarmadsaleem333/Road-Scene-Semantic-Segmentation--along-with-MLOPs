import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from road_segmentation.models.DeepVLab3 import DeepLabV3Plus
from road_segmentation.entity.config_entity import ModelTrainerConfig
from road_segmentation.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from road_segmentation.exception import RoadSegmentationException
from road_segmentation.logger import logging
# from road_segmentation.utils.visualization import plot_training_curves  # if defined

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_artifact: DataTransformationArtifact):
        self.config = config
        self.data_artifact = data_artifact

    def evaluate(self, model, val_loader, criterion, device):
        model.eval()
        val_loss = 0
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_pixels += torch.numel(labels)

        avg_loss = val_loss / len(val_loader)
        pixel_acc = total_correct / total_pixels
        return avg_loss, pixel_acc

    def train(self, model, train_loader, val_loader, device):
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0
            total_correct = 0
            total_pixels = 0

            print(f"\nEpoch [{epoch+1}/{self.config.epochs}]")

            for images, labels in tqdm(train_loader, desc="Training"):
                images, labels = images.to(device), labels.to(device).long()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_pixels += torch.numel(labels)

            avg_train_loss = train_loss / len(train_loader)
            train_pixel_acc = total_correct / total_pixels
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_pixel_acc)

            logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Pixel Acc: {train_pixel_acc:.4f}")

            val_loss, val_acc = self.evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            logging.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Pixel Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.config.model_path)
                logging.info(f"Validation accuracy improved. Model saved at: {self.config.model_path}")

        logging.info("Training complete.")
        # plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

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
            raise RoadSegmentationException(e, sys)
