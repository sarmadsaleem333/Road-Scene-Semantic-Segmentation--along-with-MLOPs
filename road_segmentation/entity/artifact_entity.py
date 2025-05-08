from  dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    train_images_path:str
    test_images_path:str
    val_images_path:str
    train_labels_path:str
    test_labels_path:str
    val_labels_path:str

@dataclass
class ModelTrainerArtifact:
    model_path: str
    training_metrics: dict

@dataclass
class ModelPusherArtifact:
    saved_model_path: str
    model_version: str


class ModelEvaluationArtifact:
    def __init__(self, pixel_accuracy: float):
        self.pixel_accuracy = pixel_accuracy

