from dataclasses import dataclass
from typing import Optional, Dict
from torch.utils.data import DataLoader

@dataclass
class DataIngestionArtifact:
    train_images_path: str
    test_images_path: str
    val_images_path: str
    train_labels_path: str
    test_labels_path: str
    val_labels_path: str


@dataclass
class DataTransformationArtifact:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
@dataclass
class ModelTrainerArtifact:
    model_path: str
    training_metrics: Optional[Dict] = None  



@dataclass
class ModelEvaluationArtifact:
    model_path: str
    pixel_accuracy: float
    mean_iou: float


from typing import Optional

@dataclass
class ModelPusherArtifact:
    saved_model_path: Optional[str] = None
    model_version: Optional[str] = None
    s3_model_path: Optional[str] = None
    s3_metrics_path: Optional[str] = None 