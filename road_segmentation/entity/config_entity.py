

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_image_dir: str
    train_label_dir: str
    val_image_dir: str
    val_label_dir: str
    test_image_dir: str
    test_label_dir: str
    batch_size: int  

@dataclass
class ModelTrainerConfig:
    model_name: str
    model_path: str
    num_classes: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int

@dataclass
class ModelEvaluationConfig:
    num_classes: int = 19
