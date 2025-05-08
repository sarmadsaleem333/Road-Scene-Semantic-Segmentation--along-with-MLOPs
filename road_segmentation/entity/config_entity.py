
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_image_dir: str
    train_label_dir: str
    val_image_dir: str
    val_label_dir: str
    test_image_dir: str
    test_label_dir: str

@dataclass
class ModelTrainerConfig:
    model_name: str
    num_classes: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int


class ModelEvaluationConfig:
    def __init__(self, num_classes: int = 19):
        self.num_classes = num_classes
