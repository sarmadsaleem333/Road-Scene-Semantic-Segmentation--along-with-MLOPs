from dataclasses import dataclass
import road_segmentation.constants as constants


@dataclass
class DataIngestionConfig:
    train_image_dir: str = constants.TRAIN_IMAGE_DIR
    train_label_dir: str = constants.TRAIN_LABEL_DIR
    val_image_dir: str = constants.VAL_IMAGE_DIR
    val_label_dir: str = constants.VAL_LABEL_DIR
    test_image_dir: str = constants.TEST_IMAGE_DIR
    test_label_dir: str = constants.TEST_LABEL_DIR
    batch_size: int = constants.BATCH_SIZE


@dataclass
class ModelTrainerConfig:
    model_name: str = constants.MODEL_NAME
    model_path: str = constants.MODEL_PATH
    num_classes: int = constants.NUM_CLASSES
    learning_rate: float = constants.LEARNING_RATE
    weight_decay: float = constants.WEIGHT_DECAY
    batch_size: int = constants.BATCH_SIZE
    epochs: int = constants.EPOCHS
    device:str=constants.DEVICE



@dataclass
class ModelEvaluationConfig:
    num_classes: int = 19

@dataclass
class ModelPusherConfig:
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_default_region: str
    bucket_name: str
    s3_model_key: str
    s3_metrics_key: str

