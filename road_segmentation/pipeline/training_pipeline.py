
from components.data_ingestion import DataIngestion
from components.model_trainer import ModelTrainer
from components.model_pusher import ModelPusher
from entity.config_entity import DataIngestionConfig, ModelTrainerConfig

def run_pipeline():
    # Step 1: Load configurations
    ingestion_config = DataIngestionConfig(
        train_image_dir="data/train/images",
        train_label_dir="data/train/labels",
        val_image_dir="data/val/images",
        val_label_dir="data/val/labels",
        test_image_dir="data/test/images",
        test_label_dir="data/test/labels"
    )

    trainer_config = ModelTrainerConfig(
        model_name="deeplabv3plus",
        num_classes=19,
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=4,
        num_epochs=50
    )

    # Step 2: Run components
    data_artifact = DataIngestion(config=ingestion_config).initiate()
    trainer_artifact = ModelTrainer(config=trainer_config, data_artifact=data_artifact).initiate()
    pusher_artifact = ModelPusher(trainer_artifact=trainer_artifact).initiate()

    print(f" Model saved to registry at {pusher_artifact.saved_model_path}")

