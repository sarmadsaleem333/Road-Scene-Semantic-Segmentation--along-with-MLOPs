
from road_segmentation.components.data_ingestion import DataIngestion
from road_segmentation.components.data_transformation import DataTransformation
from road_segmentation.components.model_trainer import ModelTrainer
from road_segmentation.entity.config_entity import DataIngestionConfig, ModelTrainerConfig
from road_segmentation.logger import logging

def run_pipeline():
    # Step 1: Data Ingestion
    ingestion_config = DataIngestionConfig(
        train_image_dir="D:/Citscapes dataset/Input/leftImg8bit/train_final",
        train_label_dir="D:/Citscapes dataset/Output/train_final",
        val_image_dir="D:/Citscapes dataset/Input/leftImg8bit/val_final",
        val_label_dir="D:/Citscapes dataset/Output/val_final",
        test_image_dir="D:/Citscapes dataset/Input/leftImg8bit/test_final",
        test_label_dir="D:/Citscapes dataset/Output/test_final",
        batch_size=4
    )
    data_ingestion = DataIngestion(config=ingestion_config)
    ingestion_artifact = data_ingestion.initiate()

    # Step 2: Data Transformation
    data_transformation = DataTransformation(config=ingestion_config, ingestion_artifact=ingestion_artifact)
    transformation_artifact = data_transformation.initiate()

    # Step 3: Model Training
    trainer_config = ModelTrainerConfig(
        model_name="DeepLabV3",
        model_path="saved_models/deeplabv3.pth",
        num_classes=19,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=4,
        epochs=25
    )
    model_trainer = ModelTrainer(config=trainer_config, data_artifact=transformation_artifact)
    trainer_artifact = model_trainer.initiate()

    logging.info("Pipeline executed successfully.")
