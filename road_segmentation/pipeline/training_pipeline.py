from road_segmentation.components.data_ingestion import DataIngestion
from road_segmentation.components.data_transformation import DataTransformation
from road_segmentation.components.model_trainer import ModelTrainer
from road_segmentation.entity.config_entity import DataIngestionConfig, ModelTrainerConfig
from road_segmentation.logger import logging
import sys
from road_segmentation.components.model_evaluation import ModelEvaluator
from road_segmentation.components.model_pusher import ModelPusher
from road_segmentation.entity.config_entity import (
    DataIngestionConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)
from road_segmentation.constants import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    S3_BUCKET_NAME,
    S3_MODEL_KEY,
    S3_METRICS_KEY
)

def run_pipeline():
    try:
        logging.info("Pipeline started.")
                
        data_ingestion = DataIngestion(config=DataIngestionConfig())
        ingestion_artifact = data_ingestion.initiate()

        data_transformation = DataTransformation(config=DataIngestionConfig(), ingestion_artifact=ingestion_artifact)
        transformation_artifact = data_transformation.initiate()

        model_trainer = ModelTrainer(config=ModelTrainerConfig(), data_artifact=transformation_artifact)
        trainer_artifact = model_trainer.initiate()

   
        model_evaluator = ModelEvaluator(
            config=ModelEvaluationConfig(num_classes=19),
            data_artifact=transformation_artifact,
            model_artifact=trainer_artifact
        )
        evaluation_artifact = model_evaluator.evaluate()

        model_pusher = ModelPusher(
            config=ModelPusherConfig(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                aws_default_region=AWS_DEFAULT_REGION,
                bucket_name=S3_BUCKET_NAME,
                s3_model_key=S3_MODEL_KEY,
                s3_metrics_key=S3_METRICS_KEY
            ),
            evaluation_artifact=evaluation_artifact
        )
        pusher_artifact = model_pusher.push_model()

        logging.info("Pipeline executed successfully.")

    except Exception as e:
        logging.error("Pipeline failed.", exc_info=True)
        raise e
