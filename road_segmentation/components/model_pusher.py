import os
import boto3
from botocore.exceptions import ClientError
from road_segmentation.logger import logging
import sys
import json
from road_segmentation.exception import RoadSegmentationException
from road_segmentation.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from road_segmentation.entity.config_entity import ModelPusherConfig

class ModelPusher:
    def __init__(self, config: ModelPusherConfig, evaluation_artifact: ModelEvaluationArtifact):
        self.config = config
        self.evaluation_artifact = evaluation_artifact
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
            region_name=self.config.aws_default_region
        )

    def push_model(self):
        try:
            logging.info(f"Model pushing to AWS started")
            # Check if a model already exists in S3
            try:
                self.s3_client.download_file(self.config.bucket_name, self.config.s3_model_key, 'temp_model.pth')
                existing_model_exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    existing_model_exists = False
                else:
                    raise

            push_model = False

            if existing_model_exists:
                # Load existing model's evaluation metrics
                # For demonstration, assume we have stored metrics in a JSON file in S3
                try:
                    self.s3_client.download_file(self.config.bucket_name, self.config.s3_metrics_key, 'temp_metrics.json')
                    
                    with open('temp_metrics.json', 'r') as f:
                        existing_metrics = json.load(f)
                    existing_miou = existing_metrics.get('mean_iou', 0)
                    existing_accuracy = existing_metrics.get('pixel_accuracy', 0)

                    if (self.evaluation_artifact.mean_iou > existing_miou) and (self.evaluation_artifact.pixel_accuracy > existing_accuracy):
                        push_model = True
                except Exception as e:
                    logging.warning("Could not retrieve existing model metrics. Proceeding to push the new model.")
                    push_model = True
            else:
                push_model = True

            if push_model:
                # Upload the model
                self.s3_client.upload_file(self.evaluation_artifact.model_path, self.config.bucket_name, self.config.s3_model_key)
                logging.info(f"Model uploaded to S3 at {self.config.s3_model_key}")

                # Upload the evaluation metrics
                metrics = {
                    'pixel_accuracy': self.evaluation_artifact.pixel_accuracy,
                    'mean_iou': self.evaluation_artifact.mean_iou
                }
               
                with open('temp_metrics.json', 'w') as f:
                    json.dump(metrics, f)
                self.s3_client.upload_file('temp_metrics.json', self.config.bucket_name, self.config.s3_metrics_key)
                logging.info(f"Model evaluation metrics uploaded to S3 at {self.config.s3_metrics_key}")

                return ModelPusherArtifact(
                    s3_model_path=f"s3://{self.config.bucket_name}/{self.config.s3_model_key}",
                    s3_metrics_path=f"s3://{self.config.bucket_name}/{self.config.s3_metrics_key}"
                )
            else:
                logging.info("Existing model in S3 has better or equal performance. Skipping upload.")
                return ModelPusherArtifact(
                    s3_model_path=None,
                    s3_metrics_path=None
                )

        except Exception as e:
            logging.error("Error during model pushing.", exc_info=True)
            raise RoadSegmentationException(e, sys)
