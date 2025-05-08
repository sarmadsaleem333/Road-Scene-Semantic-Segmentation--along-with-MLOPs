import os
import sys
from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifact
from exception import RoadSegmentationException
from road_segmentation.logger import logging


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):

        try:
            self.config = config
            self.artifact = DataIngestionArtifact()
        except Exception as e:
            raise RoadSegmentationException()
        

    def intiate (self)-> DataIngestionArtifact:
        
        try:
            logging.info("Data Ingestion started")
            return DataIngestionArtifact(
                        train_images_path=self.config.train_image_dir,
                        train_labels_path=self.config.train_label_dir,
                        val_images_path=self.config.val_image_dir,
                        val_labels_path=self.config.val_label_dir,
                        test_images_path=self.config.test_image_dir,
                        test_labels_path=self.config.test_label_dir,
                    )
        
        except Exception as e:
            raise RoadSegmentationException(e, sys)
    