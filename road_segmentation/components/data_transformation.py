from utils.dataset import CityscapesDataset
from utils.augmentations import get_train_augmentation,get_val_augmentation
from entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from entity.config_entity import DataIngestionConfig
from exception import RoadSegmentationException
import sys
import os
from road_segmentation.logger import logging
from torch.utils.data import DataLoader


class DataTransformation:
    
    def __init__(self, config: DataIngestionConfig,ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def initiate(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")
            
            train_dataset = CityscapesDataset(
                image_dir=self.ingestion_artifact.train_images_path,
                label_dir=self.ingestion_artifact.train_labels_path,
                transform=get_train_augmentation()
            )
            
            val_dataset = CityscapesDataset(
                image_dir=self.ingestion_artifact.val_images_path,
                label_dir=self.ingestion_artifact.val_labels_path,
                transform=get_val_augmentation()
            )
            
            test_dataset = CityscapesDataset(
                image_dir=self.ingestion_artifact.test_images_path,
                label_dir=self.ingestion_artifact.test_labels_path,
                transform=get_val_augmentation()
            )
            
           
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

            return DataTransformationArtifact(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader
            )
        except Exception as e:
            raise RoadSegmentationException(e, sys)
    
       