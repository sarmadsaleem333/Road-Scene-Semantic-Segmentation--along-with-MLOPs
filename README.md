# Road Scene Semantic Segmentation aligned with MLOps


## Overview

This project implements an enhanced DeepLabv3+ architecture for road scene semantic segmentation, incorporating a Selective Kernel Fusion (SK-Fusion) module to improve feature representation and segmentation accuracy. The model is designed for autonomous driving applications, with a complete MLOps pipeline for scalable, reproducible deployment on AWS.


## Features

- **Enhanced DeepLabv3+ Architecture**: Base model with MobileNetV2 backbone for efficiency
- **SK-Fusion Module**: Attention-based feature selection for improved segmentation
- **Complete MLOps Pipeline**: End-to-end pipeline for training, evaluation, and deployment
- **Cityscapes Dataset Integration**: Processing pipeline for the standard urban scene dataset
- **AWS SageMaker Deployment**: Ready-to-deploy infrastructure for production environments

## Model Architecture

Our architecture enhances the standard DeepLabv3+ model by integrating Selective Kernel Fusion (SK-Fusion) modules in both the ASPP and decoder components:

![image](https://github.com/user-attachments/assets/4fab3efe-be83-4b68-8ab8-9fba62e6d4e6)

![image](https://github.com/user-attachments/assets/bc861c9b-32d8-4193-989e-354b7e4b8690)


The SK-Fusion module improves feature representation by dynamically adjusting feature weights based on the input content, analogous to human visual attention mechanisms.

## Results

Our enhanced model achieves:

- **mIoU**: 52% 
- **Pixel Accuracy**: 90% 

Particularly strong improvements on challenging categories like poles, pedestrians, and traffic signs.


## Requirements

```
pytorch==1.9.0
torchvision==0.10.0
albumentations==1.1.0
numpy==1.21.4
pillow==8.4.0
matplotlib==3.5.0
tqdm==4.62.3
boto3==1.20.24
pyyaml==6.0
scikit-learn==1.0.1
```

## Installation

```bash
# Clone the repository
git clone https://github.com/sarmadsaleem333/Road-Scene-Semantic-Segmentation--along-with-MLOPs.git
cd Road-Scene-Semantic-Segmentation--along-with-MLOPs

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The project uses the Cityscapes dataset. To prepare the dataset:

1. Download the dataset from [Cityscapes website](https://www.cityscapes-dataset.com/)
2. Extract the data to your desired location


## Training

To train the model run the main.phy it will run training_pipeline.py:

Key configuration parameters in `config.yaml`:

```constants.py

  epochs: 60
  batch_size: 8
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "poly"

```

## Evaluation

To evaluate a trained model:


## MLOps Pipeline

Our MLOps pipeline ensures scalable, reproducible deployment with modular components:


![image](https://github.com/user-attachments/assets/42184a2e-c4e6-4841-bb3f-74d9395bf02e)



The pipeline consists of:

1. **Data Ingestion**: Load and validate Cityscapes dataset
2. **Data Transformation**: Apply preprocessing and augmentation
3. **Model Training**: Train the enhanced DeepLabv3+ model
4. **Model Evaluation**: Compute mIoU and pixel accuracy
5. **Model Pusher**: Package and deploy to AWS SageMaker

## Project Structure

```
road_segmentation/
├── components
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model/
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── model_pusher.py
│   └── constanrs.py.py
├── pipeline.py
├── config.yaml
├── requirements.txt
└── README.md
```
![image](https://github.com/user-attachments/assets/29e98400-3d0d-4285-8dbd-daed0c957e07)





## Acknowledgments

- We thank the creators of the Cityscapes dataset for providing the valuable benchmark
- Our implementation builds upon the DeepLabv3+ architecture by Chen et al.

