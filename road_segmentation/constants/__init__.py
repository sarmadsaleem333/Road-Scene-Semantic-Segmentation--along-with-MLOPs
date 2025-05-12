import os

# Base paths
ARTIFACTS_DIR = os.path.join("artifacts")
DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("saved_models")
LOG_DIR = os.path.join("logs")
# TRAIN_IMAGE_DIR = "D:/Citscapes dataset/Input/leftImg8bit/train_final"
# TRAIN_LABEL_DIR = "D:/Citscapes dataset/Output/train_final"
# VAL_IMAGE_DIR = "D:/Citscapes dataset/Input/leftImg8bit/val_final"
# VAL_LABEL_DIR = "D:/Citscapes dataset/Output/val_final"
# TEST_IMAGE_DIR = "D:/Citscapes dataset/Input/leftImg8bit/test_final"
# TEST_LABEL_DIR = "D:/Citscapes dataset/Output/test_final"
TRAIN_IMAGE_DIR = "D:/ShortCityScapes/Input/leftImg8bit/train_final"
TRAIN_LABEL_DIR = "D:/ShortCityScapes/Output/train_final"
VAL_IMAGE_DIR = "D:/ShortCityScapes/Input/leftImg8bit/train_final"
VAL_LABEL_DIR = "D:/ShortCityScapes/Output/train_final"
TEST_IMAGE_DIR = "D:/ShortCityScapes/Input/leftImg8bit/train_final"
TEST_LABEL_DIR = "D:/ShortCityScapes/Output/train_final"

# Model training constants
MODEL_NAME = "deeplabv3plus"
MODEL_PATH = os.path.join(MODEL_DIR, "deeplabv3plus.pth")
NUM_CLASSES = 19
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 3
EPOCHS = 1
DEVICE="cpu"
# AWS constants
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

S3_BUCKET_NAME = 'road-segmentation'
S3_MODEL_KEY = 'models/deeplabv3plus.pth'
S3_METRICS_KEY = 'models/deeplabv3plus_metrics.json'