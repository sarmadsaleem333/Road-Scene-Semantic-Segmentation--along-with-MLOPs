# import logging
# import os

# from from_root import from_root
# from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# log_dir = 'logs'

# logs_path = os.path.join(from_root(), log_dir, LOG_FILE)

# os.makedirs(log_dir, exist_ok=True)


# logging.basicConfig(
#     filename=logs_path,
#     format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
#     level=logging.DEBUG,
# )

import logging
import os
from from_root import from_root

# Define a single log file name (fixed, not per run)
LOG_FILE = "pipeline.log"

# Log directory setup
log_dir = os.path.join(from_root(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Full path to the log file
logs_path = os.path.join(log_dir, LOG_FILE)

# Clear existing handlers if script reruns in same session
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_path),           # Log to file
        logging.StreamHandler()                   # Log to terminal/command prompt
    ]
)


# Silence noisy 3rd-party loggers
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)