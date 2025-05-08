from road_segmentation.logger import logging
import sys

from road_segmentation.exception import RoadSegmentationException


# logging.info("Logging has been set up successfully.")


try:
    a=20/0

except Exception as e:
    raise RoadSegmentationException(e, sys)