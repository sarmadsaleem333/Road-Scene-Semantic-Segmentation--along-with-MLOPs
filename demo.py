
# import warnings
# warnings.filterwarnings("ignore")
# from road_segmentation.pipeline.training_pipeline import run_pipeline

# if __name__ == "__main__":
#     run_pipeline()

from road_segmentation.pipeline.prediction_pipeline import (
    RoadSegmentationImage,
    RoadSegmentationPredictor,
    visualize_prediction,
)

if __name__ == "__main__":
    image_path = "D:/Citscapes dataset/Input/leftImg8bit/test_final/berlin_000000_000019_leftImg8bit.png"

    predictor = RoadSegmentationPredictor()
    image_data = RoadSegmentationImage(image_path)
    tensor, original = image_data.preprocess()
    pred_mask = predictor.predict(tensor)

    visualize_prediction(original, pred_mask)
