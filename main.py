
from road_segmentation.pipeline.training_pipeline import run_pipeline
if __name__ == "__main__":
    run_pipeline()
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io
import numpy as np
from road_segmentation.pipeline.prediction_pipeline import (
    RoadSegmentationImage,
    RoadSegmentationPredictor,
    visualize_prediction,
)
from PIL import Image
import torch
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define a request model for the input image
class ImageRequest(BaseModel):
    image_path: str

# Initialize the Predictor (model loading logic can be inside this class)
predictor = RoadSegmentationPredictor()

# Helper function for in-memory image processing
def process_image(file: UploadFile):
    image = Image.open(file.file)
    image = np.array(image.convert("RGB"))
    return image

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Process the image received in the request
        original = process_image(file)
        image_data = RoadSegmentationImage(image_path=None)  # No path needed, as we load the image directly
        tensor, _ = image_data.preprocess(original)  # Pass the image directly
        pred_mask = predictor.predict(tensor)

        # Visualize prediction (optional, for testing or debugging)
        visualize_prediction(original, pred_mask)

        # Returning the mask or result as response
        # You can return a JSON with a result mask or just an image file
        return {"status": "success", "prediction_mask": pred_mask.tolist()}  # Convert np.ndarray to list

    except Exception as e:
        return {"status": "error", "message": str(e)}

# To run the FastAPI app (command line instruction)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
