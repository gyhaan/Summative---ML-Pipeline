from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import logging
import os
import shutil
from typing import List
from pathlib import Path

from src.prediction import predict_single
from src.model import train_and_retrain_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Skin Cancer Classifier API")

# Define class names based on dataset
CLASS_NAMES = ['benign', 'malignant']

@app.get("/", tags=["Welcome"])
def show_welcome():
    """Returns a welcome message."""
    return {"message": "Welcome to the Skin Cancer Classification API!"}

@app.post("/predict/", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image, makes a prediction, and returns the result.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        image_bytes = await file.read()
        
        class_name, confidence = predict_single(image_bytes)
        
        logger.info(f"Prediction successful: {class_name}, Confidence: {confidence:.2f}")
        
        return JSONResponse(content={
            "predicted_class": class_name,
            "confidence_score": float(confidence)
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

@app.post("/retrain/", tags=["Retraining"])
async def trigger_retraining(files: List[UploadFile] = File(...), class_label: str = Form(...)):
    """
    Receives new images and a class label, saves them to the specified class folder,
    and triggers the retraining process.
    """
    if class_label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid class label. Must be one of {CLASS_NAMES}.")
    
    project_root = Path(__file__).parent.parent
    NEW_DATA_DIR = project_root / "data" / "new_uploads"
    
    if NEW_DATA_DIR.exists():
        shutil.rmtree(NEW_DATA_DIR)
    NEW_DATA_DIR.mkdir(parents=True)

    logger.info("--- Starting File Save Process ---")
    saved_file_count = 0
    
    class_dir = NEW_DATA_DIR / class_label
    class_dir.mkdir(exist_ok=True)
    
    for file in files:
        if not file.content_type.startswith('image/'):
            logger.warning(f"Skipping non-image file: {file.filename}")
            continue
            
        logger.info(f"Processing file: {file.filename}")
        file_path = class_dir / file.filename
        try:
            contents = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            logger.info(f"   -> Successfully saved to: {file_path}")
            saved_file_count += 1
        except Exception as e:
            logger.warning(f"Failed to save file {file.filename}: {e}")

    logger.info(f"--- File Save Process Complete. Saved {saved_file_count} files to {class_label} class. ---")

    if saved_file_count == 0:
        raise HTTPException(status_code=400, detail="No valid image files were provided for retraining.")

    try:
        train_and_retrain_model()
        return JSONResponse(content={"message": f"Model retraining on new {class_label} images initiated successfully."})
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during retraining: {str(e)}")