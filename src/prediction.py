import tensorflow as tf
import numpy as np
from pathlib import Path
from src.preprocessing import preprocess_image

CLASS_NAMES = ['benign', 'malignant']  # Match dataset folder names

def load_model(model_path=None):
    """Loads the trained model."""
    if model_path is None:
        project_root = Path(__file__).parent.parent
        model_path = project_root / 'models' / 'skin_cancer_class.keras'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_single(image_bytes, model=None):
    """Makes a prediction on a single image."""
    if model is None:
        model = load_model()
    if model is None:
        return None, None
    processed_image = preprocess_image(image_bytes)
    prediction = model.predict(processed_image)
    
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence_score = float(np.max(prediction))
    
    return predicted_class_name, confidence_score

def predict_single_from_path(image_path, model=None):
    """Makes a prediction on a single image from a file path."""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return predict_single(image_bytes, model)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None