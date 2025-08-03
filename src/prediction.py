import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_single(file):
    MODEL_PATH = Path.cwd() / "models/skin_cancer_class.keras"
    CLASS_NAMES = ['benign', 'malignant']
    IMAGE_SIZE = (224, 224)

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    try:
        image = Image.open(file).convert('RGB')
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]

        if predicted_class_idx >= len(CLASS_NAMES):
            logger.error(f"Predicted class index {predicted_class_idx} out of range for classes {CLASS_NAMES}")
            raise ValueError(f"Predicted class index {predicted_class_idx} out of range")

        return CLASS_NAMES[predicted_class_idx], float(confidence)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise