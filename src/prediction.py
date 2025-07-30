import os
import numpy as np
import joblib
from preprocessing import preprocess_single_image, load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    """
    Load the trained XGBoost model from a file.
    
    Args:
        model_path (str): Path to the saved model file.
    
    Returns:
        XGBClassifier: Loaded model, or None if loading fails.
    """
    try:
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            return None
        model = joblib.load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None

def predict_single_image(model, image_path, target_size=(172, 251), normalize=True):
    """
    Predict the class (benign/malignant) for a single image.
    
    Args:
        model (XGBClassifier): Trained XGBoost model.
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (height, width).
        normalize (bool): Whether to normalize pixel values.
    
    Returns:
        dict: Prediction result with class label and probability, or None if prediction fails.
    """
    try:
        # Preprocess the image
        processed_image = preprocess_single_image(image_path, target_size, normalize)
        if processed_image is None:
            logging.error(f"Failed to preprocess image: {image_path}")
            return None
        
        # Reshape for XGBoost (single sample)
        processed_image = processed_image.reshape(1, -1)
        
        # Make prediction
        prob = model.predict_proba(processed_image)[0, 1]  # Probability for malignant class
        prediction = 1 if prob > 0.5 else 0
        label = 'malignant' if prediction == 1 else 'benign'
        
        result = {
            'image_path': image_path,
            'prediction': label,
            'probability': float(prob)
        }
        logging.info(f"Prediction for {image_path}: {label} (probability: {prob:.4f})")
        return result
    except Exception as e:
        logging.error(f"Error predicting for image {image_path}: {e}")
        return None

def predict_batch(model, test_dir, batch_size=32, normalize=True):
    """
    Predict classes for a batch of images in the test dataset.
    
    Args:
        model (XGBClassifier): Trained XGBoost model.
        test_dir (str): Path to test data directory.
        batch_size (int): Batch size for data loading.
        normalize (bool): Whether to normalize pixel values.
    
    Returns:
        list: List of prediction dictionaries with filenames, labels, and probabilities.
    """
    try:
        # Load test dataset without augmentation
        test_gen, num_samples = load_dataset(test_dir, batch_size=batch_size, augmentation=False, normalize=normalize)
        logging.info(f"Loaded {num_samples} test samples for batch prediction")
        
        predictions = []
        filenames = test_gen.filenames
        
        # Process all batches
        for i in range(num_samples // batch_size + (1 if num_samples % batch_size else 0)):
            batch_x, _ = next(test_gen)
            batch_probs = model.predict_proba(batch_x)[:, 1]  # Probabilities for malignant class
            batch_preds = (batch_probs > 0.5).astype(int)
            
            for j in range(len(batch_preds)):
                idx = i * batch_size + j
                if idx < num_samples:
                    label = 'malignant' if batch_preds[j] == 1 else 'benign'
                    result = {
                        'image_path': os.path.join(test_dir, filenames[idx]),
                        'prediction': label,
                        'probability': float(batch_probs[j])
                    }
                    predictions.append(result)
                    logging.info(f"Prediction for {filenames[idx]}: {label} (probability: {batch_probs[j]:.4f})")
        
        return predictions
    except Exception as e:
        logging.error(f"Error in batch prediction: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    model_path = "models/optimized_xgb_model.pkl"
    test_dir = "data/test"
    sample_image = "data/test/benign/ISIC_1431322.jpg"
    
    # Load model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Single image prediction
    result = predict_single_image(model, sample_image)
    if result:
        print(f"Single Image Prediction: {result['image_path']}")
        print(f"  Predicted: {result['prediction']}, Probability: {result['probability']:.4f}")
    
    # Batch prediction
    predictions = predict_batch(model, test_dir, batch_size=32)
    print("\nBatch Prediction Results (first 10):")
    for pred in predictions[:10]:
        print(f"Image: {pred['image_path']}, Predicted: {pred['prediction']}, Probability: {pred['probability']:.4f}")