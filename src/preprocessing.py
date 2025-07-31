import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_image(image_path, target_size=(172, 251), normalize=True):
    """
    Load and preprocess a single image for prediction or evaluation.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (height, width).
        normalize (bool): Whether to normalize pixel values.
    
    Returns:
        np.ndarray: Preprocessed image array, or None if preprocessing fails.
    """
    try:
        # Convert to absolute path
        image_path = os.path.abspath(image_path)
        logging.info(f"Attempting to load image: {image_path}")
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return None
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        if normalize:
            img_array = img_array / 255.0
        img_array = img_array.flatten()
        logging.info(f"Preprocessed image: {image_path}")
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None

def preprocess_single_image(image_path, target_size=(172, 251), normalize=True):
    """
    Wrapper for load_and_preprocess_image to maintain compatibility with prediction.py.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (height, width).
        normalize (bool): Whether to normalize pixel values.
    
    Returns:
        np.ndarray: Preprocessed image array, or None if preprocessing fails.
    """
    return load_and_preprocess_image(image_path, target_size, normalize)

def load_dataset(data_dir, batch_size=32, augmentation=True, normalize=True):
    """
    Load dataset using ImageDataGenerator.
    
    Args:
        data_dir (str): Path to data directory (containing benign/ and malignant/ subfolders).
        batch_size (int): Batch size for data loading.
        augmentation (bool): Whether to apply data augmentation.
        normalize (bool): Whether to normalize pixel values.
    
    Returns:
        tuple: (generator, number of samples).
    """
    try:
        # Convert to absolute path
        data_dir = os.path.abspath(data_dir)
        logging.info(f"Attempting to load dataset from: {data_dir}")
        
        if not os.path.exists(data_dir):
            logging.error(f"Data directory not found: {data_dir}")
            return None, 0
        
        # Check for subdirectories
        benign_dir = os.path.join(data_dir, 'benign')
        malignant_dir = os.path.join(data_dir, 'malignant')
        logging.info(f"Checking benign directory: {benign_dir}")
        logging.info(f"Checking malignant directory: {malignant_dir}")
        
        if not os.path.exists(benign_dir) or not os.path.exists(malignant_dir):
            logging.error(f"Subdirectories benign/ or malignant/ not found in {data_dir}")
            return None, 0
        
        # Count images
        benign_images = [f for f in os.listdir(benign_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        malignant_images = [f for f in os.listdir(malignant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_samples = len(benign_images) + len(malignant_images)
        logging.info(f"Found {num_samples} images in {data_dir} (benign: {len(benign_images)}, malignant: {len(malignant_images)})")
        
        if num_samples == 0:
            logging.error(f"No images found in {benign_dir} or {malignant_dir}")
            return None, 0

        # Set up data generator
        if augmentation:
            datagen = ImageDataGenerator(
                rescale=1./255 if normalize else None,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.8, 1.2],
                preprocessing_function=lambda x: x if not normalize else x / 255.0
            )
        else:
            datagen = ImageDataGenerator(
                rescale=1./255 if normalize else None,
                preprocessing_function=lambda x: x if not normalize else x / 255.0
            )

        # Load dataset
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(172, 251),
            batch_size=batch_size,
            class_mode='binary',
            classes=['benign', 'malignant'],
            shuffle=True
        )
        logging.info(f"Loaded dataset from {data_dir} with {num_samples} samples")
        return generator, num_samples
    except Exception as e:
        logging.error(f"Error loading dataset from {data_dir}: {e}")
        return None, 0

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.abspath("data/train")
    logging.info(f"Testing dataset loading from: {data_dir}")
    generator, num_samples = load_dataset(data_dir, batch_size=32, augmentation=True, normalize=True)
    if generator:
        print(f"Loaded {num_samples} samples")
        x, y = next(generator)
        print(f"Batch shape: {x.shape}, Labels: {y.shape}")