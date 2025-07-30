import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path, target_size=(172, 251)):
    """
    Load and resize an image from a file path.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired size (height, width) for resizing.
    
    Returns:
        np.array: Resized image array.
    """
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        # Ensure image is in RGB format
        if img_array.ndim == 2:  # Convert grayscale to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # Remove alpha channel if present
            img_array = img_array[:, :, :3]
        return img_array
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(img_array, normalize=True):
    """
    Preprocess an image array (normalize and flatten for XGBoost).
    
    Args:
        img_array (np.array): Input image array.
        normalize (bool): Whether to normalize pixel values to [0, 1].
    
    Returns:
        np.array: Preprocessed image array (flattened for XGBoost).
    """
    try:
        if normalize:
            img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = img_array.flatten()  # Flatten for XGBoost
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None

def create_data_generator(augmentation=True):
    """
    Create an ImageDataGenerator for data augmentation during training.
    
    Args:
        augmentation (bool): Whether to apply data augmentation.
    
    Returns:
        ImageDataGenerator: Configured data generator.
    """
    if augmentation:
        return ImageDataGenerator(
            rotation_range=20,           # Rotate images by up to 20 degrees
            width_shift_range=0.2,      # Shift width by up to 20%
            height_shift_range=0.2,     # Shift height by up to 20%
            horizontal_flip=True,       # Random horizontal flips
            vertical_flip=True,         # Random vertical flips
            brightness_range=[0.8, 1.2],# Random brightness adjustment
            preprocessing_function=lambda x: tf.image.rgb_to_grayscale(x) if np.random.rand() < 0.1 else x,  # 10% chance of grayscale
            fill_mode='nearest'
        )
    else:
        return ImageDataGenerator()

def load_dataset(data_dir, target_size=(172, 251), batch_size=32, augmentation=True, normalize=True):
    """
    Load and preprocess dataset from directory.
    
    Args:
        data_dir (str): Path to dataset directory (containing 'benign' and 'malignant' subfolders).
        target_size (tuple): Desired image size (height, width).
        batch_size (int): Batch size for data generator.
        augmentation (bool): Whether to apply data augmentation.
        normalize (bool): Whether to normalize pixel values.
    
    Returns:
        tuple: (data_generator, num_samples)
    """
    try:
        datagen = create_data_generator(augmentation)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',  # Binary classification (benign/malignant)
            shuffle=True,
            classes=['benign', 'malignant']
        )
        
        # For XGBoost, we need to preprocess images (flatten and optionally normalize)
        def preprocess_generator(gen):
            while True:
                batch_x, batch_y = next(gen)
                processed_batch_x = np.array([preprocess_image(img, normalize) for img in batch_x])
                yield processed_batch_x, batch_y
        
        num_samples = generator.samples
        return preprocess_generator(generator), num_samples
    except Exception as e:
        logging.error(f"Error loading dataset from {data_dir}: {e}")
        return None, 0

def preprocess_single_image(image_path, target_size=(172, 251), normalize=True):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (height, width).
        normalize (bool): Whether to normalize pixel values.
    
    Returns:
        np.array: Preprocessed image array (flattened).
    """
    img_array = load_image(image_path, target_size)
    if img_array is not None:
        return preprocess_image(img_array, normalize)
    return None

if __name__ == "__main__":
    # Example usage
    train_dir = "data/train"
    test_dir = "data/test"
    
    # Load training dataset with augmentation
    train_gen, train_samples = load_dataset(train_dir, augmentation=True, normalize=True)
    print(f"Loaded {train_samples} training samples")
    
    # Load test dataset without augmentation
    test_gen, test_samples = load_dataset(test_dir, augmentation=False, normalize=True)
    print(f"Loaded {test_samples} test samples")
    
    # Example: Preprocess a single image
    sample_image = "data/test/benign/ISIC_1431322.jpg"
    processed_image = preprocess_single_image(sample_image)
    if processed_image is not None:
        print(f"Processed single image shape: {processed_image.shape}")