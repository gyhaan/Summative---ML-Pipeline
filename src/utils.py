import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_split_data(train_dir, test_dir, image_size, batch_size):
    """
    Loads and splits the dataset into training, validation, and test sets.

    Args:
        train_dir (str): Path to training data directory.
        test_dir (str): Path to test data directory.
        image_size (tuple): Target size for images (height, width).
        batch_size (int): Batch size for data loading.

    Returns:
        tuple: (training_set, validation_set, test_set, class_names)
    """
    logger.info(f"Loading data from train_dir: {train_dir}, test_dir: {test_dir}")
    
    # Verify directories
    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Load training dataset
    try:
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            subset='training',
            seed=42
        )
        logger.info("Loaded training dataset.")
    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}")
        raise

    # Load validation dataset
    try:
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            subset='validation',
            seed=42
        )
        logger.info("Loaded validation dataset.")
    except Exception as e:
        logger.error(f"Failed to load validation dataset: {e}")
        raise

    # Load test dataset
    try:
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False
        )
        logger.info("Loaded test dataset.")
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        raise

    class_names = train_dataset.class_names
    logger.info(f"Class names: {class_names}")

    # Optimize datasets
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names

def count_samples(dataset, class_names):
    """
    Counts the number of samples per class in a dataset.

    Args:
        dataset: TensorFlow dataset with categorical labels.
        class_names: List of class names.

    Returns:
        dict: Dictionary with class names as keys and sample counts as values.
    """
    counts = {name: 0 for name in class_names}
    for _, labels in dataset:
        for label in labels:
            class_idx = tf.argmax(label).numpy()
            counts[class_names[class_idx]] += 1
    return counts

def plot_batch(dataset, class_names, num_images=8):
    """
    Plots a batch of images from the dataset.
    """
    plt.figure(figsize=(15, 10))
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_history(history):
    """
    Plots training history (accuracy, loss, precision, recall).
    """
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix for the given true and predicted labels.

    Args:
        y_true: True labels (list or array of class indices).
        y_pred: Predicted labels (list or array of class indices).
        class_names: List of class names.

    Returns:
        matplotlib.figure.Figure: Confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig