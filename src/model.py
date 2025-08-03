import tensorflow as tf
import os
from pathlib import Path
from utils import load_and_split_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(num_classes=2):
    input_shape = (224, 224, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def train_and_retrain_model():
    logger.info("--- Starting Model Training/Retraining ---")
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    MODEL_PATH = Path.cwd() / "models/skin_cancer_class.keras"
    DATA_DIR = Path.cwd() / "data"
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    NEW_DATA_DIR = DATA_DIR / "new_uploads"
    CLASS_NAMES = ['benign', 'malignant']

    logger.info(f"Checking directories: TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}, NEW_DATA_DIR={NEW_DATA_DIR}")
    if not os.path.exists(TRAIN_DIR):
        logger.error(f"Could not find directory {TRAIN_DIR}")
        raise FileNotFoundError(f"Could not find directory {TRAIN_DIR}")
    if not os.path.exists(TEST_DIR):
        logger.error(f"Could not find directory {TEST_DIR}")
        raise FileNotFoundError(f"Could not find directory {TEST_DIR}")

    train_subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    if set(train_subdirs) != set(CLASS_NAMES):
        logger.error(f"Training directory {TRAIN_DIR} contains unexpected classes: {train_subdirs}. Expected: {CLASS_NAMES}")
        raise ValueError(f"Training directory contains unexpected classes: {train_subdirs}. Expected: {CLASS_NAMES}")

    logger.info("Loading and preparing data...")
    try:
        training_set, validation_set, test_set, class_names = load_and_split_data(
            train_dir=str(TRAIN_DIR),
            test_dir=str(TEST_DIR),
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        if set(class_names) != set(CLASS_NAMES):
            logger.error(f"Dataset classes {class_names} do not match expected classes {CLASS_NAMES}")
            raise ValueError(f"Dataset classes {class_names} do not match expected classes {CLASS_NAMES}")
        logger.info(f"Loaded data with classes: {class_names}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    if os.path.exists(NEW_DATA_DIR):
        logger.info(f"Loading new uploads from {NEW_DATA_DIR}...")
        try:
            new_subdirs = [d for d in os.listdir(NEW_DATA_DIR) if os.path.isdir(os.path.join(NEW_DATA_DIR, d))]
            if set(new_subdirs) != set(CLASS_NAMES):
                logger.error(f"New uploads directory {NEW_DATA_DIR} contains unexpected classes: {new_subdirs}. Expected: {CLASS_NAMES}")
                raise ValueError(f"New uploads directory contains unexpected classes: {new_subdirs}. Expected: {CLASS_NAMES}")
            new_dataset = tf.keras.utils.image_dataset_from_directory(
                NEW_DATA_DIR,
                labels='inferred',
                label_mode='categorical',
                image_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            new_class_names = new_dataset.class_names
            if set(new_class_names) != set(CLASS_NAMES):
                logger.error(f"New dataset classes {new_class_names} do not match expected classes {CLASS_NAMES}")
                raise ValueError(f"New dataset classes {new_class_names} do not match expected classes {CLASS_NAMES}")
            training_set = training_set.concatenate(new_dataset).cache().prefetch(tf.data.AUTOTUNE)
            logger.info("New uploads added to training set.")
        except Exception as e:
            logger.warning(f"Failed to load new uploads: {e}. Continuing with original training data.")

    num_classes = len(CLASS_NAMES)
    logger.info(f"Expected number of classes: {num_classes} ({CLASS_NAMES})")
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from {MODEL_PATH}...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            output_shape = model.layers[-1].output_shape[-1]
            if output_shape != num_classes:
                logger.warning(f"Model output shape ({output_shape}) does not match dataset classes ({num_classes}). Rebuilding model.")
                model = build_model(num_classes=num_classes)
            else:
                logger.info("Loaded model matches dataset classes.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Building new model...")
            model = build_model(num_classes=num_classes)
    else:
        logger.info("No existing model found. Building new model...")
        model = build_model(num_classes=num_classes)
    
    logger.info("Continuing training...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    try:
        history = model.fit(
            training_set,
            validation_data=validation_set,
            epochs=10,
            callbacks=[early_stopping]
        )
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info("Evaluating model on test set...")
    results = model.evaluate(test_set, return_dict=True)
    logger.info(f"Test metrics: {results}")

    logger.info(f"Saving updated model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("--- Model training/retraining complete. ---")
    
    return history, model, CLASS_NAMES

if __name__ == '__main__':
    history, _, _ = train_and_retrain_model()