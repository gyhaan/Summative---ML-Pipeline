import tensorflow as tf
import os
from pathlib import Path
from src.utils import load_and_split_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(num_classes):
    """
    Builds and compiles a new transfer learning model using MobileNetV2.
    """
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
    """
    Loads the existing model and continues training it on the dataset, including new uploads.
    """
    logger.info("--- Starting Model Retraining ---")
    
    # 1. Define Constants
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    project_root = Path(__file__).parent.parent
    MODEL_PATH = project_root / 'models' / 'skin_cancer_class.keras'
    DATA_DIR = project_root / 'data'
    TRAIN_DIR = DATA_DIR / 'train'
    TEST_DIR = DATA_DIR / 'test'
    NEW_DATA_DIR = DATA_DIR / 'new_uploads'

    # 2. Verify Directories
    logger.info(f"Checking directories: TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}, NEW_DATA_DIR={NEW_DATA_DIR}")
    if not TRAIN_DIR.exists():
        logger.error(f"Could not find directory {TRAIN_DIR}")
        raise FileNotFoundError(f"Could not find directory {TRAIN_DIR}")
    if not TEST_DIR.exists():
        logger.error(f"Could not find directory {TEST_DIR}")
        raise FileNotFoundError(f"Could not find directory {TEST_DIR}")

    # 3. Load the Data
    logger.info("Loading and preparing data...")
    try:
        training_set, validation_set, test_set, class_names = load_and_split_data(
            train_dir=str(TRAIN_DIR),
            test_dir=str(TEST_DIR),
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        logger.info(f"Loaded data with classes: {class_names}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Include new uploads in training set if they exist
    if NEW_DATA_DIR.exists():
        logger.info(f"Loading new uploads from {NEW_DATA_DIR}...")
        try:
            new_dataset = tf.keras.utils.image_dataset_from_directory(
                NEW_DATA_DIR,
                labels='inferred',
                label_mode='categorical',
                image_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            new_class_names = new_dataset.class_names
            logger.info(f"New dataset classes: {new_class_names}")
            if set(new_class_names) != set(class_names):
                logger.warning(f"Class mismatch: Training set has {class_names}, new uploads have {new_class_names}")
                raise ValueError(f"Class mismatch: Training set has {class_names}, new uploads have {new_class_names}")
            training_set = training_set.concatenate(new_dataset).cache().prefetch(tf.data.AUTOTUNE)
            logger.info("New uploads added to training set.")
        except Exception as e:
            logger.warning(f"Failed to load new uploads: {e}. Continuing with original training data.")

    # 4. Load or Build Model
    num_classes = len(class_names)
    logger.info(f"Expected number of classes: {num_classes} ({class_names})")
    
    if MODEL_PATH.exists():
        logger.info(f"Loading existing model from {MODEL_PATH}...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            # Check if model output matches number of classes
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
    
    # 5. Continue Training the Model
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

    # 6. Save the Newly Retrained Model
    logger.info(f"Saving updated model to {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("--- Model retraining complete. ---")
    
    return history, model, class_names

if __name__ == '__main__':
    history, _, _ = train_and_retrain_model()