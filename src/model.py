import os
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import logging
from .preprocessing import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(params=None):
    """
    Create an XGBoost classifier with specified parameters.
    
    Args:
        params (dict): Optional hyperparameters for XGBoost.
    
    Returns:
        XGBClassifier: Configured XGBoost model.
    """
    default_params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
    if params:
        default_params.update(params)
    
    try:
        model = XGBClassifier(**default_params)
        logging.info("XGBoost model created successfully")
        return model
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        return None

def train_model(model, train_dir, batch_size=16, tune_hyperparameters=False):
    """
    Train the XGBoost model on the training dataset.
    
    Args:
        model (XGBClassifier): The XGBoost model to train.
        train_dir (str): Path to training data directory.
        batch_size (int): Batch size for data loading.
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning.
    
    Returns:
        XGBClassifier: Trained model.
    """
    try:
        # Load training data
        train_gen, num_samples = load_dataset(train_dir, batch_size=batch_size, augmentation=True, normalize=True)
        logging.info(f"Loaded {num_samples} training samples")
        
        # Collect all training data
        X_train, y_train = [], []
        for _ in range((num_samples // batch_size) + (1 if num_samples % batch_size else 0)):
            batch_x, batch_y = next(train_gen)
            # Reshape batch_x to 2D: (batch_size, height * width * channels)
            batch_x = batch_x.reshape(batch_x.shape[0], -1)
            X_train.append(batch_x)
            y_train.append(batch_y)
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        
        # Verify shape
        logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        if tune_hyperparameters:
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200]
            }
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, error_score='raise')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logging.info(f"Best parameters from GridSearchCV: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)
        
        logging.info("Model training completed")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def evaluate_model(model, test_dir, batch_size=16):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (XGBClassifier): Trained XGBoost model.
        test_dir (str): Path to test data directory.
        batch_size (int): Batch size for data loading.
    
    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, f1, confusion matrix).
    """
    try:
        # Load test data
        test_gen, num_samples = load_dataset(test_dir, batch_size=batch_size, augmentation=False, normalize=True)
        logging.info(f"Loaded {num_samples} test samples")
        
        # Collect all test data
        X_test, y_test = [], []
        for _ in range((num_samples // batch_size) + (1 if num_samples % batch_size else 0)):
            batch_x, batch_y = next(test_gen)
            batch_x = batch_x.reshape(batch_x.shape[0], -1)
            X_test.append(batch_x)
            y_test.append(batch_y)
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return None

def retrain_model(model, new_data_dir, model_path, batch_size=16):
    """
    Retrain the model with new data and save the updated model.
    
    Args:
        model (XGBClassifier): Existing trained model.
        new_data_dir (str): Path to new data directory.
        model_path (str): Path to save the updated model.
        batch_size (int): Batch size for data loading.
    
    Returns:
        XGBClassifier: Retrained model.
    """
    try:
        # Load new data
        new_gen, num_samples = load_dataset(new_data_dir, batch_size=batch_size, augmentation=True, normalize=True)
        logging.info(f"Loaded {num_samples} samples for retraining")
        
        # Collect new data
        X_new, y_new = [], []
        for _ in range((num_samples // batch_size) + (1 if num_samples % batch_size else 0)):
            batch_x, batch_y = next(new_gen)
            batch_x = batch_x.reshape(batch_x.shape[0], -1)
            X_new.append(batch_x)
            y_new.append(batch_y)
        X_new = np.vstack(X_new)
        y_new = np.hstack(y_new)
        
        # Retrain model
        model.fit(X_new, y_new)
        
        # Save updated model
        joblib.dump(model, model_path)
        logging.info(f"Retrained model saved to {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error retraining model: {e}")
        return None

def save_model(model, model_path):
    """
    Save the model to a file.
    
    Args:
        model (XGBClassifier): Trained model to save.
        model_path (str): Path to save the model.
    """
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def trigger_retrain(model, new_data_dir, model_path, performance_threshold=0.8, test_dir=None, batch_size=16):
    """
    Trigger retraining if model performance falls below a threshold or new data is provided.
    
    Args:
        model (XGBClassifier): Current model.
        new_data_dir (str): Path to new data directory.
        model_path (str): Path to save the updated model.
        performance_threshold (float): Accuracy threshold to trigger retraining.
        test_dir (str): Path to test data for performance evaluation (optional).
        batch_size (int): Batch size for data loading.
    
    Returns:
        XGBClassifier: Retrained model if triggered, else original model.
    """
    try:
        if test_dir:
            metrics = evaluate_model(model, test_dir, batch_size)
            if metrics and metrics['accuracy'] < performance_threshold:
                logging.info(f"Model accuracy ({metrics['accuracy']:.4f}) below threshold ({performance_threshold}). Triggering retraining.")
                return retrain_model(model, new_data_dir, model_path, batch_size)
            else:
                logging.info(f"Model accuracy ({metrics['accuracy']:.4f}) above threshold. No retraining needed.")
                return model
        else:
            # If new data is provided but no test_dir, retrain directly
            logging.info("New data provided. Triggering retraining.")
            return retrain_model(model, new_data_dir, model_path, batch_size)
    except Exception as e:
        logging.error(f"Error in trigger_retrain: {e}")
        return model

if __name__ == "__main__":
    # Example usage
    train_dir = "data/train"
    test_dir = "data/test"
    model_path = "models/optimized_xgb_model.pkl"
    new_data_dir = "data/new_data"
    
    # Create and train model
    model = create_model()
    if model:
        model = train_model(model, train_dir, batch_size=16, tune_hyperparameters=True)
        if model:
            save_model(model, model_path)
            
            # Evaluate model
            metrics = evaluate_model(model, test_dir, batch_size=16)
            if metrics:
                print("Evaluation Metrics:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            
            # Trigger retraining
            model = trigger_retrain(model, new_data_dir, model_path, performance_threshold=0.8, test_dir=test_dir)