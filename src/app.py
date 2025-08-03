import streamlit as st
from insights import get_dataset_insights
from prediction import predict_single
from model import train_and_retrain_model
from PIL import Image
import io
import os
from pathlib import Path
import logging
import shutil
from typing import List

st.set_page_config(page_title="Skin Cancer Classifier", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CLASS_NAMES = ['benign', 'malignant']
PROJECT_ROOT = Path.cwd()
UPLOAD_DIR = PROJECT_ROOT / "data/uploads"
NEW_DATA_DIR = PROJECT_ROOT / "data/new_uploads"
DATA_DIR = PROJECT_ROOT / "data"

st.title("Skin Cancer Classifier")

tab1, tab2, tab3 = st.tabs(["Prediction", "Retraining", "Data Insights"])

with tab1:
    st.header("Predict Skin Cancer")
    uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        if st.button("Predict"):
            try:
                uploaded_file.seek(0)
                class_name, confidence = predict_single(uploaded_file)
                logger.info(f"Prediction successful: {class_name}, Confidence: {confidence:.2f}")
                st.success(f"Prediction: {class_name} (Confidence: {confidence:.2%})")
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                st.error(f"Error during prediction: {e}")

with tab2:
    st.header("Retrain Model")
    uploaded_files = st.file_uploader("Upload images for retraining", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    class_label = st.selectbox("Select class label", CLASS_NAMES)
    if st.button("Start Retraining"):
        if uploaded_files:
            try:
                if NEW_DATA_DIR.exists():
                    shutil.rmtree(NEW_DATA_DIR)
                NEW_DATA_DIR.mkdir(parents=True)
                logger.info("--- Starting File Save Process ---")
                saved_file_count = 0
                class_dir = NEW_DATA_DIR / class_label
                class_dir.mkdir(exist_ok=True)
                
                for file in uploaded_files:
                    if not file.type.startswith('image/'):
                        logger.warning(f"Skipping non-image file: {file.name}")
                        continue
                    logger.info(f"Processing file: {file.name}")
                    file_path = class_dir / file.name
                    try:
                        contents = file.read()
                        with open(file_path, "wb") as buffer:
                            buffer.write(contents)
                        logger.info(f"   -> Successfully saved to: {file_path}")
                        saved_file_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to save file {file.name}: {e}")
                
                logger.info(f"--- File Save Process Complete. Saved {saved_file_count} files to {class_label} class. ---")
                
                if saved_file_count == 0:
                    st.error("No valid image files were provided for retraining.")
                else:
                    train_and_retrain_model()
                    st.success(f"Model retraining on new {class_label} images completed successfully.")
            except Exception as e:
                logger.error(f"Error during retraining: {e}")
                st.error(f"Error during retraining: {e}")
        else:
            st.error("Please upload at least one image.")

with tab3:
    st.header("Data Insights")
    insights = get_dataset_insights(DATA_DIR)
    print(DATA_DIR)
    if insights['error_message']:
        st.error(insights['error_message'])
    else:
        if 'class_distribution' in insights['chart_data']:
            st.subheader("Class Distribution")
            st.bar_chart(insights['chart_data']['class_distribution'], x='Class', y='Number of Images')
        
        if 'image_dimensions' in insights['chart_data']:
            df_dims = insights['chart_data']['image_dimensions']
            if not df_dims.empty and all(col in df_dims.columns for col in ['width', 'height', 'class']):
                st.subheader("Image Dimensions")
                st.scatter_chart(df_dims, x='width', y='height', color='class')
            else:
                st.info("Not enough data available to display image dimensions.")

        
        if 'pixel_intensity' in insights['chart_data'] and not insights['chart_data']['pixel_intensity'].empty:
            st.subheader("Pixel Intensity Distribution")
            st.line_chart(insights['chart_data']['pixel_intensity'], x='Pixel Intensity')