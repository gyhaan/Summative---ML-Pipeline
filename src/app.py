import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os

from src.insights import get_dataset_insights

st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🩺",
    layout="wide"
)

PREDICT_API_URL = "http://127.0.0.1:8000/predict/"
RETRAIN_API_URL = "http://127.0.0.1:8000/retrain/"

@st.cache_data
def load_insights():
    """Loads and caches the dataset insights."""
    train_dir = project_root / "data" / "train"
    if train_dir.exists():
        return get_dataset_insights(str(train_dir))
    return {'error_message': "Training directory not found."}

st.title("🩺 Skin Cancer Classifier")
st.write("An end-to-end MLOps project to classify skin lesions as benign or malignant from images.")

tab1, tab2, tab3 = st.tabs(["🔍 Predictor", "📊 Data Insights", "🔄 Retraining"])

with tab1:
    st.header("Predict Skin Cancer")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if st.button('Predict'):
            with st.spinner('Analyzing the image...'):
                image_bytes = uploaded_file.getvalue()
                files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                try:
                    response = requests.post(PREDICT_API_URL, files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"**Predicted Class:** {result['predicted_class']}")
                        st.info(f"**Confidence:** {result['confidence_score']:.2%}")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the API: {e}")

with tab2:
    st.header("Exploratory Data Analysis")
    insights_data = load_insights()

    if insights_data.get('error_message'):
        st.warning(f"Could not load data for insights: {insights_data['error_message']}")
    elif insights_data.get('chart_data') and insights_data['chart_data'].get('class_distribution') is not None:
        st.subheader("1. Class Distribution")
        st.bar_chart(insights_data['chart_data']['class_distribution'], x='Class', y='Number of Images')
        st.markdown("""
        **Interpretation:** This bar chart shows the number of images for each class (benign, malignant) in the training set.
        - **Story:** If the dataset is imbalanced (e.g., significantly more benign images than malignant), the model may become biased toward predicting the majority class (benign). This could lead to poor performance on malignant cases, which are critical for early detection. To address this, we should consider techniques like oversampling the minority class, using class weights in the loss function, or collecting more malignant images. Per-class metrics (e.g., precision, recall) are essential to evaluate model performance fairly.
        """)
        st.divider()

        st.subheader("2. Image Dimensions")
        st.scatter_chart(
            insights_data['chart_data']['image_dimensions'],
            x='width',
            y='height',
            color='class',
            size=50
        )
        st.markdown("""
        **Interpretation:** This scatter chart displays the width and height of a sample of images from the dataset, with points colored by class.
        - **Story:** If the images vary widely in size, it indicates the need for consistent preprocessing (e.g., resizing to 224x224 pixels) to ensure the model receives uniform input. Consistent dimensions are crucial for MobileNetV2, which expects fixed-size inputs. Variations in size may also reflect differences in image sources (e.g., clinical vs. consumer cameras), which could introduce noise if not standardized.
        """)
        st.divider()

        st.subheader("3. Pixel Intensity Distribution")
        if not insights_data['chart_data']['pixel_intensity'].empty:
            st.line_chart(insights_data['chart_data']['pixel_intensity'], x='Pixel Intensity')
            st.markdown("""
            **Interpretation:** This line chart shows the distribution of pixel intensities (grayscale, 0-255) for each class.
            - **Story:** Malignant lesions often have darker or more irregular pigmentation compared to benign ones, which may appear lighter or more uniform. If the malignant class shows a broader or shifted intensity distribution (e.g., peaks at lower intensities), it suggests that pixel intensity is a key feature for distinguishing classes. The model can leverage these differences, but we must ensure preprocessing (e.g., normalization) preserves these intensity variations for effective learning.
            """)
        else:
            st.warning("No pixel intensity data available for visualization.")
    else:
        st.warning("No valid images found in the training directory. Please ensure `data/train/benign` and `data/train/malignant` contain images.")

with tab3:
    st.header("Model Retraining")
    st.write("Upload images and select their class (benign or malignant) to retrain the model.")
    
    retrain_files = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    class_label = st.selectbox("Select the class for these images:", ["benign", "malignant"])

    if retrain_files:
        st.write(f"{len(retrain_files)} images selected for retraining as {class_label}.")
        if st.button("Start Retraining"):
            with st.spinner("Retraining process initiated..."):
                files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in retrain_files]
                data = {"class_label": class_label}
                try:
                    response = requests.post(RETRAIN_API_URL, files=files_to_upload, data=data)
                    if response.status_code == 200:
                        st.success("Model retraining process completed successfully!")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the API: {e}")