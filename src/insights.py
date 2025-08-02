import os
import numpy as np
from PIL import Image
import pathlib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataset_insights(data_dir):
    """
    Computes insights about the dataset including class distribution, image dimensions, and pixel intensity distribution.

    Args:
        data_dir (str): Path to the training data directory.

    Returns:
        dict: Dictionary containing class_distribution, image_dimensions, pixel_intensity, chart_data, and error_message.
              chart_data contains pandas DataFrames for Streamlit charts.
    """
    data_dir = pathlib.Path(data_dir)
    class_distribution = {}
    image_dimensions = []
    pixel_intensity = {}
    chart_data = {}
    error_message = None

    if not data_dir.exists():
        error_message = f"Directory {data_dir} does not exist."
        logger.error(error_message)
        return {
            'class_distribution': {},
            'image_dimensions': [],
            'pixel_intensity': {},
            'chart_data': {},
            'error_message': error_message
        }

    # Iterate through class directories (benign, malignant)
    for class_name in os.listdir(data_dir):
        class_path = data_dir / class_name
        if not class_path.is_dir():
            continue

        # 1. Class Distribution
        images = list(class_path.glob('*.[jJ][pP][gG]'))
        class_distribution[class_name] = len(images)
        logger.info(f"Found {len(images)} images in {class_name}")

        # Sample up to 100 images per class for dimensions and pixel intensity
        class_images = images[:100]
        pixel_intensity[class_name] = []

        for img_path in class_images:
            try:
                with Image.open(img_path) as img:
                    # 2. Image Dimensions
                    width, height = img.size
                    image_dimensions.append({
                        'class': class_name,
                        'width': width,
                        'height': height
                    })

                    # 3. Pixel Intensity (convert to grayscale)
                    img_gray = img.convert('L')
                    img_array = np.array(img_gray)
                    pixel_intensity[class_name].append(img_array.flatten())
            except Exception as e:
                logger.warning(f"Failed to process image {img_path}: {e}")

    if not class_distribution:
        error_message = f"No valid images found in {data_dir}."
        logger.error(error_message)
        return {
            'class_distribution': {},
            'image_dimensions': [],
            'pixel_intensity': {},
            'chart_data': {},
            'error_message': error_message
        }

    # Prepare chart data for Streamlit
    try:
        # 1. Class Distribution (for st.bar_chart)
        class_distribution_df = pd.DataFrame.from_dict(
            class_distribution, orient='index', columns=['Number of Images']
        ).reset_index().rename(columns={'index': 'Class'})
        chart_data['class_distribution'] = class_distribution_df
        logger.info("Prepared class distribution data")

        # 2. Image Dimensions (for st.scatter_chart)
        image_dimensions_df = pd.DataFrame(image_dimensions)
        chart_data['image_dimensions'] = image_dimensions_df
        logger.info("Prepared image dimensions data")

        # 3. Pixel Intensity (for st.line_chart, approximating KDE)
        pixel_intensity_dfs = []
        for class_name, intensities in pixel_intensity.items():
            if intensities:
                # Compute histogram for intensities (approximates KDE)
                hist, bins = np.histogram(intensities, bins=50, range=(0, 255), density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                df = pd.DataFrame({
                    'Pixel Intensity': bin_centers,
                    class_name: hist
                })
                pixel_intensity_dfs.append(df)
        if pixel_intensity_dfs:
            # Merge dataframes on Pixel Intensity
            pixel_intensity_df = pixel_intensity_dfs[0]
            for df in pixel_intensity_dfs[1:]:
                pixel_intensity_df = pixel_intensity_df.merge(df, on='Pixel Intensity', how='outer').fillna(0)
            chart_data['pixel_intensity'] = pixel_intensity_df
        else:
            chart_data['pixel_intensity'] = pd.DataFrame()
        logger.info("Prepared pixel intensity data")

    except Exception as e:
        logger.error(f"Failed to prepare chart data: {e}")
        error_message = f"Error preparing chart data: {e}"

    return {
        'class_distribution': class_distribution,
        'image_dimensions': image_dimensions,
        'pixel_intensity': pixel_intensity,
        'chart_data': chart_data,
        'error_message': error_message
    }