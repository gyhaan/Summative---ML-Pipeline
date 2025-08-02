import tensorflow as tf

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Preprocesses the input image bytes for prediction."""
    try:
        img = tf.image.decode_image(image_bytes, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.expand_dims(img, axis=0)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # MobileNetV2 preprocessing
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None