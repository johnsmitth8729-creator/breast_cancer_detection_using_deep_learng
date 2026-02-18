import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# ===== CONFIG =====
IMG_SIZE = 256
CLASS_NAMES = ["benign", "malignant", "normal"]

# =============================
# MODEL LOAD
# =============================
@st.cache_resource
def load_my_model(model_path):
    return load_model(model_path)

# =============================
# PREPROCESS
# =============================
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original = img.copy()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img, original

# =============================
# GRAD-CAM
# =============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# =============================
# OVERLAY
# =============================
def overlay_heatmap(heatmap, original_img, alpha=0.4):
    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0]),
    )
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
