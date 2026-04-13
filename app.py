import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🧠 Brain Tumor Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload an MRI image to detect tumor type and visualize Grad-CAM</p>",
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 About Project")

st.sidebar.info(
"""
- Model: MobileNetV2  
- Classes: Glioma, Meningioma, Pituitary, No Tumor  
- Includes Grad-CAM Explainability  
"""
)

## ----------------------------
# Download Model from Drive (only once)
# ----------------------------
MODEL_PATH = "brain_tumor_model.keras"

if not os.path.exists(MODEL_PATH):
    st.info("⬇️ Downloading model... please wait")
    
    # 🔴 Replace with your actual Google Drive file ID
    file_id = "1AbCdEfGhIjKlMn"
    
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ----------------------------
# Load Model
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]
# Preprocessing
# ----------------------------
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)
    return img, img_array

# ----------------------------
# Grad-CAM
# ----------------------------
def generate_gradcam(model, img_array, layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ----------------------------
# Upload Section
# ----------------------------
st.markdown("### 📤 Upload MRI Image")

uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

# ----------------------------
# Prediction Section
# ----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with st.spinner("🔍 Analyzing MRI..."):

        # Preprocess
        original_img, img_array = preprocess_image(image)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        label = class_names[predicted_class]

        # GradCAM
        heatmap = generate_gradcam(model, img_array)

        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_resized = cv2.resize(np.array(image), (224,224))

        overlay = heatmap_color * 0.4 + original_resized
        overlay = np.clip(overlay, 0, 255)
        overlay = overlay.astype(np.uint8)

    # ----------------------------
    # Show Prediction
    # ----------------------------
    st.markdown("### 🔍 Prediction Result")

    st.success(f"🧠 Tumor Type: {label}")
    st.metric("Confidence", f"{confidence*100:.2f}%")

    # ----------------------------
    # Show Images (Clean Layout)
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original_resized, caption="Original MRI")

    with col2:
        st.image(heatmap_color, caption="Grad-CAM Heatmap")

    with col3:
        st.image(overlay, caption="Tumor Localization")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)