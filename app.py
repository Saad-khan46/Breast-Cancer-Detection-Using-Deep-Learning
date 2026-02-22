import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageStat

# ---------------------------
# Load trained model
# ---------------------------
model = load_model(r'D:\Main\CNN\Project\breast_cancer_final_model.keras')

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Cancer Classifier", layout="centered")

st.title("🩺 Breast Cancer Detection/ Malignant / Benign / Normal Classifier(Ultrasound)")
st.write("Upload a *breast ultrasound image* and click *Predict*.")
st.subheader("Group Members:")
st.markdown("""
- **Saad Khan**  
- **Siraj Kareem**  
- **Abdullah Hussain**  
- **Zainib Syed**
""")

# ---------------------------
# Show metrics (example values)
# ---------------------------
accuracy = 0.83
loss = 0.45
val_accuracy = 0.81
val_loss = 0.50

st.subheader("📊 Model Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Loss", f"{loss:.2f}")
col3.metric("Validation Accuracy", f"{val_accuracy*100:.2f}%")
col4.metric("Validation Loss", f"{val_loss:.2f}")

st.write("---")

# Class labels
class_mapping = {
    0: "Benign",
    1: "Malignant",
    2: "Normal"
}

# ---------------------------------
# UI
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Ultrasound Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# Prediction logic
# ---------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Validating image..."):
            img = image.resize((224, 224))
            img_array = np.array(img)

            # Ultrasound grayscale validation
            gray = np.mean(img_array, axis=2)
            color_diff = (
                np.mean(np.abs(img_array[:, :, 0] - gray)) +
                np.mean(np.abs(img_array[:, :, 1] - gray)) +
                np.mean(np.abs(img_array[:, :, 2] - gray))
            )

            if color_diff > 20:
                st.error("❌ Invalid image. Please upload a breast ultrasound image only.")
                st.stop()

            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Predicting..."):
            pred_probs = model.predict(img_array)
            pred_class = np.argmax(pred_probs, axis=1)[0]
            confidence = pred_probs[0][pred_class]

            if confidence < 0.70:
                st.error(
                    "❌ The image does not appear to be a valid breast ultrasound.\n"
                    "Prediction rejected for safety."
                )
                st.stop()

            pred_label = class_mapping[pred_class]

        st.success(f"✅ Prediction: *{pred_label}*")
        st.info(f"Confidence: *{confidence * 100:.2f}%*")