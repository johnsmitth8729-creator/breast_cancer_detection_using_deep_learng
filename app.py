import streamlit as st
from utils import (
    load_my_model,
    preprocess_image,
    make_gradcam_heatmap,
    overlay_heatmap,
    CLASS_NAMES,
)

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Breast Cancer Detection",
    layout="wide",
)

# ======================
# LANGUAGE
# ======================
lang = st.sidebar.selectbox(
    "🌐 Language / Til / Язык",
    ["UZ", "EN", "RU"],
)

TEXTS = {
    "UZ": {
        "title": "Ko‘krak saratonini aniqlash",
        "upload": "Rasm yuklang",
        "result": "Natijalar",
        "mal_alert": "⚠️ Xavfli o‘sma aniqlandi",
    },
    "EN": {
        "title": "Breast Cancer Detection",
        "upload": "Upload image",
        "result": "Results",
        "mal_alert": "⚠️ Malignant tumor detected",
    },
    "RU": {
        "title": "Определение рака груди",
        "upload": "Загрузите изображение",
        "result": "Результаты",
        "mal_alert": "⚠️ Обнаружена злокачественная опухоль",
    },
}

T = TEXTS[lang]

# ======================
# MODEL
# ======================
model = load_my_model("model/breast_model.keras")
LAST_CONV_LAYER = "conv5_block16_concat"

# ======================
# TITLE
# ======================
st.title(T["title"])

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader(
    T["upload"], type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    img_array, original = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    # ===== ORIGINAL =====
    with col1:
        st.image(original, caption="Original", width=400)

    # ===== RESULT =====
    with col2:
        with st.spinner("Analyzing..."):
            preds = model.predict(img_array)[0]

        st.subheader(T["result"])

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("BENIGN", f"{preds[0]*100:.2f}%")
        with m2:
            st.metric("MALIGNANT", f"{preds[1]*100:.2f}%")
        with m3:
            st.metric("NORMAL", f"{preds[2]*100:.2f}%")

        pred_class = preds.argmax()

        # ✅ FAQAT MALIGNANT BO‘LSA
        if CLASS_NAMES[pred_class] == "malignant":
            st.error(T["mal_alert"])

            heatmap = make_gradcam_heatmap(
                img_array, model, LAST_CONV_LAYER
            )
            cam_image = overlay_heatmap(heatmap, original)

            st.image(cam_image, caption="Cancer Localization", width=400)
