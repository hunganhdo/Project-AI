# ===============================
# FOOD VN AI DETECTOR – PREMIUM VISUAL UI
# Focus: BẮT MẮT – HIỆN ĐẠI – TRÌNH DIỄN ĐỒ ÁN
# ===============================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="FoodDetector – AI Vision",
    page_icon="🍜",
    layout="wide"
)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model", "best.pt")
    return YOLO(path) if os.path.exists(path) else None

model = load_model()

# ===============================
# PREMIUM CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(180deg, #fff7f2, #ffffff);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

/* NAVBAR */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 25px;
    border-radius: 16px;
    background: linear-gradient(135deg, #ff7043, #ff5722);
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px rgba(255,87,34,.35);
}

.nav-title {
    font-size: 1.5rem;
    font-weight: 700;
}

.nav-links span {
    margin-left: 15px;
    font-weight: 500;
    opacity: .9;
}

/* HERO */
.hero {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 30px;
    align-items: center;
    margin-bottom: 35px;
}

.hero-text h1 {
    font-size: 3rem;
    font-weight: 700;
    color: #ff5722;
}

.hero-text p {
    font-size: 1.1rem;
    color: #555;
    margin-top: 10px;
}

.hero-card {
    background: white;
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(0,0,0,.15);
}

.hero-card img {
    width: 100%;
}

/* CARDS */
.card {
    background: white;
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 12px 30px rgba(0,0,0,.12);
    margin-bottom: 25px;
}

.card h3 {
    margin-bottom: 15px;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #ff7043, #ff5722);
    color: white;
    border-radius: 999px;
    padding: 12px 36px;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    box-shadow: 0 6px 20px rgba(255,87,34,.35);
}

/* BADGE */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #43a047, #66bb6a);
    color: white;
    padding: 8px 16px;
    border-radius: 999px;
    margin: 6px;
    font-size: .9rem;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# NAVBAR
# ===============================
page = st.radio("", ["🏠 Home", "ℹ️ About"], horizontal=True)

st.markdown("""
<div class="navbar">
    <div class="nav-title">🍜 FoodDetector AI</div>
    <div class="nav-links"><span>Computer Vision</span><span>YOLO</span><span>Vietnamese Food</span></div>
</div>
""", unsafe_allow_html=True)

# ===============================
# HOME
# ===============================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-text">
            <h1>Vietnamese Food<br>Detection AI</h1>
            <p>Upload image or video and let AI recognize Vietnamese dishes instantly.</p>
        </div>
        <div class="hero-card">
            <img src="https://images.unsplash.com/photo-1604908177225-6c9b9e8c2e07">
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,1.2], gap="large")

    # INPUT
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📥 Upload data")
        mode = st.radio("Mode", ["Image", "Video"])
        conf = st.slider("Confidence", 0.1, 1.0, 0.3)

        img = None
        vid = None

        if mode == "Image":
            img = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
            if img:
                image = Image.open(img)
                st.image(image, use_container_width=True)

        else:
            vid = st.file_uploader("Upload video", type=["mp4","avi"])
            if vid:
                st.video(vid)

        st.markdown('</div>', unsafe_allow_html=True)

    # OUTPUT
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Result")

        if model is None:
            st.error("Model not found")

        elif img and st.button("🚀 Detect Image"):
            res = model(np.array(image), conf=conf)
            st.image(res[0].plot(), channels="BGR", use_container_width=True)
            for b in res[0].boxes:
                st.markdown(f'<span class="badge">{model.names[int(b.cls[0])]} ({float(b.conf[0]):.0%})</span>', unsafe_allow_html=True)

        elif vid and st.button("▶️ Detect Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid.read())
            cap = cv2.VideoCapture(tfile.name)
            frame_box = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                res = model(frame, conf=conf)
                frame_box.image(cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()

        else:
            st.info("Upload data to start")

        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# ABOUT
# ===============================
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## About")
    st.write("Premium UI for Vietnamese Food Detection using YOLO.")
    st.markdown('</div>', unsafe_allow_html=True)
