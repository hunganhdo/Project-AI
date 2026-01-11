import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Ngon Luôn - AI Food Detector",
    page_icon="🍲",
    layout="wide"
)

# --- 2. CSS TÙY CHỈNH (Tạo Banner đẹp) ---
st.markdown("""
    <style>
    /* Container chứa banner */
    .banner-container {
        position: relative;
        width: 100%;
        overflow: hidden;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Ảnh nền banner */
    .banner-img {
        width: 100%;
        height: 250px;
        object-fit: cover;
        display: block;
    }
    
    /* Lớp phủ đen mờ */
    .banner-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
    }
    
    .banner-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        font-family: 'Helvetica', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .banner-subtitle {
        font-size: 1.2rem;
        margin-top: 10px;
        font-weight: 300;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (THANH BÊN TRÁI - ĐÃ SỬA THÀNH HOME) ---
with st.sidebar:
    # Logo nhỏ (nếu có)
    st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", icon_image=None)
    
    # --- THAY ĐỔI Ở ĐÂY: Đổi "Bảng điều khiển" thành "Home" ---
    st.title("🏠 Home") 
    
    st.markdown("---")
    
    # Upload file
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh", type=['jpg', 'jpeg', 'png'])
    
    st.markdown("---")
    
    # Cấu hình Model
    st.subheader("2. Settings")
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.25)
    st.caption("Điều chỉnh độ nhạy của AI.")

# --- 4. GIAO DIỆN CHÍNH (BANNER) ---

# Banner hiển thị ngay đầu trang
st.markdown("""
    <div class="banner-container">
        <img src="https://images.unsplash.com/photo-1504674900247-0877df9cc836?q=80&w=2070&auto=format&fit=crop" class="banner-img">
        <div class="banner-overlay">
            <h1 class="banner-title">Welcome to Group 😋</h1>
            <p class="banner-subtitle">An easy way to detect Vietnamese dishes!</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.write("") # Khoảng trống

# --- 5. LOGIC AI ---
model_path = 'model/best.pt'
try:
    model = YOLO(model_path)
except Exception:
    st.error(f"⚠️ Không tìm thấy file model tại {model_path}")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📸 Ảnh gốc")
        st.image(image, use_column_width=True)
        analyze_button = st.button('🚀 Phân tích ngay', type="primary", use_container_width=True)

    if analyze_button:
        with col2:
            st.write("### 🧠 Kết quả AI")
            with st.spinner('Đang soi món ăn...'):
                results = model(image, conf=conf_threshold)
                res_plotted = results[0].plot()
                st.image(res_plotted, use_column_width=True)
                
                detected_items = []
                for box in results[0].boxes:
                    item_name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    detected_items.append(f"- **{item_name}** ({conf:.1%})")
                
                if detected_items:
                    st.success("Đã nhận diện xong!")
                    with st.expander("📝 Xem danh sách"):
                        st.markdown("\n".join(detected_items))
                else:
                    st.warning("Không tìm thấy món nào.")
else:
    st.info("👈 Hãy upload ảnh bên tay trái để bắt đầu.")