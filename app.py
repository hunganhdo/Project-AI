import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. CẤU HÌNH TRANG WEB ---
# Tôi đã bỏ layout="wide" để banner tự động căn vừa đẹp hơn
st.set_page_config(
    page_title="Ngon Luôn - AI Food Detector",
    page_icon="🍲"
)

# --- 2. CSS TÙY CHỈNH (Tạo Banner đẹp tràn viền) ---
st.markdown("""
    <style>
    /* Container chính của banner - Tràn viền 100% */
    .banner-container {
        position: relative;
        width: 100%;
        overflow: hidden;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Ảnh nền banner - Phóng to để bao phủ toàn bộ */
    .banner-img {
        width: 100%;
        height: 400px; /* Tăng chiều cao lên 400px cho hoành tráng */
        object-fit: cover; /* Quan trọng: Cắt ảnh để vừa khít khung */
        display: block;
    }
    
    /* Lớp phủ đen mờ */
    .banner-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        /* Màu đen mờ dần từ trên xuống dưới */
        background: linear-gradient(to bottom, rgba(0,0,0,0.3), rgba(0,0,0,0.7));
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
        padding: 20px;
    }
    
    .banner-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
    }
    
    .banner-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        font-style: italic;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (THANH BÊN TRÁI) ---
with st.sidebar:
    st.title("🏠 Home") 
    st.markdown("---")
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh", type=['jpg', 'jpeg', 'png'])
    st.markdown("---")
    st.subheader("2. Settings")
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.25)
    st.caption("Điều chỉnh độ nhạy của AI.")

# --- 4. GIAO DIỆN CHÍNH (BANNER TRÀN VIỀN) ---

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