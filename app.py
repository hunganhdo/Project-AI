import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. CẤU HÌNH TRANG WEB (Phải để đầu tiên) ---
st.set_page_config(
    page_title="Ngon Luôn - AI Food Detector",
    page_icon="🍲",
    layout="wide"  # Quan trọng: Dùng chế độ màn hình rộng
)

# --- 2. CSS TÙY CHỈNH (Làm đẹp nhẹ) ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #FF4B4B; 
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    .sub-title {
        text-align: center;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (Cột bên trái) ---
with st.sidebar:
    st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", icon_image=None) # Ví dụ logo Python
    st.title("🎛️ Bảng Điều Khiển")
    
    st.markdown("---")
    
    # Upload file
    st.subheader("1. Chọn ảnh món ăn")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh", type=['jpg', 'jpeg', 'png'])
    
    st.markdown("---")
    
    # Cấu hình Model
    st.subheader("2. Cấu hình AI")
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.25, help="Chỉ số càng cao, AI càng khắt khe khi nhận diện.")
    
    st.info("💡 Mẹo: Nếu AI không nhận ra món ăn, hãy thử giảm độ tin cậy xuống thấp hơn.")

# --- 4. GIAO DIỆN CHÍNH (Bên phải) ---

st.markdown('<h1 class="main-title">🍲 NGON LUÔN AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Công cụ nhận diện món ăn Việt Nam sử dụng YOLOv10</p>', unsafe_allow_html=True)
st.write("") # Tạo khoảng trống

# Load Model
model_path = 'model/best.pt'
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"⚠️ Lỗi: Không tìm thấy file model tại {model_path}. Vui lòng kiểm tra lại!")
    st.stop()

# Xử lý khi có ảnh
if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    
    # Tạo 2 cột để hiển thị so sánh
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📸 Ảnh gốc")
        st.image(image, use_column_width=True)
        
        # Nút bấm nhận diện (Đặt ở cột 1 cho thuận tay)
        analyze_button = st.button('🚀 Phân tích ngay', type="primary", use_container_width=True)

    # Khi bấm nút
    if analyze_button:
        with col2:
            st.write("### 🧠 Kết quả AI")
            with st.spinner('Đang soi món ăn...'):
                # Chạy model với ngưỡng tin cậy từ slider
                results = model(image, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                # Hiển thị ảnh kết quả
                st.image(res_plotted, use_column_width=True)
                
                # Hiển thị thông tin chi tiết dưới dạng bảng
                st.success("Hoàn tất!")
                
                # Lấy danh sách vật thể để hiện ra text
                detected_items = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    item_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    detected_items.append(f"- **{item_name}** ({conf:.1%})")
                
                if detected_items:
                    with st.expander("📝 Xem chi tiết danh sách món"):
                        st.markdown("\n".join(detected_items))
                else:
                    st.warning("Không tìm thấy món nào. Thử giảm độ tin cậy xem sao?")

else:
    # Màn hình chờ khi chưa upload ảnh
    st.markdown(
        """
        <div style="text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px;">
            <h3>👈 Hãy upload ảnh ở thanh bên trái để bắt đầu</h3>
            <p>AI đang ngủ, chờ bạn đánh thức đấy...</p>
        </div>
        """, unsafe_allow_html=True
    )