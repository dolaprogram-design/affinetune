import streamlit as st
import os
import cv2
import rawpy
import numpy as np
import pandas as pd
import exiftool
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import io

# --- ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(page_title="RAW Image Analytics Dashboard", page_icon="📸", layout="wide")

# --- ฟังก์ชันช่วยประมวลผล ---
@st.cache_data(show_spinner=False)
def process_uploaded_files(uploaded_files):
    records = []
    for f_upload in uploaded_files:
        file_bytes = f_upload.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".NEF") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        try:
            with exiftool.ExifToolHelper() as et:
                meta = et.get_metadata(temp_path)[0]
            
            camera = meta.get("EXIF:Model", "Unknown Camera")
            focus_dist = float(meta.get("MakerNotes:FocusDistance", 0)) if meta.get("MakerNotes:FocusDistance") else np.nan
            roll = float(meta.get("MakerNotes:RollAngle", np.nan))
            pitch = float(meta.get("MakerNotes:PitchAngle", np.nan))

            with rawpy.imread(temp_path) as raw:
                try:
                    thumb = raw.extract_thumb()
                    thumb_img = Image.open(io.BytesIO(thumb.data)) if thumb.format == rawpy.ThumbFormat.JPEG else Image.fromarray(thumb.data)
                except:
                    thumb_img = None
                rgb_image = raw.postprocess(half_size=True)
            
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            af_w = meta.get("MakerNotes:AFImageWidth", w*2)
            af_h = meta.get("MakerNotes:AFImageHeight", h*2)
            af_x = meta.get("MakerNotes:AFAreaXPosition", None)
            af_y = meta.get("MakerNotes:AFAreaYPosition", None)
            
            cx, cy = int(af_x * (w/af_w)), int(af_y * (h/af_h)) if af_x and af_y else (w//2, h//2)
            box_size = int(min(h, w) * 0.15)
            box_size += 1 if box_size % 2 != 0 else 0
            half_box = box_size // 2
            
            sx, ex = max(0, cx - half_box), min(w, cx + half_box)
            sy, ey = max(0, cy - half_box), min(h, cy + half_box)
            
            roi = gray[sy:ey, sx:ex]
            roi_rgb = rgb_image[sy:ey, sx:ex]
            
            # Analysis values
            score = np.var(cv2.Laplacian(roi, cv2.CV_64F))
            high_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny = cv2.Canny(roi, 0.5 * high_thresh, high_thresh)
            edge_density = (np.sum(canny > 0) / canny.size) * 100
            hist = cv2.calcHist([roi_rgb], [0], None, [256], [0, 256]) # R channel for demo
            contrast = np.std(roi)

            records.append({
                "File Name": f_upload.name,
                "Camera": camera,
                "Focus Distance (m)": focus_dist,
                "Roll Angle": roll,
                "Pitch Angle": pitch,
                "Sharpness Score": round(score, 2),
                "Edge Density (%)": round(edge_density, 2),
                "Contrast (Std Dev)": round(contrast, 2),
                "Image Data": {"rgb": rgb_image, "roi_rgb": roi_rgb, "canny": canny, "box": (sx, sy, ex, ey), "center": (cx, cy), "thumb": thumb_img}
            })
        except Exception as e:
            st.error(f"Error processing {f_upload.name}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return records

# --- UI ส่วนหัวและแถบด้านข้าง ---
st.title("📸 RAW Advanced Analytics Dashboard")

with st.sidebar:
    st.header("📂 อัปโหลดไฟล์ RAW")
    uploaded_files = st.file_uploader("อัปโหลดไฟล์ภาพ .NEF หลายไฟล์", type=["nef"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("กำลังประมวลผลการวิเคราะห์ทั้งหมด..."):
        data_records = process_uploaded_files(uploaded_files)
    
    if data_records:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "Image Data"} for r in data_records])
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 1. วิเคราะห์ภาพเดี่ยว", 
            "📊 2. เปรียบเทียบ Sharpness", 
            "📐 3. Edge Detection", 
            "🎨 4. Color Histogram", 
            "📈 5. Correlations & Angles"
        ])
        
        # Tab 1: Single Image
        with tab1:
            selected_file = st.selectbox("เลือกภาพที่ต้องการดูรายละเอียด:", df["File Name"].tolist())
            img_data = next((r["Image Data"] for r in data_records if r["File Name"] == selected_file), None)
            if img_data:
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(img_data["rgb"])
                    sx, sy, ex, ey = img_data["box"]
                    ax.add_patch(patches.Rectangle((sx, sy), ex-sx, ey-sy, linewidth=2, edgecolor='cyan', facecolor='none'))
                    ax.scatter(*img_data["center"], color='red', marker='+', s=100)
                    ax.axis('off')
                    st.pyplot(fig)
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(img_data["roi_rgb"])
                    ax2.set_title("Crop Focus ROI")
                    ax2.axis('off')
                    st.pyplot(fig2)
        
        # Tab 2: Sharpness Comparison
        with tab2:
            st.subheader("ตารางสรุปผลข้อมูล")
            st.dataframe(df)
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            sns.barplot(data=df, x='File Name', y='Sharpness Score', ax=ax_bar, palette='viridis')
            ax_bar.set_title("Sharpness Score Comparison")
            st.pyplot(fig_bar)
            
        # Tab 3: Edge Detection
        with tab3:
            st.subheader("การตรวจจับขอบภาพ (Canny Edge)")
            for r in data_records:
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(r["Image Data"]["roi_rgb"])
                    ax.set_title(f"{r['File Name']} - ROI")
                    ax.axis('off')
                    st.pyplot(fig)
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(r["Image Data"]["canny"], cmap='gray')
                    ax2.set_title(f"Edge Density: {r['Edge Density (%)']:.2f}%")
                    ax2.axis('off')
                    st.pyplot(fig2)
        
        # Tab 4: Histogram
        with tab4:
            st.subheader("Color Histogram ของพื้นที่จุดโฟกัส")
            for r in data_records:
                fig, ax = plt.subplots(figsize=(8, 3))
                roi_rgb = r["Image Data"]["roi_rgb"]
                colors = ('r', 'g', 'b')
                for j, col in enumerate(colors):
                    hist = cv2.calcHist([roi_rgb], [j], None, [256], [0, 256])
                    ax.plot(hist, color=col, alpha=0.8)
                ax.set_title(f"{r['File Name']} - Contrast: {r['Contrast (Std Dev)']:.2f}")
                st.pyplot(fig)

        # Tab 5: Correlations
        with tab5:
            st.subheader("ความสัมพันธ์เชิงลึก (Correlations)")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.regplot(data=df, x='Edge Density (%)', y='Sharpness Score', ax=ax, scatter_kws={'s':100})
                ax.set_title("Sharpness vs Edge Density")
                st.pyplot(fig)
            with col2:
                fig2, ax2 = plt.subplots()
                sns.regplot(data=df, x='Roll Angle', y='Sharpness Score', ax=ax2, scatter_kws={'s':100, 'color':'red'})
                ax2.set_title("Sharpness vs Roll Angle")
                st.pyplot(fig2)
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ RAW ทางแถบด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์ครับ")
