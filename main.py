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
import subprocess
import gc

# --- ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(page_title="RAW Image Advanced Analytics", page_icon="📸", layout="wide")

# ค้นหาตำแหน่งของ Exiftool
EXIFTOOL_PATH = None
possible_paths = [
    "./exiftool_bin/exiftool",                # สำหรับ Render (โหลดมาด้วย bash script)
    "/opt/render/project/src/exiftool_bin/exiftool", 
    "exiftool",
    "/usr/bin/exiftool",
    "/usr/local/bin/exiftool"
]

for path in possible_paths:
    try:
        result = subprocess.run([path, "-ver"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            EXIFTOOL_PATH = path
            break
    except Exception:
        continue

@st.cache_data(show_spinner=False)
def process_uploaded_files(uploaded_files):
    records = []
    for f_upload in uploaded_files:
        file_bytes = f_upload.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".NEF") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        try:
            et_executable = EXIFTOOL_PATH if EXIFTOOL_PATH else "exiftool"
            with exiftool.ExifToolHelper(executable=et_executable) as et:
                meta = et.get_metadata(temp_path)[0]

            camera = meta.get("EXIF:Model", "Unknown Camera")
            lens = meta.get("MakerNotes:LensSpec", meta.get("EXIF:LensModel", "Unknown Lens"))
            iso = meta.get("EXIF:ISO", "N/A")
            aperture = meta.get("EXIF:FNumber", "N/A")
            shutter = meta.get("EXIF:ExposureTime", "N/A")
            focal = meta.get("EXIF:FocalLength", "N/A")

            focus_dist = float(meta.get("MakerNotes:FocusDistance", 0)) if meta.get("MakerNotes:FocusDistance") else np.nan
            af_mode = meta.get("MakerNotes:FocusMode", "N/A")
            af_area = meta.get("MakerNotes:AFAreaMode", "N/A")
            af_tune = meta.get("MakerNotes:AFFineTuneAdj", "N/A")
            roll = float(meta.get("MakerNotes:RollAngle", np.nan))
            pitch = float(meta.get("MakerNotes:PitchAngle", np.nan))

            # --- RAM Optimization: ปรับแต่งวิธีเปิดและประมวลผลไฟล์ RAW ---
            with rawpy.imread(temp_path) as raw:
                # ใช้ half_size=True เสมอ และยกเว้นการทำ Demosaic คุณภาพสูงเพื่อประหยัด CPU/RAM
                rgb_image = raw.postprocess(half_size=True, use_camera_wb=True, fast_close_to_rgb=True)

            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            af_w = meta.get("MakerNotes:AFImageWidth", None)
            af_h = meta.get("MakerNotes:AFImageHeight", None)
            af_x = meta.get("MakerNotes:AFAreaXPosition", None)
            af_y = meta.get("MakerNotes:AFAreaYPosition", None)

            if af_x is not None and af_y is not None and af_w is not None and af_h is not None and float(af_w) > 0 and float(af_h) > 0:
                cx = int(af_x * (w / float(af_w)))
                cy = int(af_y * (h / float(af_h)))
            elif af_x is not None and af_y is not None:
                cx, cy = int(af_x) // 2, int(af_y) // 2
            else:
                cx, cy = w // 2, h // 2

            box_size = int(min(h, w) * 0.15)
            box_size += 1 if box_size % 2 != 0 else 0
            half_box = box_size // 2

            sx, ex = max(0, cx - half_box), min(w, cx + half_box)
            sy, ey = max(0, cy - half_box), min(h, cy + half_box)

            roi = gray[sy:ey, sx:ex]
            roi_rgb = rgb_image[sy:ey, sx:ex].copy() # ทำสำเนาเฉพาะส่วน

            score = np.var(cv2.Laplacian(roi, cv2.CV_64F)) if roi.size > 0 else 0.0

            if roi.size > 0:
                high_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                canny = cv2.Canny(roi, 0.5 * high_thresh, high_thresh)
                edge_density = (np.sum(canny > 0) / canny.size) * 100
            else:
                canny = np.zeros((10,10))
                edge_density = 0.0

            contrast = np.std(roi) if roi.size > 0 else 0.0

            # สเกลย่อขนาดรูปภาพใหญ่ (Full RGB) ลงไปอีกเพื่อไม่ให้เบราว์เซอร์และฝั่ง Server ค้าง
            preview_w = int(w * 0.25)
            preview_h = int(h * 0.25)
            rgb_resized = cv2.resize(rgb_image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

            # เคลียร์หน่วยความจำที่ไม่ใช้งานทันที
            del rgb_image, gray, roi
            gc.collect()

            records.append({
                "File Name": f_upload.name,
                "Camera": camera, "Lens": lens, "ISO": iso,
                "Aperture (F-Stop)": aperture, "Shutter Speed": shutter, "Focal Length": focal,
                "Focus Mode": af_mode, "AF Area Mode": af_area, "AF Fine Tune Adj": af_tune,
                "Focus Distance (m)": focus_dist, "Roll Angle": roll, "Pitch Angle": pitch,
                "Sharpness Score": round(score, 2), "Edge Density (%)": round(edge_density, 2),
                "Contrast (Std Dev)": round(contrast, 2),
                "Image Data": {
                    "rgb_resized": rgb_resized, 
                    "roi_rgb": roi_rgb, 
                    "canny": canny, 
                    "box_scaled": (int(sx * 0.25), int(sy * 0.25), int(ex * 0.25), int(ey * 0.25)), 
                    "center_scaled": (int(cx * 0.25), int(cy * 0.25))
                }
            })
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดกับไฟล์ {f_upload.name}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            gc.collect()
    return records

st.title("📸 RAW Advanced Analytics Dashboard")

if not EXIFTOOL_PATH:
    st.error("❌ ไม่พบโปรแกรม 'exiftool' ในระบบ")

with st.sidebar:
    st.header("📂 อัปโหลดไฟล์ RAW")
    uploaded_files = st.file_uploader("อัปโหลดไฟล์ภาพ .NEF (จำกัดสูงสุด 5 ไฟล์บนคลาวด์)", type=["nef"], accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("⚠️ จำกัดระบบประมวลผลไว้ที่ 5 ไฟล์แรกเท่านั้นเพื่อป้องกัน RAM เต็มบน Render")
        uploaded_files = uploaded_files[:5]

if uploaded_files:
    with st.spinner("กำลังประมวลผลและวิเคราะห์โครงสร้างภาพ... "):
        data_records = process_uploaded_files(uploaded_files)

    if data_records:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "Image Data"} for r in data_records])
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗂️ 1. ข้อมูลอภิมานเชิงลึก",
            "🔍 2. ภาพจุดโฟกัส & ROI",
            "📊 3. เปรียบเทียบ Sharpness",
            "📐 4. Edge & Histogram",
            "📈 5. ความสัมพันธ์เชิงสถิติ"
        ])

        with tab1:
            st.dataframe(df.style.highlight_max(subset=['Sharpness Score'], color='lightgreen'))

        with tab2:
            selected_file = st.selectbox("เลือกภาพที่ต้องการวิเคราะห์:", df["File Name"].tolist())
            img_data = next((r["Image Data"] for r in data_records if r["File Name"] == selected_file), None)
            if img_data:
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(img_data["rgb_resized"])
                    sx, sy, ex, ey = img_data["box_scaled"]
                    ax.add_patch(patches.Rectangle((sx, sy), ex-sx, ey-sy, linewidth=2, edgecolor='cyan', facecolor='none'))
                    ax.scatter(*img_data["center_scaled"], color='red', marker='+', s=100)
                    ax.set_title("Full View Preview (Resized for RAM)")
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(img_data["roi_rgb"])
                    ax2.set_title("Crop Focus ROI (Original Resolution)")
                    ax2.axis('off')
                    st.pyplot(fig2)
                    plt.close(fig2)

        with tab3:
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            sns.barplot(data=df, x='File Name', y='Sharpness Score', ax=ax_bar, palette='viridis')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with tab4:
            for r in data_records:
                st.markdown(f"**🎯 {r['File Name']}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(r["Image Data"]["roi_rgb"])
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(r["Image Data"]["canny"], cmap='gray')
                    ax2.axis('off')
                    st.pyplot(fig2)
                    plt.close(fig2)
                with col3:
                    fig3, ax3 = plt.subplots()
                    for j, col in enumerate(('r', 'g', 'b')):
                        hist = cv2.calcHist([r["Image Data"]["roi_rgb"]], [j], None, [256], [0, 256])
                        ax3.plot(hist, color=col, alpha=0.8)
                    st.pyplot(fig3)
                    plt.close(fig3)
                st.divider()

        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.regplot(data=df, x='Edge Density (%)', y='Sharpness Score', ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                fig2, ax2 = plt.subplots()
                if not df['Roll Angle'].isnull().all():
                    sns.regplot(data=df, x='Roll Angle', y='Sharpness Score', ax=ax2)
                else:
                    ax2.text(0.5, 0.5, "No Roll Angle Data Available", ha='center', va='center')
                st.pyplot(fig2)
                plt.close(fig2)
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ RAW (.NEF) เพื่อประมวลผลสถิติภาพครับ")
