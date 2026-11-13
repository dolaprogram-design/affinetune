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
import subprocess

# --- ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(page_title="RAW Image Analytics Dashboard", page_icon="📸", layout="wide")

# ค้นหาและระบุตำแหน่งของ Exiftool เพื่อแก้ไขปัญหา "exiftool not found"
EXIFTOOL_PATH = None
possible_paths = [
    "/usr/bin/exiftool",
    "/usr/local/bin/exiftool",
    "/opt/render/project/src/exiftool",
    "exiftool"
]

for path in possible_paths:
    try:
        # ทดสอบเรียกเวอร์ชัน
        subprocess.run([path, "-ver"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        EXIFTOOL_PATH = path
        break
    except Exception:
        continue

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
            # ใช้งาน ExifToolHelper โดยกำหนดค่า executable ที่หาได้ในเครื่องแม่ข่าย
            et_executable = EXIFTOOL_PATH if EXIFTOOL_PATH else "exiftool"
            with exiftool.ExifToolHelper(executable=et_executable) as et:
                meta = et.get_metadata(temp_path)[0]

            # สกัดข้อมูลพารามิเตอร์กล้องและเลนส์
            camera = meta.get("EXIF:Model", "Unknown Camera")
            lens = meta.get("MakerNotes:LensSpec", meta.get("EXIF:LensModel", "Unknown Lens"))
            iso = meta.get("EXIF:ISO", "N/A")
            aperture = meta.get("EXIF:FNumber", "N/A")
            shutter = meta.get("EXIF:ExposureTime", "N/A")
            focal = meta.get("EXIF:FocalLength", "N/A")

            # ข้อมูลโฟกัสและมุมกล้อง
            focus_dist = float(meta.get("MakerNotes:FocusDistance", 0)) if meta.get("MakerNotes:FocusDistance") else np.nan
            af_mode = meta.get("MakerNotes:FocusMode", "N/A")
            af_area = meta.get("MakerNotes:AFAreaMode", "N/A")
            af_tune = meta.get("MakerNotes:AFFineTuneAdj", "N/A")
            roll = float(meta.get("MakerNotes:RollAngle", np.nan))
            pitch = float(meta.get("MakerNotes:PitchAngle", np.nan))

            with rawpy.imread(temp_path) as raw:
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

            # การวิเคราะห์ภาพ
            score = np.var(cv2.Laplacian(roi, cv2.CV_64F))
            high_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny = cv2.Canny(roi, 0.5 * high_thresh, high_thresh)
            edge_density = (np.sum(canny > 0) / canny.size) * 100
            contrast = np.std(roi)

            records.append({
                "File Name": f_upload.name,
                "Camera": camera,
                "Lens": lens,
                "ISO": iso,
                "Aperture (F-Stop)": aperture,
                "Shutter Speed": shutter,
                "Focal Length": focal,
                "Focus Mode": af_mode,
                "AF Area Mode": af_area,
                "AF Fine Tune Adj": af_tune,
                "Focus Distance (m)": focus_dist,
                "Roll Angle": roll,
                "Pitch Angle": pitch,
                "Sharpness Score": round(score, 2),
                "Edge Density (%)": round(edge_density, 2),
                "Contrast (Std Dev)": round(contrast, 2),
                "Image Data": {"rgb": rgb_image, "roi_rgb": roi_rgb, "canny": canny, "box": (sx, sy, ex, ey), "center": (cx, cy)}
            })
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดกับไฟล์ {f_upload.name}: {e}")
            st.info("คำแนะนำ: โปรดตรวจสอบว่าระบบติดตั้ง exiftool สำเร็จ (ผ่านระบบ apt.txt หรือ packages.txt ตามประเภทผู้ให้บริการคลาวด์)")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return records

# --- UI ส่วนหัวและแถบด้านข้าง ---
st.title("📸 RAW Advanced Analytics Dashboard")

if not EXIFTOOL_PATH:
    st.error("❌ ไม่พบโปรแกรม 'exiftool' ในระบบเซิร์ฟเวอร์ กรุณาตรวจสอบให้แน่ใจว่าได้ระบุการติดตั้งใน packages.txt (สำหรับ Streamlit Cloud) หรือ apt.txt (สำหรับ Render)")

with st.sidebar:
    st.header("📂 อัปโหลดไฟล์ RAW")
    st.markdown("อัปโหลดไฟล์ภาพ .NEF (จำกัดสูงสุด 20 ไฟล์)")
    uploaded_files = st.file_uploader("", type=["nef"], accept_multiple_files=True)
    if len(uploaded_files) > 20:
        st.warning(f"⚠️ จำกัดระบบประมวลผลไว้ที่ 20 ไฟล์แรกเท่านั้น")
        uploaded_files = uploaded_files[:20]

if uploaded_files:
    with st.spinner(f"กำลังวิเคราะห์พารามิเตอร์เชิงสถิติ {len(uploaded_files)} ไฟล์..."):
        data_records = process_uploaded_files(uploaded_files)

    if data_records:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "Image Data"} for r in data_records])

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗂️ 1. ข้อมูลอภิมานเชิงลึก",
            "🔍 2. ภาพจุดโฟกัส",
            "📊 3. เปรียบเทียบ Sharpness",
            "📐 4. Edge & Histogram",
            "📈 5. ความสัมพันธ์เชิงสถิติ"
        ])

        with tab1:
            st.subheader("ตารางสรุปพารามิเตอร์กล้องเชิงลึก")
            st.dataframe(df.style.highlight_max(subset=['Sharpness Score', 'Edge Density (%)'], color='lightgreen'))

        with tab2:
            selected_file = st.selectbox("เลือกภาพที่ต้องการดูรายละเอียดจุดโฟกัส:", df["File Name"].tolist())
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
                    ax2.axis('off')
                    st.pyplot(fig2)

        with tab3:
            st.subheader("เปรียบเทียบคะแนนความคมชัด (Sharpness Score)")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            sns.barplot(data=df, x='File Name', y='Sharpness Score', ax=ax_bar, palette='viridis')
            for container in ax_bar.containers:
                ax_bar.bar_label(container, fmt='%.2f', fontweight='bold', padding=5)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_bar)

        with tab4:
            st.subheader("วิเคราะห์คุณลักษณะขอบเขตภาพเฉพาะส่วน")
            for r in data_records:
                st.markdown(f"**🎯 {r['File Name']}** | Sharpness: `{r['Sharpness Score']}` | Edge Density: `{r['Edge Density (%)']}%`")
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(r["Image Data"]["roi_rgb"])
                    ax.axis('off')
                    st.pyplot(fig)
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(r["Image Data"]["canny"], cmap='gray')
                    ax2.axis('off')
                    st.pyplot(fig2)
                with col3:
                    fig3, ax3 = plt.subplots()
                    for j, col in enumerate(('r', 'g', 'b')):
                        hist = cv2.calcHist([r["Image Data"]["roi_rgb"]], [j], None, [256], [0, 256])
                        ax3.plot(hist, color=col, alpha=0.8)
                    st.pyplot(fig3)
                st.divider()

        with tab5:
            st.subheader("ความสัมพันธ์เชิงสถิติระหว่างพารามิเตอร์")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.regplot(data=df, x='Edge Density (%)', y='Sharpness Score', ax=ax)
                st.pyplot(fig)
            with col2:
                fig2, ax2 = plt.subplots()
                sns.regplot(data=df, x='Roll Angle', y='Sharpness Score', ax=ax2)
                st.pyplot(fig2)
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ RAW (.NEF) เพื่อประมวลผลข้อมูลในแดชบอร์ด")
