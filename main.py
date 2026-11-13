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

st.set_page_config(page_title="RAW Image Analytics Dashboard", page_icon="📸", layout="wide")

# ค้นหาตำแหน่งของ Exiftool รวมถึงโฟลเดอร์ที่เราโหลดมาเองผ่าน render-build.sh
EXIFTOOL_PATH = None
possible_paths = [
    "./exiftool_bin/exiftool",                # สำหรับ Render (โหลดมาด้วย bash script)
    "/opt/render/project/src/exiftool_bin/exiftool", # Path เต็มบน Render
    "exiftool",                                 # ระบบปกติ (System PATH)
    "/usr/bin/exiftool",
    "/usr/local/bin/exiftool"
]

for path in possible_paths:
    try:
        # ทดสอบการเรียกใช้งานผ่าน subprocess
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

            score = np.var(cv2.Laplacian(roi, cv2.CV_64F))
            high_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny = cv2.Canny(roi, 0.5 * high_thresh, high_thresh)
            edge_density = (np.sum(canny > 0) / canny.size) * 100

            records.append({
                "File Name": f_upload.name,
                "Camera": camera, "Lens": lens, "ISO": iso,
                "Aperture": aperture, "Shutter Speed": shutter, "Focal Length": focal,
                "Sharpness Score": round(score, 2), "Edge Density (%)": round(edge_density, 2),
                "Image Data": {"rgb": rgb_image, "roi_rgb": roi_rgb, "canny": canny, "box": (sx, sy, ex, ey), "center": (cx, cy)}
            })
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดกับไฟล์ {f_upload.name}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return records

st.title("📸 RAW Advanced Analytics Dashboard")

if not EXIFTOOL_PATH:
    st.error("❌ ไม่พบโปรแกรม 'exiftool' ในระบบ กรุณาตั้งค่า Build Command บน Render ให้เป็น `bash render-build.sh`")

with st.sidebar:
    st.header("📂 อัปโหลดไฟล์ RAW")
    uploaded_files = st.file_uploader("อัปโหลดไฟล์ภาพ .NEF", type=["nef"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner(f"กำลังประมวลผล..."):
        data_records = process_uploaded_files(uploaded_files)

    if data_records:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "Image Data"} for r in data_records])
        tab1, tab2 = st.tabs(["🗂️ ข้อมูลพารามิเตอร์", "🔍 ภาพจุดโฟกัส"])
        
        with tab1:
            st.dataframe(df)
            
        with tab2:
            selected_file = st.selectbox("เลือกภาพ:", df["File Name"].tolist())
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
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ RAW ทางแถบด้านซ้าย")
