import streamlit as st
import os
import json
import cv2
import rawpy
import numpy as np
import pandas as pd
import exiftool
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# --- ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(
    page_title="RAW Image Analytics Dashboard",
    page_icon="📸",
    layout="wide"
)

# --- 1. จัดการระบบฐานข้อมูลกล้องและเลนส์จากไฟล์ JSON (Persistent Storage) ---
DB_FILE = 'camera_db.json'

def load_camera_db():
    default_db = {
        "Brands": ["Nikon", "Canon", "Sigma", "Tamron", "Samyang", "Tokina"],
        "Models": ["NIKON D810", "NIKON D850", "NIKON D750", "Canon EOS 5D Mark IV", "Canon EOS 6D Mark II"]
    }
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default_db
    return default_db

def save_camera_db(db_data):
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

# โหลดข้อมูลเข้าสู่ Session State
if 'camera_db' not in st.session_state:
    st.session_state['camera_db'] = load_camera_db()

# --- 2. ฟังก์ชันช่วยประมวลผลสกัดอภิข้อมูลเชิงลึก ---
def extract_metadata_from_bytes(file_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        with exiftool.ExifToolHelper() as et:
            meta = et.get_metadata(temp_path)[0]

        # สกัดข้อมูลเชิงลึก
        camera = meta.get("EXIF:Model", "Unknown Camera")
        lens = meta.get("MakerNotes:LensSpec", meta.get("EXIF:LensModel", "Unknown Lens"))
        iso = meta.get("EXIF:ISO", "N/A")
        aperture = meta.get("EXIF:FNumber", "N/A")
        shutter = meta.get("EXIF:ExposureTime", "N/A")
        focal = meta.get("EXIF:FocalLength", "N/A")
        focus_mode = meta.get("MakerNotes:FocusMode", "N/A")
        af_area_mode = meta.get("MakerNotes:AFAreaMode", "N/A")
        focus_distance = meta.get("MakerNotes:FocusDistance", "N/A")

        return {
            "File Name": filename,
            "Camera": camera,
            "Lens": lens,
            "ISO": iso,
            "Aperture": f"f/{aperture}" if aperture != "N/A" else "N/A",
            "Shutter Speed": f"{shutter}s" if shutter != "N/A" else "N/A",
            "Focal Length": f"{focal}mm" if focal != "N/A" else "N/A",
            "Focus Mode": focus_mode,
            "AF Area Mode": af_area_mode,
            "Focus Distance (m)": focus_distance,
            "meta_full": meta
        }, temp_path
    except Exception as e:
        return {"File Name": filename, "Error": str(e)}, temp_path

def analyze_raw_file(temp_path, meta_info):
    try:
        with rawpy.imread(temp_path) as raw:
            # ดึงภาพ thumbnail (ถ้ามี) หรือ postprocess แบบ half_size
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    thumb_img = Image.open(io.BytesIO(thumb.data))
                else:
                    thumb_img = Image.fromarray(thumb.data)
            except rawpy.LibRawNoThumbnailError:
                thumb_img = None

            rgb_image = raw.postprocess(half_size=True)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        meta = meta_info.get("meta_full", {})
        af_w = meta.get("MakerNotes:AFImageWidth", w*2)
        af_h = meta.get("MakerNotes:AFImageHeight", h*2)
        af_x = meta.get("MakerNotes:AFAreaXPosition", None)
        af_y = meta.get("MakerNotes:AFAreaYPosition", None)

        if af_x is not None and af_y is not None:
            scale_x = w / float(af_w)
            scale_y = h / float(af_h)
            cx, cy = int(af_x * scale_x), int(af_y * scale_y)
        else:
            cx, cy = w // 2, h // 2

        min_dim = min(h, w)
        box_size = int(min_dim * 0.15)
        if box_size % 2 != 0:
            box_size += 1
        half_box = box_size // 2

        sx, ex = max(0, cx - half_box), min(w, cx + half_box)
        sy, ey = max(0, cy - half_box), min(h, cy + half_box)

        roi = gray[sy:ey, sx:ex]
        roi_rgb = rgb_image[sy:ey, sx:ex]
        score = np.var(cv2.Laplacian(roi, cv2.CV_64F))

        return rgb_image, roi_rgb, score, cx, cy, sx, sy, ex, ey, thumb_img
    except Exception as e:
        return None, None, 0, 0, 0, 0, 0, 0, 0, None

# --- 3. ส่วนหัวของเว็บไซต์ (Header UI) ---
st.title("📸 RAW Focus & Sharpness Analytics Dashboard")
st.write("ระบบสกัดและวิเคราะห์คุณภาพความคมชัดของภาพจากระบบจุดโฟกัสกล้อง DSLR (Canon/Nikon) บนเว็บแอปพลิเคชัน")

# --- 4. เมนูแถบด้านข้างสำหรับควบคุมฐานข้อมูล (Sidebar Database Manager) ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าฐานข้อมูลอุปกรณ์")
    st.write("จัดการฐานข้อมูลแบรนด์กล้องและเลนส์ที่รองรับในระบบถาวร (JSON Database)")

    new_brand = st.text_input("➕ เพิ่มยี่ห้อกล้อง/เลนส์ (เช่น Tamron, Sigma):")
    if st.button("บันทึกยี่ห้อ") and new_brand:
        if new_brand not in st.session_state['camera_db']["Brands"]:
            st.session_state['camera_db']["Brands"].append(new_brand)
            if save_camera_db(st.session_state['camera_db']):
                st.success(f"เพิ่มยี่ห้อ '{new_brand}' และบันทึกลง JSON แล้ว!")
            else:
                st.error("ไม่สามารถบันทึกฐานข้อมูลลงไฟล์ได้")

    new_model = st.text_input("➕ เพิ่มรุ่นกล้องถ่ายภาพ DSLR:")
    if st.button("บันทึกรุ่นกล้อง") and new_model:
        if new_model not in st.session_state['camera_db']["Models"]:
            st.session_state['camera_db']["Models"].append(new_model)
            if save_camera_db(st.session_state['camera_db']):
                st.success(f"เพิ่มรุ่นกล้อง '{new_model}' ลงระบบถาวรแล้ว!")
            else:
                st.error("ไม่สามารถบันทึกฐานข้อมูลลงไฟล์ได้")

    st.subheader("📂 ข้อมูลปัจจุบันในระบบ")
    st.write("**แบรนด์ที่รองรับ:**", ", ".join(st.session_state['camera_db']["Brands"]))
    st.write("**รุ่นกล้องที่ลงทะเบียน:**", ", ".join(st.session_state['camera_db']["Models"]))

# --- 5. แท็บสำหรับการทำงานหลัก 2 เมนู (Dashboard Navigation Tabs) ---
tab1, tab2 = st.tabs(["🔍 1. ทดสอบตรวจค่าภาพเดี่ยว", "📊 2. ประมวลผลเปรียบเทียบกลุ่มภาพ"])

# --- เมนู 1: ทดสอบตรวจสอบภาพเดี่ยว ---
with tab1:
    st.header("🔍 วิเคราะห์ข้อมูลและจำลองกรอบพิกัด AF อ้างอิงรายไฟล์")
    uploaded_file = st.file_uploader("กรุณาอัปโหลดภาพ RAW (.NEF)", type=["nef"], key="single_file")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        with st.spinner("กำลังวิเคราะห์ไฟล์ภาพ RAW โปรดรอสักครู่..."):
            meta_info, temp_path = extract_metadata_from_bytes(file_bytes, uploaded_file.name)

            if "Error" in meta_info:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {meta_info['Error']}")
            else:
                rgb, roi, score, cx, cy, sx, sy, ex, ey, thumb_img = analyze_raw_file(temp_path, meta_info)

                # แสดงผลแบ่งเป็น 3 คอลัมน์สำหรับรูปภาพ
                col_t, col1, col2 = st.columns([1, 2, 2])

                with col_t:
                    st.subheader("🖼️ ภาพย่อ (Thumbnail)")
                    if thumb_img:
                        st.image(thumb_img, use_container_width=True)
                    else:
                        st.info("ไม่มี Thumbnail ฝังอยู่ในไฟล์นี้")

                with col1:
                    st.subheader("🖼️ รูปถ่ายต้นฉบับย่อส่วนพร้อมกรอบโฟกัส")
                    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
                    ax1.imshow(rgb)
                    rect = patches.Rectangle((sx, sy), ex-sx, ey-sy, linewidth=2, edgecolor='cyan', facecolor='none')
                    ax1.add_patch(rect)
                    ax1.scatter(cx, cy, color='red', marker='+', s=100, label='AF Center')
                    ax1.legend()
                    ax1.axis('off')
                    st.pyplot(fig1)

                with col2:
                    st.subheader("🎯 พื้นที่ตรวจสอบความคมชัดเฉพาะจุด (ROI)")
                    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
                    ax2.imshow(roi)
                    ax2.set_title(f"Sharpness (Laplacian Var): {score:.2f}")
                    ax2.axis('off')
                    st.pyplot(fig2)

                # แสดงรายละเอียดเชิงลึก
                st.subheader("📋 สรุปรายละเอียดอภิข้อมูลอุปกรณ์ (Detailed EXIF Metadata)")

                # แบ่งข้อมูลเป็น 2 ตารางเพื่อความสวยงาม
                col_detail1, col_detail2 = st.columns(2)

                with col_detail1:
                    st.markdown("**ข้อมูลอุปกรณ์พื้นฐาน**")
                    meta_basic = {
                        "ชื่อไฟล์": meta_info.get("File Name", ""),
                        "กล้องถ่ายภาพ": meta_info.get("Camera", ""),
                        "เลนส์ที่ใช้": meta_info.get("Lens", ""),
                    }
                    st.table(pd.DataFrame(list(meta_basic.items()), columns=["คุณสมบัติ", "ค่าที่ได้"]))

                with col_detail2:
                    st.markdown("**การตั้งค่าการถ่ายภาพ (Settings & AF)**")
                    meta_settings = {
                        "ISO": meta_info.get("ISO", ""),
                        "รูรับแสง": meta_info.get("Aperture", ""),
                        "สปีดชัตเตอร์": meta_info.get("Shutter Speed", ""),
                        "ความยาวโฟกัส": meta_info.get("Focal Length", ""),
                        "โหมดโฟกัส": meta_info.get("Focus Mode", ""),
                        "โหมดพื้นที่ AF": meta_info.get("AF Area Mode", ""),
                        "ระยะโฟกัส (เมตร)": meta_info.get("Focus Distance (m)", "")
                    }
                    st.table(pd.DataFrame(list(meta_settings.items()), columns=["การตั้งค่า", "ค่าที่ได้"]))

            # ทำลายไฟล์ชั่วคราว
            if os.path.exists(temp_path):
                os.remove(temp_path)

# --- เมนู 2: เปรียบเทียบสถิติกลุ่มภาพ ---
with tab2:
    st.header("📊 วิเคราะห์และสรุปผลข้อมูลเชิงเปรียบเทียบในรูปภาพหลายไฟล์")
    uploaded_files = st.file_uploader("กรุณาอัปโหลดกลุ่มภาพ RAW (.NEF) เพื่อทำการเปรียบเทียบ", type=["nef"], accept_multiple_files=True, key="multi_file")

    if uploaded_files:
        records = []
        for f_upload in uploaded_files:
            file_bytes = f_upload.read()
            meta_info, temp_path = extract_metadata_from_bytes(file_bytes, f_upload.name)

            if "Error" not in meta_info:
                _, _, score, _, _, _, _, _, _, _ = analyze_raw_file(temp_path, meta_info)
                meta_info["Sharpness Score"] = round(score, 2)
                records.append(meta_info)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        if records:
            df = pd.DataFrame(records)

            st.subheader("📈 ตารางสรุปการประเมินวิเคราะห์ผลเชิงสถิติ (พร้อมข้อมูลการตั้งค่ากล้องเชิงลึก)")
            display_cols = ["File Name", "Camera", "Lens", "ISO", "Aperture", "Shutter Speed", "Focal Length", "Focus Mode", "Focus Distance (m)", "Sharpness Score"]
            st.dataframe(df[display_cols], use_container_width=True)

            st.subheader("📊 แผนภูมิแท่งเปรียบเทียบคะแนนความคมชัด (Sharpness Variance)")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4.5))
            ax_bar.bar(df["File Name"], df["Sharpness Score"], color='#3498DB', alpha=0.85, width=0.4)
            ax_bar.set_title("Sharpness Quality Assessment (Higher is Sharper)", fontsize=12, fontweight='bold')
            ax_bar.set_ylabel("Sharpness Score (Laplacian Var)")
            ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig_bar)
        else:
            st.warning("⚠️ ไม่สามารถวิเคราะห์สกัดคุณสมบัติไฟล์ที่อัปโหลดได้")
