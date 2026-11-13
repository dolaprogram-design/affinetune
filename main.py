import streamlit as st
import os, cv2, rawpy, numpy as np, pandas as pd, exiftool, tempfile, gc, subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# --- ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(page_title="RAW Image Advanced Analytics", page_icon="📸", layout="wide")

# ค้นหาตำแหน่งของ Exiftool
EXIFTOOL_PATH = None
possible_paths = [
    "./exiftool_bin/exiftool",
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

# Initialize Session State สำหรับเก็บประวัติการวิเคราะห์ (ไม่เก็บภาพ เพื่อประหยัด RAM)
if 'history_records' not in st.session_state:
        st.session_state['history_records'] = []
if 'current_image_data' not in st.session_state:
        st.session_state['current_image_data'] = None

def process_single_file(f_upload):
    file_bytes = f_upload.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".NEF") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # 1. อ่าน Metadata ด้วย Exiftool
        et_executable = EXIFTOOL_PATH if EXIFTOOL_PATH else "exiftool"
        with exiftool.ExifToolHelper(executable=et_executable) as et:
            meta = et.get_metadata(temp_path)[0]

        camera = meta.get("EXIF:Model", "Unknown Camera")
        lens = meta.get("MakerNotes:LensSpec", meta.get("EXIF:LensModel", "Unknown Lens"))
        iso_raw = meta.get("EXIF:ISO", "No EXIF")
        aperture = meta.get("EXIF:FNumber", "N/A")
        shutter = meta.get("EXIF:ExposureTime", "N/A")
        focal = meta.get("EXIF:FocalLength", "N/A")
        focus_dist = float(meta.get("MakerNotes:FocusDistance", 0)) if meta.get("MakerNotes:FocusDistance") else np.nan
        af_mode = meta.get("MakerNotes:FocusMode", "N/A")
        roll = float(meta.get("MakerNotes:RollAngle", np.nan))

        # 2. ประมวลผลภาพ RAW (ลดขนาดลงครึ่งหนึ่งเพื่อประหยัด RAM ทันที)
        with rawpy.imread(temp_path) as raw:
            rgb_image = raw.postprocess(half_size=True, use_camera_wb=True)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # 3. คำนวณจุดโฟกัส
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
        roi_rgb = rgb_image[sy:ey, sx:ex].copy()

        # 4. คำนวณคุณภาพ
        score = int(round(np.var(cv2.Laplacian(roi, cv2.CV_64F)))) if roi.size > 0 else 0
        contrast = int(round(np.std(roi))) if roi.size > 0 else 0

        if roi.size > 0:
            high_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny = cv2.Canny(roi, 0.5 * high_thresh, high_thresh)
            edge_density = int(round((np.sum(canny > 0) / canny.size) * 100))
        else:
            canny = np.zeros((10,10))
            edge_density = 0

        # 5. สร้างภาพ Preview ขนาดย่อสุดๆ สำหรับแสดงผลหน้าเว็บ
        preview_w, preview_h = int(w * 0.25), int(h * 0.25)
        rgb_resized = cv2.resize(rgb_image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

        # บันทึกข้อมูลสถิติ (สำหรับกราฟ)
        record_stats = {
            "File Name": f_upload.name, "Camera": camera, "Lens": lens, "ISO": iso_raw,
            "Aperture": aperture, "Shutter Speed": shutter, "Focal Length": focal,
            "Sharpness Score": score, "Edge Density (%)": edge_density, "Contrast": contrast,
            "Focus Mode": af_mode, "Roll Angle": roll
        }

        # ลบข้อมูลเก่าที่มีชื่อซ้ำออกก่อนเพิ่มใหม่
        st.session_state['history_records'] = [r for r in st.session_state['history_records'] if r['File Name'] != f_upload.name]
        st.session_state['history_records'].append(record_stats)

        # บันทึกภาพเฉพาะไฟล์ปัจจุบัน
        st.session_state['current_image_data'] = {
            "file_name": f_upload.name,
            "rgb_resized": rgb_resized,
            "roi_rgb": roi_rgb,
            "canny": canny,
            "box_scaled": (int(sx * 0.25), int(sy * 0.25), int(ex * 0.25), int(ey * 0.25)),
            "center_scaled": (int(cx * 0.25), int(cy * 0.25)),
            "stats": record_stats
        }

        # คืนหน่วยความจำทันที
        del rgb_image, gray, roi, rgb_resized, roi_rgb, canny
        gc.collect()
        return True

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผล {f_upload.name}: {e}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

# --- UI Components ---
st.title("📸 RAW Advanced Analytics (Optimized for Cloud)")
if not EXIFTOOL_PATH:
    st.warning("⚠️ ไม่พบ Exiftool ในระบบ การอ่านค่าบางอย่างอาจไม่สมบูรณ์")

with st.sidebar:
    st.header("📂 อัปโหลดไฟล์ RAW")
    st.info("💡 อัปโหลดและประมวลผลทีละ 1 ไฟล์ เพื่อป้องกันหน่วยความจำ (RAM) เต็ม ระบบจะเก็บสถิติของไฟล์ก่อนหน้าไว้เปรียบเทียบอัตโนมัติ")
    
    # บังคับให้อัปโหลดทีละไฟล์
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ .NEF (ครั้งละ 1 ไฟล์)", type=["nef"], accept_multiple_files=False)

    if uploaded_file:
        if st.button("⚡ เริ่มทำการวิเคราะห์ไฟล์นี้", type="primary"):
            with st.spinner("กำลังแยกข้อมูลและประมวลผลภาพ..."):
                success = process_single_file(uploaded_file)
                if success:
                    st.success("✅ วิเคราะห์เสร็จสิ้น!")

    st.divider()
    if st.button("🗑️ ล้างประวัติการวิเคราะห์ทั้งหมด"):
        st.session_state['history_records'] = []
        st.session_state['current_image_data'] = None
        st.rerun()

# --- Dashboard ---
if st.session_state['current_image_data']:
    curr_data = st.session_state['current_image_data']
    stats = curr_data['stats']

    st.subheader(f"🔍 ผลการวิเคราะห์ล่าสุด: {curr_data['file_name']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpness Score", stats["Sharpness Score"])
    col2.metric("Edge Density (%)", stats["Edge Density (%)"])
    col3.metric("Contrast", stats["Contrast"])

    tab1, tab2, tab3 = st.tabs(["🖼️ ภาพพรีวิว & จุดโฟกัส", "🔬 การวิเคราะห์โครงสร้าง (ROI)", "📊 สถิติและข้อมูลเปรียบเทียบ (Dashboard)"])

    with tab1:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(curr_data["rgb_resized"])
            sx, sy, ex, ey = curr_data["box_scaled"]
            ax.add_patch(patches.Rectangle((sx, sy), ex-sx, ey-sy, linewidth=2, edgecolor='cyan', facecolor='none'))
            ax.scatter(*curr_data["center_scaled"], color='red', marker='+', s=100)
            ax.set_title("Full Preview (Resized)")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            st.markdown("### ข้อมูล EXIF")
            st.write(f"**Camera:** {stats['Camera']}")
            st.write(f"**Lens:** {stats['Lens']}")
            st.write(f"**ISO:** {stats['ISO']}")
            st.write(f"**Aperture:** {stats['Aperture']}")
            st.write(f"**Shutter:** {stats['Shutter Speed']}")

    with tab2:
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            fig, ax = plt.subplots()
            ax.imshow(curr_data["roi_rgb"])
            ax.set_title("Crop Focus ROI")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        with col_y:
            fig, ax = plt.subplots()
            ax.imshow(curr_data["canny"], cmap='gray')
            ax.set_title("Canny Edges")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        with col_z:
            fig, ax = plt.subplots()
            for j, col in enumerate(('r', 'g', 'b')):
                hist = cv2.calcHist([curr_data["roi_rgb"]], [j], None, [256], [0, 256])
                ax.plot(hist, color=col, alpha=0.8)
            ax.set_title("Color Histogram")
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.markdown("### 📊 ตารางเปรียบเทียบประวัติการวิเคราะห์ทั้งหมด")
        if len(st.session_state['history_records']) > 0:
            df_history = pd.DataFrame(st.session_state['history_records'])
            st.dataframe(df_history)

            if len(df_history) > 1:
                st.markdown("### 📈 กราฟแนวโน้ม Sharpness")
                fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
                sns.barplot(data=df_history, x='File Name', y='Sharpness Score', ax=ax_bar, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_bar)
                plt.close(fig_bar)
        else:
            st.info("ยังไม่มีข้อมูลประวัติเพียงพอที่จะแสดงกราฟ")
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ที่เมนูด้านซ้ายและกดเริ่มวิเคราะห์ เพื่อดูผลลัพธ์")
