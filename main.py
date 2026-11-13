import streamlit as st
import os, cv2, rawpy, numpy as np, pandas as pd, exiftool, tempfile, gc, subprocess, math
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

# Initialize Session State สำหรับเก็บประวัติ (ไม่หายแม้อัปโหลดไฟล์ใหม่)
if 'history_records' not in st.session_state:
    st.session_state['history_records'] = []

def process_single_file(f_upload):
    file_bytes = f_upload.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".NEF") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        et_executable = EXIFTOOL_PATH if EXIFTOOL_PATH else "exiftool"
        with exiftool.ExifToolHelper(executable=et_executable) as et:
            meta = et.get_metadata(temp_path)[0]

        clean_meta = {k: v for k, v in meta.items() if isinstance(v, (int, float, str)) and len(str(v)) < 250}

        camera = meta.get("EXIF:Model", "Unknown Camera")
        lens = meta.get("MakerNotes:LensSpec", meta.get("EXIF:LensModel", "Unknown Lens"))
        iso_raw = meta.get("EXIF:ISO", "No EXIF")
        aperture = meta.get("EXIF:FNumber", "N/A")
        shutter = meta.get("EXIF:ExposureTime", "N/A")
        focal = meta.get("EXIF:FocalLength", "N/A")
        
        af_tune = meta.get("MakerNotes:AFFineTuneAdj", "N/A")
        if af_tune == "N/A" and "MakerNotes:AFFineTune" in meta:
            af_tune = meta.get("MakerNotes:AFFineTune")
            
        focus_dist = float(meta.get("MakerNotes:FocusDistance", 0)) if meta.get("MakerNotes:FocusDistance") else np.nan
        af_mode = meta.get("MakerNotes:FocusMode", "N/A")
        roll = float(meta.get("MakerNotes:RollAngle", np.nan))

        with rawpy.imread(temp_path) as raw:
            rgb_image = raw.postprocess(half_size=True, use_camera_wb=True)

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

        scan_scale = 4 
        gray_small = cv2.resize(gray, (w // scan_scale, h // scan_scale))
        lap = cv2.Laplacian(gray_small, cv2.CV_32F)
        lap_var = cv2.boxFilter(lap**2, -1, (15, 15))
        _, _, _, max_loc = cv2.minMaxLoc(lap_var)
        peak_cx = max_loc[0] * scan_scale
        peak_cy = max_loc[1] * scan_scale
        
        focus_shift_px = int(math.hypot(cx - peak_cx, cy - peak_cy))

        box_size = int(min(h, w) * 0.15)
        box_size += 1 if box_size % 2 != 0 else 0
        half_box = box_size // 2
        sx, ex = max(0, cx - half_box), min(w, cx + half_box)
        sy, ey = max(0, cy - half_box), min(h, cy + half_box)

        roi = gray[sy:ey, sx:ex]
        roi_rgb = rgb_image[sy:ey, sx:ex].copy()

        score = int(round(np.var(cv2.Laplacian(roi, cv2.CV_64F)))) if roi.size > 0 else 0
        contrast = int(round(np.std(roi))) if roi.size > 0 else 0

        if roi.size > 0:
            high_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny = cv2.Canny(roi, 0.5 * high_thresh, high_thresh)
            edge_density = int(round((np.sum(canny > 0) / canny.size) * 100))
        else:
            canny = np.zeros((10,10))
            edge_density = 0

        preview_scale = 0.25
        preview_w, preview_h = int(w * preview_scale), int(h * preview_scale)
        rgb_resized = cv2.resize(rgb_image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

        record = {
            "File Name": f_upload.name,
            "Camera": camera, 
            "Lens": lens, 
            "ISO": iso_raw,
            "Aperture": aperture, 
            "Shutter Speed": shutter, 
            "Focal Length": focal,
            "AF Fine-Tune": af_tune,
            "Focus Mode": af_mode,
            "Recorded Focus (X,Y)": f"({cx}, {cy})",
            "Actual Sharpest (X,Y)": f"({peak_cx}, {peak_cy})",
            "Focus Shift (px)": focus_shift_px,
            "Sharpness Score": score, 
            "Edge Density (%)": edge_density, 
            "Contrast": contrast,
            "Full Metadata": clean_meta,
            "Images": {
                "preview_rgb": rgb_resized,
                "roi_rgb": roi_rgb,
                "canny": canny,
                "box_scaled": (int(sx * preview_scale), int(sy * preview_scale), int(ex * preview_scale), int(ey * preview_scale)),
                "center_scaled": (int(cx * preview_scale), int(cy * preview_scale)),
                "peak_scaled": (int(peak_cx * preview_scale), int(peak_cy * preview_scale))
            }
        }

        # ลบข้อมูลเก่าที่ชื่อซ้ำออกก่อนอัปเดตใหม่
        st.session_state['history_records'] = [r for r in st.session_state['history_records'] if r['File Name'] != f_upload.name]
        st.session_state['history_records'].append(record)

        del rgb_image, gray, roi, rgb_resized, roi_rgb, canny, lap, lap_var, gray_small
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
st.title("📸 RAW Analytics: Multi-File & AF Accuracy Analysis")
if not EXIFTOOL_PATH:
    st.warning("⚠️ ไม่พบ Exiftool ในระบบ การอ่านค่าบางอย่างอาจไม่สมบูรณ์")

with st.sidebar:
    st.header("📂 คิวอัปโหลดไฟล์ RAW")
    st.error("🚨 สำคัญ: ระบบคลาวด์มีข้อจำกัดด้านหน่วยความจำ (RAM) และเวลาการอัปโหลด (Timeout)\n\n**แนะนำให้อัปโหลดทีละ 2-3 ไฟล์** \nเมื่อประมวลผลเสร็จแล้ว คุณสามารถลากไฟล์ชุดต่อไปมาวางทับได้เลย ระบบจะนำข้อมูลไปต่อท้ายตารางวิเคราะห์ให้โดยอัตโนมัติ (ข้อมูลเก่าไม่หาย)")
    
    uploaded_files = st.file_uploader("เลือกไฟล์ภาพ .NEF (แนะนำทีละ 2-3 ไฟล์)", type=["nef"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"เตรียมประมวลผล: {len(uploaded_files)} ไฟล์")
        if st.button("⚡ เริ่มทำการประมวลผลทั้งหมด", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, f in enumerate(uploaded_files):
                status_text.text(f"กำลังประมวลผล ({i+1}/{len(uploaded_files)}): {f.name} ...")
                
                # เช็คว่าเคยประมวลผลไฟล์นี้ไปแล้วหรือยัง (เช็คจากชื่อ)
                if not any(r['File Name'] == f.name for r in st.session_state['history_records']):
                    process_single_file(f)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ ประมวลผลเสร็จสมบูรณ์!")
            st.success("สามารถดูลากไฟล์ชุดต่อไปมาอัปโหลดเพิ่มได้เลย!")

    st.divider()
    if st.button("🗑️ ล้างข้อมูลประวัติทั้งหมด"):
        st.session_state['history_records'] = []
        st.rerun()

# --- Dashboard ---
if len(st.session_state['history_records']) > 0:
    st.success(f"📊 ปัจจุบันมีข้อมูลไฟล์ในระบบทั้งหมด {len(st.session_state['history_records'])} ไฟล์")
    tab_compare, tab_detail, tab_meta = st.tabs([
        "📊 1. ตารางเปรียบเทียบข้อมูลไฟล์ (Comparison)", 
        "🖼️ 2. วิเคราะห์จุดโฟกัส (Focus Accuracy)", 
        "🗂️ 3. ข้อมูล Metadata เชิงลึก"
    ])

    with tab_compare:
        st.subheader("ตารางเปรียบเทียบสถิติและคุณภาพเลนส์จากไฟล์ทั้งหมด")
        df_compare = pd.DataFrame([{
            "File Name": r["File Name"],
            "Camera": r["Camera"],
            "Lens": r["Lens"],
            "AF Fine-Tune": r["AF Fine-Tune"],
            "Focus Shift (px)": r["Focus Shift (px)"],
            "Sharpness Score": r["Sharpness Score"],
            "Edge Density (%)": r["Edge Density (%)"],
            "Contrast": r["Contrast"]
        } for r in st.session_state['history_records']])
        
        st.dataframe(df_compare.style.highlight_max(subset=['Sharpness Score', 'Edge Density (%)'], color='lightgreen')
                                     .highlight_min(subset=['Focus Shift (px)'], color='lightblue'))
        
        if len(df_compare) > 1:
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.barplot(data=df_compare, x='File Name', y='Sharpness Score', ax=ax, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                ax.set_title("เปรียบเทียบความคมชัด (Sharpness Score)")
                st.pyplot(fig)
                plt.close(fig)
            with col_chart2:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.barplot(data=df_compare, x='File Name', y='Focus Shift (px)', ax=ax, palette='magma')
                plt.xticks(rotation=45, ha='right')
                ax.set_title("เปรียบเทียบความคลาดเคลื่อนโฟกัส (Focus Shift)")
                st.pyplot(fig)
                plt.close(fig)

    with tab_detail:
        st.subheader("วิเคราะห์ความแม่นยำจุดโฟกัสรายภาพ")
        file_names = [r["File Name"] for r in st.session_state['history_records']]
        selected_file = st.selectbox("เลือกภาพที่ต้องการดูรายละเอียด:", file_names)
        
        curr_data = next(r for r in st.session_state['history_records'] if r["File Name"] == selected_file)
        imgs = curr_data["Images"]
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("AF Fine-Tune", curr_data["AF Fine-Tune"])
        col_m2.metric("Focus Shift (คลาดเคลื่อน)", f"{curr_data['Focus Shift (px)']} px")
        col_m3.metric("Sharpness", curr_data["Sharpness Score"])
        col_m4.metric("Edge Density", f"{curr_data['Edge Density (%)']}%")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(imgs["preview_rgb"])
            
            sx, sy, ex, ey = imgs["box_scaled"]
            ax.add_patch(patches.Rectangle((sx, sy), ex-sx, ey-sy, linewidth=1.5, edgecolor='cyan', facecolor='none'))
            
            cx, cy = imgs["center_scaled"]
            ax.scatter(cx, cy, color='red', marker='+', s=150, linewidths=2, label='Recorded AF Point (กล้องบันทึก)')
            
            px, py = imgs["peak_scaled"]
            ax.scatter(px, py, color='lime', marker='o', s=100, facecolors='none', linewidths=2, label='Actual Sharpest Point (ชัดที่สุดจริง)')
            
            if math.hypot(cx - px, cy - py) > 10:
                ax.plot([cx, px], [cy, py], color='yellow', linestyle='--', linewidth=1)
                
            ax.legend(loc='upper right')
            ax.set_title(f"Focus Analysis: {curr_data['File Name']}")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
        with col_b:
            fig_roi, ax_roi = plt.subplots()
            ax_roi.imshow(imgs["roi_rgb"])
            ax_roi.set_title("Crop Focus ROI (Original Res)")
            ax_roi.axis('off')
            st.pyplot(fig_roi)
            plt.close(fig_roi)
            
            fig_canny, ax_canny = plt.subplots()
            ax_canny.imshow(imgs["canny"], cmap='gray')
            ax_canny.set_title("Canny Edges (ROI)")
            ax_canny.axis('off')
            st.pyplot(fig_canny)
            plt.close(fig_canny)

    with tab_meta:
        st.subheader("ข้อมูล Metadata เชิงลึกทั้งหมด (Raw Data)")
        selected_meta_file = st.selectbox("เลือกภาพที่ต้องการดู Metadata:", file_names, key='meta_select')
        curr_meta_data = next(r for r in st.session_state['history_records'] if r["File Name"] == selected_meta_file)
        
        st.write(f"**Camera:** {curr_meta_data['Camera']} | **Lens:** {curr_meta_data['Lens']}")
        
        meta_df = pd.DataFrame(list(curr_meta_data["Full Metadata"].items()), columns=["Metadata Tag", "Value"])
        st.dataframe(meta_df, use_container_width=True, height=500)
        
else:
    st.info("👈 กรุณาเลือกไฟล์และกด 'เริ่มทำการประมวลผล' ที่แถบด้านซ้าย เพื่อดูข้อมูลเปรียบเทียบ")
