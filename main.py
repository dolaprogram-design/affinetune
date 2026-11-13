import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import subprocess
import json
import tempfile
import gc
import warnings
warnings.filterwarnings('ignore')

# สร้าง Temp Directory สำหรับเก็บภาพ Thumbnail เพื่อลดการใช้ RAM ของเบราว์เซอร์
TEMP_DIR = tempfile.mkdtemp()

# ==========================================
# Backend Functions สำหรับ Dashboard
# ==========================================

def extract_preview(fpath, temp_dir):
    """ดึงภาพ Preview จากไฟล์ RAW โดยใช้ ExifTool เพื่อไม่ให้เปลือง RAM"""
    try:
        cmd_img = ['exiftool', '-b', '-JpgFromRaw', fpath]
        res_img = subprocess.run(cmd_img, stdout=subprocess.PIPE)
        if not res_img.stdout:
            cmd_img = ['exiftool', '-b', '-PreviewImage', fpath]
            res_img = subprocess.run(cmd_img, stdout=subprocess.PIPE)
            
        if res_img.stdout:
            nparr = np.frombuffer(res_img.stdout, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Error extracting preview: {e}")
    return None

def process_dashboard_images(file_paths, progress=gr.Progress()):
    if not file_paths:
        return [], pd.DataFrame(), pd.DataFrame()

    data = []
    gallery_items = []
    
    # วนลูปพร้อมแสดง Progress Bar
    for i, fpath in progress.tqdm(enumerate(file_paths), desc="กำลังวิเคราะห์ภาพ...", total=len(file_paths)):
        filename = os.path.basename(fpath)

        iso_value, focal_length, exposure_time = None, None, None
        camera_model, af_point, focus_dist, lens_info = "N/A", "N/A", "N/A", "N/A"
        sharpness = 0.0

        try:
            # 1. อ่านข้อมูล MakerNotes ด้วย ExifTool
            cmd = ['exiftool', '-j', fpath]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            exif_data = json.loads(res.stdout)[0]

            iso_val = exif_data.get('ISO', exif_data.get('ISOSpeedRatings', None))
            if iso_val is not None:
                try: iso_value = int(iso_val)
                except: pass

            fl = exif_data.get('FocalLength', None)
            if fl:
                try: focal_length = float(str(fl).replace(' mm', '').strip())
                except: pass

            et = exif_data.get('ExposureTime', None)
            if et:
                try:
                    if isinstance(et, str) and '/' in et:
                        num, den = et.split('/')
                        exposure_time = float(num) / float(den)
                    else:
                        exposure_time = float(et)
                except: pass

            camera_model = exif_data.get('Model', "N/A")
            af_point = exif_data.get('PrimaryAFPoint', exif_data.get('AFPointsUsed', "N/A"))
            focus_dist = exif_data.get('FocusDistance', exif_data.get('ApproximateFocusDistance', "N/A"))
            lens_info = exif_data.get('LensID', exif_data.get('Lens', "N/A"))
            
            # 2. คำนวณความคมชัด
            img_preview = extract_preview(fpath, TEMP_DIR)
            if img_preview is not None:
                gray = cv2.cvtColor(img_preview, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                
                # บันทึกภาพย่อสำหรับ Gallery (ลดขนาดเพื่อป้องกันหน้าเว็บค้าง)
                preview_h, preview_w = gray.shape
                scale = min(1.0, 800.0 / max(preview_h, preview_w))
                if scale < 1.0:
                    img_preview = cv2.resize(img_preview, (int(preview_w * scale), int(preview_h * scale)))
                
                temp_jpg_path = os.path.join(TEMP_DIR, f"thumb_{i}.jpg")
                cv2.imwrite(temp_jpg_path, img_preview)
                
                rating = "🔴 Blur" if sharpness < 100 else "🟡 Fair" if sharpness < 500 else "🟢 Good" if sharpness < 1500 else "⭐ Excellent"
                caption = f"{filename}\nScore: {sharpness:.2f} ({rating})"
                gallery_items.append((temp_jpg_path, caption))
                
                del gray, img_preview
            else:
                rating = "⚠️ Cannot Read"

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            rating = "⚠️ Error"

        shutter = "N/A"
        if exposure_time:
            shutter = f"{exposure_time:.1f}s" if exposure_time >= 1 else f"1/{int(round(1/exposure_time))}"

        data.append({
            "Filename": filename,
            "Model": camera_model,
            "Sharpness": round(sharpness, 2),
            "Quality": rating,
            "ISO": iso_value,
            "FocalLength_mm": focal_length,
            "ShutterSpeed": shutter,
            "ExposureTime_sec": exposure_time,
            "AF_Point": af_point,
            "Focus_Dist": focus_dist,
            "Lens": lens_info
        })
        
        # คืนหน่วยความจำทันทีหลังจบแต่ละลูป
        gc.collect()

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by='Sharpness', ascending=False)

    return gallery_items, df, df

def update_plots(df):
    if df is None or df.empty:
        return None, None, None, None

    fig_iso = plt.figure(figsize=(6, 4))
    df_iso = df.dropna(subset=['ISO']).copy()
    if not df_iso.empty:
        df_iso_grp = df_iso.groupby('ISO')['Sharpness'].mean().reset_index()
        plt.bar([str(int(x)) for x in df_iso_grp['ISO']], df_iso_grp['Sharpness'], color='#8b5cf6')
        plt.title('Average Sharpness by ISO', fontweight='bold')
        plt.xlabel('ISO')
        plt.ylabel('Sharpness')
    plt.tight_layout()

    fig_focal = plt.figure(figsize=(6, 4))
    df_foc = df.dropna(subset=['FocalLength_mm']).copy()
    if not df_foc.empty:
        plt.scatter(df_foc['FocalLength_mm'], df_foc['Sharpness'], color='#10b981', s=80, alpha=0.8)
        plt.title('Focal Length vs Sharpness', fontweight='bold')
        plt.xlabel('Focal Length (mm)')
        plt.ylabel('Sharpness Score')
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig_shutter = plt.figure(figsize=(6, 4))
    df_shut = df.dropna(subset=['ExposureTime_sec']).copy()
    if not df_shut.empty:
        plt.scatter(df_shut['ExposureTime_sec'], df_shut['Sharpness'], color='#f59e0b', s=80, alpha=0.8)
        plt.title('Shutter Speed vs Sharpness', fontweight='bold')
        plt.xlabel('Exposure Time (sec) [Log Scale]')
        plt.ylabel('Sharpness Score')
        plt.xscale('log')
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig_af = plt.figure(figsize=(6, 4))
    df_af = df[df['AF_Point'] != 'N/A'].dropna(subset=['AF_Point', 'Sharpness']).copy()
    if not df_af.empty:
        df_af_grp = df_af.groupby('AF_Point')['Sharpness'].mean().reset_index()
        df_af_grp = df_af_grp.sort_values('Sharpness', ascending=False)
        plt.bar(df_af_grp['AF_Point'].astype(str), df_af_grp['Sharpness'], color='#ec4899')
        plt.title('Average Sharpness by AF Point', fontweight='bold')
        plt.xlabel('AF Point')
        plt.ylabel('Average Sharpness')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig_iso, fig_focal, fig_shutter, fig_af

def analyze_roi(gray_img, start_x, start_y, crop_w, crop_h):
    roi = gray_img[start_y:start_y+crop_h, start_x:start_x+crop_w]
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    return np.var(laplacian)

def fine_tune_lens(image, center_roi_percent, edge_roi_percent):
    if image is None:
        return None, "กรุณาอัปโหลดภาพทดสอบ"

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_disp = image.copy()
    results = []

    center_crop_h, center_crop_w = int(h * (center_roi_percent/100)), int(w * (center_roi_percent/100))
    c_start_y, c_start_x = (h - center_crop_h) // 2, (w - center_crop_w) // 2

    center_score = analyze_roi(gray, c_start_x, c_start_y, center_crop_w, center_crop_h)
    cv2.rectangle(img_disp, (c_start_x, c_start_y), (c_start_x+center_crop_w, c_start_y+center_crop_h), (50, 205, 50), max(2, int(w*0.005)))
    cv2.putText(img_disp, f"Center: {center_score:.0f}", (c_start_x, c_start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2)
    results.append(f"📍 Center Score: {center_score:.2f}")

    edge_crop_h, edge_crop_w = int(h * (edge_roi_percent/100)), int(w * (edge_roi_percent/100))
    offset_y, offset_x = int(h * 0.05), int(w * 0.05)

    corners = {
        "Top-Left": (offset_x, offset_y),
        "Top-Right": (w - edge_crop_w - offset_x, offset_y),
        "Bottom-Left": (offset_x, h - edge_crop_h - offset_y),
        "Bottom-Right": (w - edge_crop_w - offset_x, h - edge_crop_h - offset_y)
    }

    edge_scores = []
    for name, (start_x, start_y) in corners.items():
        score = analyze_roi(gray, start_x, start_y, edge_crop_w, edge_crop_h)
        edge_scores.append(score)
        cv2.rectangle(img_disp, (start_x, start_y), (start_x+edge_crop_w, start_y+edge_crop_h), (255, 165, 0), max(2, int(w*0.005)))
        cv2.putText(img_disp, f"{score:.0f}", (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        results.append(f"📐 {name} Score: {score:.2f}")

    avg_edge_score = np.mean(edge_scores)
    results.append(f"\n📊 Average Edge Score: {avg_edge_score:.2f}")

    status = "🟢 Center ผ่านเกณฑ์ (คมชัด)" if center_score > 500 else "🔴 Center ต่ำกว่าเกณฑ์"
    edge_status = "🟢 Edges คมชัดดี" if avg_edge_score > (center_score * 0.6) else "🟡 Edges ดรอป (ขอบเบลอ)"

    result_text = "[ ผลการประเมินความคมชัด ]\n" + "\n".join(results) + f"\n\n[ การวินิจฉัย ]\n- {status}\n- {edge_status}"

    return img_disp, result_text

def analyze_af_finetune_series(file_paths, progress=gr.Progress()):
    if not file_paths:
        return None, pd.DataFrame(), "ไม่พบข้อมูล"

    af_tune_data = []

    for i, fpath in progress.tqdm(enumerate(file_paths), desc="กำลังวิเคราะห์ AF Fine-tune...", total=len(file_paths)):
        filename = os.path.basename(fpath)

        cmd_exif = ['exiftool', '-j', '-AFFineTuneAdj', '-AFFineTune', fpath]
        res_exif = subprocess.run(cmd_exif, stdout=subprocess.PIPE, text=True)

        af_tune_val = None
        af_tune_status = "Off"
        try:
            exif_info = json.loads(res_exif.stdout)[0]
            af_tune_status = exif_info.get('AFFineTune', 'Off')
            if 'AFFineTuneAdj' in exif_info:
                af_tune_val = int(exif_info['AFFineTuneAdj'])
        except:
            pass

        sharpness = 0.0
        img_preview = extract_preview(fpath, TEMP_DIR)
        if img_preview is not None:
            gray = cv2.cvtColor(img_preview, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            del gray, img_preview
            
        af_tune_data.append({
            'Filename': filename,
            'AF Fine-tune Status': af_tune_status,
            'AF Fine-tune Adj': af_tune_val if af_tune_val is not None else "N/A",
            'Sharpness Score': round(sharpness, 2)
        })
        
        gc.collect()

    df_af = pd.DataFrame(af_tune_data)

    if df_af.empty:
        return None, df_af, "ไม่สามารถวิเคราะห์ข้อมูลได้"

    fig = plt.figure(figsize=(8, 5))
    df_plot = df_af[df_af['AF Fine-tune Adj'] != 'N/A'].copy()

    best_af = "ไม่พบข้อมูลเพียงพอ"
    if not df_plot.empty:
        df_plot['AF Fine-tune Adj'] = df_plot['AF Fine-tune Adj'].astype(int)
        df_plot = df_plot.sort_values('AF Fine-tune Adj')

        plt.plot(df_plot['AF Fine-tune Adj'], df_plot['Sharpness Score'],
                 marker='o', color='purple', linestyle='-', linewidth=2, markersize=8)

        for _, row in df_plot.iterrows():
            plt.annotate(row['Filename'],
                         (row['AF Fine-tune Adj'], row['Sharpness Score']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        plt.title('AF Fine-tune vs Sharpness', fontweight='bold')
        plt.xlabel('AF Fine-tune Adjustment Value')
        plt.ylabel('Sharpness Score')
        plt.grid(True, linestyle='--', alpha=0.5)

        if len(df_plot) > 0:
            best_af_val = df_plot.loc[df_plot['Sharpness Score'].idxmax(), 'AF Fine-tune Adj']
            plt.axvline(x=best_af_val, color='green', linestyle=':', label=f'Best Focus: {best_af_val}')
            plt.legend()
            best_af = f"ค่า AF Fine-tune ที่คมชัดที่สุดคือ: {best_af_val}"

    plt.tight_layout()
    return fig, df_af, best_af


# ==========================================
# Gradio User Interface
# ==========================================

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=(gr.themes.GoogleFont("Inter"), "sans-serif"),
)

custom_css = """
.gradio-container { background-color: #f8fafc; }
.gr-button-primary { background: linear-gradient(to right, #6366f1, #8b5cf6); border: none; box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3); }
.gr-button-primary:hover { box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4); }
.gr-panel { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); border-radius: 12px; }
.tabs button { color: #334155 !important; font-weight: 500; }
.tabs button.selected { color: #4f46e5 !important; font-weight: 700; }
"""

with gr.Blocks(theme=custom_theme, css=custom_css) as dashboard:
    gr.Markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1e293b; font-size: 2.5rem; margin-bottom: 0.5rem;">📸 Pro Lens & Camera Analytics</h1>
        <p style="color: #64748b; font-size: 1.1rem;">ระบบวิเคราะห์คุณภาพไฟล์ภาพและเลนส์เชิงลึกแบบอัตโนมัติ</p>
    </div>
    """)

    df_state = gr.State(pd.DataFrame())

    with gr.Tab("1. 📊 ภาพรวมและประเมินผล (Overview)"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### อัปโหลดไฟล์เพื่อวิเคราะห์")
                file_input = gr.File(file_count="multiple", type="filepath", label="ลากไฟล์มาวางที่นี่ (JPEG, NEF, CR2...)")
                process_btn = gr.Button("🚀 เริ่มวิเคราะห์ข้อมูลทันที", variant="primary", size="lg")
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ ภาพตัวอย่างและผลคะแนน")
                gallery = gr.Gallery(label="Image Gallery", columns=4, height=350, object_fit="contain")

        gr.Markdown("### 📋 ตารางสรุปพารามิเตอร์เชิงลึก (MakerNotes)")
        data_table = gr.Dataframe(interactive=False)

    with gr.Tab("2. 📈 กราฟวิเคราะห์ความสัมพันธ์ (Analytics)"):
        gr.Markdown("ดูความสัมพันธ์ของตัวแปรต่างๆ จากข้อมูลที่คุณเพิ่งอัปโหลดในหน้า Overview")
        refresh_btn = gr.Button("🔄 อัปเดตกราฟจากข้อมูลล่าสุด", variant="secondary")
        with gr.Row():
            plot_iso = gr.Plot(label="ISO vs Sharpness")
            plot_focal = gr.Plot(label="Focal Length vs Sharpness")
        with gr.Row():
            plot_shutter = gr.Plot(label="Shutter Speed vs Sharpness")
            plot_af = gr.Plot(label="AF Point vs Sharpness")

    with gr.Tab("3. 🎯 ทดสอบความคมชัดขอบเลนส์ (Lens Calibration)"):
        gr.Markdown("อัปโหลดภาพ Focus Chart หรือภาพระนาบตรง เพื่อเปรียบเทียบความคมชัดตรงกลางและขอบภาพแบบละเอียด")
        with gr.Row():
            with gr.Column():
                ft_image_input = gr.Image(label="อัปโหลดภาพทดสอบเลนส์")
                center_roi_slider = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="ขนาดพื้นที่กลางภาพ (Center ROI %)")
                edge_roi_slider = gr.Slider(minimum=5, maximum=30, value=10, step=5, label="ขนาดพื้นที่ขอบภาพ (Edge ROI %)")
                ft_btn = gr.Button("🔍 ประเมินความคมชัดจุดกลางและขอบ", variant="primary")
            with gr.Column():
                ft_image_output = gr.Image(label="ผลการวิเคราะห์ (กรอบโฟกัส)", interactive=False)
                ft_result = gr.Textbox(label="สรุปผลการวินิจฉัยเลนส์", lines=10)

    with gr.Tab("4. ⚙️ วิเคราะห์ AF Fine-tune"):
        gr.Markdown("อัปโหลดชุดภาพที่ถ่ายแบบไล่ระดับค่า AF Fine-tune เพื่อค้นหาค่าที่ให้ความคมชัดดีที่สุด")
        with gr.Row():
            with gr.Column(scale=1):
                af_files_input = gr.File(file_count="multiple", type="filepath", label="อัปโหลดรูปภาพทดสอบ AF (เช่น RAW NEF)")
                af_analyze_btn = gr.Button("🔍 วิเคราะห์หาค่า AF ที่ดีที่สุด", variant="primary")
                af_conclusion = gr.Textbox(label="ข้อสรุปจากการวิเคราะห์", lines=2)
            with gr.Column(scale=2):
                af_plot = gr.Plot(label="AF Fine-tune vs Sharpness Trend")

        af_table = gr.Dataframe(interactive=False)

    process_btn.click(
        fn=process_dashboard_images,
        inputs=[file_input],
        outputs=[gallery, data_table, df_state]
    )

    refresh_btn.click(
        fn=update_plots,
        inputs=[df_state],
        outputs=[plot_iso, plot_focal, plot_shutter, plot_af]
    )

    ft_btn.click(
        fn=fine_tune_lens,
        inputs=[ft_image_input, center_roi_slider, edge_roi_slider],
        outputs=[ft_image_output, ft_result]
    )

    af_analyze_btn.click(
        fn=analyze_af_finetune_series,
        inputs=[af_files_input],
        outputs=[af_plot, af_table, af_conclusion]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    dashboard.launch(server_name="0.0.0.0", server_port=port)
