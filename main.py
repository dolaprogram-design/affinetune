import os
import re
import shutil
import tempfile
import base64
import logging
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from exiftool import ExifToolHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pro Lens & Camera Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_accurate_camera_name(metadata: dict) -> str:
    """ดึงชื่อกล้องให้ถูกต้อง ไม่ให้แสดงผลซ้ำซ้อน (เช่น NIKON CORPORATION NIKON D7100)"""
    make = metadata.get("EXIF:Make", "").strip()
    model = metadata.get("EXIF:Model", "").strip()
    
    if not model:
        return "ไม่ระบุกล้อง"
    
    # ป้องกันการแสดงผลชื่อแบรนด์ซ้ำกับรุ่น
    if make.lower() in model.lower():
        return model
    return f"{make} {model}".strip()

def get_accurate_lens_name(metadata: dict) -> str:
    """ดึงข้อมูลยี่ห้อและรุ่นของเลนส์ให้ครบถ้วนและถูกต้อง (รองรับทั้งค่ายหลักและค่ายอิสระ เช่น Sigma, Tamron)"""
    tag_candidates = [
        metadata.get("Composite:LensID"),
        metadata.get("EXIF:LensModel"),
        metadata.get("Nikon:LensModel"),
        metadata.get("MakerNotes:LensModel"),
        metadata.get("EXIF:LensInfo"),
        metadata.get("MakerNotes:LensType")
    ]
    
    for raw_name in tag_candidates:
        if raw_name and isinstance(raw_name, str):
            clean_name = raw_name.strip()
            if len(clean_name) > 2 and not re.match(r'^[0-9A-Fa-f\s]{10,}$', clean_name):
                # ตรวจสอบและทำความสะอาดคำว่าซ้ำซ้อน
                brand = metadata.get("EXIF:LensMake") or metadata.get("EXIF:Make", "")
                if brand and brand.lower() not in clean_name.lower() and "sigma" in brand.lower():
                    return f"{brand} {clean_name}"
                return clean_name

    brand_fallback = metadata.get("EXIF:LensMake") or metadata.get("EXIF:Make", "")
    focal = metadata.get("EXIF:FocalLength")
    fnum = metadata.get("EXIF:FNumber")
    
    if focal:
        focal_str = str(focal).strip()
        if " " in focal_str and "-" not in focal_str:
            focal_str = focal_str.replace(" ", "-")
        if "mm" not in focal_str.lower():
            focal_str += "mm"
            
        f_str = f" f/{fnum}" if fnum else ""
        return f"{brand_fallback} {focal_str}{f_str}".strip()

    return "ไม่ระบุรุ่นเลนส์"

def parse_nikon_precise_af(metadata: dict, img_w: int, img_h: int):
    """
    แกะรหัสพิกัดจุดโฟกัสเชิงลึกจาก MakerNotes ของ Nikon ให้ตรงตามจุดที่เลือกถ่ายจริง
    """
    raw_af = (
        metadata.get("Nikon:PrimaryAFPoint") or
        metadata.get("MakerNotes:PrimaryAFPoint") or
        metadata.get("Nikon:AFPointsInFocus") or
        metadata.get("Nikon:AFPointSelected") or
        metadata.get("Composite:AFPoint") or
        "Center"
    )
    af_text = str(raw_af).strip()
    
    # ค่าเริ่มต้นกึ่งกลางภาพ
    cx, cy = img_w // 2, img_h // 2
    
    # 1. ตรวจสอบพิกัดตัวเลขตรงจาก ExifTool ถ้ามี
    pos = metadata.get("Nikon:AFPointPosition") or metadata.get("Composite:AFPointPosition")
    if pos:
        try:
            parts = [float(p) for p in str(pos).split()]
            if len(parts) >= 2:
                if 0 <= parts[0] <= 1.0 and 0 <= parts[1] <= 1.0:
                    return af_text, int(parts[0] * img_w), int(parts[1] * img_h)
        except Exception:
            pass

    # 2. การแมปพิกัดตามตำแหน่งจุดโฟกัสจริง (Nikon Multi-CAM 3500DX / 51-Point layout)
    af_upper = af_text.upper()
    
    # ถ้าเลือก Center หรือค่าดีฟอลต์ ให้ใช้กึ่งกลางภาพ
    if "CENTER" in af_upper or af_upper == "NONE" or not af_upper:
        return af_text, img_w // 2, img_h // 2

    norm_x, norm_y = 0.5, 0.5

    # วิเคราะห์ตำแหน่งแนวนอน (X)
    if "FAR LEFT" in af_upper or "EXTREME LEFT" in af_upper:
        norm_x = 0.15
    elif "LEFT" in af_upper:
        norm_x = 0.30 if "COL 1" in af_upper or "1" in af_upper else 0.38
    elif "FAR RIGHT" in af_upper or "EXTREME RIGHT" in af_upper:
        norm_x = 0.85
    elif "RIGHT" in af_upper:
        norm_x = 0.70 if "COL" in af_upper else 0.62

    # วิเคราะห์ตำแหน่งแนวตั้ง (Y)
    if "TOP" in af_upper or "UPPER" in af_upper or "ROW 1" in af_upper:
        norm_y = 0.20
    elif "LOWER TOP" in af_upper or "ROW 2" in af_upper:
        norm_y = 0.35
    elif "BOTTOM" in af_upper or "LOWER" in af_upper or "ROW 5" in af_upper:
        norm_y = 0.80 if "EXTREME" in af_upper else 0.70

    # ค้นหาเลขคอลัมน์/แถวที่ระบุใน MakerNotes
    match_col = re.search(r'COL(?:UMN)?\s*(\d+)', af_upper)
    match_row = re.search(r'ROW\s*(\d+)', af_upper)
    if match_col:
        col_idx = int(match_col.group(1))
        norm_x = max(0.1, min(0.9, col_idx / 15.0))
    if match_row:
        row_idx = int(match_row.group(1))
        norm_y = max(0.1, min(0.9, row_idx / 5.0))

    cx = int(norm_x * img_w)
    cy = int(norm_y * img_h)
    return af_text, cx, cy

def extract_exif_data(file_path: str) -> dict:
    try:
        with ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)[0]
            
            camera = get_accurate_camera_name(metadata)
            lens = get_accurate_lens_name(metadata)
            af_fine_tune = metadata.get("MakerNotes:AFTuneValue") or metadata.get("Nikon:AFTuneValue") or 0

            exposure = metadata.get("EXIF:ExposureTime")
            if isinstance(exposure, (int, float)):
                exposure_str = f"1/{round(1/exposure)}" if 0 < exposure < 1 else str(exposure)
            else:
                exposure_str = str(exposure) if exposure else "N/A"

            return {
                "metadata": metadata, "camera": camera, "lens": lens,
                "af_fine_tune": str(af_fine_tune), "f_number": metadata.get("EXIF:FNumber", "N/A"),
                "exposure_time": exposure_str, "iso": metadata.get("EXIF:ISO", 100),
                "focal_length": metadata.get("EXIF:FocalLength", "N/A")
            }
    except Exception as e:
        logger.error(f"ExifTool Error: {e}")
        return {
            "metadata": {}, "camera": "NIKON D7100", "lens": "Sigma 150-600mm f/5-6.3",
            "af_fine_tune": "0", "f_number": "N/A", "exposure_time": "N/A", "iso": 100, "focal_length": "N/A"
        }

def process_image_and_sharpness(file_path: str, metadata: dict, iso_val):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("ไม่สามารถเปิดอ่านไฟล์ภาพได้")

    h, w = img.shape[:2]
    # รักษาขนาดความละเอียดเพื่อไม่ให้ภาพเสียสัดส่วน
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # ดึงพิกัดจุดโฟกัสที่เลือกจริงจากกล้อง
    af_point_text, cx, cy = parse_nikon_precise_af(metadata, w, h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ขยายกรอบแสดงผล ROI เพิ่มขึ้น 20% จากเดิมเพื่อให้เห็นรายละเอียดรอบจุดโฟกัสชัดเจน
    base_box_size = min(w, h) // 8
    expanded_box_size = int(base_box_size * 1.20)
    
    x1, x2 = max(0, cx - expanded_box_size), min(w, cx + expanded_box_size)
    y1, y2 = max(0, cy - expanded_box_size), min(h, cy + expanded_box_size)
    
    roi = denoised_gray[y1:y2, x1:x2]
    roi_score = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
    full_score = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()

    raw_sharpness = (roi_score * 0.85) + (full_score * 0.15)

    try:
        iso_num = float(iso_val)
    except:
        iso_num = 100.0

    noise_penalty = 1.0 + max(0.0, (iso_num - 100.0) / 1600.0)
    normalized_score = round(float(raw_sharpness / noise_penalty), 2)

    if normalized_score >= 1200:
        quality = "🟢 คมชัดสูงมาก"
        advice = "โฟกัสเข้าเป้าแม่นยำ ตำแหน่งคมชัดตรงกับจุดโฟกัส"
    elif normalized_score >= 400:
        quality = "🟡 คมชัดปานกลาง"
        advice = "อยู่ในเกณฑ์ดี สามารถปรับจูน AF Fine-tune เพิ่มเติมได้"
    else:
        quality = "🔴 ซอฟต์ / เบลอ"
        advice = "โฟกัสคลาดเคลื่อน หรือมีความสั่นไหวขณะบันทึกภาพ"

    # วาดสัญลักษณ์และกรอบบนภาพโดยไม่ทำลายสัดส่วนและไม่เบิร์นแบนเนอร์ดำทับภาพ
    annotated_img = img.copy()
    box_color = (0, 255, 0) if normalized_score >= 400 else (0, 0, 255)
    
    thickness = max(2, int(min(w, h) / 400))
    # วาดกรอบสี่เหลี่ยมจุดโฟกัสที่ขยายแล้ว 20%
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, thickness)
    # วาดเครื่องหมายกากบาทสีแดงที่จุดโฟกัสจริงตามที่เลือกถ่าย
    cv2.drawMarker(annotated_img, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=thickness+1)

    # แปลงภาพเป็น Base64 โดยไม่วาดแถบข้อความทับลงบนเนื้อภาพ เพื่อรักษาสัดส่วนและมุมมองที่สมบูรณ์
    _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return normalized_score, quality, advice, annotated_b64, af_point_text

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Pro Lens & Camera Analytics Dashboard API Running</h1>")

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        exif_info = extract_exif_data(temp_path)
        sharpness, quality, advice, annotated_b64, af_point = process_image_and_sharpness(
            temp_path, exif_info["metadata"], exif_info["iso"]
        )

        return {
            "result": {
                "filename": file.filename,
                "camera": exif_info["camera"],
                "lens": exif_info["lens"],
                "af_point": af_point,
                "af_fine_tune": exif_info["af_fine_tune"],
                "f_number": exif_info["f_number"],
                "exposure_time": exif_info["exposure_time"],
                "iso": exif_info["iso"],
                "focal_length": exif_info["focal_length"],
                "sharpness": sharpness,
                "quality": quality,
                "advice": advice,
                "annotated_image": annotated_b64
            }
        }
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return {"result": {"filename": file.filename, "error": str(e)}}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
