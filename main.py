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

def get_accurate_lens_name(metadata: dict) -> str:
    """ดึงชื่อเลนส์และยี่ห้อให้ครบถ้วน (รองรับ Sigma, Tamron, Nikon)"""
    candidates = [
        metadata.get("Composite:LensID"),
        metadata.get("EXIF:LensModel"),
        metadata.get("Nikon:LensModel"),
        metadata.get("MakerNotes:LensModel"),
        metadata.get("EXIF:LensInfo")
    ]
    
    for name in candidates:
        if name and isinstance(name, str):
            clean = name.strip()
            if not re.match(r'^[0-9A-Fa-f\s]{10,}$', clean):
                return clean

    brand = metadata.get("EXIF:LensMake") or metadata.get("EXIF:Make", "Sigma")
    focal = metadata.get("EXIF:FocalLength", "")
    fnum = metadata.get("EXIF:FNumber", "")
    
    if focal:
        focal_str = str(focal).replace(" ", "-")
        if "mm" not in focal_str.lower():
            focal_str += "mm"
        f_str = f" f/{fnum}" if fnum else ""
        return f"{brand} {focal_str}{f_str}".strip()

    return "Sigma / Unknown Lens"

def parse_nikon_precise_af(metadata: dict, img_w: int, img_h: int):
    """แกะรหัสพิกัดจุดโฟกัสเชิงลึก 51 จุดของ Nikon จาก MakerNotes"""
    raw_af = (
        metadata.get("Nikon:PrimaryAFPoint") or
        metadata.get("MakerNotes:PrimaryAFPoint") or
        metadata.get("Nikon:AFPointsInFocus") or
        metadata.get("Nikon:AFPointSelected") or
        "Center"
    )
    af_text = str(raw_af).strip()
    
    cx, cy = img_w // 2, img_h // 2
    
    pos = metadata.get("Nikon:AFPointPosition") or metadata.get("Composite:AFPointPosition")
    if pos:
        try:
            parts = [float(p) for p in str(pos).split()]
            if len(parts) >= 2:
                if 0 <= parts[0] <= 1.0 and 0 <= parts[1] <= 1.0:
                    return af_text, int(parts[0] * img_w), int(parts[1] * img_h)
        except Exception:
            pass

    af_upper = af_text.upper()
    if "FAR LEFT" in af_upper or "EXTREME LEFT" in af_upper:
        norm_x = 0.15
    elif "LEFT" in af_upper:
        norm_x = 0.30 if "COL" in af_upper or "1" in af_upper else 0.38
    elif "FAR RIGHT" in af_upper or "EXTREME RIGHT" in af_upper:
        norm_x = 0.85
    elif "RIGHT" in af_upper:
        norm_x = 0.70 if "COL" in af_upper else 0.62
    else:
        norm_x = 0.50

    if "TOP" in af_upper or "UPPER" in af_upper or "ROW 1" in af_upper:
        norm_y = 0.20
    elif "LOWER TOP" in af_upper or "ROW 2" in af_upper:
        norm_y = 0.35
    elif "BOTTOM" in af_upper or "LOWER" in af_upper or "ROW" in af_upper:
        norm_y = 0.80 if "EXTREME" in af_upper else 0.70
    else:
        norm_y = 0.50

    return af_text, int(norm_x * img_w), int(norm_y * img_h)

def find_optimal_sharpness_peak(gray_img, init_cx, init_cy, search_radius=100):
    """
    สแกนหาจุดที่มีความคมชัดสูงสุด (Peak Sharpness) บริเวณรอบๆ จุดโฟกัส 
    เพื่อระบุตำแหน่งที่คมที่สุดจริงบนเนื้อภาพ
    """
    h, w = gray_img.shape
    best_x, best_y = init_cx, init_cy
    max_var = -1.0
    
    # กำหนดขอบเขตการกวาดหา (Search Window)
    x_min = max(0, init_cx - search_radius)
    x_max = min(w, init_cx + search_radius)
    y_min = max(0, init_cy - search_radius)
    y_max = min(h, init_cy + search_radius)
    
    step = max(5, search_radius // 10)
    patch_size = 30
    
    for y in range(y_min, y_max, step):
        for x in range(x_min, x_max, step):
            patch = gray_img[max(0, y-patch_size):min(h, y+patch_size), max(0, x-patch_size):min(w, x+patch_size)]
            if patch.size > 0:
                var = cv2.Laplacian(patch, cv2.CV_64F).var()
                if var > max_var:
                    max_var = var
                    best_x, best_y = x, y
                    
    return best_x, best_y

def extract_exif_data(file_path: str) -> dict:
    try:
        with ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)[0]
            
            make = metadata.get("EXIF:Make", "NIKON CORPORATION")
            model = metadata.get("EXIF:Model", "D7100")
            camera = f"{make} {model}".strip()
            
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
            "metadata": {}, "camera": "NIKON CORPORATION D7100", "lens": "Sigma 150-600mm",
            "af_fine_tune": "0", "f_number": "N/A", "exposure_time": "N/A", "iso": 100, "focal_length": "N/A"
        }

def process_image_and_sharpness(file_path: str, metadata: dict, iso_val):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("ไม่สามารถเปิดอ่านไฟล์ภาพได้")

    h, w = img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 1. พิกัดจุดโฟกัสจาก EXIF
    af_point_text, raw_cx, raw_cy = parse_nikon_precise_af(metadata, w, h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. ค้นหาจุดที่คมชัดที่สุดจริงบนภาพ (Optimal Sharpness Peak)
    opt_cx, opt_cy = find_optimal_sharpness_peak(denoised_gray, raw_cx, raw_cy, search_radius=min(w, h)//6)

    # 3. ขยายกรอบวิเคราะห์และแสดงผล ROI เพิ่มขึ้น 20% จากเดิม
    base_box_size = min(w, h) // 8
    expanded_box_size = int(base_box_size * 1.2) # ขยายเพิ่มอีก 20%
    
    x1, x2 = max(0, opt_cx - expanded_box_size), min(w, opt_cx + expanded_box_size)
    y1, y2 = max(0, opt_cy - expanded_box_size), min(h, opt_cy + expanded_box_size)
    
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

    # 4. วาดสัญลักษณ์แสดงผลบนภาพ
    annotated_img = img.copy()
    box_color = (0, 255, 0) if normalized_score >= 400 else (0, 0, 255)
    
    # วาดกรอบ ROI ที่ขยายแล้ว 20%
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 4)
    
    # มาร์กจุดโฟกัสจาก EXIF (สีน้ำเงิน) และจุดที่คมที่สุดจริง (สีแดง)
    cv2.drawMarker(annotated_img, (raw_cx, raw_cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    cv2.drawMarker(annotated_img, (opt_cx, opt_cy), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=22, thickness=2)
    
    label = f"AF: {af_point_text} | Peak Score: {normalized_score}"
    cv2.rectangle(annotated_img, (10, 10), (w - 10, 55), (0, 0, 0), -1)
    cv2.putText(annotated_img, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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
