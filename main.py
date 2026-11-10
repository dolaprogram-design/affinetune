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

# เปิดระบบแจ้งเตือน Error ไว้ดูใน Log ของ Render
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
    """ดึงชื่อเลนส์ให้ครบถ้วน รองรับเลนส์ค่ายอิสระ (Sigma, Tamron)"""
    # ExifTool จะสังเคราะห์ชื่อที่ถูกต้องที่สุดไว้ใน Composite:LensID
    best_name = metadata.get("Composite:LensID") or metadata.get("MakerNotes:LensType") or metadata.get("EXIF:LensModel")
    
    if best_name and isinstance(best_name, str):
        clean_name = best_name.strip()
        # กรองข้อมูลรหัส Hex ออก
        if not re.match(r'^[0-9A-Fa-f\s]{10,}$', clean_name):
            return clean_name

    # หากหาชื่อเต็มไม่เจอ ให้ประกอบชื่อขึ้นมาเองจากสเปก
    brand = metadata.get("EXIF:LensMake", "")
    focal = metadata.get("EXIF:FocalLength", "")
    fnum = metadata.get("EXIF:FNumber", "")
    
    if focal and fnum:
        # กรณีที่เป็น "150 600" ระบบจะแปลงเป็น "150-600mm" ให้
        focal_str = str(focal).replace(" ", "-") + "mm"
        return f"{brand} {focal_str} f/{fnum}".strip()

    return "ไม่ระบุรุ่นเลนส์"

def get_nikon_af_coordinates(af_text: str, img_w: int, img_h: int):
    """
    จำลองจุดพิกัด 51 จุดของ Nikon 
    คำนวณตำแหน่ง X, Y จากคำอธิบาย MakerNotes
    """
    af_upper = str(af_text).upper().strip()
    
    # ค่าเริ่มต้น (กึ่งกลาง)
    norm_x, norm_y = 0.5, 0.5
    
    # คำนวณแกน Y (บน-ล่าง)
    if "TOP" in af_upper or "UPPER" in af_upper:
        norm_y = 0.30
    elif "BOTTOM" in af_upper or "LOWER" in af_upper:
        norm_y = 0.70
        
    # คำนวณแกน X (ซ้าย-ขวา)
    if "LEFT" in af_upper:
        norm_x = 0.25 if "FAR" in af_upper else 0.35
    elif "RIGHT" in af_upper:
        norm_x = 0.75 if "FAR" in af_upper else 0.65

    return int(norm_x * img_w), int(norm_y * img_h)

def parse_af_point(metadata: dict, img_w: int, img_h: int):
    """ดึงจุดโฟกัสเชิงลึกและคำนวณตำแหน่งพิกเซล"""
    raw_af = (
        metadata.get("MakerNotes:PrimaryAFPoint") or
        metadata.get("Nikon:PrimaryAFPoint") or 
        metadata.get("Nikon:AFPointsInFocus") or 
        metadata.get("Canon:AFPointUsed") or 
        "Center (Default)"
    )
    af_text = str(raw_af).strip()
    
    # ดึงพิกัด (X,Y) จำลอง
    cx, cy = get_nikon_af_coordinates(af_text, img_w, img_h)
    return af_text, cx, cy

def extract_exif_data(file_path: str) -> dict:
    try:
        with ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)[0]
            
            make = metadata.get("EXIF:Make", "")
            model = metadata.get("EXIF:Model", "")
            camera = f"{make} {model}".strip() or "ไม่ระบุ"
            
            lens = get_accurate_lens_name(metadata)
            
            # ค่า AF Fine-tune
            af_fine_tune = (
                metadata.get("MakerNotes:AFTuneValue") or
                metadata.get("Nikon:AFTuneValue") or
                metadata.get("Canon:AFMicroAdjustment") or 0
            )

            exposure = metadata.get("EXIF:ExposureTime")
            if isinstance(exposure, (int, float)):
                exposure_str = f"1/{round(1/exposure)}" if 0 < exposure < 1 else str(exposure)
            else:
                exposure_str = str(exposure) if exposure else "N/A"

            return {
                "metadata": metadata,
                "camera": camera,
                "lens": lens,
                "af_fine_tune": str(af_fine_tune),
                "f_number": metadata.get("EXIF:FNumber", "N/A"),
                "exposure_time": exposure_str,
                "iso": metadata.get("EXIF:ISO", 100),
                "focal_length": metadata.get("EXIF:FocalLength", "N/A")
            }
    except Exception as e:
        logger.error(f"ExifTool Execution Error: {e}")
        return {
            "metadata": {},
            "camera": "ไม่พบข้อมูล (ExifTool Error)",
            "lens": "ไม่พบข้อมูล (ExifTool Error)",
            "af_fine_tune": "0",
            "f_number": "N/A",
            "exposure_time": "N/A",
            "iso": 100,
            "focal_length": "N/A"
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

    # ดึงพิกัดที่ผ่านการคำนวณ Mapping 51 จุดแล้ว
    af_point_text, cx, cy = parse_af_point(metadata, w, h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # กรอบ Focus ROI ขนาด 15% ของภาพ
    box_size = min(w, h) // 7
    x1, x2 = max(0, cx - box_size), min(w, cx + box_size)
    y1, y2 = max(0, cy - box_size), min(h, cy + box_size)
    
    roi = denoised_gray[y1:y2, x1:x2]
    roi_score = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
    full_score = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()

    raw_sharpness = (roi_score * 0.8) + (full_score * 0.2)

    try:
        iso_num = float(iso_val)
    except:
        iso_num = 100.0

    noise_penalty_factor = 1.0 + max(0.0, (iso_num - 100.0) / 1600.0)
    normalized_score = round(float(raw_sharpness / noise_penalty_factor), 2)

    if normalized_score >= 1200:
        quality = "🟢 คมชัดสูงมาก"
        advice = "โฟกัสแม่นยำ เลนส์ทำงานเต็มประสิทธิภาพ"
    elif normalized_score >= 400:
        quality = "🟡 คมชัดปานกลาง"
        advice = "ยอมรับได้ ตรวจสอบ AF Fine-tune หากต้องการคมชัดขึ้น"
    else:
        quality = "🔴 ซอฟต์ / เบลอ"
        advice = "โฟกัสคลาดเคลื่อน หรือมีความสั่นไหว"

    # วาดเป้าพิกัดบนภาพ
    annotated_img = img.copy()
    box_color = (0, 255, 0) if normalized_score >= 400 else (0, 0, 255)
    
    # วาดกรอบและจุดกึ่งกลางเป้า
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 4)
    cv2.drawMarker(annotated_img, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    # แถบพื้นหลังข้อความให้อ่านง่ายขึ้น
    label = f"AF Point: {af_point_text} | Score: {normalized_score}"
    cv2.rectangle(annotated_img, (10, 10), (w - 10, 50), (0, 0, 0), -1)
    cv2.putText(annotated_img, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return normalized_score, quality, advice, annotated_b64, af_point_text

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
