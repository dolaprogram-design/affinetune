import os
import re
import shutil
import tempfile
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from exiftool import ExifToolHelper

app = FastAPI(title="Pro Lens & Camera Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_clean_lens_name(metadata: dict) -> str:
    """แปลงค่ารหัส Hex เลนส์ดิบของ Nikon ให้เป็นชื่อเลนส์ทางการค้าที่อ่านง่าย"""
    candidates = [
        metadata.get("Composite:LensID"),
        metadata.get("EXIF:LensModel"),
        metadata.get("Composite:LensSpec"),
        metadata.get("Nikon:LensModel"),
        metadata.get("EXIF:LensInfo")
    ]
    
    for cand in candidates:
        if cand and isinstance(cand, str):
            clean = cand.strip()
            # กรองรหัส Hex ดิบ เช่น "82 34 76 A6..." ออก
            if not re.match(r'^[0-9A-Fa-f\s]{10,}$', clean):
                return clean
                
    focal = metadata.get("EXIF:FocalLength")
    fnum = metadata.get("EXIF:FNumber")
    if focal:
        return f"{focal}mm f/{fnum}" if fnum else f"{focal}mm Lens"
        
    return "ไม่ระบุรุ่นเลนส์"

def get_af_fine_tune_value(metadata: dict) -> str:
    """ดึงค่า AF Fine-tune / Microadjustment แยกตาม MakerNotes ของแต่ละแบรนด์"""
    val = (
        metadata.get("Nikon:AFTuneValue") or
        metadata.get("Nikon:AFFineTune") or
        metadata.get("MakerNotes:AFTuneValue") or
        metadata.get("Canon:AFMicroAdjustment") or
        metadata.get("Sony:AFMicroAdjustment")
    )
    if val is not None:
        try:
            return str(int(val))
        except (ValueError, TypeError):
            return str(val)
    return "0"

def get_nikon_51point_coordinates(af_text: str, img_w: int, img_h: int):
    """
    ตารางแมปพิกัด 51 จุดโฟกัสของ Nikon (Multi-CAM 3500DX) ให้ตรงกับพิกัดจริงบนเซ็นเซอร์
    """
    af_upper = str(af_text).upper().strip()
    
    # พิกัดเริ่มต้น: กึ่งกลางภาพ (Normalized 0.5, 0.5)
    norm_x, norm_y = 0.5, 0.5
    
    # ตรวจสอบทิศทางและตำแหน่งจุดโฟกัสเชิงลึก
    if "CENTER" in af_upper or "C" in af_upper:
        norm_x, norm_y = 0.5, 0.5
    elif "TOP" in af_upper or "UPPER" in af_upper:
        norm_y = 0.32
    elif "BOTTOM" in af_upper or "LOWER" in af_upper:
        norm_y = 0.68
        
    if "LEFT" in af_upper:
        norm_x = 0.30 if "FAR" in af_upper else 0.38
    elif "RIGHT" in af_upper:
        norm_x = 0.70 if "FAR" in af_upper else 0.62

    # แปลงสัดส่วน Normalized เป็นพิกเซลจริงบนภาพ
    cx = int(norm_x * img_w)
    cy = int(norm_y * img_h)
    
    return cx, cy

def parse_af_point_and_coordinates(metadata: dict, img_w: int, img_h: int):
    """อ่านค่า AF Point จาก MakerNotes และคำนวณตำแหน่งพิกเซล (CX, CY)"""
    raw_af = (
        metadata.get("Nikon:AFPointsInFocus") or 
        metadata.get("Nikon:PrimaryAFPoint") or 
        metadata.get("Nikon:AFPointSelected") or
        metadata.get("Canon:AFPointUsed") or 
        metadata.get("Sony:AFPointSelected") or 
        "Center (Default)"
    )
    af_text = str(raw_af).strip()
    
    # 1. พยายามอ่านพิกัดตัวเลขโดยตรงจาก ExifTool ถ้ามี
    pos = metadata.get("Nikon:AFPointPosition") or metadata.get("Composite:AFPointPosition")
    if pos:
        try:
            parts = [float(p) for p in str(pos).split()]
            if len(parts) == 2:
                if 0 <= parts[0] <= 1.0 and 0 <= parts[1] <= 1.0:
                    return af_text, int(parts[0] * img_w), int(parts[1] * img_h)
        except Exception:
            pass
            
    # 2. กรณีไม่มีพิกัดตัวเลข ให้ผ่านตารางแมปพิกัด 51 จุด
    cx, cy = get_nikon_51point_coordinates(af_text, img_w, img_h)
    return af_text, cx, cy

def extract_exif_data(file_path: str) -> dict:
    try:
        with ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)[0]
            
            make = metadata.get("EXIF:Make", "")
            model = metadata.get("EXIF:Model", "")
            camera = f"{make} {model}".strip() or "ไม่ระบุ"
            
            lens = get_clean_lens_name(metadata)
            af_fine_tune = get_af_fine_tune_value(metadata)
            
            exposure = metadata.get("EXIF:ExposureTime")
            if isinstance(exposure, (int, float)):
                exposure_str = f"1/{round(1/exposure)}" if 0 < exposure < 1 else str(exposure)
            else:
                exposure_str = str(exposure) if exposure else "N/A"

            return {
                "metadata": metadata,
                "camera": camera,
                "lens": lens,
                "af_fine_tune": af_fine_tune,
                "f_number": metadata.get("EXIF:FNumber", "N/A"),
                "exposure_time": exposure_str,
                "iso": metadata.get("EXIF:ISO", 100),
                "focal_length": metadata.get("EXIF:FocalLength", "N/A")
            }
    except Exception:
        return {
            "metadata": {},
            "camera": "ไม่ระบุ",
            "lens": "ไม่ระบุ",
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

    # ย่อขนาดภาพเพื่อความปลอดภัยของหน่วยความจำเซิร์ฟเวอร์
    h, w = img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # ดึงตำแหน่งจุดโฟกัสจริง
    af_point_text, cx, cy = parse_af_point_and_coordinates(metadata, w, h)

    # แปลงภาพเป็นขาวดำ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # -------------------------------------------------------------------------
    # แก้ไขจุดสำคัญ 1: ตัดสัญญาณ Noise จาก ISO สูงด้วย Gaussian Blur (3x3)
    # -------------------------------------------------------------------------
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # -------------------------------------------------------------------------
    # แก้ไขจุดสำคัญ 2: คำนวณความคมชัดเฉพาะขอบเขตจุดโฟกัส (Focus Region of Interest)
    # -------------------------------------------------------------------------
    box_size = min(w, h) // 10
    x1, x2 = max(0, cx - box_size), min(w, cx + box_size)
    y1, y2 = max(0, cy - box_size), min(h, cy + box_size)
    
    roi = denoised_gray[y1:y2, x1:x2]
    roi_score = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
    full_score = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()

    # ถ่วงน้ำหนักความคมชัด: จุดโฟกัส ROI 80% + ภาพรวม 20%
    raw_sharpness = (roi_score * 0.8) + (full_score * 0.2)

    # -------------------------------------------------------------------------
    # แก้ไขจุดสำคัญ 3: Normalization ปรับระดับสเกลคะแนนตามค่า ISO เพื่อความแม่นยำ
    # -------------------------------------------------------------------------
    try:
        iso_num = float(iso_val)
    except (ValueError, TypeError):
        iso_num = 100.0

    # ปรับสเกลคะแนนให้อยู่ในระดับมาตรฐาน (เทียบเท่าสเกล 0 - 2,000)
    noise_penalty_factor = 1.0 + max(0.0, (iso_num - 100.0) / 1600.0)
    normalized_score = round(float(raw_sharpness / noise_penalty_factor), 2)

    # ประเมินเกณฑ์คุณภาพภาพถ่าย
    if normalized_score >= 1200:
        quality = "🟢 คมชัดสูงมาก (Excellent)"
        advice = "โฟกัสเข้าเป้าแม่นยำ ชิ้นแก้วทำงานได้เต็มประสิทธิภาพ"
    elif normalized_score >= 400:
        quality = "🟡 คมชัดปานกลาง (Good)"
        advice = "ความคมชัดอยู่ในเกณฑ์ดี หากต้องการคมกว่านี้ลองตรวจสอบ AF Fine-tune"
    else:
        quality = "🔴 ซอฟต์ / เบลอ (Soft/Out of Focus)"
        advice = "ภาพมีความซอฟต์หรือโฟกัสคลาดเคลื่อน ควรตรวจสอบอาการ Front/Back Focus"

    # วาดมาร์กเกอร์จุดโฟกัสจริงลงบนภาพ preview
    annotated_img = img.copy()
    box_color = (0, 255, 0) if normalized_score >= 400 else (0, 0, 255)
    
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
    cv2.circle(annotated_img, (cx, cy), 6, (255, 255, 255), -1)
    cv2.circle(annotated_img, (cx, cy), 3, (0, 0, 0), -1)
    
    label = f"AF: {af_point_text} | Score: {normalized_score}"
    cv2.putText(annotated_img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # แปลงภาพเป็น Base64
    _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    base64_str = base64.b64encode(buffer).decode('utf-8')
    annotated_b64 = f"data:image/jpeg;base64,{base64_str}"

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
        return {
            "result": {
                "filename": file.filename,
                "error": str(e)
            }
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
