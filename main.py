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

app = FastAPI(title="Lens Calibration & Camera Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_clean_lens_name(metadata: dict) -> str:
    """
    แปลงค่าเลนส์จาก ExifTool ให้เป็นชื่อทางการค้า 
    กรองรหัส Hex ดิบของ Nikon (เช่น '82 34 76 A6...') ออก
    """
    candidates = [
        metadata.get("Composite:LensID"),
        metadata.get("EXIF:LensModel"),
        metadata.get("Composite:LensSpec"),
        metadata.get("Nikon:LensModel"),
        metadata.get("EXIF:LensInfo")
    ]
    
    for cand in candidates:
        if cand and isinstance(cand, str):
            clean_cand = cand.strip()
            # เช็กว่าไม่ใช่รหัส Hex ดิบ เช่น "82 34 76 A6 38 40 4B 0E"
            if not re.match(r'^[0-9A-Fa-f\s]{10,}$', clean_cand):
                return clean_cand
                
    # กรณีดึงชื่อเลนส์ไม่ได้ ให้สร้างชื่อชั่วคราวจากทางยาวโฟกัสและรูรับแสง
    focal = metadata.get("EXIF:FocalLength")
    fnum = metadata.get("EXIF:FNumber")
    if focal:
        return f"{focal}mm f/{fnum}" if fnum else f"{focal}mm Lens"
        
    return "ไม่ระบุรุ่นเลนส์"

def get_af_fine_tune_value(metadata: dict) -> str:
    """
    ดึงค่า AF Fine-tune / Microadjustment แยกตามค่ายกล้อง (Nikon, Canon, Sony)
    """
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

def parse_af_point_and_coordinates(metadata: dict, img_w: int, img_h: int):
    """
    ดึงจุดโฟกัสเชิงลึกและคำนวณพิกัด (X, Y) ที่แท้จริงบนภาพ
    """
    raw_af = (
        metadata.get("Nikon:AFPointsInFocus") or 
        metadata.get("Nikon:PrimaryAFPoint") or 
        metadata.get("Nikon:AFPointSelected") or
        metadata.get("Canon:AFPointUsed") or 
        metadata.get("Sony:AFPointSelected") or 
        "Center"
    )
    af_text = str(raw_af).strip()
    
    # ค่า Default คือจุดศูนย์กลางภาพ
    cx, cy = img_w // 2, img_h // 2
    
    # ตรวจสอบว่า ExifTool คืนค่าพิกัดจุดโฟกัสมาให้หรือไม่ (Nikon AFPointPosition)
    pos = metadata.get("Nikon:AFPointPosition") or metadata.get("Composite:AFPointPosition")
    
    if pos:
        try:
            parts = [float(p) for p in str(pos).split()]
            if len(parts) == 2:
                # กรณีเป็นสัดส่วน (0.0 - 1.0)
                if 0 <= parts[0] <= 1.0 and 0 <= parts[1] <= 1.0:
                    cx = int(parts[0] * img_w)
                    cy = int(parts[1] * img_h)
                # กรณีเป็นพิกัดพิกเซลจริงเทียบกับขนาดเซ็นเซอร์
                elif parts[0] > 1 and parts[1] > 1:
                    af_w = metadata.get("Nikon:AFImageWidth") or img_w
                    af_h = metadata.get("Nikon:AFImageHeight") or img_h
                    cx = int((parts[0] / af_w) * img_w)
                    cy = int((parts[1] / af_h) * img_h)
        except Exception:
            pass
    else:
        # คำนวณตำแหน่งแบบประมาณการตามชื่อทิศทางจุดโฟกัส (กรณีไม่มีพิกัดตัวเลข)
        af_lower = af_text.lower()
        if "left" in af_lower:
            cx = int(img_w * 0.3)
        elif "right" in af_lower:
            cx = int(img_w * 0.7)
            
        if "top" in af_lower or "upper" in af_lower:
            cy = int(img_h * 0.3)
        elif "bottom" in af_lower or "lower" in af_lower:
            cy = int(img_h * 0.7)

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
                "iso": metadata.get("EXIF:ISO", "N/A"),
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
            "iso": "N/A",
            "focal_length": "N/A"
        }

def process_image_and_sharpness(file_path: str, metadata: dict):
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

    # คำนวณจุดโฟกัสและพิกัด (X, Y) จริงบนภาพ
    af_point_text, cx, cy = parse_af_point_and_coordinates(metadata, w, h)

    # ตัดคำนวณความคมชัดเฉพาะบริเวณรอบจุดโฟกัส (ROI) ร่วมกับภาพรวม
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box_size = min(w, h) // 8
    
    x1, x2 = max(0, cx - box_size), min(w, cx + box_size)
    y1, y2 = max(0, cy - box_size), min(h, cy + box_size)
    
    roi = gray[y1:y2, x1:x2]
    roi_sharpness = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
    full_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # คำนวณคะแนนความคมชัดถ่วงน้ำหนัก (เน้นพื้นที่จุดโฟกัส 70% + ภาพรวม 30%)
    sharpness_score = round(float((roi_sharpness * 0.7) + (full_sharpness * 0.3)), 2)

    # ประเมินระดับคุณภาพ
    if sharpness_score >= 1500:
        quality = "🟢 คมชัดสูงมาก (Excellent)"
        advice = "โฟกัสเข้าเป้าแม่นยำ ชิ้นแก้วทำงานได้เต็มประสิทธิภาพ"
    elif sharpness_score >= 500:
        quality = "🟡 คมชัดปานกลาง (Good)"
        advice = "ความคมชัดอยู่ในเกณฑ์ยอมรับได้ หากต้องการคมกว่านี้ลองปรับจูน AF Fine-tune"
    else:
        quality = "🔴 ซอฟต์ / เบลอ (Soft/Out of Focus)"
        advice = "ภาพมีความซอฟต์หรือโฟกัสคลาดเคลื่อน ควรตรวจสอบอาการ Front/Back Focus"

    # วาดมาร์กเกอร์ตรงพิกัดจุดโฟกัสจริง
    annotated_img = img.copy()
    box_color = (0, 255, 0) if sharpness_score >= 500 else (0, 0, 255)
    
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
    cv2.circle(annotated_img, (cx, cy), 6, (255, 255, 255), -1)
    cv2.circle(annotated_img, (cx, cy), 3, (0, 0, 0), -1)
    
    label = f"AF: {af_point_text} | Score: {sharpness_score}"
    cv2.putText(annotated_img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # แปลงภาพเป็น Base64 ส่งกลับไปยังหน้าต่าง Modal
    _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    base64_str = base64.b64encode(buffer).decode('utf-8')
    annotated_b64 = f"data:image/jpeg;base64,{base64_str}"

    return sharpness_score, quality, advice, annotated_b64, af_point_text

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
            temp_path, exif_info["metadata"]
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
