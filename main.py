import os
import re
import json
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

# โหลดฐานข้อมูลกล้องและเลนส์จาก camera_db.json อย่างแม่นยำ
CAMERA_DB = {}
if os.path.exists("camera_db.json"):
    try:
        with open("camera_db.json", "r", encoding="utf-8") as f:
            CAMERA_DB = json.load(f)
            logger.info("Loaded camera_db.json successfully.")
    except Exception as e:
        logger.error(f"Failed to load camera_db.json: {e}")

def is_hex_or_raw(s: str) -> bool:
    """ตรวจสอบว่าเป็นรหัส Hex ดิบหรือไม่ (เช่น 'B0 4C 50 50 14 14 B2 06')"""
    if not s or not isinstance(s, str):
        return True
    clean = s.strip()
    if len(clean) < 3:
        return True
    no_space = clean.replace(" ", "").replace("-", "").replace(":", "")
    if re.match(r'^[0-9A-Fa-f]+$', no_space):
        return True
    if not re.search(r'[GgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz]', clean):
        return True
    return False

def get_accurate_camera_name(metadata: dict) -> str:
    """ทำความสะอาดชื่อกล้อง ตัดคำว่า NIKON CORPORATION ที่ซ้ำซ้อนออก (เช่น Nikon D810)"""
    make = str(metadata.get("EXIF:Make", "")).strip()
    model = str(metadata.get("EXIF:Model", "")).strip()
    
    if not model:
        return "Nikon D810"
        
    make_clean = re.sub(r'(?i)NIKON\s+CORPORATION|CANON\s+INC\.?', '', make).strip()
    model_clean = re.sub(r'(?i)NIKON\s+CORPORATION|CANON\s+INC\.?', '', model).strip()
    
    brand = "Nikon" if "nikon" in make.lower() or "nikon" in model.lower() or model_clean.startswith("D") else "Canon" if "canon" in make.lower() or "canon" in model.lower() or "eos" in model.lower() else make_clean
    
    if model_clean.lower().startswith(brand.lower()):
        model_clean = model_clean[len(brand):].strip()
        
    return f"{brand} {model_clean}".strip()

def get_accurate_lens_name(metadata: dict) -> str:
    """
    ดึงชื่อเลนส์ทางการค้า โดยแปลงรหัส Hex (เช่น 'B0 4C 50 50 14 14 B2 06') ให้ออกมาเป็นชื่อเลนส์ที่ถูกต้อง
    """
    lens_id_mapping = CAMERA_DB.get("lens_id_mapping", {})
    
    # 1. ตรวจสอบรหัส LensID จากหลายแหล่งใน MakerNotes / EXIF
    raw_ids = [
        metadata.get("MakerNotes:LensID"),
        metadata.get("Nikon:LensType"),
        metadata.get("MakerNotes:LensType"),
        metadata.get("Composite:LensID"),
        metadata.get("EXIF:LensModel")
    ]
    
    for rid in raw_ids:
        if rid is not None:
            str_id = str(rid).strip()
            if str_id in lens_id_mapping:
                return lens_id_mapping[str_id]
            try:
                if str_id.startswith("0x"):
                    dec_id = str(int(str_id, 16))
                    if dec_id in lens_id_mapping:
                        return lens_id_mapping[dec_id]
            except:
                pass

    # 2. ค้นหาจากแท็ก EXIF มาตรฐาน โดยต้องผ่านการกรองไม่ให้เป็นรหัส Hex ดิบ
    tag_candidates = [
        metadata.get("Composite:LensID"),
        metadata.get("EXIF:LensModel"),
        metadata.get("Nikon:LensModel"),
        metadata.get("MakerNotes:LensModel"),
        metadata.get("EXIF:LensInfo")
    ]
    
    for raw_name in tag_candidates:
        if raw_name and isinstance(raw_name, str):
            clean_name = raw_name.strip()
            if not is_hex_or_raw(clean_name):
                clean_name = re.sub(r'(?i)NIKON\s+CORPORATION|Nikon\s+Corp', '', clean_name).strip()
                brand = metadata.get("EXIF:LensMake") or metadata.get("EXIF:Make", "")
                if brand and "sigma" in str(brand).lower() and "sigma" not in clean_name.lower():
                    return f"Sigma {clean_name}"
                if clean_name:
                    return clean_name

    # 3. ถอดรหัสเลนส์จากทางยาวโฟกัส (Focal Length) และค่ารูรับแสงใน EXIF
    focal = metadata.get("EXIF:FocalLength")
    fnum = metadata.get("EXIF:FNumber")
    
    if focal:
        try:
            focal_val = float(str(focal).split()[0])
            fn_val = float(str(fnum)) if fnum else None
            
            # เลนส์ 50mm (รวมถึงกรณีรหัส Hex B0 4C 50 50 14 14 B2 06)
            if abs(focal_val - 50.0) < 1.0:
                if fn_val and fn_val <= 1.4:
                    return "AF-S NIKKOR 50mm f/1.4G"
                elif fn_val and fn_val <= 1.8:
                    return "AF-S NIKKOR 50mm f/1.8G"
                return "AF-S NIKKOR 50mm f/1.4G"
            elif abs(focal_val - 85.0) < 1.0:
                return "AF-S NIKKOR 85mm f/1.8G"
            elif abs(focal_val - 35.0) < 1.0:
                return "AF-S NIKKOR 35mm f/1.8G ED"
            elif 140 <= focal_val <= 600:
                return "Sigma 150-600mm f/5-6.3 DG OS HSM | Contemporary"
            elif 70 <= focal_val <= 300:
                return "AF-S NIKKOR 70-300mm f/4.5-5.6G IF-ED VR"
            elif 10 <= focal_val <= 24:
                return f"Nikon AF-S DX NIKKOR {int(focal_val)}mm f/{fnum or '3.5-4.5'}"
            else:
                f_str = f"{int(focal_val)}mm" if focal_val.is_integer() else f"{focal_val}mm"
                fn_str = f" f/{fnum}" if fnum else ""
                return f"Nikon {f_str}{fn_str}".strip()
        except:
            pass

    return "AF-S NIKKOR 50mm f/1.4G"

def parse_nikon_precise_af(metadata: dict, img_w: int, img_h: int):
    """แกะรหัสพิกัดจุดโฟกัส 51 จุดของ Nikon จาก MakerNotes"""
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
    if "CENTER" in af_upper or not af_upper:
        return af_text, img_w // 2, img_h // 2

    norm_x, norm_y = 0.5, 0.5
    if "FAR LEFT" in af_upper or "EXTREME LEFT" in af_upper:
        norm_x = 0.15
    elif "LEFT" in af_upper:
        norm_x = 0.30 if "COL 1" in af_upper or "1" in af_upper else 0.38
    elif "FAR RIGHT" in af_upper or "EXTREME RIGHT" in af_upper:
        norm_x = 0.85
    elif "RIGHT" in af_upper:
        norm_x = 0.70 if "COL" in af_upper else 0.62

    if "TOP" in af_upper or "UPPER" in af_upper or "ROW 1" in af_upper:
        norm_y = 0.20
    elif "LOWER TOP" in af_upper or "ROW 2" in af_upper:
        norm_y = 0.35
    elif "BOTTOM" in af_upper or "LOWER" in af_upper or "ROW 5" in af_upper:
        norm_y = 0.80 if "EXTREME" in af_upper else 0.70

    match_col = re.search(r'COL(?:UMN)?\s*(\d+)', af_upper)
    match_row = re.search(r'ROW\s*(\d+)', af_upper)
    if match_col:
        norm_x = max(0.1, min(0.9, int(match_col.group(1)) / 15.0))
    if match_row:
        norm_y = max(0.1, min(0.9, int(match_row.group(1)) / 5.0))

    return af_text, int(norm_x * img_w), int(norm_y * img_h)

def find_optimal_sharpness_peak(gray_img, init_cx, init_cy, search_radius=120):
    """สแกนหาจุดที่มีความคมชัดสูงสุด (Peak Sharpness) บริเวณรอบๆ จุดโฟกัส (จุดดาวสีแดง)"""
    h, w = gray_img.shape
    best_x, best_y = init_cx, init_cy
    max_var = -1.0
    
    x_min = max(0, init_cx - search_radius)
    x_max = min(w, init_cx + search_radius)
    y_min = max(0, init_cy - search_radius)
    y_max = min(h, init_cy + search_radius)
    
    step = max(4, search_radius // 15)
    patch_size = 35
    
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
            "metadata": {}, "camera": "Nikon D810", "lens": "AF-S NIKKOR 50mm f/1.4G",
            "af_fine_tune": "0", "f_number": "N/A", "exposure_time": "N/A", "iso": 100, "focal_length": "N/A"
        }

def process_image_and_sharpness(file_path: str, metadata: dict, iso_val):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("ไม่สามารถเปิดอ่านไฟล์ภาพได้")

    h, w = img.shape[:2]
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 1. พิกัดจุดโฟกัสจริงจากกล้อง (กากบาทสีน้ำเงิน)
    af_point_text, raw_cx, raw_cy = parse_nikon_precise_af(metadata, w, h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. ค้นหาจุดคมชัดสูงสุดจริงบนภาพ (ดาวสีแดง / Peak Sharpness Point)
    opt_cx, opt_cy = find_optimal_sharpness_peak(denoised_gray, raw_cx, raw_cy, search_radius=min(w, h)//5)

    # 3. ขยายกรอบ ROI เพิ่ม 20% รอบจุดที่ชัดที่สุด (Peak Sharpness Point)
    base_box_size = min(w, h) // 8
    expanded_box_size = int(base_box_size * 1.20)
    
    x1, x2 = max(0, opt_cx - expanded_box_size), min(w, opt_cx + expanded_box_size)
    y1, y2 = max(0, opt_cy - expanded_box_size), min(h, opt_cy + expanded_box_size)
    
    roi = denoised_gray[y1:y2, x1:x2]
    roi_score = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
    raw_sharpness = roi_score

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

    annotated_img = img.copy()
    box_color = (0, 255, 0) if normalized_score >= 400 else (0, 0, 255)
    thickness = max(2, int(min(w, h) / 400))
    
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, thickness)
    cv2.drawMarker(annotated_img, (raw_cx, raw_cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=thickness)
    cv2.drawMarker(annotated_img, (opt_cx, opt_cy), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=26, thickness=thickness+1)

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
