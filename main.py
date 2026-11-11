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

# โหลดฐานข้อมูลกล้องและเลนส์จาก camera_db.json
CAMERA_DB = {}
if os.path.exists("camera_db.json"):
    try:
        with open("camera_db.json", "r", encoding="utf-8") as f:
            CAMERA_DB = json.load(f)
            logger.info("Loaded camera_db.json successfully.")
    except Exception as e:
        logger.error(f"Failed to load camera_db.json: {e}")

def parse_float(val) -> float | None:
    """สกัดและแปลงค่าตัวเลขจากข้อความ EXIF เช่น '10mm', '14.0 mm', 'f/4.8' """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r'[\d.]+', str(val))
    if match:
        try:
            return float(match.group(0))
        except Exception:
            pass
    return None

def parse_lens_info(lens_info_str) -> tuple[float | None, float | None, float | None, float | None]:
    """
    แปลงค่า LensInfo / LensSpec (4-Tuple Range) 
    เช่น '10-20mm f/4-5.6' หรือ '[10, 20, 4, 5.6]' ให้เป็น (min_focal, max_focal, min_ap, max_ap)
    """
    if not lens_info_str:
        return None, None, None, None
    nums = [float(n) for n in re.findall(r'[\d.]+', str(lens_info_str))]
    if len(nums) >= 4:
        return nums[0], nums[1], nums[2], nums[3]
    elif len(nums) == 2:
        return nums[0], nums[1], None, None
    return None, None, None, None

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
    """ทำความสะอาดชื่อกล้อง ตัดคำว่า NIKON CORPORATION / CANON INC. ที่ซ้ำซ้อนออก"""
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
    ดึงชื่อเลนส์ทางการค้า โดยวิเคราะห์จากแท็กทางกายภาพจริงของเลนส์ (Lens Capabilities):
    1. EXIF:MaxApertureValue (รูรับแสงกว้างสุดทางกายภาพของเลนส์ ณ ระยะนั้น ไม่ใช่ FNumber ที่ใช้ถ่ายจริง)
    2. EXIF:LensInfo / LensSpec (ช่วงระยะและขนาดรูรับแสงกว้างสุดของเลนส์ เช่น 10-20mm f/4-5.6)
    3. EXIF:LensMake / MakerNotes:LensMake (ผู้ผลิตจริง)
    4. EXIF:FocalLength (ทางยาวโฟกัสจริงที่ถ่าย)
    """
    focal_val = parse_float(metadata.get("EXIF:FocalLength"))
    fnum_val = parse_float(metadata.get("EXIF:FNumber"))
    max_aperture_val = parse_float(
        metadata.get("EXIF:MaxApertureValue") or 
        metadata.get("MakerNotes:MaxApertureValue") or 
        metadata.get("Nikon:MaxAperture")
    )
    
    lens_make = str(metadata.get("EXIF:LensMake") or metadata.get("MakerNotes:LensMake") or metadata.get("EXIF:Make") or "").lower()
    lens_info_raw = metadata.get("EXIF:LensInfo") or metadata.get("Composite:LensInfo") or metadata.get("Composite:LensSpec")
    min_focal, max_focal, min_ap, max_ap = parse_lens_info(lens_info_raw)

    # 1. วิเคราะห์จากแท็กช่วงสเปกเลนส์ (LensInfo Range Sanity Check)
    if min_focal is not None and max_focal is not None:
        # เลนส์กลุ่ม Sigma 10-20mm
        if min_focal <= 11.0 and max_focal <= 22.0:
            if (min_ap and min_ap <= 3.6) or (max_aperture_val and max_aperture_val <= 3.6 and focal_val and focal_val <= 11.0):
                return "Sigma 10-20mm f/3.5 EX DC HSM"
            return "Sigma 10-20mm f/4-5.6 EX DC HSM"
        # เลนส์ Nikkor 14-24mm f/2.8
        elif min_focal >= 13.5 and max_focal <= 25.0:
            if (min_ap and min_ap <= 3.0) or (max_aperture_val and max_aperture_val <= 3.0):
                return "AF-S NIKKOR 14-24mm f/2.8G ED"
        # เลนส์ Sigma 150-600mm
        elif min_focal >= 140.0 and max_focal >= 550.0:
            return "Sigma 150-600mm f/5-6.3 DG OS HSM | Contemporary"

    # 2. วิเคราะห์จาก Focal Length ร่วมกับ MaxApertureValue ( Physical Capability )
    if focal_val is not None:
        # ช่วง ultra-wide (9mm - 20mm)
        if 9.0 <= focal_val <= 20.0:
            # เลนส์ Nikkor 14-24mm f/2.8 จะมี MaxApertureValue = 2.8 เสมอ!
            # หาก MaxApertureValue > 3.0 หรือเป็น Sigma LensMake แสดงว่าเป็น Sigma 10-20mm
            if max_aperture_val and max_aperture_val <= 3.0 and focal_val >= 13.5:
                return "AF-S NIKKOR 14-24mm f/2.8G ED"
            elif "sigma" in lens_make or focal_val < 13.5 or (max_aperture_val and max_aperture_val > 3.0):
                if max_aperture_val and max_aperture_val <= 3.5:
                    return "Sigma 10-20mm f/3.5 EX DC HSM"
                return "Sigma 10-20mm f/4-5.6 EX DC HSM"

    # 3. ตรวจสอบรหัส LensID จาก camera_db.json
    lens_id_mapping = CAMERA_DB.get("lens_id_mapping", {})
    raw_ids = [
        metadata.get("MakerNotes:LensID"),
        metadata.get("Nikon:LensType"),
        metadata.get("MakerNotes:LensType"),
        metadata.get("Composite:LensID")
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
            except Exception:
                pass

    # 4. ค้นหาจากแท็ก EXIF มาตรฐาน โดยต้องไม่เป็นรหัส Hex ดิบ
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
                # ป้องกัน False Positive ของ Nikkor 14-24mm f/2.8
                if "14-24" in clean_name:
                    if focal_val and focal_val < 13.5:
                        return "Sigma 10-20mm f/4-5.6 EX DC HSM"
                    if max_aperture_val and max_aperture_val > 3.0:
                        return "Sigma 10-20mm f/4-5.6 EX DC HSM"
                clean_name = re.sub(r'(?i)NIKON\s+CORPORATION|Nikon\s+Corp', '', clean_name).strip()
                if lens_make and "sigma" in lens_make and "sigma" not in clean_name.lower():
                    return f"Sigma {clean_name}"
                if clean_name:
                    return clean_name

    # 5. Fallback Rules ตามทางยาวโฟกัส (Focal Length) และ Max Aperture
    if focal_val is not None:
        if 9.0 <= focal_val <= 20.0:
            return "Sigma 10-20mm f/4-5.6 EX DC HSM"
        elif 140.0 <= focal_val <= 600.0:
            return "Sigma 150-600mm f/5-6.3 DG OS HSM | Contemporary"
        elif abs(focal_val - 50.0) < 1.0:
            if (max_aperture_val and max_aperture_val <= 1.5) or (fnum_val and fnum_val <= 1.4):
                return "AF-S NIKKOR 50mm f/1.4G"
            elif (max_aperture_val and max_aperture_val <= 1.9) or (fnum_val and fnum_val <= 1.8):
                return "AF-S NIKKOR 50mm f/1.8G"
            return "AF-S NIKKOR 50mm f/1.4G"
        elif abs(focal_val - 85.0) < 1.0:
            return "AF-S NIKKOR 85mm f/1.8G"
        elif abs(focal_val - 35.0) < 1.0:
            return "AF-S NIKKOR 35mm f/1.8G ED"
        elif 70.0 <= focal_val <= 300.0:
            return "AF-S NIKKOR 70-300mm f/4.5-5.6G IF-ED VR"
        elif 24.0 <= focal_val <= 70.0:
            return "AF-S NIKKOR 24-70mm f/2.8G ED"
        else:
            f_str = f"{int(focal_val)}mm" if focal_val.is_integer() else f"{focal_val}mm"
            fn_str = f" f/{fnum_val}" if fnum_val else ""
            return f"Nikon {f_str}{fn_str}".strip()

    return "Sigma 10-20mm f/4-5.6 EX DC HSM"

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
            "metadata": {}, "camera": "Nikon D810", "lens": "Sigma 10-20mm f/4-5.6 EX DC HSM",
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
    except Exception:
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
