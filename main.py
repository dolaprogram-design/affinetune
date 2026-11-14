import os
import re
import json
import shutil
import tempfile
import base64
import logging
import subprocess
import cv2
import numpy as np
from typing import Optional, Tuple
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from exiftool import ExifToolHelper
except ImportError:
    ExifToolHelper = None

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

# ฐานข้อมูลแบรนด์และรุ่นกล้องที่รองรับ
SUPPORTED_BRANDS = ["Nikon", "Canon", "Sigma", "Tamron", "Samyang", "Tokina"]
SUPPORTED_MODELS = [
    "NIKON D810",
    "NIKON D850",
    "NIKON D750",
    "Canon EOS 5D Mark IV",
    "Canon EOS 6D Mark II",
]


def parse_float(val) -> Optional[float]:
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


def format_num_clean(val: Optional[float]) -> str:
    if val is None:
        return ""
    rounded = round(val, 1)
    return str(int(rounded)) if rounded.is_integer() else str(rounded)


def normalize_lens_brand(name: str) -> str:
    if not name:
        return ""
    clean = re.sub(r'(?i)\s*(CORPORATION|INC\.?|CO\.,\s*LTD\.?)\s*', ' ', name).strip()
    clean = re.sub(r'\s+', ' ', clean)

    # ปรับตัวพิมพ์ใหญ่-เล็กของชื่อยี่ห้อให้ตรงตามฐานข้อมูล
    for brand in SUPPORTED_BRANDS:
        if re.search(rf'{re.escape(brand)}', clean, re.IGNORECASE):
            clean = re.sub(rf'{re.escape(brand)}', brand, clean, flags=re.IGNORECASE)

    # เติมชื่อแบรนด์ด้านหน้าหาก EXIF ระบุเฉพาะชื่อรุ่นเลนส์ (เช่น AF-S Nikkor หรือ EF)
    has_brand = any(re.search(rf'{re.escape(b)}', clean, re.IGNORECASE) for b in SUPPORTED_BRANDS)
    if not has_brand:
        if re.match(r'^(AF-S|AF-P|AF|AI-S|AI|NIKKOR|PC-E|1 NIKKOR)', clean, re.IGNORECASE):
            clean = f"Nikon {clean}"
        elif re.match(r'^(EF|EF-S|EF-M|RF|RF-S|TS-E|MP-E)', clean, re.IGNORECASE):
            clean = f"Canon {clean}"

    return clean.strip()


def format_brand_casing(name: str) -> str:
    if not name:
        return ""
    for brand in SUPPORTED_BRANDS:
        if re.search(rf'{re.escape(brand)}', name, re.IGNORECASE):
            name = re.sub(rf'{re.escape(brand)}', brand, name, flags=re.IGNORECASE)
    return name.strip()


def is_invalid_lens_tag(val) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if len(s) < 3:
        return True
    s_lower = s.lower()
    if s_lower in ("n/a", "none", "unknown", "manual lens", "no lens", "----", "0.0 mm f/0.0", "0mm f/0"):
        return True
    if s_lower.startswith("unknown (") or s_lower.startswith("unknown lens"):
        return True
    # ปฏิเสธค่าที่เป็นรหัสฐานสิบหกหรือตัวเลขล้วน เช่น "168 42 54 92 14 14 78 6" หรือ "A8 48 3C 8E"
    no_space = s.replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    if re.match(r'^[0-9A-Fa-f]+$', no_space) and not re.search(r'[G-Zg-z]', s):
        return True
    return False


def matches_focal_range(lens_str: str, focal_val: Optional[float]) -> bool:
    if not focal_val or focal_val <= 0:
        return True
    # ตรวจสอบช่วงซูม เช่น "24-70mm", "150-600 mm", "24.0-85.0 mm"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*mm', lens_str, re.IGNORECASE)
    if m:
        min_f, max_f = float(m.group(1)), float(m.group(2))
        if min_f > 0 and max_f >= min_f:
            # เผื่อระยะกรณีต่อ Teleconverter 1.4x หรือ 2.0x
            return (min_f * 0.85) <= focal_val <= (max_f * 2.1)
    # ตรวจสอบเลนส์ฟิกซ์ เช่น "50mm", "85 mm"
    m_prime = re.search(r'(\d+(?:\.\d+)?)\s*mm', lens_str, re.IGNORECASE)
    if m_prime:
        p_f = float(m_prime.group(1))
        if p_f > 0:
            return (p_f * 0.85) <= focal_val <= (p_f * 2.1)
    return True


def get_accurate_camera_name(metadata: dict) -> str:
    make = metadata.get("EXIF:Make") or metadata.get("Make") or metadata.get("MakerNotes:Make") or ""
    model = metadata.get("EXIF:Model") or metadata.get("Model") or metadata.get("MakerNotes:Model") or ""

    make_clean = re.sub(r'(?i)\s*(CORPORATION|INC\.?|CO\.,\s*LTD\.?)\s*', '', str(make)).strip()
    model_clean = re.sub(r'(?i)\s*(CORPORATION|INC\.?|CO\.,\s*LTD\.?)\s*', '', str(model)).strip()

    if not make_clean and not model_clean:
        return "Nikon D810"
    if not make_clean:
        return format_brand_casing(model_clean)
    if not model_clean:
        return format_brand_casing(make_clean)

    if model_clean.lower().startswith(make_clean.lower()):
        return format_brand_casing(model_clean)

    return format_brand_casing(f"{make_clean} {model_clean}")


def get_accurate_lens_name(metadata: dict) -> str:
    focal_val = parse_float(metadata.get("EXIF:FocalLength") or metadata.get("FocalLength"))

    # 1. ตรวจสอบแท็ก XMP/EXIF LensModel ที่มีชื่อแบรนด์ระบุชัดเจน (เช่น โปรไฟล์เลนส์จาก Lightroom/ACR)
    for xmp_key in ["XMP:Lens", "XMP:LensModel", "EXIF:LensModel", "LensModel"]:
        val = metadata.get(xmp_key)
        if not is_invalid_lens_tag(val):
            val_str = str(val).strip()
            if any(re.search(rf'{re.escape(b)}', val_str, re.IGNORECASE) for b in SUPPORTED_BRANDS):
                if matches_focal_range(val_str, focal_val):
                    return normalize_lens_brand(val_str)

    # 2. ตรวจสอบแท็ก LensID ที่ ExifTool ถอดรหัสชื่อรุ่นเลนส์จริงออกมาแล้ว
    tag_candidates = [
        metadata.get("Composite:LensID"),
        metadata.get("LensID"),
        metadata.get("MakerNotes:LensID"),
        metadata.get("EXIF:LensModel"),
        metadata.get("LensModel"),
        metadata.get("XMP:Lens"),
        metadata.get("Composite:LensSpec"),
        metadata.get("MakerNotes:Lens"),
        metadata.get("EXIF:Lens"),
    ]

    for cand in tag_candidates:
        if not is_invalid_lens_tag(cand):
            cand_str = str(cand).strip()
            if matches_focal_range(cand_str, focal_val):
                return normalize_lens_brand(cand_str)

    # 3. กรณีเป็นเลนส์รุ่นใหม่ที่ยังไม่ระบุชื่อรุ่นในฐานข้อมูล ให้สร้างสเปกเลนส์จริงจาก MakerNotes / LensInfo
    min_f = parse_float(metadata.get("MakerNotes:MinFocalLength") or metadata.get("MinFocalLength"))
    max_f = parse_float(metadata.get("MakerNotes:MaxFocalLength") or metadata.get("MaxFocalLength"))
    min_a = parse_float(metadata.get("MakerNotes:MaxApertureAtMinFocal") or metadata.get("MaxApertureAtMinFocal"))
    max_a = parse_float(metadata.get("MakerNotes:MaxApertureAtMaxFocal") or metadata.get("MaxApertureAtMaxFocal"))

    if not (min_f and max_f):
        lens_info = metadata.get("EXIF:LensInfo") or metadata.get("LensInfo")
        if lens_info:
            nums = [float(n) for n in re.findall(r'[\d.]+', str(lens_info))]
            if len(nums) >= 4:
                min_f, max_f, min_a, max_a = nums[0], nums[1], nums[2], nums[3]
            elif len(nums) >= 2:
                min_f, max_f = nums[0], nums[1]

    if min_f and max_f and min_f > 0 and max_f > 0:
        f_str = (
            f"{format_num_clean(min_f)}-{format_num_clean(max_f)}mm"
            if abs(min_f - max_f) > 0.1
            else f"{format_num_clean(min_f)}mm"
        )
        a_str = ""
        if min_a and max_a and min_a > 0 and max_a > 0:
            a_str = (
                f" f/{format_num_clean(min_a)}-{format_num_clean(max_a)}"
                if abs(min_a - max_a) > 0.05
                else f" f/{format_num_clean(min_a)}"
            )
        elif min_a and min_a > 0:
            a_str = f" f/{format_num_clean(min_a)}"
        lens_type = str(metadata.get("MakerNotes:LensType") or "").strip()
        type_suffix = f" ({lens_type})" if lens_type and lens_type.lower() not in ("n/a", "none") else ""
        return f"Lens {f_str}{a_str}{type_suffix}".strip()

    # 4. Fallback สุดท้าย (เช่น เลนส์มือหมุนรุ่นเก่าที่ไม่ส่งค่าช่วงซูมให้กล้อง)
    if focal_val and focal_val > 0:
        max_ap = parse_float(metadata.get("EXIF:MaxApertureValue") or metadata.get("MaxApertureValue"))
        f_str = f"{format_num_clean(focal_val)}mm"
        a_str = f" f/{format_num_clean(max_ap)}" if max_ap and max_ap > 0 else ""
        return f"Unknown Lens {f_str}{a_str}".strip()

    return "N/A"


def load_image_robust(file_path: str) -> np.ndarray:
    img = cv2.imread(file_path)
    if img is not None:
        return img
    try:
        for tag in ["-JpgFromRaw", "-PreviewImage", "-ThumbnailImage"]:
            cmd = ["exiftool", "-b", tag, file_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and len(res.stdout) > 2000:
                nparr = np.frombuffer(res.stdout, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return decoded
    except Exception as e:
        logger.error(f"RAW Preview Extraction Failed: {e}")
    raise ValueError("ไม่สามารถเปิดอ่านไฟล์ภาพได้")


def parse_nikon_precise_af(metadata: dict, img_w: int, img_h: int):
    raw_af = (
        metadata.get("Nikon:PrimaryAFPoint")
        or metadata.get("MakerNotes:PrimaryAFPoint")
        or metadata.get("PrimaryAFPoint")
        or metadata.get("Nikon:AFPointsInFocus")
        or metadata.get("MakerNotes:AFPointsInFocus")
        or metadata.get("Nikon:AFPointSelected")
        or "Center"
    )
    af_text = str(raw_af).strip()

    pos = (
        metadata.get("Nikon:AFPointPosition")
        or metadata.get("Composite:AFPointPosition")
        or metadata.get("AFPointPosition")
    )
    if pos:
        try:
            pos_str = str(pos).strip()
            parts = re.findall(r'[\d.]+', pos_str)
            if len(parts) >= 2:
                px, py = float(parts[0]), float(parts[1])
                if "%" in pos_str:
                    px /= 100.0
                    py /= 100.0
                if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                    return af_text, int(px * img_w), int(py * img_h)
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
            patch = gray_img[
                max(0, y - patch_size) : min(h, y + patch_size),
                max(0, x - patch_size) : min(w, x + patch_size),
            ]
            if patch.size > 0:
                var = cv2.Laplacian(patch, cv2.CV_64F).var()
                if var > max_var:
                    max_var = var
                    best_x, best_y = x, y

    return best_x, best_y


def read_exiftool_metadata(file_path: str) -> dict:
    # รัน exiftool โดยเปิด Print Conversion (ไม่ใส่ -n) เพื่อให้ถอดรหัส Composite:LensID เป็นชื่อรุ่นเลนส์จริง
    try:
        cmd = ["exiftool", "-j", "-G", "-charset", "utf8", file_path]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, list) and len(data) > 0:
                return data[0]
    except Exception as e:
        logger.warning(f"Subprocess exiftool fallback triggered: {e}")

    if ExifToolHelper is not None:
        try:
            with ExifToolHelper() as et:
                return et.get_metadata(file_path)[0]
        except Exception as e:
            logger.error(f"ExifToolHelper Error: {e}")

    return {}


def extract_exif_data(file_path: str) -> dict:
    try:
        metadata = read_exiftool_metadata(file_path)
        camera = get_accurate_camera_name(metadata)
        lens = get_accurate_lens_name(metadata)

        af_fine_tune = (
            metadata.get("MakerNotes:AFTuneValue")
            or metadata.get("Nikon:AFTuneValue")
            or metadata.get("AFFineTuneAdj")
            or 0
        )

        exposure = metadata.get("EXIF:ExposureTime") or metadata.get("ExposureTime")
        if isinstance(exposure, (int, float)):
            exposure_str = f"1/{round(1 / exposure)}" if 0 < exposure < 1 else str(exposure)
        elif exposure is not None:
            exposure_str = re.sub(r'\s*s(?:ec)?\.?$', '', str(exposure).strip(), flags=re.IGNORECASE)
        else:
            exposure_str = "N/A"

        # ส่งค่าตัวเลขรูรับแสงแบบสะอาด (ไม่ซ้ำ f/f/) เพื่อให้ตรงกับ template หน้าเว็บ
        f_num = metadata.get("EXIF:FNumber") or metadata.get("FNumber")
        if f_num is not None:
            f_clean = re.sub(r'^f/', '', str(f_num).strip(), flags=re.IGNORECASE)
            f_number_str = f_clean if f_clean else "N/A"
        else:
            f_number_str = "N/A"

        focal_raw = metadata.get("EXIF:FocalLength") or metadata.get("FocalLength")
        focal_val = parse_float(focal_raw)
        focal_str = format_num_clean(focal_val) if focal_val is not None else "N/A"

        return {
            "metadata": metadata,
            "camera": camera,
            "lens": lens,
            "af_fine_tune": str(af_fine_tune),
            "f_number": f_number_str,
            "exposure_time": exposure_str,
            "iso": metadata.get("EXIF:ISO") or metadata.get("ISO") or 100,
            "focal_length": focal_str,
        }
    except Exception as e:
        logger.error(f"ExifTool Error: {e}")
        return {
            "metadata": {},
            "camera": "Nikon D810",
            "lens": "N/A",
            "af_fine_tune": "0",
            "f_number": "N/A",
            "exposure_time": "N/A",
            "iso": 100,
            "focal_length": "N/A",
        }


def process_image_and_sharpness(file_path: str, metadata: dict, iso_val):
    img = load_image_robust(file_path)

    h, w = img.shape[:2]
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    af_point_text, raw_cx, raw_cy = parse_nikon_precise_af(metadata, w, h)
    opt_cx, opt_cy = find_optimal_sharpness_peak(denoised_gray, raw_cx, raw_cy, search_radius=min(w, h) // 5)

    faces = []
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(face_cascade_path):
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(denoised_gray, scaleFactor=1.03, minNeighbors=2, minSize=(15, 15))

    if len(faces) == 0:
        profile_cascade_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        if os.path.exists(profile_cascade_path):
            profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
            faces = profile_cascade.detectMultiScale(denoised_gray, scaleFactor=1.03, minNeighbors=2, minSize=(15, 15))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        fx, fy, fw, fh = faces[0]

        roi = denoised_gray[fy : fy + fh, fx : fx + fw]
        raw_sharpness = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0

        x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
        opt_cx, opt_cy = fx + fw // 2, fy + fh // 2
        composition_summary = "📸 องค์ประกอบภาพ: ตรวจพบตัวบุคคล (วัดความคมชัดจากบริเวณใบหน้าหลัก)"
    else:
        base_box_size = min(w, h) // 8
        expanded_box_size = int(base_box_size * 1.20)

        x1, x2 = max(0, opt_cx - expanded_box_size), min(w, opt_cx + expanded_box_size)
        y1, y2 = max(0, opt_cy - expanded_box_size), min(h, opt_cy + expanded_box_size)

        roi = denoised_gray[y1:y2, x1:x2]
        raw_sharpness = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
        composition_summary = f"🎯 องค์ประกอบภาพ: ไม่มีตัวบุคคล (วัดความคมชัดอ้างอิงพิกัดกล้อง: {af_point_text})"

    try:
        iso_num = float(iso_val)
    except Exception:
        iso_num = 100.0

    noise_penalty = 1.0 + max(0.0, (iso_num - 100.0) / 1600.0)
    normalized_score = round(float(raw_sharpness / noise_penalty), 2)

    if normalized_score >= 1200:
        quality = "🟢 คมชัดสูงมาก"
        base_advice = "โฟกัสเข้าเป้าแม่นยำ ตำแหน่งคมชัดตรงจุด"
    elif normalized_score >= 400:
        quality = "🟡 คมชัดปานกลาง"
        base_advice = "อยู่ในเกณฑ์ดี สามารถปรับจูน AF Fine-tune เพิ่มเติมได้"
    else:
        quality = "🔴 ซอฟต์ / เบลอ"
        base_advice = "โฟกัสคลาดเคลื่อน หรือมีความสั่นไหวขณะบันทึกภาพ"

    advice = f"{composition_summary} | {base_advice}"

    annotated_img = img.copy()
    box_color = (0, 255, 0) if normalized_score >= 400 else (0, 0, 255)
    thickness = max(2, int(min(w, h) / 400))

    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, thickness)
    if len(faces) == 0:
        cv2.drawMarker(
            annotated_img,
            (raw_cx, raw_cy),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=thickness,
        )
    cv2.drawMarker(
        annotated_img,
        (opt_cx, opt_cy),
        (0, 0, 255),
        markerType=cv2.MARKER_STAR,
        markerSize=26,
        thickness=thickness + 1,
    )

    _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
                "annotated_image": annotated_b64,
            }
        }
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return {"result": {"filename": file.filename, "error": str(e)}}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
