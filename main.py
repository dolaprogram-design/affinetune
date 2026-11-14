# Pro Lens & Camera Optical Sharpness & Quality Analytics API v2.1 (Production Ready)
import os
import re
import gc
import json
import math
import asyncio
import shutil
import tempfile
import base64
import logging
import subprocess
import cv2
import numpy as np
from typing import Optional, Tuple, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from exiftool import ExifToolHelper
except ImportError:
    ExifToolHelper = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ขีดจำกัดเพื่อป้องกันหน่วยความจำเต็ม (Render 512MB - 2GB RAM Safe Guard)
MAX_FILE_SIZE_MB = 60
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_RECOMMENDED_BATCH = 10
MAX_SESSION_ITEMS = 20

# ประมวลผลภาพทีละ 1 ไฟล์บนเซิร์ฟเวอร์ เพื่อป้องกัน Peak RAM พุ่งสูงเกินขีดจำกัดเมื่ออัปโหลดหลายไฟล์
ANALYSIS_SEMAPHORE = asyncio.Semaphore(1)

app = FastAPI(
    title="Pro Lens & Camera Optical Sharpness & Quality Analytics API",
    version="2.1.0",
    description="ระบบวิเคราะห์ความคมชัดเชิงแสง สอบเทียบจุดโฟกัส และประเมินคุณภาพภาพถ่ายอัตโนมัติ (RAW & JPEG)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ฐานข้อมูลแบรนด์และรุ่นกล้องที่รองรับ
SUPPORTED_BRANDS = ["Nikon", "Canon", "Sigma", "Tamron", "Samyang", "Tokina", "Sony", "Fujifilm"]
SUPPORTED_MODELS = [
    "NIKON D810",
    "NIKON D850",
    "NIKON D750",
    "Canon EOS 5D Mark IV",
    "Canon EOS 6D Mark II",
]


def get_tag_value(metadata: dict, candidate_keys: List[str]):
    """ดึงค่าจาก metadata โดยไม่ข้ามค่า 0 หรือ False และรองรับทั้งแบบมีและไม่มี Group prefix"""
    for key in candidate_keys:
        if key in metadata and metadata[key] is not None and str(metadata[key]).strip() != "":
            return metadata[key]
    for key in candidate_keys:
        suffix = f":{key.split(':')[-1]}"
        for m_key, m_val in metadata.items():
            if m_key.endswith(suffix) and m_val is not None and str(m_val).strip() != "":
                return m_val
    return None


def parse_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(val))
    if match:
        try:
            f = float(match.group(0))
            return None if (math.isnan(f) or math.isinf(f)) else f
        except Exception:
            pass
    return None


def parse_exposure_seconds(val) -> Optional[float]:
    """แปลงค่า Shutter Speed (เช่น '1/400', '1/60 s', 0.0025, '2') เป็นวินาที (float)"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if f > 0 else None
    s = re.sub(r'\s*s(?:ec)?\.?$', '', str(val).strip(), flags=re.IGNORECASE)
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            num = parse_float(parts[0])
            den = parse_float(parts[1])
            if num is not None and den is not None and den != 0:
                res = num / den
                return res if res > 0 else None
    return parse_float(s)


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

    for brand in SUPPORTED_BRANDS:
        if re.search(rf'\b{re.escape(brand)}\b', clean, re.IGNORECASE):
            clean = re.sub(rf'\b{re.escape(brand)}\b', brand, clean, flags=re.IGNORECASE)

    has_brand = any(re.search(rf'\b{re.escape(b)}\b', clean, re.IGNORECASE) for b in SUPPORTED_BRANDS)
    if not has_brand:
        if re.match(r'^(AF-S|AF-P|AF|AI-S|AI|NIKKOR|PC-E|1 NIKKOR)\b', clean, re.IGNORECASE):
            clean = f"Nikon {clean}"
        elif re.match(r'^(EF|EF-S|EF-M|RF|RF-S|TS-E|MP-E)\b', clean, re.IGNORECASE):
            clean = f"Canon {clean}"

    return clean.strip()


def format_brand_casing(name: str) -> str:
    if not name:
        return ""
    for brand in SUPPORTED_BRANDS:
        if re.search(rf'\b{re.escape(brand)}\b', name, re.IGNORECASE):
            name = re.sub(rf'\b{re.escape(brand)}\b', brand, name, flags=re.IGNORECASE)
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
    no_space = s.replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    if re.match(r'^[0-9A-Fa-f]+$', no_space) and not re.search(r'[G-Zg-z]', s):
        return True
    return False


def matches_focal_range(lens_str: str, focal_val: Optional[float]) -> bool:
    if not focal_val or focal_val <= 0:
        return True
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*mm', lens_str, re.IGNORECASE)
    if m:
        min_f, max_f = float(m.group(1)), float(m.group(2))
        if min_f > 0 and max_f >= min_f:
            return (min_f * 0.85) <= focal_val <= (max_f * 2.1)
    m_prime = re.search(r'\b(\d+(?:\.\d+)?)\s*mm\b', lens_str, re.IGNORECASE)
    if m_prime:
        p_f = float(m_prime.group(1))
        if p_f > 0:
            return (p_f * 0.85) <= focal_val <= (p_f * 2.1)
    return True


def get_accurate_camera_name(metadata: dict) -> str:
    make = get_tag_value(metadata, ["EXIF:Make", "Make", "MakerNotes:Make"]) or ""
    model = get_tag_value(metadata, ["EXIF:Model", "Model", "MakerNotes:Model"]) or ""

    make_clean = re.sub(r'(?i)\s*(CORPORATION|INC\.?|CO\.,\s*LTD\.?)\s*', '', str(make)).strip()
    model_clean = re.sub(r'(?i)\s*(CORPORATION|INC\.?|CO\.,\s*LTD\.?)\s*', '', str(model)).strip()

    if not make_clean and not model_clean:
        return "Unknown Camera"
    if not make_clean:
        return format_brand_casing(model_clean)
    if not model_clean:
        return format_brand_casing(make_clean)

    if model_clean.lower().startswith(make_clean.lower()):
        return format_brand_casing(model_clean)

    return format_brand_casing(f"{make_clean} {model_clean}")


def get_accurate_lens_name(metadata: dict) -> str:
    focal_val = parse_float(get_tag_value(metadata, ["EXIF:FocalLength", "FocalLength"]))

    for xmp_key in ["XMP:Lens", "XMP:LensModel", "EXIF:LensModel", "LensModel"]:
        val = get_tag_value(metadata, [xmp_key])
        if not is_invalid_lens_tag(val):
            val_str = str(val).strip()
            if any(re.search(rf'\b{re.escape(b)}\b', val_str, re.IGNORECASE) for b in SUPPORTED_BRANDS):
                if matches_focal_range(val_str, focal_val):
                    return normalize_lens_brand(val_str)

    tag_candidates = [
        get_tag_value(metadata, ["Composite:LensID", "LensID"]),
        get_tag_value(metadata, ["MakerNotes:LensID"]),
        get_tag_value(metadata, ["EXIF:LensModel", "LensModel"]),
        get_tag_value(metadata, ["XMP:Lens"]),
        get_tag_value(metadata, ["Composite:LensSpec"]),
        get_tag_value(metadata, ["MakerNotes:Lens", "EXIF:Lens"]),
    ]

    for cand in tag_candidates:
        if not is_invalid_lens_tag(cand):
            cand_str = str(cand).strip()
            if matches_focal_range(cand_str, focal_val):
                return normalize_lens_brand(cand_str)

    min_f = parse_float(get_tag_value(metadata, ["MakerNotes:MinFocalLength", "MinFocalLength"]))
    max_f = parse_float(get_tag_value(metadata, ["MakerNotes:MaxFocalLength", "MaxFocalLength"]))
    min_a = parse_float(get_tag_value(metadata, ["MakerNotes:MaxApertureAtMinFocal", "MaxApertureAtMinFocal"]))
    max_a = parse_float(get_tag_value(metadata, ["MakerNotes:MaxApertureAtMaxFocal", "MaxApertureAtMaxFocal"]))

    if not (min_f and max_f):
        lens_info = get_tag_value(metadata, ["EXIF:LensInfo", "LensInfo"])
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
        lens_type = str(get_tag_value(metadata, ["MakerNotes:LensType"]) or "").strip()
        type_suffix = f" ({lens_type})" if lens_type and lens_type.lower() not in ("n/a", "none") else ""
        return f"Lens {f_str}{a_str}{type_suffix}".strip()

    if focal_val and focal_val > 0:
        max_ap = parse_float(get_tag_value(metadata, ["EXIF:MaxApertureValue", "MaxApertureValue"]))
        f_str = f"{format_num_clean(focal_val)}mm"
        a_str = f" f/{format_num_clean(max_ap)}" if max_ap and max_ap > 0 else ""
        return f"Unknown Lens {f_str}{a_str}".strip()

    return "N/A"


def extract_af_fine_tune(metadata: dict) -> str:
    """สกัดค่า AF Fine-tune / AF Microadjustment จริงจาก MakerNotes ของไฟล์ภาพ"""
    af_status = get_tag_value(metadata, ["MakerNotes:AFFineTune", "Nikon:AFFineTune", "AFFineTune"])
    af_adj = get_tag_value(metadata, [
        "MakerNotes:AFFineTuneAdj",
        "Nikon:AFFineTuneAdj",
        "MakerNotes:AFTuneValue",
        "Nikon:AFTuneValue",
        "AFFineTuneAdj",
        "AFTuneValue",
    ])
    af_index = get_tag_value(metadata, ["MakerNotes:AFFineTuneIndex", "Nikon:AFFineTuneIndex", "AFFineTuneIndex"])

    canon_mode = get_tag_value(metadata, ["MakerNotes:AFMicroAdjMode", "Canon:AFMicroAdjMode", "AFMicroAdjMode"])
    canon_val = get_tag_value(metadata, ["MakerNotes:AFMicroAdjValue", "Canon:AFMicroAdjValue", "AFMicroAdjValue"])

    generic_adj = get_tag_value(metadata, ["MakerNotes:AFMicroAdj", "XMP:AFFineTune", "Composite:AFFineTune"])

    def fmt_signed(v) -> str:
        s = str(v).strip()
        try:
            num = int(round(float(s)))
            if num > 0:
                return f"+{num}"
            return str(num)
        except Exception:
            return s

    if af_adj is not None or af_status is not None:
        status_str = str(af_status).strip() if af_status is not None else ""
        adj_str = fmt_signed(af_adj) if af_adj is not None else "0"
        if status_str.lower().startswith("off"):
            if adj_str != "0":
                return f"Off (ค่าที่บันทึก: {adj_str})"
            return "Off (0)"
        elif status_str.lower().startswith("on"):
            idx_part = f" [Saved #{af_index}]" if af_index is not None and str(af_index).strip() not in ("", "0") else ""
            return f"On ({adj_str}){idx_part}"
        elif af_adj is not None:
            return adj_str

    if canon_val is not None or canon_mode is not None:
        mode_str = str(canon_mode).strip() if canon_mode is not None else ""
        val_str = fmt_signed(canon_val) if canon_val is not None else "0"
        if "disable" in mode_str.lower():
            return f"Off ({val_str})" if val_str != "0" else "Off (0)"
        elif canon_val is not None:
            return f"{val_str} ({mode_str})" if mode_str else val_str

    if generic_adj is not None:
        return fmt_signed(generic_adj)

    return "ไม่พบข้อมูลในไฟล์"


def extract_deep_makernotes(metadata: dict) -> dict:
    """สกัดข้อมูล MakerNotes เชิงลึกจากไฟล์ RAW / JPEG"""
    focus_mode = get_tag_value(metadata, ["MakerNotes:FocusMode", "Nikon:FocusMode", "Canon:FocusMode", "FocusMode"])
    af_area_mode = get_tag_value(metadata, ["MakerNotes:AFAreaMode", "Nikon:AFAreaMode", "Canon:AFAreaMode", "AFAreaMode"])
    focus_dist_raw = get_tag_value(metadata, ["MakerNotes:FocusDistance", "Nikon:FocusDistance", "Composite:FocusDistance", "FocusDistance"])
    roll_raw = get_tag_value(metadata, ["MakerNotes:RollAngle", "Nikon:RollAngle", "RollAngle"])
    pitch_raw = get_tag_value(metadata, ["MakerNotes:PitchAngle", "Nikon:PitchAngle", "PitchAngle"])
    white_balance = get_tag_value(metadata, ["EXIF:WhiteBalance", "MakerNotes:WhiteBalance", "WhiteBalance"])
    metering_mode = get_tag_value(metadata, ["EXIF:MeteringMode", "MeteringMode"])
    exp_comp = get_tag_value(metadata, ["EXIF:ExposureCompensation", "ExposureCompensation"])

    focus_dist_val = parse_float(focus_dist_raw)
    roll_val = parse_float(roll_raw)
    pitch_val = parse_float(pitch_raw)

    return {
        "focus_mode": str(focus_mode).strip() if focus_mode is not None else "N/A",
        "af_area_mode": str(af_area_mode).strip() if af_area_mode is not None else "N/A",
        "focus_distance_m": round(focus_dist_val, 2) if focus_dist_val is not None else None,
        "focus_distance_text": f"{round(focus_dist_val, 2)} m" if focus_dist_val is not None else (str(focus_dist_raw).strip() if focus_dist_raw is not None else "N/A"),
        "roll_angle": round(roll_val, 1) if roll_val is not None else None,
        "pitch_angle": round(pitch_val, 1) if pitch_val is not None else None,
        "white_balance": str(white_balance).strip() if white_balance is not None else "N/A",
        "metering_mode": str(metering_mode).strip() if metering_mode is not None else "N/A",
        "exposure_comp": str(exp_comp).strip() if exp_comp is not None else "0",
    }


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


def parse_camera_af_point(metadata: dict, img_w: int, img_h: int) -> Tuple[str, int, int]:
    """อ่านพิกัดจุดโฟกัสจริงที่กล้องบันทึกไว้ใน EXIF/MakerNotes"""
    raw_af = get_tag_value(metadata, [
        "MakerNotes:PrimaryAFPoint",
        "Nikon:PrimaryAFPoint",
        "PrimaryAFPoint",
        "MakerNotes:AFPointsInFocus",
        "Nikon:AFPointsInFocus",
        "MakerNotes:AFPointSelected",
        "Nikon:AFPointSelected",
        "Canon:AFPointsInFocus",
    ])
    af_text = str(raw_af).strip() if raw_af is not None else "Center"

    af_img_w = parse_float(get_tag_value(metadata, ["MakerNotes:AFImageWidth", "AFImageWidth"]))
    af_img_h = parse_float(get_tag_value(metadata, ["MakerNotes:AFImageHeight", "AFImageHeight"]))
    af_area_x = parse_float(get_tag_value(metadata, ["MakerNotes:AFAreaXPosition", "AFAreaXPosition"]))
    af_area_y = parse_float(get_tag_value(metadata, ["MakerNotes:AFAreaYPosition", "AFAreaYPosition"]))

    if af_img_w and af_img_h and af_area_x is not None and af_area_y is not None:
        if af_img_w > 0 and af_img_h > 0:
            if 0 <= af_area_x <= af_img_w and 0 <= af_area_y <= af_img_h:
                nx = af_area_x / af_img_w
                ny = af_area_y / af_img_h
                return af_text, int(nx * img_w), int(ny * img_h)
            elif abs(af_area_x) <= af_img_w / 2 and abs(af_area_y) <= af_img_h / 2:
                nx = 0.5 + (af_area_x / af_img_w)
                ny = 0.5 - (af_area_y / af_img_h)
                return af_text, int(nx * img_w), int(ny * img_h)

    pos = get_tag_value(metadata, ["Composite:AFPointPosition", "Nikon:AFPointPosition", "AFPointPosition"])
    if pos is not None:
        try:
            pos_str = str(pos).strip()
            parts = re.findall(r'[-+]?\d+(?:\.\d+)?', pos_str)
            if len(parts) >= 2:
                px, py = float(parts[0]), float(parts[1])
                if "%" in pos_str or px > 1.0 or py > 1.0:
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
        norm_y = 0.22
    elif "LOWER TOP" in af_upper or "ROW 2" in af_upper:
        norm_y = 0.36
    elif "BOTTOM" in af_upper or "LOWER" in af_upper or "ROW 5" in af_upper:
        norm_y = 0.78 if "EXTREME" in af_upper else 0.68

    match_col = re.search(r'COL(?:UMN)?\s*(\d+)', af_upper)
    match_row = re.search(r'ROW\s*(\d+)', af_upper)
    if match_col:
        norm_x = max(0.1, min(0.9, int(match_col.group(1)) / 15.0))
    if match_row:
        norm_y = max(0.1, min(0.9, int(match_row.group(1)) / 5.0))

    return af_text, int(norm_x * img_w), int(norm_y * img_h)


def compute_patch_sharpness(gray_img: np.ndarray, cx: int, cy: int, half_size: int) -> float:
    """คำนวณค่าความคมชัดเชิงแสง (Laplacian Variance) ในกรอบพื้นที่ที่กำหนด"""
    h, w = gray_img.shape
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size)
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size)
    patch = gray_img[y1:y2, x1:x2]
    if patch.size < 100:
        return 0.0
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def describe_offset_direction(dx: int, dy: int, diag: float) -> str:
    dist = math.hypot(dx, dy)
    if dist / diag <= 0.03:
        return "ตรงกับจุดโฟกัสกล้อง"
    dirs = []
    if dy < -diag * 0.02:
        dirs.append("บน")
    elif dy > diag * 0.02:
        dirs.append("ล่าง")
    if dx < -diag * 0.02:
        dirs.append("ซ้าย")
    elif dx > diag * 0.02:
        dirs.append("ขวา")
    return "เยื้องไปทาง" + "-".join(dirs) if dirs else "ใกล้จุดโฟกัสกล้อง"


def extract_highres_crop_b64(
    orig_img: np.ndarray,
    center_x_work: int,
    center_y_work: int,
    scale: float,
    max_crop_px: int = 860,
    jpeg_quality: int = 96,
) -> str:
    """ตัดภาพรายละเอียด 100% จากภาพต้นฉบับความละเอียดสูง โดยคงความคมชัดสูงสุด"""
    orig_h, orig_w = orig_img.shape[:2]
    orig_cx = int(center_x_work / scale)
    orig_cy = int(center_y_work / scale)
    orig_half = max(240, min(orig_w, orig_h) // 6)

    x1 = max(0, orig_cx - orig_half)
    x2 = min(orig_w, orig_cx + orig_half)
    y1 = max(0, orig_cy - orig_half)
    y2 = min(orig_h, orig_cy + orig_half)

    crop = orig_img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = orig_img

    ch, cw = crop.shape[:2]
    if max(ch, cw) > max_crop_px:
        c_scale = max_crop_px / float(max(ch, cw))
        crop = cv2.resize(crop, (int(cw * c_scale), int(ch * c_scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"


def compute_texture_and_histogram(
    orig_img: np.ndarray,
    work_img: np.ndarray,
    gray_img: np.ndarray,
    cx: int,
    cy: int,
    half_box: int,
    scale: float,
) -> dict:
    """คำนวณ Edge Density (Canny), Sobel Gradient, Contrast, Histogram 3 ช่องสี และภาพ ROI ความละเอียดสูง"""
    h, w = gray_img.shape
    x1 = max(0, cx - half_box)
    x2 = min(w, cx + half_box)
    y1 = max(0, cy - half_box)
    y2 = min(h, cy + half_box)

    roi_gray = gray_img[y1:y2, x1:x2]
    roi_bgr = work_img[y1:y2, x1:x2]

    if roi_gray.size == 0:
        roi_gray = gray_img
        roi_bgr = work_img

    # 1. Edge Density (%) ผ่าน Otsu + Canny
    high_thresh, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(roi_gray, 0.5 * high_thresh, high_thresh)
    edge_density_pct = round(float((np.sum(canny > 0) / max(1, canny.size)) * 100.0), 1)

    # 2. Mean Gradient Strength (Sobel)
    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_gradient = round(float(np.mean(np.hypot(gx, gy))), 1)

    # 3. Contrast (Std Dev) & Mean Brightness
    contrast_std = round(float(np.std(roi_gray)), 1)
    mean_brightness = round(float(np.mean(roi_gray)), 1)

    # 4. Exposure Clipping (ตรวจสอบทั้งเฟรม)
    total_pixels = max(1, gray_img.size)
    highlight_clipping_pct = round(float((np.sum(gray_img >= 250) / total_pixels) * 100.0), 2)
    shadow_crushing_pct = round(float((np.sum(gray_img <= 5) / total_pixels) * 100.0), 2)

    # 5. RGB Histogram (64 bins normalized % สำหรับแสดงกราฟสวยงามและน้ำหนักเบา)
    bins = 64
    hist_r = cv2.calcHist([roi_bgr], [2], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([roi_bgr], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([roi_bgr], [0], None, [bins], [0, 256]).flatten()
    roi_px = max(1, roi_gray.size)
    hist_data = {
        "bins": [int(i * 4) for i in range(bins)],
        "r": [round(float(v / roi_px * 100.0), 2) for v in hist_r],
        "g": [round(float(v / roi_px * 100.0), 2) for v in hist_g],
        "b": [round(float(v / roi_px * 100.0), 2) for v in hist_b],
    }

    # 6. ภาพตัวอย่าง ROI และ Canny Edge Map ความละเอียดสูง (560x560px จากภาพต้นฉบับ)
    orig_h, orig_w = orig_img.shape[:2]
    ox1 = max(0, int(x1 / scale))
    ox2 = min(orig_w, int(x2 / scale))
    oy1 = max(0, int(y1 / scale))
    oy2 = min(orig_h, int(y2 / scale))
    orig_roi = orig_img[oy1:oy2, ox1:ox2] if (ox2 > ox1 and oy2 > oy1) else roi_bgr

    interp_roi = cv2.INTER_AREA if orig_roi.shape[0] >= 560 else cv2.INTER_LANCZOS4
    roi_disp = cv2.resize(orig_roi, (560, 560), interpolation=interp_roi) if orig_roi.shape[0] > 0 else roi_bgr
    canny_disp = cv2.resize(canny, (560, 560), interpolation=cv2.INTER_NEAREST) if canny.shape[0] > 0 else canny

    _, roi_buf = cv2.imencode(".jpg", roi_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    _, canny_buf = cv2.imencode(".png", canny_disp)

    af_roi_b64 = f"data:image/jpeg;base64,{base64.b64encode(roi_buf).decode('utf-8')}"
    canny_b64 = f"data:image/png;base64,{base64.b64encode(canny_buf).decode('utf-8')}"

    del gx, gy, canny, roi_disp, canny_disp
    return {
        "edge_density_pct": edge_density_pct,
        "mean_gradient": mean_gradient,
        "contrast_std": contrast_std,
        "mean_brightness": mean_brightness,
        "highlight_clipping_pct": highlight_clipping_pct,
        "shadow_crushing_pct": shadow_crushing_pct,
        "rgb_histogram": hist_data,
        "af_roi_image": af_roi_b64,
        "canny_edge_image": canny_b64,
    }


def evaluate_photo_quality(
    af_sharpness: float,
    peak_sharpness: float,
    sharpness_ratio_pct: float,
    focus_offset_pct: float,
    edge_density_pct: float,
    contrast_std: float,
    highlight_clipping_pct: float,
    shadow_crushing_pct: float,
    iso_num: float,
    exposure_sec: Optional[float],
    focal_val: Optional[float],
    roll_angle: Optional[float],
) -> dict:
    """ระบบให้คะแนนคุณภาพภาพถ่ายแบบถ่วงน้ำหนัก (Composite Quality Scoring) และจำแนกสาเหตุภาพเสีย"""

    # 1. Focus & Sharpness Index (น้ำหนัก 50%)
    base_focus_score = min(100.0, (af_sharpness / 1250.0) * 85.0 + (15.0 if af_sharpness >= 1250 else 0.0))
    if af_sharpness >= 600:
        base_focus_score = max(base_focus_score, min(96.0, 68.0 + ((af_sharpness - 600.0) / 800.0) * 28.0))

    if edge_density_pct < 6.0 and 250 <= af_sharpness < 750:
        base_focus_score = min(100.0, base_focus_score + 12.0)

    if focus_offset_pct > 6.0 and sharpness_ratio_pct < 85.0:
        shift_penalty = min(32.0, (100.0 - sharpness_ratio_pct) * 0.55)
        base_focus_score = max(0.0, base_focus_score - shift_penalty)

    focus_index = round(base_focus_score, 1)

    # 2. Contrast & Exposure Index (น้ำหนัก 30%)
    contrast_score = 100.0
    if contrast_std < 35.0:
        contrast_score -= (35.0 - contrast_std) * 1.4
    elif contrast_std > 82.0:
        contrast_score -= (contrast_std - 82.0) * 0.8

    if highlight_clipping_pct > 3.0:
        contrast_score -= min(35.0, (highlight_clipping_pct - 3.0) * 2.5)
    if shadow_crushing_pct > 8.0:
        contrast_score -= min(25.0, (shadow_crushing_pct - 8.0) * 1.2)

    contrast_exposure_index = round(max(0.0, min(100.0, contrast_score)), 1)

    # 3. Technical Stability Index (น้ำหนัก 20%)
    stability_score = 100.0
    shutter_risk = False
    recommended_shutter_str = "N/A"
    if focal_val and focal_val > 0:
        safe_sec = 1.0 / focal_val
        recommended_shutter_str = f"1/{int(round(focal_val))}s"
        if exposure_sec and exposure_sec > safe_sec * 1.5:
            shutter_risk = True
            ratio_slow = exposure_sec / safe_sec
            raw_pen = min(35.0, math.log2(max(1.0, ratio_slow)) * 12.0)
            if af_sharpness >= 850:
                raw_pen *= 0.3
            stability_score -= raw_pen

    if iso_num > 400:
        if iso_num <= 1600:
            stability_score -= ((iso_num - 400.0) / 1200.0) * 10.0
        else:
            stability_score -= 10.0 + min(30.0, ((iso_num - 1600.0) / 4800.0) * 30.0)

    if roll_angle is not None and abs(roll_angle) > 4.0:
        stability_score -= min(6.0, (abs(roll_angle) - 4.0) * 1.0)

    stability_index = round(max(0.0, min(100.0, stability_score)), 1)

    # คะแนนคุณภาพรวม (Composite Quality Score)
    quality_score = round(
        (0.50 * focus_index) + (0.30 * contrast_exposure_index) + (0.20 * stability_index),
        1,
    )

    if quality_score >= 85.0:
        grade_code = "A"
        grade_label = "Grade A — ดีเยี่ยม (พร้อมใช้งาน)"
    elif quality_score >= 70.0:
        grade_code = "B"
        grade_label = "Grade B — ดี (ผ่านมาตรฐาน)"
    elif quality_score >= 50.0:
        grade_code = "C"
        grade_label = "Grade C — ปานกลาง (ควรตรวจสอบ)"
    else:
        grade_code = "D"
        grade_label = "Reject / Grade D — ไม่ผ่านเกณฑ์"

    # จำแนกสาเหตุข้อบกพร่อง (Issue Classification)
    issues = []
    if peak_sharpness < 350 and af_sharpness < 350:
        if shutter_risk:
            issues.append("💨 ภาพสั่นไหว (Motion Blur / Shutter ช้ากว่าทางยาวโฟกัส)")
        else:
            issues.append("🔴 หลุดโฟกัสทั้งเฟรม (Out-of-Focus)")
    elif focus_offset_pct > 6.0 and sharpness_ratio_pct < 85.0:
        issues.append("🟠 โฟกัสคลาดเคลื่อน (Focus Shift / Front-Back Focus)")

    if iso_num >= 3200:
        issues.append(f"📶 สัญญาณรบกวนสูง (ISO {int(iso_num)})")
    if highlight_clipping_pct > 5.0:
        issues.append(f"☀️ แสงจ้าทะลุ (Highlight Clipping {highlight_clipping_pct}%)")
    if shadow_crushing_pct > 12.0:
        issues.append(f"🌑 เงามืดจม (Shadow Crushing {shadow_crushing_pct}%)")
    if contrast_std < 30.0:
        issues.append(f"🌫️ คอนทราสต์ต่ำ/ภาพแบน (Contrast {contrast_std})")
    if roll_angle is not None and abs(roll_angle) > 4.0:
        issues.append(f"📐 ระนาบกล้องเอียง ({roll_angle}°)")

    if not issues:
        issues.append("✅ สมบูรณ์ตามมาตรฐาน (ไม่พบข้อบกพร่อง)")

    return {
        "quality_score": quality_score,
        "grade_code": grade_code,
        "grade_label": grade_label,
        "focus_index": focus_index,
        "contrast_exposure_index": contrast_exposure_index,
        "stability_index": stability_index,
        "recommended_shutter": recommended_shutter_str,
        "issues_detected": issues,
    }


def read_exiftool_metadata(file_path: str) -> dict:
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
        af_fine_tune = extract_af_fine_tune(metadata)
        deep_notes = extract_deep_makernotes(metadata)

        exposure = get_tag_value(metadata, ["EXIF:ExposureTime", "ExposureTime"])
        exposure_sec = parse_exposure_seconds(exposure)
        if isinstance(exposure, (int, float)):
            exposure_str = f"1/{round(1 / exposure)}" if 0 < exposure < 1 else str(exposure)
        elif exposure is not None:
            exposure_str = re.sub(r'\s*s(?:ec)?\.?$', '', str(exposure).strip(), flags=re.IGNORECASE)
        else:
            exposure_str = "N/A"

        f_num = get_tag_value(metadata, ["EXIF:FNumber", "FNumber"])
        if f_num is not None:
            f_clean = re.sub(r'^f/', '', str(f_num).strip(), flags=re.IGNORECASE)
            f_number_str = f_clean if f_clean else "N/A"
        else:
            f_number_str = "N/A"

        focal_raw = get_tag_value(metadata, ["EXIF:FocalLength", "FocalLength"])
        focal_val = parse_float(focal_raw)
        focal_str = format_num_clean(focal_val) if focal_val is not None else "N/A"

        iso_val = get_tag_value(metadata, ["EXIF:ISO", "ISO"])
        if iso_val is None:
            iso_val = 100

        return {
            "metadata": metadata,
            "camera": camera,
            "lens": lens,
            "af_fine_tune": af_fine_tune,
            "f_number": f_number_str,
            "exposure_time": exposure_str,
            "exposure_sec": exposure_sec,
            "iso": iso_val,
            "focal_length": focal_str,
            "focal_val": focal_val,
            "deep_notes": deep_notes,
        }
    except Exception as e:
        logger.error(f"ExifTool Error: {e}")
        return {
            "metadata": {},
            "camera": "Unknown Camera",
            "lens": "N/A",
            "af_fine_tune": "ไม่พบข้อมูลในไฟล์",
            "f_number": "N/A",
            "exposure_time": "N/A",
            "exposure_sec": None,
            "iso": 100,
            "focal_length": "N/A",
            "focal_val": None,
            "deep_notes": {
                "focus_mode": "N/A",
                "af_area_mode": "N/A",
                "focus_distance_m": None,
                "focus_distance_text": "N/A",
                "roll_angle": None,
                "pitch_angle": None,
                "white_balance": "N/A",
                "metering_mode": "N/A",
                "exposure_comp": "0",
            },
        }


def process_image_and_sharpness(file_path: str, exif_info: dict):
    """วิเคราะห์ความคมชัดเชิงแสงทั้งภาพ เปรียบเทียบจุดโฟกัส วิเคราะห์ Edge/Histogram และสร้างภาพตัวอย่างความละเอียดสูง"""
    metadata = exif_info["metadata"]
    iso_val = exif_info["iso"]
    orig_img = load_image_robust(file_path)
    orig_h, orig_w = orig_img.shape[:2]

    # เพิ่มความละเอียดภาพทำงานเป็น 2048px เพื่อให้ภาพตัวอย่างชัดเจนขึ้นมาก
    max_dim = 2048
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        work_img = cv2.resize(orig_img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        work_img = orig_img.copy()

    h, w = work_img.shape[:2]
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1. อ่านพิกัดจุดโฟกัสจากกล้อง (Camera AF Point)
    af_point_text, af_cx, af_cy = parse_camera_af_point(metadata, w, h)
    half_box = max(36, min(w, h) // 10)

    try:
        iso_num = float(parse_float(iso_val) or 100.0)
    except Exception:
        iso_num = 100.0
    noise_penalty = 1.0 + max(0.0, (iso_num - 100.0) / 1600.0)

    raw_af_sharpness = compute_patch_sharpness(denoised_gray, af_cx, af_cy, half_box)
    af_sharpness = round(raw_af_sharpness / noise_penalty, 2)

    # 2. ค้นหาจุดที่มีความคมชัดที่สุดในบริเวณใกล้จุดโฟกัสกล้องก่อน
    local_rad = max(half_box, min(w, h) // 6)
    local_best_x, local_best_y = af_cx, af_cy
    local_best_raw = raw_af_sharpness
    step_local = max(6, local_rad // 8)
    for y in range(max(half_box, af_cy - local_rad), min(h - half_box, af_cy + local_rad) + 1, step_local):
        for x in range(max(half_box, af_cx - local_rad), min(w - half_box, af_cx + local_rad) + 1, step_local):
            val = compute_patch_sharpness(denoised_gray, x, y, half_box)
            if val > local_best_raw:
                local_best_raw = val
                local_best_x, local_best_y = x, y

    # 3. ค้นหาจุดที่มีความคมชัดที่สุดทั่วทั้งภาพ (Global Optical Sharpness Search)
    margin_x = max(half_box, int(w * 0.04))
    margin_y = max(half_box, int(h * 0.04))
    step_global = max(12, half_box // 2)

    global_best_x, global_best_y = local_best_x, local_best_y
    global_best_raw = local_best_raw

    for y in range(margin_y, max(margin_y + 1, h - margin_y), step_global):
        for x in range(margin_x, max(margin_x + 1, w - margin_x), step_global):
            val = compute_patch_sharpness(denoised_gray, x, y, half_box)
            if val > global_best_raw:
                global_best_raw = val
                global_best_x, global_best_y = x, y

    fine_rad = step_global
    fine_step = max(3, fine_rad // 4)
    for y in range(max(half_box, global_best_y - fine_rad), min(h - half_box, global_best_y + fine_rad) + 1, fine_step):
        for x in range(max(half_box, global_best_x - fine_rad), min(w - half_box, global_best_x + fine_rad) + 1, fine_step):
            val = compute_patch_sharpness(denoised_gray, x, y, half_box)
            if val > global_best_raw:
                global_best_raw = val
                global_best_x, global_best_y = x, y

    if local_best_raw >= global_best_raw * 0.88:
        peak_cx, peak_cy = local_best_x, local_best_y
        peak_raw = local_best_raw
    else:
        peak_cx, peak_cy = global_best_x, global_best_y
        peak_raw = global_best_raw

    peak_sharpness = round(peak_raw / noise_penalty, 2)

    # 4. คำนวณระยะห่างและเปรียบเทียบระหว่างจุดโฟกัสกล้องกับจุดที่คมชัดที่สุด
    diag = math.hypot(w, h)
    dx = peak_cx - af_cx
    dy = peak_cy - af_cy
    dist_px = math.hypot(dx, dy)
    focus_offset_pct = round((dist_px / diag) * 100.0, 1)
    offset_dir = describe_offset_direction(dx, dy, diag)
    sharpness_ratio_pct = round(min(100.0, (af_sharpness / peak_sharpness) * 100.0), 1) if peak_sharpness > 0 else 100.0

    af_norm_x = round((af_cx / w) * 100, 1)
    af_norm_y = round((af_cy / h) * 100, 1)
    peak_norm_x = round((peak_cx / w) * 100, 1)
    peak_norm_y = round((peak_cy / h) * 100, 1)

    af_coords_text = f"X:{round(af_norm_x)}%, Y:{round(af_norm_y)}%"
    peak_coords_text = f"X:{round(peak_norm_x)}%, Y:{round(peak_norm_y)}%"

    if peak_sharpness < 350 and af_sharpness < 350:
        quality = "🔴 ซอฟต์ / เบลอทั้งภาพ"
        focus_status = "ไม่พบจุดคมชัดสูงในภาพ"
        advice = (
            f"ความคมชัดจุดโฟกัสกล้อง ({af_sharpness}) เทียบจุดสูงสุด ({peak_sharpness}) ต่ำกว่าเกณฑ์ "
            f"| อาจเกิดจากความเร็วชัตเตอร์ต่ำ กล้องสั่นไหว หรือหลุดระยะโฟกัสทั้งเฟรม"
        )
    elif focus_offset_pct <= 6.0 or sharpness_ratio_pct >= 85.0:
        if af_sharpness >= 1000:
            quality = "🟢 โฟกัสเข้าเป้า & คมชัดสูงมาก"
        else:
            quality = "🟡 โฟกัสเข้าเป้า (คมชัดปานกลาง)"
        focus_status = f"เข้าเป้าแม่นยำ (คลาดเคลื่อน {focus_offset_pct}%)"
        advice = (
            f"จุดชัดที่สุดตรงกับพิกัดกล้อง ({af_point_text}) | ความคมชัดจุดกล้อง {af_sharpness} "
            f"(คิดเป็น {sharpness_ratio_pct}% ของจุดสูงสุด {peak_sharpness})"
        )
    else:
        quality = "🟠 โฟกัสคลาดเคลื่อน (Focus Shift)"
        focus_status = f"จุดชัดสุด{offset_dir} ({focus_offset_pct}%)"
        advice = (
            f"พบจุดคมชัดกว่า ({peak_sharpness} ที่ {peak_coords_text}) อยู่นอกพิกัดกล้อง ({af_sharpness} ที่ {af_point_text}) "
            f"| แนะนำตรวจสอบ Front/Back Focus หรือปรับชดเชย AF Fine-tune"
        )

    # 5. วิเคราะห์ Edge Density, Contrast และ RGB Histogram ความละเอียดสูง (สำหรับเมนูที่ 2, 3, 4)
    texture_metrics = compute_texture_and_histogram(orig_img, work_img, gray, af_cx, af_cy, half_box, scale)

    # 6. คำนวณคะแนนคุณภาพรวม (Composite Quality Score) และคัดกรองข้อบกพร่อง (สำหรับเมนูที่ 2)
    quality_eval = evaluate_photo_quality(
        af_sharpness=af_sharpness,
        peak_sharpness=peak_sharpness,
        sharpness_ratio_pct=sharpness_ratio_pct,
        focus_offset_pct=focus_offset_pct,
        edge_density_pct=texture_metrics["edge_density_pct"],
        contrast_std=texture_metrics["contrast_std"],
        highlight_clipping_pct=texture_metrics["highlight_clipping_pct"],
        shadow_crushing_pct=texture_metrics["shadow_crushing_pct"],
        iso_num=iso_num,
        exposure_sec=exif_info.get("exposure_sec"),
        focal_val=exif_info.get("focal_val"),
        roll_angle=exif_info["deep_notes"].get("roll_angle"),
    )

    # 7. สร้างภาพครอปความละเอียดสูง 100% ทั้งจุดที่ชัดที่สุด (Peak) และจุดโฟกัสกล้อง (AF Point)
    sharpest_crop_b64 = extract_highres_crop_b64(orig_img, peak_cx, peak_cy, scale, max_crop_px=860, jpeg_quality=96)
    af_crop_b64 = extract_highres_crop_b64(orig_img, af_cx, af_cy, scale, max_crop_px=860, jpeg_quality=96)

    # 8. สร้างภาพเต็มเฟรมแบบสะอาด (Clean High-Res Preview) สำหรับเครื่องมือเลือกครอปส่วนภาพเองในตาราง
    _, clean_buf = cv2.imencode(".jpg", work_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    clean_preview_b64 = f"data:image/jpeg;base64,{base64.b64encode(clean_buf).decode('utf-8')}"

    # 9. วาดกรอบเปรียบเทียบลงบนภาพเต็ม (พิกัดจุดโฟกัสกล้อง vs พิกัดจุดที่คมชัดที่สุด)
    annotated_img = work_img.copy()
    thickness = max(2, int(min(w, h) / 400))

    af_color = (255, 120, 0)
    ax1, ax2 = max(0, af_cx - half_box), min(w, af_cx + half_box)
    ay1, ay2 = max(0, af_cy - half_box), min(h, af_cy + half_box)
    cv2.rectangle(annotated_img, (ax1, ay1), (ax2, ay2), af_color, thickness)
    cv2.drawMarker(
        annotated_img,
        (af_cx, af_cy),
        af_color,
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=thickness + 1,
    )

    if dist_px > 12:
        cv2.line(annotated_img, (af_cx, af_cy), (peak_cx, peak_cy), (0, 220, 255), thickness)

    peak_color = (0, 220, 0) if (focus_offset_pct <= 6.0 or sharpness_ratio_pct >= 85.0) and peak_sharpness >= 350 else (0, 60, 255)
    px1, px2 = max(0, peak_cx - half_box), min(w, peak_cx + half_box)
    py1, py2 = max(0, peak_cy - half_box), min(h, peak_cy + half_box)
    cv2.rectangle(annotated_img, (px1, py1), (px2, py2), peak_color, thickness + 1)
    cv2.drawMarker(
        annotated_img,
        (peak_cx, peak_cy),
        peak_color,
        markerType=cv2.MARKER_STAR,
        markerSize=30,
        thickness=thickness + 1,
    )

    _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # คืนค่า RAM ทันทีหลังประมวลผลภาพเสร็จ (รองรับ Render 512MB RAM)
    del orig_img, work_img, gray, denoised_gray, annotated_img
    gc.collect()

    return {
        # ฟิลด์หลัก (Menu 1)
        "sharpness": af_sharpness,
        "af_sharpness": af_sharpness,
        "peak_sharpness": peak_sharpness,
        "sharpness_ratio_pct": sharpness_ratio_pct,
        "focus_offset_pct": focus_offset_pct,
        "focus_status": focus_status,
        "af_coords_text": af_coords_text,
        "peak_coords_text": peak_coords_text,
        "af_norm_coords": {"x": af_norm_x, "y": af_norm_y},
        "peak_norm_coords": {"x": peak_norm_x, "y": peak_norm_y},
        "quality": quality,
        "advice": advice,
        "annotated_image": annotated_b64,
        "clean_preview_image": clean_preview_b64,
        "sharpest_crop_image": sharpest_crop_b64,
        "af_crop_image": af_crop_b64,
        "af_point": af_point_text,
        # ฟิลด์ต่อยอดสำหรับ Menu 2, 3, 4
        "resolution": f"{orig_w} x {orig_h}",
        "texture": texture_metrics,
        "quality_eval": quality_eval,
    }


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Pro Lens & Camera Optical Sharpness API v2.1 Running</h1>")


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok",
        "version": "2.1.0",
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "recommended_batch_files": MAX_RECOMMENDED_BATCH,
            "max_session_items": MAX_SESSION_ITEMS,
        },
        "modules": ["AF_Calibration", "Quality_Scoring", "Edge_Histogram", "Statistical_Insights"],
    }


async def _save_upload_with_limit(file: UploadFile) -> str:
    """บันทึกไฟล์อัปโหลดลงดิสก์ชั่วคราวแบบแบ่งก้อน (Chunked) และตรวจสอบขนาดไม่ให้เกินขีดจำกัด RAM/Disk"""
    ext = os.path.splitext(file.filename or "")[1]
    total_bytes = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        temp_path = tmp.name
        try:
            while True:
                chunk = await file.read(1024 * 1024)  # อ่านทีละ 1 MB
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"ไฟล์มีขนาดเกิน {MAX_FILE_SIZE_MB} MB — ข้ามการประมวลผลเพื่อป้องกันหน่วยความจำเต็ม",
                    )
                tmp.write(chunk)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    return temp_path


async def _handle_image_analysis(file: UploadFile) -> dict:
    temp_path = None
    try:
        temp_path = await _save_upload_with_limit(file)
        async with ANALYSIS_SEMAPHORE:
            loop = asyncio.get_running_loop()
            exif_info = await loop.run_in_executor(None, extract_exif_data, temp_path)
            analysis = await loop.run_in_executor(None, process_image_and_sharpness, temp_path, exif_info)

        return {
            "result": {
                "filename": file.filename,
                "camera": exif_info["camera"],
                "lens": exif_info["lens"],
                "af_point": analysis["af_point"],
                "af_fine_tune": exif_info["af_fine_tune"],
                "f_number": exif_info["f_number"],
                "exposure_time": exif_info["exposure_time"],
                "iso": exif_info["iso"],
                "focal_length": exif_info["focal_length"],
                "resolution": analysis["resolution"],
                "sharpness": analysis["sharpness"],
                "af_sharpness": analysis["af_sharpness"],
                "peak_sharpness": analysis["peak_sharpness"],
                "sharpness_ratio_pct": analysis["sharpness_ratio_pct"],
                "focus_offset_pct": analysis["focus_offset_pct"],
                "focus_status": analysis["focus_status"],
                "af_coords_text": analysis["af_coords_text"],
                "peak_coords_text": analysis["peak_coords_text"],
                "af_norm_coords": analysis["af_norm_coords"],
                "peak_norm_coords": analysis["peak_norm_coords"],
                "quality": analysis["quality"],
                "advice": analysis["advice"],
                "annotated_image": analysis["annotated_image"],
                "clean_preview_image": analysis["clean_preview_image"],
                "sharpest_crop_image": analysis["sharpest_crop_image"],
                "af_crop_image": analysis["af_crop_image"],
                "deep_notes": exif_info["deep_notes"],
                "texture": analysis["texture"],
                "quality_eval": analysis["quality_eval"],
            }
        }
    except HTTPException as he:
        return {"result": {"filename": file.filename, "error": he.detail}}
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return {"result": {"filename": file.filename, "error": str(e)}}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    return await _handle_image_analysis(file)


@app.post("/api/v1/quality-evaluate")
async def api_quality_evaluate(file: UploadFile = File(...)):
    """REST API Endpoint สำหรับเชื่อมต่อระบบภายนอก (คืนค่าคะแนนคุณภาพและตัวชี้วัด JSON)"""
    full_res = await _handle_image_analysis(file)
    res = full_res.get("result", {})
    if "error" in res:
        return res
    return {
        "filename": res["filename"],
        "camera": res["camera"],
        "lens": res["lens"],
        "resolution": res["resolution"],
        "exif": {
            "f_number": res["f_number"],
            "exposure_time": res["exposure_time"],
            "iso": res["iso"],
            "focal_length": res["focal_length"],
            "af_point": res["af_point"],
            "af_fine_tune": res["af_fine_tune"],
            "deep_notes": res["deep_notes"],
        },
        "metrics": {
            "af_sharpness": res["af_sharpness"],
            "peak_sharpness": res["peak_sharpness"],
            "sharpness_ratio_pct": res["sharpness_ratio_pct"],
            "focus_offset_pct": res["focus_offset_pct"],
            "edge_density_pct": res["texture"]["edge_density_pct"],
            "mean_gradient": res["texture"]["mean_gradient"],
            "contrast_std": res["texture"]["contrast_std"],
            "highlight_clipping_pct": res["texture"]["highlight_clipping_pct"],
            "shadow_crushing_pct": res["texture"]["shadow_crushing_pct"],
        },
        "quality_evaluation": res["quality_eval"],
    }
