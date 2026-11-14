import os
import re
import gc
import json
import math
import shutil
import tempfile
import base64
import logging
import subprocess
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from exiftool import ExifToolHelper
except ImportError:
    ExifToolHelper = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pro Lens & Camera Optical Sharpness & Quality Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_BRANDS = ["Nikon", "Canon", "Sigma", "Tamron", "Samyang", "Tokina"]
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
        if math.isnan(float(val)) or math.isinf(float(val)):
            return None
        return float(val)
    s = str(val).strip()
    frac_match = re.match(r'^([-+]?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$', s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                return num / den
        except Exception:
            pass
    match = re.search(r'[-+]?\d+(?:\.\d+)?', s)
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


def extract_af_fine_tune(metadata: dict) -> Tuple[str, Optional[int]]:
    """สกัดค่า AF Fine-tune / AF Microadjustment จริงจาก MakerNotes (คืนค่าทั้งข้อความและตัวเลข)"""
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

    def parse_signed_int(v) -> Optional[int]:
        try:
            return int(round(float(str(v).strip())))
        except Exception:
            return None

    def fmt_signed(v) -> str:
        num = parse_signed_int(v)
        if num is not None:
            return f"+{num}" if num > 0 else str(num)
        return str(v).strip()

    if af_adj is not None or af_status is not None:
        status_str = str(af_status).strip() if af_status is not None else ""
        num_val = parse_signed_int(af_adj) if af_adj is not None else 0
        adj_str = fmt_signed(af_adj) if af_adj is not None else "0"
        if status_str.lower().startswith("off"):
            if adj_str != "0":
                return f"Off (ค่าที่บันทึก: {adj_str})", 0
            return "Off (0)", 0
        elif status_str.lower().startswith("on"):
            idx_part = f" [Saved #{af_index}]" if af_index is not None and str(af_index).strip() not in ("", "0") else ""
            return f"On ({adj_str}){idx_part}", num_val
        elif af_adj is not None:
            return adj_str, num_val

    if canon_val is not None or canon_mode is not None:
        mode_str = str(canon_mode).strip() if canon_mode is not None else ""
        num_val = parse_signed_int(canon_val) if canon_val is not None else 0
        val_str = fmt_signed(canon_val) if canon_val is not None else "0"
        if "disable" in mode_str.lower():
            return (f"Off ({val_str})" if val_str != "0" else "Off (0)"), 0
        elif canon_val is not None:
            return (f"{val_str} ({mode_str})" if mode_str else val_str), num_val

    if generic_adj is not None:
        return fmt_signed(generic_adj), parse_signed_int(generic_adj)

    return "ไม่พบข้อมูลในไฟล์", None


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
        af_fine_tune, af_tune_numeric = extract_af_fine_tune(metadata)

        exposure = get_tag_value(metadata, ["EXIF:ExposureTime", "ExposureTime"])
        shutter_seconds = parse_float(exposure)
        if isinstance(exposure, (int, float)):
            exposure_str = f"1/{round(1 / exposure)}" if 0 < exposure < 1 else str(exposure)
        elif exposure is not None:
            exposure_str = re.sub(r'\s*s(?:ec)?\.?$', '', str(exposure).strip(), flags=re.IGNORECASE)
        else:
            exposure_str = "N/A"

        f_num = get_tag_value(metadata, ["EXIF:FNumber", "FNumber"])
        f_number_numeric = parse_float(f_num)
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
        iso_numeric = parse_float(iso_val) or 100.0

        focus_mode_raw = get_tag_value(metadata, ["MakerNotes:FocusMode", "Nikon:FocusMode", "Canon:FocusMode", "FocusMode"])
        focus_mode = str(focus_mode_raw).strip() if focus_mode_raw is not None else "N/A"

        af_area_raw = get_tag_value(metadata, ["MakerNotes:AFAreaMode", "Nikon:AFAreaMode", "Canon:AFAreaMode", "AFAreaMode"])
        af_area_mode = str(af_area_raw).strip() if af_area_raw is not None else "N/A"

        focus_dist_raw = get_tag_value(metadata, [
            "MakerNotes:FocusDistance",
            "Nikon:FocusDistance",
            "Composite:FocusDistance",
            "MakerNotes:ApproximateFocusDistance",
            "FocusDistance",
        ])
        focus_distance_m = parse_float(focus_dist_raw)
        if focus_distance_m is not None:
            focus_distance_m = round(focus_distance_m, 2)

        roll_angle = parse_float(get_tag_value(metadata, ["MakerNotes:RollAngle", "Nikon:RollAngle", "RollAngle"]))
        if roll_angle is not None:
            roll_angle = round(roll_angle, 2)

        pitch_angle = parse_float(get_tag_value(metadata, ["MakerNotes:PitchAngle", "Nikon:PitchAngle", "PitchAngle"]))
        if pitch_angle is not None:
            pitch_angle = round(pitch_angle, 2)

        return {
            "metadata": metadata,
            "camera": camera,
            "lens": lens,
            "af_fine_tune": af_fine_tune,
            "af_tune_numeric": af_tune_numeric,
            "f_number": f_number_str,
            "f_number_numeric": f_number_numeric,
            "exposure_time": exposure_str,
            "shutter_seconds": shutter_seconds,
            "iso": iso_val,
            "iso_numeric": iso_numeric,
            "focal_length": focal_str,
            "focal_length_numeric": focal_val,
            "focus_mode": focus_mode,
            "af_area_mode": af_area_mode,
            "focus_distance_m": focus_distance_m,
            "roll_angle": roll_angle,
            "pitch_angle": pitch_angle,
        }
    except Exception as e:
        logger.error(f"ExifTool Error: {e}")
        return {
            "metadata": {},
            "camera": "Unknown Camera",
            "lens": "N/A",
            "af_fine_tune": "ไม่พบข้อมูลในไฟล์",
            "af_tune_numeric": None,
            "f_number": "N/A",
            "f_number_numeric": None,
            "exposure_time": "N/A",
            "shutter_seconds": None,
            "iso": 100,
            "iso_numeric": 100.0,
            "focal_length": "N/A",
            "focal_length_numeric": None,
            "focus_mode": "N/A",
            "af_area_mode": "N/A",
            "focus_distance_m": None,
            "roll_angle": None,
            "pitch_angle": None,
        }


def analyze_roi_optical_features(roi_bgr: np.ndarray, roi_gray: np.ndarray) -> dict:
    """คำนวณคุณลักษณะเชิงแสงในกรอบ ROI: Canny Edge Density, Sobel Gradient, Contrast, Brightness และ RGB Histogram"""
    if roi_gray.size == 0 or roi_bgr.size == 0:
        return {
            "edge_density_pct": 0.0,
            "mean_gradient_strength": 0.0,
            "contrast_std": 0.0,
            "mean_brightness": 0.0,
            "highlight_clipping_pct": 0.0,
            "shadow_crushing_pct": 0.0,
            "rgb_histogram": {"r": [0.0] * 64, "g": [0.0] * 64, "b": [0.0] * 64},
            "canny_edge_image": "",
            "roi_original_image": "",
        }

    # 1. Canny Edge Density ด้วย Otsu Adaptive Thresholding
    high_thresh, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = max(10.0, 0.5 * float(high_thresh))
    canny = cv2.Canny(roi_gray, low_thresh, max(20.0, float(high_thresh)))
    edge_density_pct = round(float((np.sum(canny > 0) / canny.size) * 100.0), 2)

    # 2. Sobel Mean Gradient Strength
    sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobelx, sobely)
    mean_gradient_strength = round(float(np.mean(grad_mag)), 2)

    # 3. Contrast (Std Dev) & Mean Brightness
    contrast_std = round(float(np.std(roi_gray)), 2)
    mean_brightness = round(float(np.mean(roi_gray)), 1)

    # 4. Exposure Clipping (Highlight >= 250, Shadow <= 5)
    highlight_clipping_pct = round(float((np.sum(roi_gray >= 250) / roi_gray.size) * 100.0), 2)
    shadow_crushing_pct = round(float((np.sum(roi_gray <= 5) / roi_gray.size) * 100.0), 2)

    # 5. RGB Histogram (64 bins normalized %)
    rgb_hist = {}
    for ch_idx, ch_name in enumerate(["b", "g", "r"]):
        hist = cv2.calcHist([roi_bgr], [ch_idx], None, [64], [0, 256]).flatten()
        total_px = max(1.0, float(np.sum(hist)))
        rgb_hist[ch_name] = [round(float(v / total_px) * 100.0, 3) for v in hist]

    # 6. Encode ROI Original & Canny Edge Map เป็น Base64
    _, orig_buf = cv2.imencode(".jpg", roi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    roi_original_b64 = f"data:image/jpeg;base64,{base64.b64encode(orig_buf).decode('utf-8')}"

    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    _, canny_buf = cv2.imencode(".jpg", canny_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    canny_b64 = f"data:image/jpeg;base64,{base64.b64encode(canny_buf).decode('utf-8')}"

    return {
        "edge_density_pct": edge_density_pct,
        "mean_gradient_strength": mean_gradient_strength,
        "contrast_std": contrast_std,
        "mean_brightness": mean_brightness,
        "highlight_clipping_pct": highlight_clipping_pct,
        "shadow_crushing_pct": shadow_crushing_pct,
        "rgb_histogram": rgb_hist,
        "canny_edge_image": canny_b64,
        "roi_original_image": roi_original_b64,
    }


def evaluate_photo_quality(
    af_sharpness: float,
    peak_sharpness: float,
    sharpness_ratio_pct: float,
    focus_offset_pct: float,
    optical_features: dict,
    exif_info: dict,
) -> dict:
    """คำนวณคะแนนคุณภาพภาพถ่ายรวม (Composite Quality Score 0-100) และจัดเกรดอัตโนมัติ (เมนูที่ 2)"""
    edge_density = optical_features["edge_density_pct"]
    grad_strength = optical_features["mean_gradient_strength"]
    contrast_std = optical_features["contrast_std"]
    mean_brightness = optical_features["mean_brightness"]
    hi_clip = optical_features["highlight_clipping_pct"]
    sh_crush = optical_features["shadow_crushing_pct"]

    iso_num = exif_info.get("iso_numeric") or 100.0
    shutter_sec = exif_info.get("shutter_seconds")
    focal_mm = exif_info.get("focal_length_numeric")
    roll_angle = exif_info.get("roll_angle")

    # 1. Focus & Optical Sharpness Index (น้ำหนัก 50%)
    raw_focus_pts = 100.0 * (1.0 - math.exp(-af_sharpness / 520.0))
    if edge_density < 4.5 and grad_strength > 18.0 and af_sharpness >= 300:
        raw_focus_pts = min(100.0, raw_focus_pts + 8.0)

    if focus_offset_pct > 6.0 and sharpness_ratio_pct < 85.0:
        shift_penalty = min(35.0, (85.0 - sharpness_ratio_pct) * 0.65)
        raw_focus_pts = max(0.0, raw_focus_pts - shift_penalty)

    focus_index = round(min(100.0, max(0.0, raw_focus_pts)), 1)

    # 2. Exposure & Tonal Contrast Index (น้ำหนัก 30%)
    if 32.0 <= contrast_std <= 78.0:
        contrast_pts = 100.0
    elif contrast_std < 32.0:
        contrast_pts = max(30.0, 100.0 - (32.0 - contrast_std) * 2.2)
    else:
        contrast_pts = max(45.0, 100.0 - (contrast_std - 78.0) * 1.4)

    brightness_penalty = 0.0
    if mean_brightness < 65.0:
        brightness_penalty = min(25.0, (65.0 - mean_brightness) * 0.5)
    elif mean_brightness > 195.0:
        brightness_penalty = min(25.0, (mean_brightness - 195.0) * 0.5)

    clip_penalty = min(35.0, max(0.0, hi_clip - 3.0) * 2.5 + max(0.0, sh_crush - 5.0) * 2.0)
    exposure_index = round(min(100.0, max(0.0, contrast_pts - brightness_penalty - clip_penalty)), 1)

    # 3. Technical Stability & Noise Index (น้ำหนัก 20%)
    stability_pts = 100.0
    reciprocal_ok: Optional[bool] = None
    rec_min_shutter = "N/A"

    if focal_mm and focal_mm > 0:
        min_shutter_sec = 1.0 / focal_mm
        rec_min_shutter = f"1/{max(1, round(focal_mm))}s"
        if shutter_sec and shutter_sec > 0:
            if shutter_sec <= min_shutter_sec * 1.15:
                reciprocal_ok = True
            elif shutter_sec <= min_shutter_sec * 2.5:
                reciprocal_ok = True
                stability_pts -= 10.0
            else:
                reciprocal_ok = False
                ratio = min(5.0, shutter_sec / min_shutter_sec)
                stability_pts -= min(45.0, (ratio - 1.0) * 14.0)

    if iso_num > 800:
        iso_stops = math.log2(iso_num / 800.0)
        stability_pts -= min(35.0, iso_stops * 9.0)

    if roll_angle is not None and abs(roll_angle) > 3.5:
        stability_pts -= min(10.0, (abs(roll_angle) - 3.5) * 1.5)

    stability_index = round(min(100.0, max(0.0, stability_pts)), 1)

    total_score = round(
        0.50 * focus_index + 0.30 * exposure_index + 0.20 * stability_index,
        1,
    )

    if total_score >= 82.0 and af_sharpness >= 500:
        grade = "A"
        grade_label = "เกรด A — ดีเยี่ยม (พร้อมใช้งาน)"
    elif total_score >= 65.0 and af_sharpness >= 300:
        grade = "B"
        grade_label = "เกรด B — ดี (ผ่านเกณฑ์)"
    elif total_score >= 48.0:
        grade = "C"
        grade_label = "เกรด C — พอใช้ (ควรตรวจสอบ)"
    else:
        grade = "Reject"
        grade_label = "ไม่ผ่านเกณฑ์ (Reject)"

    recommendations = []
    if peak_sharpness >= 400 and focus_offset_pct > 6.0 and sharpness_ratio_pct < 85.0:
        issue_category = "Focus Shift"
        issue_label_th = "🟠 โฟกัสคลาดเคลื่อน (Focus Shift)"
        recommendations.append(
            f"จุดชัดสุดเยื้องจากจุดโฟกัสกล้อง {focus_offset_pct}% (สัดส่วนคมชัด {sharpness_ratio_pct}%) แนะนำตรวจสอบ Front/Back Focus หรือปรับ AF Fine-tune"
        )
    elif af_sharpness < 450 and reciprocal_ok is False:
        issue_category = "Motion Blur Risk"
        issue_label_th = "🔴 เสี่ยงภาพสั่นไหว (Motion Blur)"
        recommendations.append(
            f"ความเร็วชัตเตอร์ช้ากว่าเกณฑ์ 1/{max(1, round(focal_mm or 50))}s แนะนำเพิ่มความเร็วชัตเตอร์เป็น {rec_min_shutter} ขึ้นไป"
        )
    elif af_sharpness < 350 and peak_sharpness < 350:
        issue_category = "Out of Focus"
        issue_label_th = "🔴 หลุดโฟกัส / ภาพซอฟต์ทั้งเฟรม"
        recommendations.append(
            "ไม่พบจุดที่มีความคมชัดสูงในเฟรม แนะนำตรวจสอบระยะโฟกัสขั้นต่ำของเลนส์หรือใช้โหมด AF-S/จุดโฟกัสเดี่ยว"
        )
    elif iso_num >= 3200 and edge_density < 4.0:
        issue_category = "High ISO Noise"
        issue_label_th = "🟡 สัญญาณรบกวนสูง (High-ISO Noise)"
        recommendations.append(
            f"ค่า ISO {int(iso_num)} ทำให้เกิดเกรนรบกวนรายละเอียดขอบภาพ แนะนำลด ISO หรือเปิดรูรับแสงให้กว้างขึ้น"
        )
    elif hi_clip > 8.0 or sh_crush > 12.0:
        issue_category = "Exposure Clipping"
        issue_label_th = "🟡 แสงจ้าทะลุ / มืดจม (Exposure Clipping)"
        if hi_clip > 8.0:
            recommendations.append(f"พบส่วนสว่างทะลุ (Highlight Clipping) {hi_clip}% แนะนำลดชดเชยแสง (-EV)")
        if sh_crush > 12.0:
            recommendations.append(f"พบส่วนมืดจม (Shadow Crushing) {sh_crush}% แนะนำเพิ่มชดเชยแสง (+EV)")
    else:
        issue_category = "Optimal"
        issue_label_th = "🟢 คุณภาพสมบูรณ์ตามเกณฑ์"
        recommendations.append("ความคมชัดจุดโฟกัส การวัดแสง และเสถียรภาพชัตเตอร์อยู่ในเกณฑ์เหมาะสม")

    if roll_angle is not None and abs(roll_angle) > 3.0:
        recommendations.append(f"ระนาบกล้องเอียง (Roll Angle: {roll_angle}°) ควรปรับระดับให้ตรง")

    return {
        "total_score": total_score,
        "grade": grade,
        "grade_label": grade_label,
        "focus_index_score": focus_index,
        "exposure_index_score": exposure_index,
        "stability_index_score": stability_index,
        "reciprocal_rule_ok": reciprocal_ok,
        "recommended_min_shutter": rec_min_shutter,
        "issue_category": issue_category,
        "issue_label_th": issue_label_th,
        "detailed_recommendations": recommendations,
    }


def process_image_and_sharpness(file_path: str, metadata: dict, exif_info: dict):
    """วิเคราะห์ความคมชัดเชิงแสงทั้งภาพ เปรียบเทียบจุดโฟกัสกล้องกับจุดที่คมชัดที่สุด และสกัดคุณลักษณะเชิงลึก"""
    orig_img = load_image_robust(file_path)
    orig_h, orig_w = orig_img.shape[:2]

    max_dim = 1600
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

    iso_num = exif_info.get("iso_numeric") or 100.0
    noise_penalty = 1.0 + max(0.0, (iso_num - 100.0) / 1600.0)

    # วัดความคมชัด ณ จุดโฟกัสกล้อง
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

    af_coords_text = f"X:{round((af_cx / w) * 100)}%, Y:{round((af_cy / h) * 100)}%"
    peak_coords_text = f"X:{round((peak_cx / w) * 100)}%, Y:{round((peak_cy / h) * 100)}%"

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

    # 5. สกัดคุณลักษณะเชิงแสงในกรอบ ROI จุดโฟกัสกล้อง (Edge Density, Sobel, Contrast, RGB Histogram)
    ax1, ax2 = max(0, af_cx - half_box), min(w, af_cx + half_box)
    ay1, ay2 = max(0, af_cy - half_box), min(h, af_cy + half_box)
    af_roi_bgr = work_img[ay1:ay2, ax1:ax2]
    af_roi_gray = gray[ay1:ay2, ax1:ax2]
    optical_features = analyze_roi_optical_features(af_roi_bgr, af_roi_gray)

    # 6. ประเมินคุณภาพภาพถ่ายเชิงลึก (เมนูที่ 2)
    quality_eval = evaluate_photo_quality(
        af_sharpness=af_sharpness,
        peak_sharpness=peak_sharpness,
        sharpness_ratio_pct=sharpness_ratio_pct,
        focus_offset_pct=focus_offset_pct,
        optical_features=optical_features,
        exif_info=exif_info,
    )

    # 7. ครอบภาพส่วนที่คมชัดที่สุดจากภาพต้นฉบับความละเอียดเต็ม (Clean 100% Detail Crop)
    orig_peak_cx = int(peak_cx / scale)
    orig_peak_cy = int(peak_cy / scale)
    orig_crop_half = max(160, min(orig_w, orig_h) // 8)
    cx1 = max(0, orig_peak_cx - orig_crop_half)
    cx2 = min(orig_w, orig_peak_cx + orig_crop_half)
    cy1 = max(0, orig_peak_cy - orig_crop_half)
    cy2 = min(orig_h, orig_peak_cy + orig_crop_half)

    sharpest_crop = orig_img[cy1:cy2, cx1:cx2]
    if sharpest_crop.size == 0:
        sharpest_crop = work_img[
            max(0, peak_cy - half_box) : min(h, peak_cy + half_box),
            max(0, peak_cx - half_box) : min(w, peak_cx + half_box),
        ]

    _, crop_buf = cv2.imencode(".jpg", sharpest_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    sharpest_crop_b64 = f"data:image/jpeg;base64,{base64.b64encode(crop_buf).decode('utf-8')}"

    # 8. วาดกรอบเปรียบเทียบลงบนภาพเต็ม
    annotated_img = work_img.copy()
    thickness = max(2, int(min(w, h) / 400))

    af_color = (255, 120, 0)
    cv2.rectangle(annotated_img, (ax1, ay1), (ax2, ay2), af_color, thickness)
    cv2.drawMarker(
        annotated_img,
        (af_cx, af_cy),
        af_color,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
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
        markerSize=26,
        thickness=thickness + 1,
    )

    _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    del orig_img, work_img, gray, denoised_gray, af_roi_bgr, af_roi_gray, sharpest_crop, annotated_img
    gc.collect()

    return {
        "sharpness": af_sharpness,
        "af_sharpness": af_sharpness,
        "peak_sharpness": peak_sharpness,
        "sharpness_ratio_pct": sharpness_ratio_pct,
        "focus_offset_pct": focus_offset_pct,
        "focus_status": focus_status,
        "af_coords_text": af_coords_text,
        "peak_coords_text": peak_coords_text,
        "quality": quality,
        "advice": advice,
        "annotated_image": annotated_b64,
        "sharpest_crop_image": sharpest_crop_b64,
        "af_point": af_point_text,
        "optical_features": optical_features,
        "quality_evaluation": quality_eval,
    }


def compute_pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    arr_x = np.array(xs, dtype=np.float64)
    arr_y = np.array(ys, dtype=np.float64)
    if np.std(arr_x) == 0 or np.std(arr_y) == 0:
        return None
    r = float(np.corrcoef(arr_x, arr_y)[0, 1])
    if math.isnan(r) or math.isinf(r):
        return None
    return round(r, 4)


def compute_batch_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """คำนวณสถิติเชิงกลุ่ม ความสัมพันธ์ Pearson r และเส้นโค้งสอบเทียบ AF Fine-tune (เมนูที่ 4 และ 5)"""
    if not records:
        return {
            "iso_groups": [],
            "af_tune_curve": [],
            "optimal_af_tune": None,
            "correlations": {},
            "metadata_variance": [],
        }

    # 1. จัดกลุ่มสถิติตามค่า ISO
    iso_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        iso_val = r.get("iso_numeric") or parse_float(r.get("iso"))
        if iso_val is not None:
            iso_key = int(round(iso_val))
            iso_buckets.setdefault(iso_key, []).append(r)

    iso_groups = []
    for iso_key in sorted(iso_buckets.keys()):
        items = iso_buckets[iso_key]
        avg_af = round(float(np.mean([x.get("af_sharpness", 0) for x in items])), 1)
        avg_peak = round(float(np.mean([x.get("peak_sharpness", 0) for x in items])), 1)
        avg_score = round(float(np.mean([x.get("quality_evaluation", {}).get("total_score", 0) for x in items])), 1)
        iso_groups.append({
            "iso": iso_key,
            "label": f"ISO {iso_key}",
            "count": len(items),
            "avg_af_sharpness": avg_af,
            "avg_peak_sharpness": avg_peak,
            "avg_quality_score": avg_score,
        })

    # 2. เส้นโค้งสอบเทียบ AF Fine-Tune Adjustment
    tune_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        tune_num = r.get("af_tune_numeric")
        if tune_num is not None:
            tune_buckets.setdefault(int(tune_num), []).append(r)

    af_tune_curve = []
    for t_val in sorted(tune_buckets.keys()):
        items = tune_buckets[t_val]
        avg_af = round(float(np.mean([x.get("af_sharpness", 0) for x in items])), 1)
        avg_ratio = round(float(np.mean([x.get("sharpness_ratio_pct", 0) for x in items])), 1)
        af_tune_curve.append({
            "af_tune": t_val,
            "label": f"+{t_val}" if t_val > 0 else str(t_val),
            "count": len(items),
            "avg_af_sharpness": avg_af,
            "avg_sharpness_ratio_pct": avg_ratio,
        })

    optimal_af_tune = None
    if af_tune_curve:
        best_pt = max(af_tune_curve, key=lambda x: x["avg_af_sharpness"])
        optimal_af_tune = {
            "recommended_value": best_pt["label"],
            "numeric_value": best_pt["af_tune"],
            "avg_af_sharpness": best_pt["avg_af_sharpness"],
            "tested_steps": len(af_tune_curve),
        }

    # 3. ค่าสหสัมพันธ์ Pearson r
    sharp_list = [float(r.get("af_sharpness", 0)) for r in records]
    edge_list = [float(r.get("optical_features", {}).get("edge_density_pct", 0)) for r in records]
    r_sharp_edge = compute_pearson_correlation(edge_list, sharp_list)

    roll_pairs = [(float(r["roll_angle"]), float(r["af_sharpness"])) for r in records if r.get("roll_angle") is not None]
    r_sharp_roll = compute_pearson_correlation([p[0] for p in roll_pairs], [p[1] for p in roll_pairs])

    iso_pairs = [(float(r["iso_numeric"]), float(r["af_sharpness"])) for r in records if r.get("iso_numeric") is not None]
    r_sharp_iso = compute_pearson_correlation([p[0] for p in iso_pairs], [p[1] for p in iso_pairs])

    # 4. ตรวจสอบความผันแปรของพารามิเตอร์ (Metadata Variance Detector)
    param_specs = [
        ("AF Sharpness", lambda x: x.get("af_sharpness")),
        ("Peak Sharpness", lambda x: x.get("peak_sharpness")),
        ("Quality Score (0-100)", lambda x: x.get("quality_evaluation", {}).get("total_score")),
        ("Edge Density (%)", lambda x: x.get("optical_features", {}).get("edge_density_pct")),
        ("Contrast (Std Dev)", lambda x: x.get("optical_features", {}).get("contrast_std")),
        ("ISO", lambda x: x.get("iso_numeric")),
        ("F-Number (Aperture)", lambda x: x.get("f_number_numeric")),
        ("Shutter Speed (s)", lambda x: x.get("shutter_seconds")),
        ("Focal Length (mm)", lambda x: x.get("focal_length_numeric")),
        ("AF Fine-tune Adj", lambda x: x.get("af_tune_numeric")),
        ("Focus Distance (m)", lambda x: x.get("focus_distance_m")),
        ("Roll Angle (deg)", lambda x: x.get("roll_angle")),
        ("Pitch Angle (deg)", lambda x: x.get("pitch_angle")),
    ]

    metadata_variance = []
    for label, extractor in param_specs:
        vals = [float(extractor(r)) for r in records if extractor(r) is not None]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            std_v = round(float(np.std(arr)), 4)
            metadata_variance.append({
                "parameter": label,
                "valid_count": len(vals),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "mean": round(float(np.mean(arr)), 4),
                "std_dev": std_v,
                "is_varied": bool(std_v > 0),
            })

    metadata_variance.sort(key=lambda x: (not x["is_varied"], -x["std_dev"]))

    return {
        "iso_groups": iso_groups,
        "af_tune_curve": af_tune_curve,
        "optimal_af_tune": optimal_af_tune,
        "correlations": {
            "sharpness_vs_edge_density": r_sharp_edge,
            "sharpness_vs_roll_angle": r_sharp_roll,
            "sharpness_vs_iso": r_sharp_iso,
        },
        "metadata_variance": metadata_variance,
    }


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Pro Lens & Camera Optical Sharpness API Running</h1>")


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        exif_info = extract_exif_data(temp_path)
        analysis = process_image_and_sharpness(
            temp_path, exif_info["metadata"], exif_info
        )

        return {
            "result": {
                "filename": file.filename,
                "camera": exif_info["camera"],
                "lens": exif_info["lens"],
                "af_point": analysis["af_point"],
                "af_fine_tune": exif_info["af_fine_tune"],
                "af_tune_numeric": exif_info["af_tune_numeric"],
                "f_number": exif_info["f_number"],
                "f_number_numeric": exif_info["f_number_numeric"],
                "exposure_time": exif_info["exposure_time"],
                "shutter_seconds": exif_info["shutter_seconds"],
                "iso": exif_info["iso"],
                "iso_numeric": exif_info["iso_numeric"],
                "focal_length": exif_info["focal_length"],
                "focal_length_numeric": exif_info["focal_length_numeric"],
                "focus_mode": exif_info["focus_mode"],
                "af_area_mode": exif_info["af_area_mode"],
                "focus_distance_m": exif_info["focus_distance_m"],
                "roll_angle": exif_info["roll_angle"],
                "pitch_angle": exif_info["pitch_angle"],
                "sharpness": analysis["sharpness"],
                "af_sharpness": analysis["af_sharpness"],
                "peak_sharpness": analysis["peak_sharpness"],
                "sharpness_ratio_pct": analysis["sharpness_ratio_pct"],
                "focus_offset_pct": analysis["focus_offset_pct"],
                "focus_status": analysis["focus_status"],
                "af_coords_text": analysis["af_coords_text"],
                "peak_coords_text": analysis["peak_coords_text"],
                "quality": analysis["quality"],
                "advice": analysis["advice"],
                "annotated_image": analysis["annotated_image"],
                "sharpest_crop_image": analysis["sharpest_crop_image"],
                "optical_features": analysis["optical_features"],
                "quality_evaluation": analysis["quality_evaluation"],
            }
        }
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return {"result": {"filename": file.filename, "error": str(e)}}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()


@app.post("/batch-statistics")
async def batch_statistics(payload: dict = Body(...)):
    records = payload.get("records", [])
    return {"statistics": compute_batch_statistics(records)}
