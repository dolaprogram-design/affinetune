import os
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

# ตั้งค่า CORS สำหรับรองรับการเชื่อมต่อจากหน้าเว็บ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_exif_data(file_path: str) -> dict:
    """
    ดึงข้อมูล EXIF เชิงลึกและ MakerNotes จากไฟล์ภาพด้วย ExifTool
    (รองรับพิกัดจุดโฟกัส AF Point, รุ่นกล้อง, รุ่นเลนส์ และค่าปรับจูน AF Fine-tune)
    """
    try:
        with ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)[0]
            
            # 1. ยี่ห้อและรุ่นกล้อง
            make = metadata.get("EXIF:Make", "")
            model = metadata.get("EXIF:Model", "")
            camera = f"{make} {model}".strip() or "ไม่ระบุ"
            
            # 2. รุ่นเลนส์ (Fallback ตามโครงสร้างค่ายกล้อง)
            lens = (
                metadata.get("EXIF:LensModel") or 
                metadata.get("Composite:LensID") or 
                metadata.get("Nikon:LensSpec") or 
                metadata.get("EXIF:LensInfo") or 
                "ไม่ระบุ"
            )
            
            # 3. จุดโฟกัสที่บันทึกจริงใน MakerNotes
            af_point = (
                metadata.get("Nikon:AFPointsInFocus") or 
                metadata.get("Nikon:PrimaryAFPoint") or 
                metadata.get("Canon:AFPointUsed") or 
                metadata.get("Sony:AFPointSelected") or 
                "Center (Default)"
            )
            
            # 4. ค่าชดเชยไมโครแอดจัสต์ (AF Fine-tune / AF Microadjustment)
            af_fine_tune = (
                metadata.get("Nikon:AFTuneValue") or 
                metadata.get("Canon:AFMicroAdjustment") or 
                0
            )

            # จัดฟอร์แมตความเร็วชัตเตอร์ (เช่น 1/250s)
            exposure = metadata.get("EXIF:ExposureTime")
            if isinstance(exposure, (int, float)):
                exposure_str = f"1/{round(1/exposure)}" if 0 < exposure < 1 else str(exposure)
            else:
                exposure_str = str(exposure) if exposure else "N/A"

            return {
                "camera": camera,
                "lens": lens,
                "af_point": str(af_point),
                "af_fine_tune": str(af_fine_tune),
                "f_number": metadata.get("EXIF:FNumber", "N/A"),
                "exposure_time": exposure_str,
                "iso": metadata.get("EXIF:ISO", "N/A"),
                "focal_length": metadata.get("EXIF:FocalLength", "N/A")
            }
    except Exception as e:
        # Fallback กรณีอ่าน EXIF ล้มเหลวหรือไฟล์ไม่มี Metadata
        return {
            "camera": "ไม่ระบุ",
            "lens": "ไม่ระบุ",
            "af_point": "Center (Default)",
            "af_fine_tune": "0",
            "f_number": "N/A",
            "exposure_time": "N/A",
            "iso": "N/A",
            "focal_length": "N/A"
        }

def process_image_and_sharpness(file_path: str, af_point_text: str):
    """
    คำนวณคะแนนความคมชัดด้วย OpenCV (Laplacian Variance) 
    พร้อมย่อขนาดภาพเพื่อป้องกันแรมเต็ม 512MB และวาดเป้าจุดโฟกัสแปลงเป็น Base64
    """
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("ไม่สามารถเปิดอ่านไฟล์ภาพได้")

    # ย่อขนาดภาพกรณีไฟล์ใหญ่เกินไป เพื่อประหยัด RAM บนเซิร์ฟเวอร์
    h, w = img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # คำนวณความคมชัดด้วย Laplacian Variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    # เกณฑ์ประเมินความคมชัดและคำแนะนำ
    if sharpness_score >= 1500:
        quality = "🟢 คมชัดสูงมาก (Excellent)"
        advice = "โฟกัสเข้าเป้าแม่นยำ ชิ้นแก้วทำงานได้เต็มประสิทธิภาพ"
    elif sharpness_score >= 500:
        quality = "🟡 คมชัดปานกลาง (Good)"
        advice = "ความคมชัดอยู่ในเกณฑ์ยอมรับได้ หากต้องการคมกว่านี้ลองปรับจูน AF Fine-tune"
    else:
        quality = "🔴 ซอฟต์ / เบลอ (Soft/Out of Focus)"
        advice = "ภาพมีความซอฟต์หรือโฟกัสคลาดเคลื่อน ควรตรวจสอบอาการ Front/Back Focus"

    # วาดมาร์กเกอร์แสดงจุดโฟกัสลงบนภาพ (Annotated Preview Image)
    annotated_img = img.copy()
    cx, cy = w // 2, h // 2
    box_size = min(w, h) // 10
    
    # วาดเป้าสี่เหลี่ยมจุดโฟกัส
    box_color = (0, 255, 0) if sharpness_score >= 1000 else (0, 0, 255)
    cv2.rectangle(annotated_img, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), box_color, 3)
    cv2.circle(annotated_img, (cx, cy), 6, (255, 255, 255), -1)
    cv2.circle(annotated_img, (cx, cy), 3, (0, 0, 0), -1)
    
    # ข้อความระบุรายละเอียดบนภาพ
    label = f"AF: {af_point_text} | Score: {sharpness_score}"
    cv2.putText(annotated_img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # แปลงภาพเป็น Base64 Data URL สำหรับส่งกลับไปแสดงใน Modal หน้าบ้าน
    _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    base64_str = base64.b64encode(buffer).decode('utf-8')
    annotated_b64 = f"data:image/jpeg;base64,{base64_str}"

    return sharpness_score, quality, advice, annotated_b64

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """แสดงผลหน้าแรกด้วยไฟล์ index.html"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Pro Lens & Camera Analytics Dashboard API Running</h1>")

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Endpoint รับไฟล์ภาพเพื่อทำการประมวลผล EXIF และความคมชัดแบบ Serial
    """
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # 1. อ่านค่า EXIF และ MakerNotes ด้วย ExifTool
        exif_info = extract_exif_data(temp_path)

        # 2. คำนวณความคมชัด และสร้างภาพ Overlay จุดโฟกัส
        sharpness, quality, advice, annotated_b64 = process_image_and_sharpness(
            temp_path, exif_info["af_point"]
        )

        # 3. จัดส่ง JSON โครงสร้างตรงตามที่ index.html ต้องการ
        return {
            "result": {
                "filename": file.filename,
                "camera": exif_info["camera"],
                "lens": exif_info["lens"],
                "af_point": exif_info["af_point"],
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
        # ลบไฟล์ชั่วคราวทิ้งทุกครั้งหลังจากประมวลผลเสร็จสิ้น
        if os.path.exists(temp_path):
            os.remove(temp_path)
