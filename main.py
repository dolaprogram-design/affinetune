from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import subprocess
import json
import os
import tempfile

app = FastAPI(title="Lens Focus & AF Fine-Tune Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>ไม่พบไฟล์ index.html กรุณาอัปโหลดขึ้นระบบ</h3>"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # บันทึกไฟล์ที่อัปโหลดชั่วคราวเพื่อใช้กับ ExifTool และ OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        contents = await file.read()
        temp.write(contents)
        temp_path = temp.name

    try:
        # 1. ใช้ ExifTool ดึงข้อมูลเชิงลึกและ MakerNotes ทั้งหมดแบบ JSON (-j)
        cmd = ['exiftool', '-j', temp_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        exif_data = {}
        if res.returncode == 0 and res.stdout:
            exif_list = json.loads(res.stdout)
            if exif_list:
                exif_data = exif_list[0]

        # ดึงค่าเฉพาะที่สำคัญ
        camera_model = exif_data.get('Model', 'N/A')
        iso = exif_data.get('ISO', exif_data.get('ISOSpeedRatings', 'N/A'))
        f_number = exif_data.get('FNumber', 'N/A')
        focal_length = exif_data.get('FocalLength', 'N/A')
        
        # ดึงค่า AF Fine-tune จาก MakerNotes (รองรับหลายรูปแบบ Tag)
        af_fine_tune = exif_data.get('AFFineTuneAdj', exif_data.get('AFMicroAdj', exif_data.get('Microadjustment', 'N/A')))
        af_status = exif_data.get('AFFineTune', 'Off')

        # 2. คำนวณความคมชัดด้วย OpenCV (รองรับการดึงภาพ Preview จากไฟล์ RAW เช่น .NEF)
        sharpness_score = 0.0
        img_cv = None
        
        if file.filename.lower().endswith(('.nef', '.cr2', '.arw')):
            # ดึงภาพ Preview จากไฟล์ RAW
            cmd_img = ['exiftool', '-b', '-JpgFromRaw', temp_path]
            res_img = subprocess.run(cmd_img, stdout=subprocess.PIPE)
            if not res_img.stdout:
                cmd_img = ['exiftool', '-b', '-PreviewImage', temp_path]
                res_img = subprocess.run(cmd_img, stdout=subprocess.PIPE)
            
            if res_img.stdout:
                nparr = np.frombuffer(res_img.stdout, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            nparr = np.frombuffer(contents, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img_cv is not None:
            sharpness_score = float(cv2.Laplacian(img_cv, cv2.CV_64F).var())

        return {
            "filename": file.filename,
            "camera_model": camera_model,
            "sharpness_score": round(sharpness_score, 2),
            "af_fine_tune": af_fine_tune,
            "af_status": af_status,
            "exif_data": {
                "ISO": iso,
                "FNumber": f_number,
                "FocalLength": focal_length,
                "Lens": exif_data.get('LensID', exif_data.get('Lens', 'N/A'))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
