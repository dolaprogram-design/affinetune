from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import exifread
import io
import os

app = FastAPI(title="Lens Focus Analysis API", description="API สำหรับวิเคราะห์ความคมชัดของภาพและอ่านค่า EXIF")

# เปิดใช้งาน CORS เพื่อให้หน้าเว็บเชื่อมต่อได้อย่างไร้รอยต่อ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    เมื่อเข้าหน้าเว็บหลัก จะทำการเปิดและแสดงผลหน้า index.html 
    ที่เป็น Dashboard 4 ส่วนหลักอัตโนมัติ
    """
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return f.read()
        else:
            return """
            <html>
                <head><title>Lens Focus Analysis</title></head>
                <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                    <h2>API ระบบวิเคราะห์ความคมชัดพร้อมใช้งานแล้ว</h2>
                    <p>ยังไม่พบไฟล์ <b>index.html</b> ในระบบ กรุณาอัปโหลดไฟล์ index.html เพื่อแสดงหน้า Dashboard</p>
                </body>
            </html>
            """
    except Exception as e:
        return f"Error loading index.html: {str(e)}"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # อนุญาตให้รับไฟล์ภาพทั่วไปและไฟล์ RAW
    if not file.content_type.startswith("image/") and not file.filename.lower().endswith(('.nef', '.cr2', '.tiff', '.raw')):
        raise HTTPException(status_code=400, detail="File must be an image or RAW file.")

    try:
        # อ่านข้อมูลไฟล์
        contents = await file.read()

        # 1. การอ่านค่า EXIF ด้วย exifread (รองรับไฟล์ RAW เช่น .NEF)
        extracted_exif = {}
        try:
            f_io = io.BytesIO(contents)
            tags = exifread.process_file(f_io, details=False)
            
            if 'EXIF FNumber' in tags:
                extracted_exif["FNumber"] = str(tags['EXIF FNumber'])
                
            if 'EXIF FocalLength' in tags:
                extracted_exif["FocalLength"] = str(tags['EXIF FocalLength'])
                
            if 'EXIF ISOSpeedRatings' in tags:
                extracted_exif["ISO"] = int(str(tags['EXIF ISOSpeedRatings']))
                    
        except Exception as e:
            extracted_exif["error"] = f"Could not parse EXIF data: {str(e)}"

        # 2. การวิเคราะห์ความคมชัดด้วย OpenCV (Laplacian Variance)
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            # กรณีไฟล์ RAW บางประเภทที่ OpenCV ถอดรหัสตรงๆ ไม่ได้ ให้กำหนดค่าเริ่มต้นป้องกันแครช
            sharpness_score = 0.0
        else:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ส่งผลลัพธ์กลับในรูปแบบ JSON
        return {
            "filename": file.filename,
            "sharpness_score": sharpness_score,
            "exif_data": extracted_exif
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
