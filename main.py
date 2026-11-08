from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import exifread
import io

app = FastAPI(title="Lens Focus Analysis API", description="API สำหรับวิเคราะห์ความคมชัดของภาพและอ่านค่า EXIF")

@app.get("/")
def read_root():
    return {"message": "Welcome to Lens Focus Analysis API. Use POST /analyze to upload an image."}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # อนุญาตให้รับไฟล์ RAW ซึ่ง browser/client อาจส่งมาเป็น application/octet-stream แทน image/
    if not file.content_type.startswith("image/") and not file.filename.lower().endswith(('.nef', '.cr2', '.tiff', '.raw')):
        raise HTTPException(status_code=400, detail="File must be an image or RAW file.")

    try:
        # อ่านข้อมูลไฟล์
        contents = await file.read()

        # 1. การอ่านค่า EXIF ด้วย exifread (รองรับ RAW file เช่น .NEF) แทน PIL/piexif
        extracted_exif = {}
        try:
            f_io = io.BytesIO(contents)
            tags = exifread.process_file(f_io, details=False)
            
            if 'EXIF FNumber' in tags:
                # แปลงเป็น string เพื่อให้อ่านง่าย (เช่น 14/5)
                extracted_exif["FNumber"] = str(tags['EXIF FNumber'])
                
            if 'EXIF FocalLength' in tags:
                extracted_exif["FocalLength"] = str(tags['EXIF FocalLength'])
                
            if 'EXIF ISOSpeedRatings' in tags:
                extracted_exif["ISO"] = int(str(tags['EXIF ISOSpeedRatings']))
                    
        except Exception as e:
            extracted_exif["error"] = f"Could not parse EXIF data: {str(e)}"

        # 2. การวิเคราะห์ความคมชัดด้วย OpenCV (Laplacian Variance)
        # แปลง bytes กลับเป็น numpy array เพื่อให้ OpenCV อ่านได้
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            raise ValueError("OpenCV cannot decode the image directly. If this is a RAW file, consider adding rawpy integration.")

        # แปลงเป็น Grayscale แล้วหาค่า Variance ของ Laplacian (ค่ายิ่งสูงยิ่งคมชัด)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # สร้างผลลัพธ์ JSON
        return {
            "filename": file.filename,
            "sharpness_score": float(sharpness_score), # คะแนนความคมชัดสำหรับการปรับจูนเลนส์
            "exif_data": extracted_exif
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
