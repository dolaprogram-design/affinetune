from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import piexif
import io

app = FastAPI(title="Lens Focus Analysis API", description="API สำหรับวิเคราะห์ความคมชัดของภาพและอ่านค่า EXIF")

@app.get("/")
def read_root():
    return {"message": "Welcome to Lens Focus Analysis API. Use POST /analyze to upload an image."}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # อ่านข้อมูลไฟล์
        contents = await file.read()

        # 1. การอ่านค่า EXIF ด้วย Pillow และ piexif
        image_pil = Image.open(io.BytesIO(contents))
        exif_dict = {"Exif": {}}
        extracted_exif = {}
        
        try:
            if "exif" in image_pil.info:
                exif_dict = piexif.load(image_pil.info["exif"])
                
                # ดึงข้อมูล EXIF พื้นฐานที่สำคัญ (ต้องแปลงค่าจาก Rational/Tuple เป็นตัวเลข)
                def format_rational(val):
                    if isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                        return val[0] / val[1]
                    return val

                if "Exif" in exif_dict and exif_dict["Exif"]:
                    exif_data = exif_dict["Exif"]
                    
                    # 33437: FNumber (รูรับแสง)
                    if piexif.ExifIFD.FNumber in exif_data:
                        extracted_exif["FNumber"] = format_rational(exif_data[piexif.ExifIFD.FNumber])
                    
                    # 37386: FocalLength (ระยะโฟกัส)
                    if piexif.ExifIFD.FocalLength in exif_data:
                        extracted_exif["FocalLength"] = format_rational(exif_data[piexif.ExifIFD.FocalLength])
                        
                    # 34855: ISOSpeedRatings
                    if piexif.ExifIFD.ISOSpeedRatings in exif_data:
                        extracted_exif["ISO"] = exif_data[piexif.ExifIFD.ISOSpeedRatings]
                        
        except Exception as e:
            # หากไม่มี EXIF หรืออ่านไม่ได้ ให้ข้ามไป
            extracted_exif["error"] = "Could not parse EXIF data"

        # 2. การวิเคราะห์ความคมชัดด้วย OpenCV (Laplacian Variance)
        # แปลง bytes กลับเป็น numpy array เพื่อให้ OpenCV อ่านได้
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            raise ValueError("OpenCV cannot decode the image.")

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
