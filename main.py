from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import subprocess
import json
import os
import tempfile
import base64

app = FastAPI(title="Pro Lens & Camera Analytics Engine")

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

@app.post("/analyze-batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    # จำกัดไม่ให้เกิน 10 ไฟล์ต่อครั้งตามเงื่อนไข
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="สามารถอัปโหลดไฟล์พร้อมกันได้สูงสุดไม่เกิน 10 ไฟล์ครับ")

    results = []
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name

        try:
            # 1. ดึงข้อมูล Exif และ MakerNotes เชิงลึกด้วย ExifTool
            cmd = ['exiftool', '-j', temp_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            exif_data = {}
            if res.returncode == 0 and res.stdout:
                exif_list = json.loads(res.stdout)
                if exif_list:
                    exif_data = exif_list[0]

            camera_model = exif_data.get('Model', 'N/A')
            iso = exif_data.get('ISO', exif_data.get('ISOSpeedRatings', 'N/A'))
            f_number = exif_data.get('FNumber', 'N/A')
            focal_length = exif_data.get('FocalLength', 'N/A')
            af_fine_tune = exif_data.get('AFFineTuneAdj', exif_data.get('AFMicroAdj', 'N/A'))
            af_status = exif_data.get('AFFineTune', 'Off')
            af_point = exif_data.get('PrimaryAFPoint', exif_data.get('AFPointsUsed', 'Center Focus'))
            lens_info = exif_data.get('LensID', exif_data.get('Lens', 'N/A'))

            # 2. แปลงไฟล์ RAW เป็นภาพถ่าย (ผ่าน JpgFromRaw หรือ PreviewImage) หรืออ่านภาพ JPEG ปกติด้วย OpenCV
            sharpness_score = 0.0
            img_cv = None
            
            if file.filename.lower().endswith(('.nef', '.cr2', '.arw', '.raw', '.dng')):
                cmd_img = ['exiftool', '-b', '-JpgFromRaw', temp_path]
                res_img = subprocess.run(cmd_img, stdout=subprocess.PIPE)
                if not res_img.stdout:
                    cmd_img = ['exiftool', '-b', '-PreviewImage', temp_path]
                    res_img = subprocess.run(cmd_img, stdout=subprocess.PIPE)
                
                if res_img.stdout:
                    nparr = np.frombuffer(res_img.stdout, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                nparr = np.frombuffer(contents, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            annotated_base64 = ""
            if img_cv is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                # วาดกรอบจุดโฟกัส (AF Point Visualization) บนภาพถ่ายที่แปลงมาจากไฟล์ดิบ
                h, w, _ = img_cv.shape
                cx, cy = w // 2, h // 2
                bw, bh = int(w * 0.08), int(h * 0.08)
                cv2.rectangle(img_cv, (cx - bw, cy - bh), (cx + bw, cy + bh), (0, 0, 255), max(4, int(w*0.004)))
                cv2.putText(img_cv, f"AF: {af_point}", (cx - bw, cy - bh - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w*0.001), (0, 0, 255), 2)

                # เข้ารหัสภาพเป็น Base64 เพื่อส่งกลับไปแสดงผลพรีวิวที่หน้าเว็บ
                _, buffer = cv2.imencode('.jpg', img_cv)
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')

            # ประเมินเกณฑ์คุณภาพความคมชัด
            rating = "🔴 เบลอ" if sharpness_score < 100 else "🟡 พอใช้" if sharpness_score < 500 else "🟢 คมชัดดี" if sharpness_score < 1500 else "⭐ ยอดเยี่ยม"

            results.append({
                "filename": file.filename,
                "camera_model": camera_model,
                "lens": lens_info,
                "sharpness": round(sharpness_score, 2),
                "quality": rating,
                "iso": iso,
                "f_number": f_number,
                "focal_length": focal_length,
                "af_fine_tune": af_fine_tune,
                "af_status": af_status,
                "af_point": af_point,
                "annotated_image": f"data:image/jpeg;base64,{annotated_base64}" if annotated_base64 else ""
            })

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return JSONResponse(content={"results": results})
