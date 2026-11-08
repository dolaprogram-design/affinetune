from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import rawpy
import numpy as np
import exiftool
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Pro Lens Calibration API on Render is running!"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        af_x_coord, af_y_coord = None, None
        afma_value = None
        lens_model = "Unknown"
        focal_length = "Unknown"
        camera_model = "Unknown"
        aperture = "Unknown"
        shutter = "Unknown"
        iso = "Unknown"

        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(temp_filename)[0]

            for key, value in metadata.items():
                k_lower = key.lower()
                
                # 1. ค้นหาพิกัดจุดโฟกัส
                if "afareaxposition" in k_lower or "afpointpositionx" in k_lower:
                    af_x_coord = int(value)
                if "afareayposition" in k_lower or "afpointpositiony" in k_lower:
                    af_y_coord = int(value)
                
                # 2. ค้นหาค่า AF Fine Tune (ครอบคลุมทุกค่าย)
                if any(x in k_lower for x in ["affinetune", "afmicroadj", "aftuning"]):
                    try:
                        afma_value = float(value)
                    except:
                        pass
                
                # 3. ค้นหาข้อมูลเลนส์และรุ่นกล้อง
                if "lensmodel" in k_lower or "lenstype" in k_lower:
                    lens_model = str(value)
                if "model" in k_lower and "camera" not in k_lower and camera_model == "Unknown":
                    camera_model = str(value)
                if "focallength" in k_lower and "35mm" not in k_lower:
                    focal_length = str(value)
                if "fnumber" in k_lower or "aperture" in k_lower:
                    aperture = str(value)
                if "exposuretime" in k_lower or "shutterspeed" in k_lower:
                    shutter = str(value)
                if "iso" in k_lower:
                    iso = str(value)

        # จัดรูปแบบข้อความเลนส์และทางยาวโฟกัส
        if focal_length != "Unknown" and not focal_length.endswith("mm"):
            focal_length += " mm"

        is_raw = temp_filename.lower().endswith(('.nef', '.cr2', '.arw', '.cr3', '.dng'))
        
        if is_raw:
            with rawpy.imread(temp_filename) as raw:
                rgb_image = raw.postprocess(half_size=True) 
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            img = cv2.imread(temp_filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        h, w = gray.shape
        scaling_factor = 2 if is_raw else 1

        if af_x_coord is not None and af_y_coord is not None:
            center_x = af_x_coord // scaling_factor
            center_y = af_y_coord // scaling_factor
            focus_status = "Used AF Point Coordinates"
        else:
            center_x = w // 2
            center_y = h // 2
            focus_status = "Used Center Crop (AF Point not found)"

        box_size = 400
        half_box = box_size // 2
        start_x = max(0, center_x - half_box)
        end_x = min(w, center_x + half_box)
        start_y = max(0, center_y - half_box)
        end_y = min(h, center_y + half_box)

        roi = gray[start_y:end_y, start_x:end_x]
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        sharpness_score = float(np.var(laplacian))

        result = {
            "filename": file.filename,
            "sharpness_score": round(sharpness_score, 2),
            "afma_value": afma_value,
            "lens_model": lens_model,
            "camera_model": camera_model,
            "focal_length": focal_length,
            "aperture": aperture,
            "shutter": shutter,
            "iso": iso,
            "focus_status": focus_status,
            "af_coords": {"x": af_x_coord, "y": af_y_coord}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return result
