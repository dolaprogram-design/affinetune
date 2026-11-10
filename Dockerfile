# ใช้ Python Image พื้นฐาน
FROM python:3.10-slim

# ติดตั้ง ExifTool และไลบรารีที่จำเป็นสำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    libimage-exiftool-perl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# รัน FastAPI ผ่าน Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
