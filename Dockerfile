FROM python:3.9-slim

# ติดตั้ง System dependencies สำหรับ OpenCV และ ExifTool[cite: 16]
# เพิ่มคำสั่งลบแคช apt เพื่อลดขนาด Image
RUN apt-get update && apt-get install -y \
    exiftool \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอกและติดตั้ง Python dependencies[cite: 16]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดทั้งหมด[cite: 16]
COPY . .

# เริ่มต้นเซิร์ฟเวอร์[cite: 16]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
