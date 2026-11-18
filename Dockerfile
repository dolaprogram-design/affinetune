FROM python:3.9-slim

# ติดตั้ง System dependencies สำหรับ OpenCV และ ExifTool[cite: 16]
# เพิ่มคำสั่งลบแคช apt เพื่อลดขนาด Image
RUN pip install -r requirements.txt && \
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless && \
pip install --no-cache-dir opencv-python-headless
WORKDIR /app

# คัดลอกและติดตั้ง Python dependencies[cite: 16]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดทั้งหมด[cite: 16]
COPY . .

# เริ่มต้นเซิร์ฟเวอร์[cite: 16]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
