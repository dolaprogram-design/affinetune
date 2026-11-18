FROM python:3.9-slim

# 1. ติดตั้งแพ็กเกจระบบ (ส่วน apt-get เดิมของคุณ)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. กำหนดโฟลเดอร์ทำงานก่อน
WORKDIR /app

# 3. คัดลอกไฟล์ requirements.txt เข้ามาก่อนสั่ง pip install
COPY requirements.txt .

# 4. ติดตั้งไลบรารีและล้าง OpenCV ตัวที่ซ้ำซ้อน
RUN pip install --no-cache-dir -r requirements.txt && \
    (pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true) && \
    pip install --no-cache-dir opencv-python-headless

# 5. คัดลอกโค้ดทั้งหมดของโปรเจกต์เข้า Container
COPY . .

# 6. คำสั่งรันแอป (คงไว้ตามเดิมของคุณ เช่น uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
