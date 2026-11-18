FROM python:3.9-slim

# บังคับให้แสดง Python Log ทันทีไม่ค้างใน Buffer
ENV PYTHONUNBUFFERED=1

# ติดตั้งไลบรารีระบบที่จำเป็นสำหรับ OpenCV และคำสั่ง exiftool
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ติดตั้งไลบรารีทั้งหมดและป้องกันแพ็กเกจ OpenCV ชนกัน
RUN pip install --no-cache-dir -r requirements.txt && \
    (pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true) && \
    pip install --no-cache-dir "opencv-python-headless==4.10.0.84"

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
