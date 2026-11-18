FROM python:3.9-slim

# บังคับให้แสดง Python Log ทันทีไม่ค้างใน Buffer
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ติดตั้งไลบรารี และล็อก OpenCV ไว้ที่เวอร์ชัน 4.10.0.84 (ไม่ให้เด้งไป 5.x / NumPy 2)
RUN pip install --no-cache-dir -r requirements.txt && \
    (pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true) && \
    pip install --no-cache-dir "opencv-python-headless==4.10.0.84"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
