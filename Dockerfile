FROM python:3.10-slim

# ติดตั้งเฉพาะ ExifTool สำหรับอ่านค่าพิกัดโฟกัสเชิงลึก
RUN apt-get update && apt-get install -y \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
