FROM python:3.9-slim

# อัปเดตและติดตั้ง exiftool รวมถึงไลบรารีระบบที่ OpenCV จำเป็นต้องใช้
RUN apt-get update && apt-get install -y \
    exiftool \
    libimage-exiftool-perl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอกและติดตั้งไลบรารี Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดและไฟล์ทั้งหมดลง Container
COPY . .

EXPOSE 8501

# คำสั่งรัน Streamlit เมื่อ Container เริ่มทำงาน
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
