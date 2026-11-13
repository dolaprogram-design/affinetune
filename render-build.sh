#!/usr/bin/env bash
# exit on error
set -o errexit

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🛠️ Downloading and setting up ExifTool for Render..."
EXIFTOOL_VERSION=12.76
wget https://exiftool.org/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz
tar -zxvf Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz

# ย้ายโฟลเดอร์ที่แตกไฟล์แล้วมาตั้งชื่อใหม่ว่า exiftool_bin
mv Image-ExifTool-${EXIFTOOL_VERSION} exiftool_bin

# ลบไฟล์ zip ทิ้งเพื่อประหยัดพื้นที่
rm Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz

echo "✅ ExifTool setup completed."
