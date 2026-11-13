#!/usr/bin/env bash
set -o errexit
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo "🛠️ Downloading and setting up ExifTool from GitHub Mirror..."
EXIFTOOL_VERSION=12.76
wget https://github.com/exiftool/exiftool/archive/refs/tags/${EXIFTOOL_VERSION}.tar.gz -O Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz
tar -zxvf Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz
mv exiftool-${EXIFTOOL_VERSION} exiftool_bin
rm Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz
echo "✅ ExifTool setup completed successfully!"
