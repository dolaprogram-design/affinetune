#!/usr/bin/env bash
# exit on error
set -o errexit

echo "ሲ Installing Python dependencies..."
pip install -r requirements.txt

echo "ቀ Downloading and setting up ExifTool for Render..."
EXIFTOOL_VERSION=12.76
# Ł     URL Ł                                                      archive Ł                                                                                                                  
if wget -q --spider https://exiftool.org/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz; then
    wget https://exiftool.org/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz
else
    wget https://exiftool.org/archive/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz
fi

tar -zxvf Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz

mv Image-ExifTool-${EXIFTOOL_VERSION} exiftool_bin
rm Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz

echo "ā ExifTool setup completed."
