from exiftool import ExifToolHelper

def extract_exif_data(file_path: str) -> dict:
    """
    ดึงข้อมูล EXIF เชิงลึกและ MakerNotes จากไฟล์ภาพด้วย ExifTool
    """
    with ExifToolHelper() as et:
        # อ่าน Metadata ทั้งหมดของไฟล์
        metadata = et.get_metadata(file_path)[0]
        
        # 1. ดึงยี่ห้อและรุ่นกล้อง
        make = metadata.get("EXIF:Make", "")
        model = metadata.get("EXIF:Model", "")
        camera = f"{make} {model}".strip() or "ไม่ระบุ"
        
        # 2. ดึงรุ่นเลนส์ (Fallback ตาม Tag ที่พบบ่อยในแต่ละค่าย)
        lens = (
            metadata.get("EXIF:LensModel") or 
            metadata.get("Composite:LensID") or 
            metadata.get("Nikon:LensSpec") or 
            metadata.get("EXIF:LensInfo") or 
            "ไม่ระบุ"
        )
        
        # 3. ดึงจุดโฟกัสที่ใช้งานจริงตอนถ่าย (MakerNotes เชิงลึก)
        af_point = (
            metadata.get("Nikon:AFPointsInFocus") or 
            metadata.get("Nikon:PrimaryAFPoint") or 
            metadata.get("Canon:AFPointUsed") or 
            metadata.get("Sony:AFPointSelected") or 
            "Center (Default)"
        )
        
        # 4. ดึงค่า AF Fine-tune / Microadjustment (ถ้ามี)
        af_fine_tune = (
            metadata.get("Nikon:AFTuneValue") or 
            metadata.get("Canon:AFMicroAdjustment") or 
            0
        )

        return {
            "camera": camera,
            "lens": lens,
            "af_point": str(af_point),
            "af_fine_tune": af_fine_tune,
            "f_number": metadata.get("EXIF:FNumber"),
            "exposure_time": metadata.get("EXIF:ExposureTime"),
            "iso": metadata.get("EXIF:ISO"),
            "focal_length": metadata.get("EXIF:FocalLength")
        }
