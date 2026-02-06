from roboflow import Roboflow
import os

# הגדרת נתיב היעד לפי המבנה שלנו
target_dir = "detection-model/dataset"
os.makedirs(target_dir, exist_ok=True)

# החלף כאן את המפתח במפתח הפרטי שלך מ-Roboflow
rf = Roboflow(api_key="CNU1dSHFkhSLLwQyFIe8")

print("⏳ מוריד את הדאטהסט לזיהוי נזקים (YOLO)...")

# הורדת הפרויקט הספציפי מהקישור שנתת
project = rf.workspace("francesco-delli-paoli-jqidv").project("yolo-car-damage-detection")
dataset = project.version(1).download("yolov8", location=target_dir)

print(f"✅ ההורדה הושלמה! הקבצים נמצאים בתיקייה: {target_dir}")