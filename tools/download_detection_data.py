from roboflow import Roboflow
import os

target_dir = "detection-model/dataset"
os.makedirs(target_dir, exist_ok=True)

rf = Roboflow(api_key="CNU1dSHFkhSLLwQyFIe8")

print("Downloading the damage detection dataset (YOLO)...")

project = rf.workspace("francesco-delli-paoli-jqidv").project("yolo-car-damage-detection")
dataset = project.version(1).download("yolov8", location=target_dir)

print(f"Download complete! Files are in directory: {target_dir}")