import cv2
import numpy as np
import json
from pathlib import Path
import glob
from PIL import Image
class LabelingTool:
    def __init__(self, image_dir, output_dir, progress_file='labeling_progress.json'):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted(glob.glob(str(self.image_dir / "*.jpg")))
        if not self.images:
            raise ValueError(f"No images found in {image_dir}")
        print(f" Found {len(self.images)} images")
        self.progress_file = progress_file
        self.progress = self.load_progress()
        self.current_idx = self.progress.get('last_index', 0)
        self.boxes = []
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.window_name = "Car Damage Labeling Tool"
        self.scale = 1.0
    def load_progress(self):
        if Path(self.progress_file).exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'last_index': 0, 'labeled_images': []}
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    def load_image(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        h, w = img.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            self.scale = max_dim / max(h, w)
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            img = cv2.resize(img, (new_w, new_h))
        else:
            self.scale = 1.0
        return img, img_path
    def load_existing_labels(self, img_path):
        label_path = self.output_dir / f"{Path(img_path).stem}.txt"
        self.boxes = []
        if label_path.exists():
            img = Image.open(img_path)
            img_w, img_h = img.size
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x_center = float(parts[1]) * img_w
                        y_center = float(parts[2]) * img_h
                        width = float(parts[3]) * img_w
                        height = float(parts[4]) * img_h
                        x1 = int((x_center - width/2) * self.scale)
                        y1 = int((y_center - height/2) * self.scale)
                        x2 = int((x_center + width/2) * self.scale)
                        y2 = int((y_center + height/2) * self.scale)
                        self.boxes.append((x1, y1, x2, y2))
    def save_labels(self, img_path):
        if len(self.boxes) == 0:
            return
        img = Image.open(img_path)
        img_w, img_h = img.size
        label_path = self.output_dir / f"{Path(img_path).stem}.txt"
        with open(label_path, 'w') as f:
            for box in self.boxes:
                x1, y1, x2, y2 = box
                x1 = x1 / self.scale
                y1 = y1 / self.scale
                x2 = x2 / self.scale
                y2 = y2 / self.scale
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f" Saved {len(self.boxes)} boxes to {label_path}")
    def draw_interface(self, img):
        display = img.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.drawing and self.start_point and self.current_point:
            cv2.rectangle(display, self.start_point, self.current_point, (0, 255, 255), 2)
        h, w = display.shape[:2]
        info = f"Image {self.current_idx + 1}/{len(self.images)} | Boxes: {len(self.boxes)}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        controls = "n:Next | p:Prev | d:Delete | c:Clear | s:Skip | q:Quit"
        cv2.putText(display, controls, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = (x, y)
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    self.boxes.append((x1, y1, x2, y2))
                    print(f" Added box {len(self.boxes)}: ({x1}, {y1}) -> ({x2}, {y2})")
                self.start_point = None
                self.current_point = None
    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        print("\n" + "="*60)
        print(" Car Damage Labeling Tool")
        print("="*60)
        print(__doc__)
        print("="*60 + "\n")
        while True:
            img, img_path = self.load_image(self.current_idx)
            self.load_existing_labels(img_path)
            print(f"\n Image {self.current_idx + 1}/{len(self.images)}: {Path(img_path).name}")
            print(f"   Loaded {len(self.boxes)} existing boxes")
            while True:
                display = self.draw_interface(img)
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n'):
                    self.save_labels(img_path)
                    self.progress['last_index'] = self.current_idx
                    if img_path not in self.progress['labeled_images']:
                        self.progress['labeled_images'].append(img_path)
                    self.save_progress()
                    if self.current_idx < len(self.images) - 1:
                        self.current_idx += 1
                    else:
                        print("\n Reached last image!")
                    break
                elif key == ord('p'):
                    if self.current_idx > 0:
                        self.current_idx -= 1
                    break
                elif key == ord('d'):
                    if self.boxes:
                        deleted = self.boxes.pop()
                        print(f" Deleted box: {deleted}")
                elif key == ord('c'):
                    self.boxes = []
                    print(" Cleared all boxes")
                elif key == ord('s'):
                    print("  Skipped (no damage)")
                    if self.current_idx < len(self.images) - 1:
                        self.current_idx += 1
                    break
                elif key == ord('q'):
                    self.save_labels(img_path)
                    self.progress['last_index'] = self.current_idx
                    self.save_progress()
                    print("\n Saved progress and exiting...")
                    cv2.destroyAllWindows()
                    return
        cv2.destroyAllWindows()
def main():
    import sys
    image_dir = "/Users/idolevi/.cache/huggingface/hub/datasets--harpreetsahota--CarDD/snapshots/56900bde8dddfe00eb7c03114a1d46e9105e3cdb/data"
    output_dir = "/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/manual_labels"
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    print(f" Image directory: {image_dir}")
    print(f" Output directory: {output_dir}")
    tool = LabelingTool(image_dir, output_dir)
    tool.run()
    print("\n" + "="*60)
    print(f" Labeling complete!")
    print(f"   Total images: {len(tool.images)}")
    print(f"   Labeled: {len(tool.progress['labeled_images'])}")
    print(f"   Labels saved to: {output_dir}")
    print("="*60)
if __name__ == "__main__":
    main()