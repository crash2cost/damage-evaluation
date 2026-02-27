"""
Simple Web-based Annotation Tool for Car Damage Detection
Run: python annotation_tool.py
Then open: http://localhost:8000
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import base64

# Configuration
IMAGE_DIR = Path("detection-model/dataset-multiclass/train/images")
LABEL_DIR = Path("detection-model/real_labels")
CLASSES = [
    "hood_dent", "hood_scratch", "hood_crack",
    "bumper_dent", "bumper_scratch", "bumper_crack", 
    "door_dent", "door_scratch", "door_crack",
    "fender_dent", "fender_scratch", "fender_crack",
    "trunk_dent", "trunk_scratch", "trunk_crack",
    "glass_crack",
    "headlight_crack", "headlight_broken",
    "taillight_crack", "taillight_broken",
    "mirror_crack", "mirror_broken"
]

# Ensure label directory exists
LABEL_DIR.mkdir(parents=True, exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚗 Car Damage Annotation Tool</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 24px; }
        .progress {
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            width: 200px;
            height: 20px;
        }
        .progress-bar {
            background: linear-gradient(90deg, #00d4aa, #00ff88);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }
        .container {
            display: flex;
            height: calc(100vh - 60px);
            padding: 15px;
            gap: 15px;
            overflow: hidden;
        }
        .canvas-container {
            flex: 1;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 15px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        #canvas {
            cursor: crosshair;
            max-width: 100%;
            max-height: 100%;
            border-radius: 10px;
        }
        .sidebar {
            width: 240px;
            min-width: 240px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
            max-height: calc(100vh - 90px);
        }
        .panel {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 12px;
        }
        .panel h3 {
            margin-bottom: 10px;
            color: #00d4aa;
            font-size: 15px;
        }
        .class-btn {
            width: 100%;
            padding: 8px 10px;
            margin: 3px 0;
            border: 2px solid transparent;
            border-radius: 7px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
            font-size: 13px;
        }
        .class-btn:hover {
            background: rgba(255,255,255,0.2);
        }
        .class-btn.selected {
            border-color: #00d4aa;
            background: rgba(0, 212, 170, 0.3);
        }
        .box-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .box-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
            margin: 5px 0;
        }
        .delete-btn {
            background: #ff4444;
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        .nav-btn {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .nav-btn:hover { transform: scale(1.05); }
        .nav-btn:active { transform: scale(0.95); }
        .prev-btn { background: #ff6b6b; color: white; }
        .next-btn { background: #4ecdc4; color: white; }
        .save-btn { background: #00d4aa; color: white; }
        .severity-select {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: none;
            font-size: 14px;
            margin-top: 5px;
        }
        .status {
            text-align: center;
            padding: 10px;
            background: rgba(0,212,170,0.2);
            border-radius: 8px;
        }
        .keyboard-hint {
            font-size: 12px;
            color: rgba(255,255,255,0.6);
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 Car Damage Annotation</h1>
        <div>
            <span id="imageCounter">Image 1/100</span>
            <div class="progress">
                <div class="progress-bar" id="progressBar" style="width: 1%"></div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="canvas-container">
            <canvas id="canvas"></canvas>
        </div>
        
        <div class="sidebar">
            <div class="panel">
                <h3>📦 Select Class</h3>
                <div id="classButtons" style="max-height: 280px; overflow-y: auto;"></div>
            </div>
            
            <div class="panel">
                <h3>🎯 Severity (1-5)</h3>
                <select class="severity-select" id="severitySelect">
                    <option value="1">1 - Minor scratch</option>
                    <option value="2">2 - Light damage</option>
                    <option value="3" selected>3 - Moderate</option>
                    <option value="4">4 - Severe</option>
                    <option value="5">5 - Total loss</option>
                </select>
            </div>
            
            <div class="panel">
                <h3>📋 Annotations</h3>
                <div class="box-list" id="boxList"></div>
            </div>
            
            <div class="nav-buttons">
                <button class="nav-btn prev-btn" onclick="prevImage()">← Previous</button>
                <button class="nav-btn save-btn" onclick="saveLabels()">💾 Save</button>
                <button class="nav-btn next-btn" onclick="nextImage()">Next →</button>
            </div>
            
            <div class="status" id="status">Ready to annotate</div>
            
            <div class="keyboard-hint">
                <strong>Shortcuts:</strong> 1-9 = Select class • S = Save • ←/→ = Navigate<br>
                Click & drag to draw boxes • Right-click box to delete
            </div>
        </div>
    </div>

    <script>
        const CLASSES = CLASSES_PLACEHOLDER;
        let images = [];
        let currentIndex = 0;
        let currentClass = 0;
        let boxes = [];
        let isDrawing = false;
        let startX, startY;
        
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let img = new Image();
        let scale = 1;
        
        // Initialize class buttons
        const classButtonsDiv = document.getElementById('classButtons');
        CLASSES.forEach((cls, i) => {
            const btn = document.createElement('button');
            btn.className = 'class-btn' + (i === 0 ? ' selected' : '');
            btn.textContent = `${i+1}. ${cls}`;
            btn.onclick = () => selectClass(i);
            classButtonsDiv.appendChild(btn);
        });
        
        function selectClass(index) {
            currentClass = index;
            document.querySelectorAll('.class-btn').forEach((btn, i) => {
                btn.classList.toggle('selected', i === index);
            });
        }
        
        // Load images list
        fetch('/api/images')
            .then(r => r.json())
            .then(data => {
                images = data.images;
                loadImage(0);
            });
        
        function loadImage(index) {
            if (index < 0 || index >= images.length) return;
            currentIndex = index;
            
            img.onload = () => {
                const maxWidth = canvas.parentElement.clientWidth - 30;
                const maxHeight = canvas.parentElement.clientHeight - 30;
                scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1.5);
                
                canvas.width = img.width * scale;
                canvas.height = img.height * scale;
                
                drawCanvas();
                loadLabels();
            };
            img.src = '/api/image/' + images[index];
            
            document.getElementById('imageCounter').textContent = 
                `Image ${index + 1}/${images.length}`;
            document.getElementById('progressBar').style.width = 
                ((index + 1) / images.length * 100) + '%';
        }
        
        function loadLabels() {
            const imageName = images[currentIndex].replace(/\.[^/.]+$/, '');
            fetch('/api/labels/' + imageName)
                .then(r => r.json())
                .then(data => {
                    boxes = data.boxes || [];
                    updateBoxList();
                    drawCanvas();
                });
        }
        
        function drawCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Draw existing boxes
            boxes.forEach((box, i) => {
                const x = box.x * canvas.width;
                const y = box.y * canvas.height;
                const w = box.w * canvas.width;
                const h = box.h * canvas.height;
                
                ctx.strokeStyle = getClassColor(box.class);
                ctx.lineWidth = 3;
                ctx.strokeRect(x - w/2, y - h/2, w, h);
                
                // Label
                ctx.fillStyle = getClassColor(box.class);
                ctx.fillRect(x - w/2, y - h/2 - 22, 
                    ctx.measureText(CLASSES[box.class]).width + 10, 22);
                ctx.fillStyle = '#fff';
                ctx.font = '14px Arial';
                ctx.fillText(CLASSES[box.class], x - w/2 + 5, y - h/2 - 6);
            });
        }
        
        function getClassColor(classIndex) {
            const hue = (classIndex * 360 / CLASSES.length) % 360;
            return `hsl(${hue}, 80%, 50%)`;
        }
        
        function updateBoxList() {
            const list = document.getElementById('boxList');
            list.innerHTML = boxes.map((box, i) => `
                <div class="box-item">
                    <span style="color: ${getClassColor(box.class)}">${CLASSES[box.class]} (sev: ${box.severity})</span>
                    <button class="delete-btn" onclick="deleteBox(${i})">✕</button>
                </div>
            `).join('');
        }
        
        function deleteBox(index) {
            boxes.splice(index, 1);
            updateBoxList();
            drawCanvas();
        }
        
        // Drawing handlers
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            isDrawing = true;
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;
            
            drawCanvas();
            ctx.strokeStyle = getClassColor(currentClass);
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
            ctx.setLineDash([]);
        });
        
        canvas.addEventListener('mouseup', (e) => {
            if (!isDrawing) return;
            isDrawing = false;
            
            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;
            
            const x = (startX + endX) / 2 / canvas.width;
            const y = (startY + endY) / 2 / canvas.height;
            const w = Math.abs(endX - startX) / canvas.width;
            const h = Math.abs(endY - startY) / canvas.height;
            
            if (w > 0.01 && h > 0.01) {  // Minimum size
                boxes.push({
                    class: currentClass,
                    x: x, y: y, w: w, h: h,
                    severity: parseInt(document.getElementById('severitySelect').value)
                });
                updateBoxList();
            }
            drawCanvas();
        });
        
        function saveLabels() {
            const imageName = images[currentIndex].replace(/\.[^/.]+$/, '');
            fetch('/api/labels/' + imageName, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({boxes: boxes})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('status').textContent = '✓ Saved!';
                setTimeout(() => {
                    document.getElementById('status').textContent = 'Ready to annotate';
                }, 2000);
            });
        }
        
        function nextImage() {
            saveLabels();
            loadImage(currentIndex + 1);
        }
        
        function prevImage() {
            loadImage(currentIndex - 1);
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key >= '1' && e.key <= '9') {
                const index = parseInt(e.key) - 1;
                if (index < CLASSES.length) selectClass(index);
            }
            if (e.key === 's' || e.key === 'S') saveLabels();
            if (e.key === 'ArrowRight') nextImage();
            if (e.key === 'ArrowLeft') prevImage();
        });
    </script>
</body>
</html>
""".replace('CLASSES_PLACEHOLDER', json.dumps(CLASSES))


class AnnotationHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
            
        elif parsed.path == '/api/images':
            images = sorted([f.name for f in IMAGE_DIR.glob('*.jpg')] + 
                          [f.name for f in IMAGE_DIR.glob('*.png')] +
                          [f.name for f in IMAGE_DIR.glob('*.jpeg')])
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'images': images}).encode())
            
        elif parsed.path.startswith('/api/image/'):
            image_name = parsed.path.split('/')[-1]
            image_path = IMAGE_DIR / image_name
            if image_path.exists():
                self.send_response(200)
                content_type = 'image/jpeg' if image_name.lower().endswith('.jpg') else 'image/png'
                self.send_header('Content-Type', content_type)
                self.end_headers()
                with open(image_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
                
        elif parsed.path.startswith('/api/labels/'):
            label_name = parsed.path.split('/')[-1]
            if not self._is_safe_label_name(label_name):
                self.send_error(400, "Invalid label name")
                return
            label_path = LABEL_DIR / f"{label_name}.json"
            
            data = {'boxes': []}
            if label_path.exists():
                with open(label_path) as f:
                    data = json.load(f)
                    
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)
    
    @staticmethod
    def _is_safe_label_name(name):
        import re
        return bool(re.match(r'^[a-zA-Z0-9_.\-]+$', name)) and '..' not in name

    def do_POST(self):
        if self.path.startswith('/api/labels/'):
            label_name = self.path.split('/')[-1]
            if not self._is_safe_label_name(label_name):
                self.send_error(400, "Invalid label name")
                return
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            # Save JSON labels
            label_path = LABEL_DIR / f"{label_name}.json"
            with open(label_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Also save YOLO format
            yolo_path = LABEL_DIR / f"{label_name}.txt"
            with open(yolo_path, 'w') as f:
                for box in data.get('boxes', []):
                    f.write(f"{box['class']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}\n")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress logging


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    port = 8000
    print(f"""
╔════════════════════════════════════════════════════════════╗
║     🚗 Car Damage Annotation Tool                          ║
║                                                            ║
║     Open your browser to: http://localhost:{port}           ║
║                                                            ║
║     Classes: {len(CLASSES)} damage types                             ║
║     Images: {len(list(IMAGE_DIR.glob('*.jpg')))} files                                    ║
║                                                            ║
║     Press Ctrl+C to stop                                   ║
╚════════════════════════════════════════════════════════════╝
""")
    server = HTTPServer(('', port), AnnotationHandler)
    server.serve_forever()
