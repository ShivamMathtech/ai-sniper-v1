# 🤖 AI Sniper System

A real-time computer vision application for person detection, head stability tracking, ROI (Region of Interest) selection, and precise coordinate extraction using MediaPipe and OpenCV.

---

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [Architecture](#architecture)
- [Output Format](#output-format)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Real-time Person Detection** | Detects persons using MediaPipe Pose estimation |
| **Target Selection** | Click-to-select any detected person |
| **Head Stability Analysis** | Tracks nose position variance over time (0-100% score) |
| **ROI Selection** | Draw polygonal regions of interest with mouse |
| **Coordinate Extraction** | Exports pixel + normalized coordinates as JSON |
| **Visual Feedback** | Color-coded stability indicators, crosshairs, info panels |

---

## 📦 Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

---

## 🚀 Installation

### 1. Clone or Download

```bash
git clone <repository-url>
cd ai-sniper
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4. Verify Installation

```bash
python -c "import cv2; import mediapipe; import numpy; print('All dependencies installed!')"
```

---

## 🎮 Usage

### Basic Usage (Webcam)

```bash
python ai_sniper.py
```

### Use Video File Instead of Webcam

Edit `ai_sniper.py` and change line:

```python
cap = cv2.VideoCapture(0)  # 0 = webcam
# Change to:
cap = cv2.VideoCapture("path/to/your/video.mp4")
```

### Adjust Camera Resolution

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height
```

---

## 🕹️ Controls

| Key / Action | Function |
|-------------|----------|
| **Left Click** | Select a person as target |
| **R** | Toggle ROI drawing mode |
| **Enter** | Confirm ROI polygon (while in ROI mode) |
| **C** | Clear target selection |
| **S** | Save current target coordinates to JSON file |
| **D** | Toggle debug/info panel |
| **Q** or **Esc** | Quit application |

### Workflow

1. **Start the app** — Persons are automatically detected
2. **Click on a person** — They become the locked target (green box)
3. **Watch stability** — Wait for score to reach 80%+ (STABLE)
4. **(Optional) Set ROI** — Press `R`, click points to define area, press `Enter`
5. **Save coordinates** — Press `S` to export target data as JSON

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AI Sniper System                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │   Webcam    │───▶│   Frame     │───▶│  Detect   │  │
│  │   / Video   │    │   Capture   │    │  Persons  │  │
│  └─────────────┘    └─────────────┘    └─────┬─────┘  │
│                                               │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────▼─────┐  │
│  │  Stability  │◀───│   Select    │◀───│  Target   │  │
│  │  Analysis   │    │   Target    │    │  Lock     │  │
│  └──────┬──────┘    └─────────────┘    └───────────┘  │
│         │                                               │
│  ┌──────▼──────┐    ┌─────────────┐    ┌───────────┐  │
│  │    ROI      │◀───│  Coordinate │───▶│   Save    │  │
│  │  Selection  │    │  Extraction │    │   JSON    │  │
│  └─────────────┘    └─────────────┘    └───────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              OpenCV Visualization                 │   │
│  │  (Bounding boxes, crosshairs, info panel, ROI)   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Class Structure

```
AISniper
├── detect_persons(frame)          → Detect persons via MediaPipe Pose
├── analyze_head_stability(id, pos) → Calculate stability score
├── set_roi(points)                 → Define polygonal ROI
├── point_in_roi(point)            → Check if point inside ROI
├── get_target_coordinates(shape)   → Extract target data
├── render(frame)                  → Main visualization pipeline
│   ├── draw_crosshair()
│   ├── draw_stability_bar()
│   ├── draw_roi()
│   └── draw_info_panel()
```

---

## 📤 Output Format

When you press **S**, a JSON file is saved:

```json
{
  "pixel": {
    "nose": [640, 360],
    "center": [640, 400],
    "bbox": [520, 240, 760, 560]
  },
  "normalized": {
    "nose": [0.5, 0.5],
    "center": [0.5, 0.5556]
  },
  "in_roi": true,
  "stability": {
    "score": 92.5,
    "status": "STABLE",
    "variance": 12.34,
    "velocity": 1.23,
    "max_velocity": 5.67,
    "samples": 30
  },
  "timestamp": 12345.6789
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `pixel.nose` | Nose position in pixel coordinates (x, y) |
| `pixel.center` | Bounding box center in pixel coordinates |
| `pixel.bbox` | Bounding box [x_min, y_min, x_max, y_max] |
| `normalized.nose` | Nose position normalized to [0, 1] |
| `in_roi` | Whether target is inside defined ROI |
| `stability.score` | Stability score 0-100% (higher = more stable) |
| `stability.status` | `STABLE` / `MODERATE` / `UNSTABLE` |
| `stability.variance` | Position variance (lower = more stable) |
| `stability.velocity` | Average movement speed between frames |
| `timestamp` | Capture time in seconds |

---

## ⚙️ Customization

### Adjust Stability Sensitivity

```python
# In __init__
sniper = AISniper(stability_window=60)  # More frames = smoother but slower
```

### Change Detection Confidence

```python
self.pose = self.mp_pose.Pose(
    min_detection_confidence=0.7,  # Increase for fewer false positives
    min_tracking_confidence=0.5
)
```

### Modify Colors

```python
self.colors = {
    'selected': (0, 255, 0),     # Green
    'unselected': (0, 165, 255),  # Orange
    'unstable': (0, 0, 255),      # Red
    'roi': (255, 0, 255),        # Magenta
    'crosshair': (0, 255, 255)   # Cyan
}
```

### Disable Mirror Effect

```python
# Remove or comment out:
frame = cv2.flip(frame, 1)
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'mediapipe'"

```bash
pip install mediapipe
# Or with specific version:
pip install mediapipe==0.10.0
```

### "Error: Could not open video source"

- Check if webcam is connected
- Try different camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- On Linux, check permissions: `sudo usermod -a -G video $USER`

### Low FPS / Laggy Performance

```python
# Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Or reduce MediaPipe complexity
self.pose = self.mp_pose.Pose(model_complexity=0)  # 0=light, 1=full, 2=heavy
```

### Stability Score Always Low

- Ensure good lighting
- Stand still for 1-2 seconds after selection
- Increase `stability_window` for smoother readings
- Check if camera is mounted securely (no vibration)

### Multiple People Not Detected

MediaPipe Pose detects one person per frame by default. For multi-person detection, consider using:
- MediaPipe Holistic
- YOLO + pose estimation
- Or run multiple Pose instances

---

## 📊 Performance Tips

| Setting | Recommended Value | Effect |
|---------|------------------|--------|
| Resolution | 640x480 or 1280x720 | Balance quality/speed |
| Model Complexity | 0 or 1 | 0=fastest, 2=most accurate |
| Stability Window | 30 frames | ~1 second at 30 FPS |
| Detection Confidence | 0.5 | Lower = more detections, more false positives |

---

## 📝 License

MIT License — Free for personal and commercial use.

---

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests. Common improvements:

- [ ] Multi-person detection support
- [ ] Head orientation (yaw/pitch/roll) tracking
- [ ] Auto-target selection (closest to center)
- [ ] UDP/TCP streaming of coordinates
- [ ] GUI with tkinter/PyQt
- [ ] Record video with overlays

---

## 📧 Support

For issues or questions, please open an issue on the repository.

---

**Built with ❤️ using OpenCV + MediaPipe + NumPy**
