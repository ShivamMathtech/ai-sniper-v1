
import sys
import random
import math
import time
import json
from datetime import datetime
from collections import deque
from enum import Enum, auto

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QGroupBox, QTextEdit, QPlainTextEdit, QSlider, QMenu,
    QInputDialog, QMessageBox, QTreeWidget, QTreeWidgetItem,
    QStatusBar, QComboBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPoint, QPointF, QRectF
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont,
    QLinearGradient, QRadialGradient, QPolygonF, QKeyEvent
)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================
class SystemMode(Enum):
    STANDBY = auto()
    SCANNING = auto()
    TRACKING = auto()
    LOCKED = auto()
    SOLVED = auto()
    FIRING = auto()
    COOLDOWN = auto()

class TargetPriority(Enum):
    LOW = (1, "#8b9bb4")
    MEDIUM = (2, "#ffaa00")
    HIGH = (3, "#ff4444")
    CRITICAL = (4, "#ff00ff")

class ROIShape(Enum):
    RECTANGLE = auto()
    CIRCLE = auto()
    POLYGON = auto()

# ============================================================================
# YOLO DETECTOR
# ============================================================================
class YOLODetector:
    def __init__(self, use_real_model=False, model_path="yolov8n.pt"):
        self.use_real = use_real_model
        self.model_path = model_path
        self.confidence_threshold = 0.5
        self.classes = ['person', 'vehicle', 'equipment', 'unknown']
        self.colors = {
            'person': (0, 255, 136),
            'vehicle': (255, 165, 0),
            'equipment': (0, 191, 255),
            'unknown': (255, 255, 255)
        }
        self.model = None

        if self.use_real:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[SYSTEM] Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load YOLO model: {e}")
                self.use_real = False

    def detect(self, frame, roi=None):
        if self.use_real and self.model is not None:
            return self._real_detect(frame, roi)
        return self._simulate_detect(frame, roi)

    def _real_detect(self, frame, roi=None):
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if roi is not None:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if not roi.contains(QPointF(cx, cy)):
                        continue

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': self.model.names[cls],
                    'id': f'T{len(detections)+1:02d}',
                    'priority': self._calculate_priority(cls, conf),
                    'distance': random.uniform(200, 1500),
                    'speed': random.uniform(0, 3.5),
                    'direction': random.uniform(0, 360)
                })
        return detections

    def _simulate_detect(self, frame, roi=None):
        h, w = frame.shape[:2]
        detections = []

        if roi is not None:
            bounds = roi.boundingRect()
            min_x, max_x = int(bounds.x()), int(bounds.x() + bounds.width())
            min_y, max_y = int(bounds.y()), int(bounds.y() + bounds.height())
        else:
            min_x, max_x = int(w*0.15), int(w*0.85)
            min_y, max_y = int(h*0.15), int(h*0.75)

        num_targets = random.randint(1, 5)

        for i in range(num_targets):
            cx = random.randint(min_x, max_x)
            cy = random.randint(min_y, max_y)
            bw = random.randint(40, 140)
            bh = random.randint(80, 220)

            x1 = max(0, cx - bw//2)
            y1 = max(0, cy - bh//2)
            x2 = min(w, cx + bw//2)
            y2 = min(h, cy + bh//2)

            confidence = random.uniform(0.55, 0.99)
            class_name = random.choice(self.classes)
            priority = self._calculate_priority(class_name, confidence)

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class': class_name,
                'id': f'T{i+1:02d}',
                'priority': priority,
                'distance': random.uniform(200, 1500),
                'speed': random.uniform(0, 3.5),
                'direction': random.uniform(0, 360),
                'vitals': random.choice(['HEAD', 'CHEST', 'HEART/LUNGS', 'PELVIS', 'EXTREMITY']),
                'lethality': random.uniform(0.3, 0.99)
            })

        return detections

    def _calculate_priority(self, cls, conf):
        if cls == 'person' and conf > 0.9:
            return TargetPriority.CRITICAL
        elif cls == 'person':
            return TargetPriority.HIGH
        elif conf > 0.85:
            return TargetPriority.MEDIUM
        return TargetPriority.LOW

# ============================================================================
# BALLISTICS ENGINE
# ============================================================================
class BallisticsEngine:
    def __init__(self):
        self.gravity = 9.80665
        self.muzzle_velocity = 850
        self.bullet_mass = 0.010
        self.bullet_diameter = 0.00782

    def calculate_solution(self, distance, wind_speed, wind_dir, temp, humidity, pressure, 
                          altitude=0, target_speed=0, target_angle=0):
        temp_k = temp + 273.15
        pressure_pa = pressure * 100
        rho = (pressure_pa / (287.05 * temp_k)) * (1 - 0.0065 * altitude / temp_k) ** 5.2561

        v = self.muzzle_velocity
        dt = 0.001
        t = 0
        x = 0
        y = 0
        vy = 0

        while x < distance and v > 100:
            v_mag = math.sqrt(v**2 + vy**2)
            drag = 0.5 * rho * v_mag**2 * 0.295 * (math.pi * (self.bullet_diameter/2)**2)
            ax = -drag / self.bullet_mass * (v / v_mag)
            ay = -self.gravity - drag / self.bullet_mass * (vy / v_mag)
            v += ax * dt
            vy += ay * dt
            x += v * dt
            y += vy * dt
            t += dt

        time_of_flight = t
        drop_mrad = math.atan(abs(y) / distance) * 1000

        wind_rad = math.radians(wind_dir)
        wind_x = wind_speed * math.sin(wind_rad)
        wind_deflection = wind_x * t * 0.1
        windage_mrad = math.atan(abs(wind_deflection) / distance) * 1000
        wind_sign = 1 if wind_rad < math.pi else -1

        lead = target_speed * time_of_flight * math.sin(math.radians(target_angle))
        lead_mrad = math.atan(lead / distance) * 1000

        impact_vel = math.sqrt(v**2 + vy**2)
        impact_energy = 0.5 * self.bullet_mass * impact_vel**2

        confidence = min(99.9, 95 + random.uniform(-2, 4))
        if distance > 1000:
            confidence -= (distance - 1000) / 100
        if wind_speed > 10:
            confidence -= wind_speed * 0.5

        return {
            'distance': distance,
            'time_of_flight': round(time_of_flight, 3),
            'drop': round(drop_mrad, 2),
            'windage': round(windage_mrad * wind_sign, 2),
            'lead': round(lead_mrad, 2),
            'impact_velocity': round(impact_vel, 1),
            'impact_energy': round(impact_energy, 1),
            'confidence': round(max(0, confidence), 1),
            'air_density': round(rho, 4),
            'holdover': round(drop_mrad, 2),
            'windage_hold': round(abs(windage_mrad), 2),
            'coriolis': round(distance * 0.00001, 4),
            'spin_drift': round(distance * 0.00005, 4)
        }

    def calculate_trajectory_points(self, distance, steps=50):
        points = []
        max_dist = max(distance * 1.3, 100)
        for i in range(steps):
            t = i / (steps - 1)
            x = t * max_dist
            y = -0.5 * self.gravity * (x / self.muzzle_velocity)**2
            points.append((x, y))
        return points

# ============================================================================
# ENVIRONMENTAL SENSOR
# ============================================================================
class EnvironmentSensor:
    def __init__(self):
        self.wind_speed = 4.2
        self.wind_gust = 6.8
        self.wind_dir = 315
        self.temperature = 18.6
        self.humidity = 62
        self.pressure = 1013
        self.visibility = 8.7
        self.altitude = 245
        self.history = {
            'wind_speed': deque(maxlen=100),
            'wind_dir': deque(maxlen=100),
            'temperature': deque(maxlen=100),
            'pressure': deque(maxlen=100)
        }

    def update(self):
        base_wind = self.wind_speed + random.uniform(-0.2, 0.2)
        self.wind_speed = max(0, base_wind)
        self.wind_gust = self.wind_speed + random.uniform(1, 4)
        self.wind_dir = (self.wind_dir + random.uniform(-3, 3)) % 360
        self.temperature += random.uniform(-0.05, 0.05)
        self.humidity = max(0, min(100, self.humidity + random.uniform(-0.5, 0.5)))
        self.pressure += random.uniform(-0.2, 0.2)
        self.visibility = max(0, min(20, self.visibility + random.uniform(-0.05, 0.05)))

        self.history['wind_speed'].append(self.wind_speed)
        self.history['wind_dir'].append(self.wind_dir)
        self.history['temperature'].append(self.temperature)
        self.history['pressure'].append(self.pressure)

    def get_data(self):
        return {
            'wind_speed': round(self.wind_speed, 1),
            'wind_gust': round(self.wind_gust, 1),
            'wind_direction': int(self.wind_dir),
            'temperature': round(self.temperature, 1),
            'humidity': int(self.humidity),
            'pressure': int(self.pressure),
            'visibility': round(self.visibility, 1),
            'altitude': self.altitude
        }

    def get_wind_stability(self):
        if len(self.history['wind_speed']) < 10:
            return 100
        speeds = list(self.history['wind_speed'])[-20:]
        variance = sum((s - sum(speeds)/len(speeds))**2 for s in speeds) / len(speeds)
        return round(max(0, 100 - variance * 20), 1)

# ============================================================================
# ROI MANAGER
# ============================================================================
class ROIManager:
    def __init__(self):
        self.rois = []
        self.active_roi = None
        self.colors = [
            QColor(0, 255, 136, 80),
            QColor(255, 165, 0, 80),
            QColor(0, 191, 255, 80),
            QColor(255, 0, 255, 80),
            QColor(255, 255, 0, 80),
        ]
        self.color_index = 0

    def add_roi(self, name, shape, geometry):
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1

        roi = {
            'id': f'ROI_{len(self.rois)+1:02d}',
            'name': name,
            'shape': shape,
            'geometry': geometry,
            'active': True,
            'color': color,
            'created': datetime.now().strftime("%H:%M:%S")
        }
        self.rois.append(roi)
        return roi

    def remove_roi(self, roi_id):
        self.rois = [r for r in self.rois if r['id'] != roi_id]
        if self.active_roi and self.active_roi['id'] == roi_id:
            self.active_roi = None

    def set_active(self, roi_id):
        for roi in self.rois:
            roi['active'] = (roi['id'] == roi_id)
        self.active_roi = next((r for r in self.rois if r['id'] == roi_id), None)

    def get_active_roi(self):
        return self.active_roi

    def clear_all(self):
        self.rois = []
        self.active_roi = None

    def draw_rois(self, frame):
        for roi in self.rois:
            if not roi['active']:
                continue
            color = (roi['color'].red(), roi['color'].green(), roi['color'].blue())

            if roi['shape'] == ROIShape.RECTANGLE:
                x, y, w, h = roi['geometry']
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, roi['name'], (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif roi['shape'] == ROIShape.CIRCLE:
                cx, cy, r = roi['geometry']
                cv2.circle(frame, (cx, cy), r, color, 2)
                cv2.putText(frame, roi['name'], (cx-r, cy-r-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif roi['shape'] == ROIShape.POLYGON:
                points = np.array(roi['geometry'], np.int32)
                cv2.polylines(frame, [points], True, color, 2)
                M = cv2.moments(points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame, roi['name'], (cx-20, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

# ============================================================================
# VIDEO CAPTURE THREAD
# ============================================================================
class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    detections_ready = pyqtSignal(list)

    def __init__(self, source=0, roi_manager=None, parent=None):
        super().__init__(parent)
        self.source = source
        self.running = True
        self.detector = YOLODetector(use_real_model=False)
        self.roi_manager = roi_manager
        self.frame_count = 0
        self.scan_mode = False
        self.scan_progress = 0
        self.zoom_level = 1.0

    def run(self):
        while self.running:
            frame = self.generate_synthetic_frame()
            processed = self.process_frame(frame)
            self.frame_ready.emit(processed)

            if self.frame_count % 5 == 0:
                roi = self.roi_manager.get_active_roi() if self.roi_manager else None
                detections = self.detector.detect(frame, 
                    roi['geometry'] if roi else None)
                self.detections_ready.emit(detections)

            self.frame_count += 1
            self.msleep(33)

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        if self.zoom_level != 1.0:
            new_w = int(w / self.zoom_level)
            new_h = int(h / self.zoom_level)
            x1 = max(0, (w - new_w) // 2)
            y1 = max(0, (h - new_h) // 2)
            frame = frame[y1:y1+new_h, x1:x1+new_w]
            frame = cv2.resize(frame, (w, h))

        if self.scan_mode:
            self.scan_progress = (self.scan_progress + 2) % h
            cv2.line(frame, (0, self.scan_progress), (w, self.scan_progress), (0, 255, 136), 2)
            for i in range(-10, 11):
                y = self.scan_progress + i
                if 0 <= y < h:
                    alpha = 1 - abs(i) / 10
                    frame[y, :] = cv2.addWeighted(
                        frame[y, :], 1,
                        np.full_like(frame[y, :], (0, 255, 136)), alpha, 0
                    )

        if self.roi_manager:
            frame = self.roi_manager.draw_rois(frame)

        return frame

    def generate_synthetic_frame(self):
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        hour = datetime.now().hour
        if 6 <= hour < 18:
            base_intensity = 40
        else:
            base_intensity = 15

        for y in range(h//2):
            intensity = int(base_intensity + (y / (h//2)) * 20)
            frame[y, :] = [intensity, intensity, intensity + 5]

        for y in range(h//2, h):
            intensity = int(base_intensity + 10 + ((y - h//2) / (h//2)) * 30)
            frame[y, :] = [intensity, intensity, intensity]

        buildings = [
            (80, 250, 180, 420, 0.7),
            (320, 180, 240, 490, 0.4),
            (650, 220, 200, 450, 0.6),
            (920, 280, 160, 390, 0.3),
            (1150, 320, 120, 350, 0.8)
        ]

        for bx, by, bw, bh, damage in buildings:
            color = (15, 15, 18) if hour >= 6 else (8, 8, 12)
            cv2.rectangle(frame, (bx, h//2 + by), (bx + bw, h), color, -1)

            for wx in range(bx + 15, bx + bw - 25, 35):
                for wy in range(h//2 + by + 25, h - 25, 45):
                    if random.random() > 0.4:
                        lit = random.random() > 0.6
                        win_color = (50, 55, 40) if lit else (20, 22, 18)
                        cv2.rectangle(frame, (wx, wy), (wx + 22, wy + 28), win_color, -1)

            if damage > 0.5:
                for _ in range(int(damage * 5)):
                    dx = bx + random.randint(0, bw)
                    dy = h//2 + by + random.randint(0, bh)
                    cv2.circle(frame, (dx, dy), random.randint(10, 30), (5, 5, 8), -1)

        if random.random() > 0.7:
            vx = int((self.frame_count * 2) % w)
            vy = h - 80
            cv2.rectangle(frame, (vx, vy), (vx + 60, vy + 30), (25, 25, 28), -1)
            cv2.circle(frame, (vx + 15, vy + 30), 8, (15, 15, 18), -1)
            cv2.circle(frame, (vx + 45, vy + 30), 8, (15, 15, 18), -1)

        fog = np.zeros_like(frame)
        fog[:, :] = (30, 35, 40)
        frame = cv2.addWeighted(frame, 0.9, fog, 0.1, 0)

        noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        frame[::2, :] = (frame[::2, :] * 0.85).astype(np.uint8)

        center = (w//2, h//2)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        vignette = np.clip(1 - dist / 900, 0, 1)
        for c in range(3):
            frame[:, :, c] = (frame[:, :, c] * vignette).astype(np.uint8)

        return frame

    def set_zoom(self, level):
        self.zoom_level = max(1.0, min(20.0, level))

    def set_scan_mode(self, active):
        self.scan_mode = active
        self.scan_progress = 0

    def stop(self):
        self.running = False
        self.wait()

# ============================================================================
# INTERACTIVE VIDEO WIDGET
# ============================================================================
class InteractiveVideoWidget(QLabel):
    target_selected = pyqtSignal(dict)
    roi_defined = pyqtSignal(str, ROIShape, tuple)
    zoom_changed = pyqtSignal(float)

    def __init__(self, roi_manager, parent=None):
        super().__init__(parent)
        self.roi_manager = roi_manager
        self.setMinimumSize(880, 495)
        self.setStyleSheet("background-color: #000000; border: 1px solid #1b3a5c;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

        self.mode = "view"
        self.detections = []
        self.selected_target = None
        self.hovered_target = None

        self.drawing = False
        self.draw_start = None
        self.draw_current = None
        self.polygon_points = []

        self.zoom_level = 1.0
        self.pan_active = False
        self.pan_start = None
        self.pan_offset = QPoint(0, 0)

        self.measure_start = None
        self.measure_end = None

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def set_mode(self, mode):
        self.mode = mode
        cursors = {
            "view": Qt.CursorShape.ArrowCursor,
            "select": Qt.CursorShape.CrossCursor,
            "roi_rect": Qt.CursorShape.CrossCursor,
            "roi_circle": Qt.CursorShape.CrossCursor,
            "roi_poly": Qt.CursorShape.CrossCursor,
            "measure": Qt.CursorShape.CrossCursor
        }
        self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))
        self.drawing = False
        self.draw_start = None
        self.polygon_points = []
        self.update()

    def set_detections(self, detections):
        self.detections = detections
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.map_to_image(event.pos())

            if self.mode == "select":
                self.select_target_at(pos)
            elif self.mode in ["roi_rect", "roi_circle"]:
                self.drawing = True
                self.draw_start = pos
                self.draw_current = pos
            elif self.mode == "roi_poly":
                self.polygon_points.append((pos.x(), pos.y()))
                if len(self.polygon_points) > 2:
                    first = QPoint(self.polygon_points[0][0], self.polygon_points[0][1])
                    if (pos - first).manhattanLength() < 15:
                        self.finish_polygon_roi()
            elif self.mode == "measure":
                self.measure_start = pos
                self.measure_end = pos
            elif self.mode == "view":
                target = self.get_target_at(pos)
                if target:
                    self.select_target_at(pos)
                else:
                    self.pan_active = True
                    self.pan_start = event.pos()

        elif event.button() == Qt.MouseButton.RightButton:
            self.show_target_context_menu(event.pos())

    def mouseMoveEvent(self, event):
        pos = self.map_to_image(event.pos())

        if self.mode in ["select", "view"]:
            self.hovered_target = self.get_target_at(pos)
            self.update()
        elif self.drawing and self.mode in ["roi_rect", "roi_circle"]:
            self.draw_current = pos
            self.update()
        elif self.mode == "measure" and self.measure_start:
            self.measure_end = pos
            self.update()
        elif self.pan_active:
            delta = event.pos() - self.pan_start
            self.pan_offset += delta
            self.pan_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing:
                if self.mode == "roi_rect":
                    self.finish_rect_roi()
                elif self.mode == "roi_circle":
                    self.finish_circle_roi()
                self.drawing = False
                self.draw_start = None
                self.draw_current = None
            elif self.pan_active:
                self.pan_active = False
                self.pan_start = None
            elif self.mode == "measure" and self.measure_start:
                self.measure_start = None
                self.measure_end = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_level = min(20.0, self.zoom_level * 1.1)
        else:
            self.zoom_level = max(1.0, self.zoom_level / 1.1)
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def map_to_image(self, widget_pos):
        if self.pixmap() is None or self.pixmap().isNull():
            return widget_pos
        pix = self.pixmap()
        scaled_w = pix.width() * self.zoom_level
        scaled_h = pix.height() * self.zoom_level
        x_offset = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        y_offset = (self.height() - scaled_h) / 2 + self.pan_offset.y()
        img_x = (widget_pos.x() - x_offset) / self.zoom_level
        img_y = (widget_pos.y() - y_offset) / self.zoom_level
        return QPoint(int(img_x), int(img_y))

    def map_from_image(self, img_pos):
        if self.pixmap() is None or self.pixmap().isNull():
            return img_pos
        pix = self.pixmap()
        scaled_w = pix.width() * self.zoom_level
        scaled_h = pix.height() * self.zoom_level
        x_offset = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        y_offset = (self.height() - scaled_h) / 2 + self.pan_offset.y()
        widget_x = img_pos.x() * self.zoom_level + x_offset
        widget_y = img_pos.y() * self.zoom_level + y_offset
        return QPoint(int(widget_x), int(widget_y))

    def get_target_at(self, pos):
        for det in self.detections:
            x1, y1, x2, y2 = det['bbox']
            if x1 <= pos.x() <= x2 and y1 <= pos.y() <= y2:
                return det
        return None

    def select_target_at(self, pos):
        target = self.get_target_at(pos)
        if target:
            self.selected_target = target
            self.target_selected.emit(target)
            self.set_mode("view")
            self.update()

    def finish_rect_roi(self):
        if self.draw_start and self.draw_current:
            x1 = min(self.draw_start.x(), self.draw_current.x())
            y1 = min(self.draw_start.y(), self.draw_current.y())
            x2 = max(self.draw_start.x(), self.draw_current.x())
            y2 = max(self.draw_start.y(), self.draw_current.y())
            w, h = x2 - x1, y2 - y1
            if w > 20 and h > 20:
                name, ok = QInputDialog.getText(self, "ROI Name", "Enter region name:")
                if ok and name:
                    self.roi_defined.emit(name, ROIShape.RECTANGLE, (x1, y1, w, h))

    def finish_circle_roi(self):
        if self.draw_start and self.draw_current:
            dx = self.draw_current.x() - self.draw_start.x()
            dy = self.draw_current.y() - self.draw_start.y()
            r = int(math.sqrt(dx**2 + dy**2))
            if r > 10:
                name, ok = QInputDialog.getText(self, "ROI Name", "Enter region name:")
                if ok and name:
                    self.roi_defined.emit(name, ROIShape.CIRCLE, 
                                        (self.draw_start.x(), self.draw_start.y(), r))

    def finish_polygon_roi(self):
        if len(self.polygon_points) > 2:
            name, ok = QInputDialog.getText(self, "ROI Name", "Enter region name:")
            if ok and name:
                self.roi_defined.emit(name, ROIShape.POLYGON, self.polygon_points)
            self.polygon_points = []
            self.set_mode("view")

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #0a1628; color: #e0e6ed; border: 1px solid #1b3a5c; }
            QMenu::item:selected { background-color: #1b3a5c; }
        """)

        roi_menu = menu.addMenu("Define ROI")
        rect_act = QAction("Rectangle", self)
        rect_act.triggered.connect(lambda: self.set_mode("roi_rect"))
        roi_menu.addAction(rect_act)

        circle_act = QAction("Circle", self)
        circle_act.triggered.connect(lambda: self.set_mode("roi_circle"))
        roi_menu.addAction(circle_act)

        poly_act = QAction("Polygon", self)
        poly_act.triggered.connect(lambda: self.set_mode("roi_poly"))
        roi_menu.addAction(poly_act)

        menu.addSeparator()

        measure_act = QAction("Measure Distance", self)
        measure_act.triggered.connect(lambda: self.set_mode("measure"))
        menu.addAction(measure_act)

        select_act = QAction("Select Target", self)
        select_act.triggered.connect(lambda: self.set_mode("select"))
        menu.addAction(select_act)

        menu.addSeparator()

        zoom_in = QAction("Zoom In", self)
        zoom_in.triggered.connect(lambda: self.set_zoom(self.zoom_level * 1.2))
        menu.addAction(zoom_in)

        zoom_out = QAction("Zoom Out", self)
        zoom_out.triggered.connect(lambda: self.set_zoom(self.zoom_level / 1.2))
        menu.addAction(zoom_out)

        zoom_reset = QAction("Reset Zoom", self)
        zoom_reset.triggered.connect(self.reset_zoom)
        menu.addAction(zoom_reset)

        menu.exec(self.mapToGlobal(pos))

    def show_target_context_menu(self, pos):
        img_pos = self.map_to_image(pos)
        target = self.get_target_at(img_pos)

        if target:
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu { background-color: #0a1628; color: #e0e6ed; border: 1px solid #1b3a5c; }
                QMenu::item:selected { background-color: #1b3a5c; }
            """)

            lock_act = QAction(f"Lock Target {target['id']}", self)
            lock_act.triggered.connect(lambda: self.target_selected.emit(target))
            menu.addAction(lock_act)

            track_act = QAction(f"Track Target {target['id']}", self)
            menu.addAction(track_act)

            info_act = QAction("Target Info", self)
            menu.addAction(info_act)

            menu.addSeparator()

            priority_menu = menu.addMenu("Priority")
            for p in TargetPriority:
                act = QAction(p.name, self)
                act.triggered.connect(lambda checked, pri=p: self.set_target_priority(target, pri))
                priority_menu.addAction(act)

            menu.exec(self.mapToGlobal(pos))

    def set_target_priority(self, target, priority):
        target['priority'] = priority
        self.update()

    def set_zoom(self, level):
        self.zoom_level = max(1.0, min(20.0, level))
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.zoom_changed.emit(1.0)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.pixmap() or self.pixmap().isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for det in self.detections:
            x1, y1, x2, y2 = det['bbox']
            p1 = self.map_from_image(QPoint(x1, y1))
            p2 = self.map_from_image(QPoint(x2, y2))

            is_selected = self.selected_target and self.selected_target['id'] == det['id']
            is_hovered = self.hovered_target and self.hovered_target['id'] == det['id']

            if is_selected:
                pen = QPen(QColor(0, 255, 136), 3)
                painter.setPen(pen)
                painter.drawRect(QRect(p1, p2))

                info_x = p2.x() + 10
                info_y = p1.y()
                info_w = 150
                info_h = 90

                painter.fillRect(info_x, info_y, info_w, info_h, QColor(10, 20, 35, 220))
                painter.setPen(QColor(0, 255, 136))
                painter.drawRect(info_x, info_y, info_w, info_h)

                painter.setPen(QColor(255, 255, 255))
                font = QFont("Consolas", 8)
                painter.setFont(font)
                painter.drawText(info_x + 5, info_y + 15, f"TARGET: {det['id']}")
                painter.drawText(info_x + 5, info_y + 30, f"DIST: {det.get('distance', 0):.1f}m")
                painter.drawText(info_x + 5, info_y + 45, f"CONF: {det['confidence']*100:.1f}%")
                painter.drawText(info_x + 5, info_y + 60, f"SPD: {det.get('speed', 0):.1f}m/s")
                painter.drawText(info_x + 5, info_y + 75, f"PRIORITY: {det.get('priority', TargetPriority.LOW).name}")

            elif is_hovered:
                pen = QPen(QColor(255, 255, 255, 150), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(QRect(p1, p2))

        if self.drawing and self.draw_start and self.draw_current:
            p1 = self.map_from_image(self.draw_start)
            p2 = self.map_from_image(self.draw_current)

            if self.mode == "roi_rect":
                painter.setPen(QPen(QColor(0, 255, 136, 180), 2, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(0, 255, 136, 40)))
                painter.drawRect(QRect(p1, p2))
                w = abs(p2.x() - p1.x())
                h = abs(p2.y() - p1.y())
                painter.setPen(QColor(255, 255, 255))
                painter.drawText((p1.x()+p2.x())//2 - 20, (p1.y()+p2.y())//2, f"{w}x{h}")
            elif self.mode == "roi_circle":
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                r = int(math.sqrt(dx**2 + dy**2))
                painter.setPen(QPen(QColor(0, 191, 255, 180), 2, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(0, 191, 255, 40)))
                painter.drawEllipse(p1, r, r)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(p1.x() - 10, p1.y() - r - 5, f"R={r}px")

        if self.mode == "roi_poly" and self.polygon_points:
            points = [self.map_from_image(QPoint(p[0], p[1])) for p in self.polygon_points]
            if len(points) > 1:
                painter.setPen(QPen(QColor(255, 165, 0, 180), 2))
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i+1])
            for p in points:
                painter.setBrush(QBrush(QColor(255, 165, 0)))
                painter.drawEllipse(p, 4, 4)

        if self.mode == "measure" and self.measure_start and self.measure_end:
            p1 = self.map_from_image(self.measure_start)
            p2 = self.map_from_image(self.measure_end)
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawLine(p1, p2)
            dx = self.measure_end.x() - self.measure_start.x()
            dy = self.measure_end.y() - self.measure_start.y()
            dist = math.sqrt(dx**2 + dy**2)
            mid_x = (p1.x() + p2.x()) // 2
            mid_y = (p1.y() + p2.y()) // 2
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(mid_x - 30, mid_y - 10, f"{dist:.1f}px")

        painter.setPen(QColor(139, 155, 180))
        font = QFont("Consolas", 9)
        painter.setFont(font)
        painter.drawText(10, 20, f"ZOOM: {self.zoom_level:.1f}x")
        painter.drawText(10, 35, f"MODE: {self.mode.upper()}")

        if self.selected_target:
            painter.setPen(QColor(0, 255, 136))
            painter.drawText(10, 50, f"SELECTED: {self.selected_target['id']}")

# ============================================================================
# CUSTOM WIDGETS
# ============================================================================
class StatusIndicator(QFrame):
    def __init__(self, label, status, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.status_text = status
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        self.label = QLabel(self.label_text)
        self.label.setStyleSheet("color: #8b9bb4; font-size: 11px; font-weight: bold;")

        self.status = QLabel(self.status_text)
        self.update_status_color(self.status_text)

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.status)

        self.setStyleSheet("""
            StatusIndicator {
                background-color: #0d1b2a;
                border: 1px solid #1b3a5c;
                border-radius: 3px;
            }
        """)

    def update_status_color(self, status):
        colors = {
            'ONLINE': '#00ff88', 'ACTIVE': '#00ff88', 'READY': '#00ff88',
            'STABLE': '#00ff88', 'SECURE': '#00ff88', 'LOCKED': '#ff4444',
            'DETECTED': '#ffaa00', 'STANDBY': '#ffaa00', 'OFFLINE': '#ff4444',
            'SCANNING': '#00ffff', 'TRACKING': '#00ff88', 'SOLVED': '#00aaff',
            'FIRING': '#ff0000', 'COOLDOWN': '#ffaa00'
        }
        color = colors.get(status, '#ffffff')
        self.status.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        self.status.setText(status)

class CircularGauge(QFrame):
    def __init__(self, title, value, unit, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = value
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.current_value = value
        self.target_value = value
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(140, 140)
        self.setStyleSheet("background-color: transparent;")
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.smooth_update)
        self.update_timer.start(50)

    def set_value(self, val):
        self.target_value = val

    def smooth_update(self):
        diff = self.target_value - self.current_value
        if abs(diff) > 0.01:
            self.current_value += diff * 0.1
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = QPointF(70, 70)
        radius = 55

        pen = QPen(QColor(30, 50, 70))
        pen.setWidth(8)
        painter.setPen(pen)
        painter.drawArc(int(center.x() - radius), int(center.y() - radius), 
                       int(radius*2), int(radius*2), 225*16, -270*16)

        ratio = (self.current_value - self.min_val) / (self.max_val - self.min_val)
        ratio = max(0, min(1, ratio))
        span = int(-270 * 16 * ratio)

        if ratio > 0.8:
            color_start = QColor(255, 68, 68)
            color_end = QColor(255, 100, 100)
        elif ratio > 0.5:
            color_start = QColor(255, 165, 0)
            color_end = QColor(255, 200, 50)
        else:
            color_start = QColor(0, 255, 136)
            color_end = QColor(0, 200, 255)

        gradient = QLinearGradient(0, 0, 140, 0)
        gradient.setColorAt(0, color_start)
        gradient.setColorAt(1, color_end)

        pen = QPen(QBrush(gradient), 8)
        painter.setPen(pen)
        painter.drawArc(int(center.x() - radius), int(center.y() - radius),
                       int(radius*2), int(radius*2), 225*16, span)

        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 16, QFont.Weight.Bold)
        painter.setFont(font)
        text = f"{self.current_value:.1f}"
        rect = painter.boundingRect(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        font = QFont("Consolas", 9)
        painter.setFont(font)
        painter.setPen(QColor(139, 155, 180))
        painter.drawText(0, 95, 140, 20, Qt.AlignmentFlag.AlignCenter, self.unit)

        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.drawText(0, 115, 140, 20, Qt.AlignmentFlag.AlignCenter, self.title)

class WindCompass(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wind_speed = 4.2
        self.wind_gust = 6.8
        self.wind_dir = 315
        self.history = deque(maxlen=30)
        self.setFixedSize(160, 160)
        self.setStyleSheet("background-color: transparent;")

    def set_data(self, speed, gust, direction):
        self.wind_speed = speed
        self.wind_gust = gust
        self.wind_dir = direction
        self.history.append((speed, direction))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = QPointF(80, 80)
        radius = 65

        pen = QPen(QColor(50, 80, 110))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(10, 20, 35)))
        painter.drawEllipse(center, radius, radius)

        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            x1 = center.x() + (radius - 8) * math.sin(rad)
            y1 = center.y() - (radius - 8) * math.cos(rad)
            x2 = center.x() + radius * math.sin(rad)
            y2 = center.y() - radius * math.cos(rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.setPen(QColor(139, 155, 180))
        font = QFont("Consolas", 10, QFont.Weight.Bold)
        painter.setFont(font)

        dirs = {'N': (80, 22), 'S': (80, 142), 'E': (138, 84), 'W': (22, 84)}
        for d, (x, y) in dirs.items():
            painter.drawText(x-6, y+4, 12, 12, Qt.AlignmentFlag.AlignCenter, d)

        if len(self.history) > 1:
            for i, (speed, dir) in enumerate(self.history):
                alpha = int(255 * (i / len(self.history)) * 0.3)
                rad = math.radians(dir)
                trail_len = 30 + speed * 3
                end_x = center.x() + trail_len * math.sin(rad)
                end_y = center.y() - trail_len * math.cos(rad)
                pen = QPen(QColor(0, 255, 136, alpha))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(int(center.x()), int(center.y()), int(end_x), int(end_y))

        angle = math.radians(self.wind_dir)
        arrow_len = 35 + self.wind_speed * 4

        end_x = center.x() + arrow_len * math.sin(angle)
        end_y = center.y() - arrow_len * math.cos(angle)

        pen = QPen(QColor(0, 255, 136))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(int(center.x()), int(center.y()), int(end_x), int(end_y))

        painter.setBrush(QBrush(QColor(0, 255, 136)))
        painter.drawEllipse(QPointF(end_x, end_y), 5, 5)

        if self.wind_gust > self.wind_speed + 1:
            gust_len = 35 + self.wind_gust * 4
            gust_x = center.x() + gust_len * math.sin(angle)
            gust_y = center.y() - gust_len * math.cos(angle)
            pen = QPen(QColor(255, 165, 0, 180))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(int(center.x()), int(center.y()), int(gust_x), int(gust_y))

        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(40, 68, 80, 20, Qt.AlignmentFlag.AlignCenter, f"{self.wind_speed:.1f}")

        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.setPen(QColor(139, 155, 180))
        painter.drawText(40, 82, 80, 15, Qt.AlignmentFlag.AlignCenter, "m/s")
        painter.drawText(40, 96, 80, 15, Qt.AlignmentFlag.AlignCenter, f"{int(self.wind_dir)}°")

        if self.wind_gust > self.wind_speed + 1:
            painter.setPen(QColor(255, 165, 0))
            painter.drawText(40, 110, 80, 15, Qt.AlignmentFlag.AlignCenter, f"GUST: {self.wind_gust:.1f}")

class TrajectoryGraph(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.distance = 612.4
        self.drop = 1.28
        self.wind_drift = 0.42
        self.trajectory_points = []
        self.setFixedSize(300, 180)
        self.setStyleSheet("background-color: transparent;")

    def set_data(self, distance, drop, wind_drift=0, trajectory_points=None):
        self.distance = distance
        self.drop = drop
        self.wind_drift = wind_drift
        if trajectory_points:
            self.trajectory_points = trajectory_points
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 35
        graph_w = w - 2 * margin
        graph_h = h - 2 * margin

        pen = QPen(QColor(30, 50, 70))
        pen.setWidth(1)
        painter.setPen(pen)

        for i in range(5):
            x = margin + graph_w * i / 4
            painter.drawLine(int(x), margin, int(x), h - margin)

        for i in range(4):
            y = margin + graph_h * i / 3
            painter.drawLine(margin, int(y), w - margin, int(y))

        painter.setPen(QColor(100, 120, 140))
        font = QFont("Consolas", 8)
        painter.setFont(font)

        max_dist = max(self.distance * 1.3, 100)
        for i in range(5):
            x = margin + graph_w * i / 4
            val = int(max_dist * i / 4)
            painter.drawText(int(x) - 15, h - margin + 15, 30, 12, 
                           Qt.AlignmentFlag.AlignCenter, str(val))

        for i in range(4):
            y = margin + graph_h * i / 3
            val = 2 - i
            painter.drawText(5, int(y) - 6, 25, 12,
                           Qt.AlignmentFlag.AlignCenter, f"{val}")

        if self.trajectory_points:
            points = []
            for x, y in self.trajectory_points:
                px = margin + (x / max_dist) * graph_w
                py = h - margin - graph_h * 0.5 - y * 50
                points.append(QPointF(px, py))

            if len(points) > 1:
                for i in range(len(points) - 1):
                    ratio = i / len(points)
                    color = QColor(
                        int(0 + ratio * 255),
                        int(255 - ratio * 100),
                        int(136 - ratio * 136)
                    )
                    pen = QPen(color, 2)
                    painter.setPen(pen)
                    painter.drawLine(points[i], points[i+1])

        target_x = margin + (self.distance / max_dist) * graph_w
        target_y = h - margin - graph_h * 0.5 - (-self.drop) * 20

        pen = QPen(QColor(255, 68, 68, 150))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(target_x), int(target_y), int(target_x), int(h - margin - graph_h * 0.5))

        painter.setPen(QPen(QColor(255, 68, 68), 2))
        painter.setBrush(QBrush(QColor(255, 68, 68, 100)))
        painter.drawEllipse(QPointF(target_x, target_y), 6, 6)

        if self.wind_drift != 0:
            drift_x = target_x + self.wind_drift * 30
            pen = QPen(QColor(255, 165, 0, 150))
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(target_x), int(target_y), int(drift_x), int(target_y))
            painter.drawText(int(drift_x), int(target_y) - 10, 40, 12,
                           Qt.AlignmentFlag.AlignLeft, f"W:{self.wind_drift}")

        painter.fillRect(int(target_x) + 10, int(target_y) - 40, 80, 35, QColor(10, 20, 35, 200))
        painter.setPen(QColor(0, 255, 136))
        painter.drawRect(int(target_x) + 10, int(target_y) - 40, 80, 35)
        painter.drawText(int(target_x) + 15, int(target_y) - 28, 70, 12,
                        Qt.AlignmentFlag.AlignLeft, f"{self.distance:.1f}m")
        painter.drawText(int(target_x) + 15, int(target_y) - 15, 70, 12,
                        Qt.AlignmentFlag.AlignLeft, f"D:{self.drop}MRAD")

class TargetVitals(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.probability = 98.7
        self.vital_area = "HEART/LUNGS"
        self.status = "LETHAL"
        self.lethality = 0.95
        self.setFixedSize(200, 220)
        self.setStyleSheet("background-color: transparent;")

    def set_data(self, probability, vital_area, status, lethality=0.95):
        self.probability = probability
        self.vital_area = vital_area
        self.status = status
        self.lethality = lethality
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        center_x = w // 2
        head_y = 35

        painter.setPen(QPen(QColor(100, 120, 140), 2))
        painter.setBrush(QBrush(QColor(20, 30, 45)))
        painter.drawEllipse(center_x - 12, head_y, 24, 28)
        painter.drawRect(center_x - 6, head_y + 28, 12, 8)

        body_points = [
            QPointF(center_x - 22, head_y + 36),
            QPointF(center_x + 22, head_y + 36),
            QPointF(center_x + 28, head_y + 90),
            QPointF(center_x - 28, head_y + 90)
        ]
        painter.drawPolygon(QPolygonF(body_points))

        painter.drawLine(int(center_x - 22), int(head_y + 40), 
                        int(center_x - 38), int(head_y + 80))
        painter.drawLine(int(center_x + 22), int(head_y + 40), 
                        int(center_x + 38), int(head_y + 80))

        painter.drawLine(int(center_x - 12), int(head_y + 90), 
                        int(center_x - 18), int(head_y + 145))
        painter.drawLine(int(center_x + 12), int(head_y + 90), 
                        int(center_x + 18), int(head_y + 145))

        if self.vital_area == "HEART/LUNGS":
            gradient = QRadialGradient(center_x, head_y + 55, 25)
            gradient.setColorAt(0, QColor(255, 68, 68, 200))
            gradient.setColorAt(1, QColor(255, 68, 68, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center_x - 22, head_y + 40, 44, 35)
        elif self.vital_area == "HEAD":
            gradient = QRadialGradient(center_x, head_y + 14, 18)
            gradient.setColorAt(0, QColor(255, 68, 68, 200))
            gradient.setColorAt(1, QColor(255, 68, 68, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center_x - 14, head_y - 2, 28, 32)
        elif self.vital_area == "PELVIS":
            gradient = QRadialGradient(center_x, head_y + 85, 20)
            gradient.setColorAt(0, QColor(255, 68, 68, 200))
            gradient.setColorAt(1, QColor(255, 68, 68, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center_x - 20, head_y + 70, 40, 30)

        painter.setPen(QColor(139, 155, 180))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.drawText(0, 5, w, 20, Qt.AlignmentFlag.AlignCenter, "VITAL AREA")

        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 155, w, 20, Qt.AlignmentFlag.AlignCenter, self.vital_area)

        painter.setPen(QColor(0, 255, 136))
        font = QFont("Consolas", 16, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 175, w, 28, Qt.AlignmentFlag.AlignCenter, f"{self.probability:.1f}%")

        color = '#00ff88' if self.status == 'LETHAL' else '#ffaa00'
        painter.setPen(QColor(color))
        font = QFont("Consolas", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 200, w, 22, Qt.AlignmentFlag.AlignCenter, self.status)

        bar_w = 120
        bar_h = 4
        bar_x = (w - bar_w) // 2
        bar_y = 208

        painter.fillRect(bar_x, bar_y, bar_w, bar_h, QColor(30, 50, 70))
        filled = int(bar_w * self.lethality)
        painter.fillRect(bar_x, bar_y, filled, bar_h, QColor(0, 255, 136))

class ROIListWidget(QTreeWidget):
    roi_selected = pyqtSignal(str)
    roi_activated = pyqtSignal(str, bool)
    roi_deleted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["ID", "Name", "Shape", "Status", "Detections"])
        self.setColumnWidth(0, 50)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 70)
        self.setColumnWidth(4, 80)
        self.setFixedHeight(200)
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #0a1628;
                border: 1px solid #1b3a5c;
                color: #e0e6ed;
                font-size: 10px;
            }
            QTreeWidget::item:selected {
                background-color: #1b3a5c;
            }
            QHeaderView::section {
                background-color: #0d2137;
                color: #8b9bb4;
                padding: 4px;
                border: 1px solid #1b3a5c;
                font-size: 9px;
            }
        """)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def update_rois(self, rois):
        self.clear()
        for roi in rois:
            item = QTreeWidgetItem([
                roi['id'],
                roi['name'],
                roi['shape'].name,
                "ACTIVE" if roi['active'] else "INACTIVE",
                str(random.randint(0, 5))
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, roi['id'])
            if roi['active']:
                item.setBackground(0, QColor(0, 255, 136, 50))
            self.addTopLevelItem(item)

    def show_context_menu(self, pos):
        item = self.itemAt(pos)
        if item:
            roi_id = item.data(0, Qt.ItemDataRole.UserRole)
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu { background-color: #0a1628; color: #e0e6ed; border: 1px solid #1b3a5c; }
                QMenu::item:selected { background-color: #1b3a5c; }
            """)

            activate = QAction("Activate", self)
            activate.triggered.connect(lambda: self.roi_activated.emit(roi_id, True))
            menu.addAction(activate)

            deactivate = QAction("Deactivate", self)
            deactivate.triggered.connect(lambda: self.roi_activated.emit(roi_id, False))
            menu.addAction(deactivate)

            menu.addSeparator()

            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self.roi_deleted.emit(roi_id))
            menu.addAction(delete)

            menu.exec(self.mapToGlobal(pos))

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        item = self.itemAt(event.pos())
        if item:
            self.roi_selected.emit(item.data(0, Qt.ItemDataRole.UserRole))


# ============================================================================
# MAIN WINDOW
# ============================================================================
class AISniperSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI SNIPER SYSTEM v4.2.1 - ADVANCED TACTICAL INTERFACE")
        self.setGeometry(50, 50, 1680, 980)

        self.system_mode = SystemMode.STANDBY
        self.ballistics = BallisticsEngine()
        self.environment = EnvironmentSensor()
        self.roi_manager = ROIManager()
        self.detections = []
        self.locked_target = None
        self.selected_target = None
        self.tracked_target = None
        self.system_logs = deque(maxlen=100)
        self.frame_count = 0
        self.zoom_level = 1.0
        self.ammo_count = 20
        self.max_ammo = 20
        self.shots_fired = 0
        self.hits = 0

        self.setup_stylesheet()
        self.init_ui()
        self.init_video()
        self.init_timers()
        self.add_log("System Initialized - Advanced Tactical Interface v4.2.1")
        self.add_log("YOLO Detection Engine: Standby Mode")
        self.add_log("Ballistics Engine: Calibrated")
        self.add_log("Environmental Sensors: Online")

    def setup_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #050a10; }
            QWidget { background-color: #050a10; color: #e0e6ed; font-family: 'Consolas', 'Courier New', monospace; }
            QFrame { border: 1px solid #1b3a5c; background-color: #0a1628; }
            QLabel { color: #e0e6ed; font-size: 12px; }
            QPushButton {
                background-color: #0d2137; color: #8b9bb4; border: 1px solid #1b3a5c;
                border-radius: 4px; padding: 8px 16px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1b3a5c; color: #ffffff; border: 1px solid #00aaff; }
            QPushButton:checked { background-color: #1b3a5c; color: #00ff88; border: 2px solid #00ff88; }
            QPushButton#fire_button {
                background-color: #3a0a0a; color: #ff4444; border: 2px solid #ff4444; font-size: 14px;
            }
            QPushButton#fire_button:hover { background-color: #5a0a0a; }
            QPushButton#fire_button:disabled { background-color: #1a0a0a; color: #555555; border: 2px solid #333333; }
            QPushButton#active_button { background-color: #0a3a1a; color: #00ff88; border: 2px solid #00ff88; }
            QPushButton#warning_button { background-color: #3a2a0a; color: #ffaa00; border: 2px solid #ffaa00; }
            QTableWidget {
                background-color: #0a1628; border: 1px solid #1b3a5c;
                gridline-color: #1b3a5c; color: #e0e6ed; font-size: 11px;
            }
            QTableWidget::item { padding: 4px; border-bottom: 1px solid #1b3a5c; }
            QHeaderView::section {
                background-color: #0d2137; color: #8b9bb4; padding: 6px;
                border: 1px solid #1b3a5c; font-size: 10px; font-weight: bold;
            }
            QScrollArea { border: none; background-color: transparent; }
            QTextEdit, QPlainTextEdit {
                background-color: #0a1628; border: 1px solid #1b3a5c;
                color: #8b9bb4; font-size: 10px; padding: 5px;
            }
            QSlider::groove:horizontal { background: #1b3a5c; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #00ff88; width: 12px; height: 12px; border-radius: 6px; }
            QSlider::sub-page:horizontal { background: #00aaff; height: 4px; border-radius: 2px; }
            QComboBox {
                background-color: #0d2137; color: #e0e6ed; border: 1px solid #1b3a5c;
                padding: 4px; font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #1b3a5c; margin-top: 10px;
                font-size: 11px; font-weight: bold; color: #00aaff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QStatusBar { background-color: #0d2137; color: #8b9bb4; font-size: 10px; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # TOP BAR
        top_bar = QHBoxLayout()

        logo = QLabel("◉ AI SNIPER SYSTEM v4.2.1")
        logo.setStyleSheet("color: #00ff88; font-size: 14px; font-weight: bold; border: none;")
        top_bar.addWidget(logo)

        top_bar.addStretch()

        self.mode_indicator = QLabel("STANDBY")
        self.mode_indicator.setStyleSheet("""
            color: #ffaa00; font-size: 12px; font-weight: bold; 
            border: 2px solid #ffaa00; border-radius: 4px; padding: 4px 12px;
        """)
        top_bar.addWidget(self.mode_indicator)

        top_bar.addSpacing(20)

        self.tab_buttons = {}
        tabs = ["DASHBOARD", "TARGETS", "WEAPON", "ENVIRONMENT", "ANALYTICS", "SYSTEM"]
        for tab in tabs:
            btn = QPushButton(tab)
            btn.setFixedWidth(120)
            btn.setCheckable(True)
            if tab == "DASHBOARD":
                btn.setChecked(True)
                btn.setStyleSheet("background-color: #1b3a5c; color: #00ff88;")
            btn.clicked.connect(lambda checked, t=tab: self.switch_tab(t))
            self.tab_buttons[tab] = btn
            top_bar.addWidget(btn)

        top_bar.addStretch()

        self.sys_status_label = QLabel("● SYSTEM ONLINE")
        self.sys_status_label.setStyleSheet("color: #00ff88; font-size: 12px; font-weight: bold; border: none;")
        top_bar.addWidget(self.sys_status_label)

        main_layout.addLayout(top_bar)

        # MAIN CONTENT
        content = QHBoxLayout()
        content.setSpacing(5)

        # LEFT PANEL
        left_panel = QVBoxLayout()
        left_panel.setSpacing(5)

        # System Status
        sys_status = QFrame()
        sys_status.setFixedWidth(280)
        sys_layout = QVBoxLayout(sys_status)
        sys_layout.setSpacing(3)

        title = QLabel("SYSTEM STATUS")
        title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        sys_layout.addWidget(title)

        self.status_indicators = {}
        statuses = [
            ("AI MODULE", "ONLINE"), ("TARGET ACQUISITION", "STANDBY"),
            ("WEAPON SYSTEM", "READY"), ("SENSOR ARRAY", "ONLINE"),
            ("ENVIRONMENT", "STABLE"), ("COMM LINK", "SECURE")
        ]
        for label, status in statuses:
            ind = StatusIndicator(label, status)
            self.status_indicators[label] = ind
            sys_layout.addWidget(ind)

        ai_title = QLabel("AI PROCESSING")
        ai_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none; margin-top: 10px;")
        sys_layout.addWidget(ai_title)

        self.neural_status = StatusIndicator("NEURAL NETWORK", "STANDBY")
        sys_layout.addWidget(self.neural_status)

        conf_layout = QHBoxLayout()
        conf_label = QLabel("CONFIDENCE LEVEL")
        conf_label.setStyleSheet("color: #8b9bb4; font-size: 10px;")
        self.conf_value = QLabel("--")
        self.conf_value.setStyleSheet("color: #8b9bb4; font-size: 12px; font-weight: bold;")
        conf_layout.addWidget(conf_label)
        conf_layout.addStretch()
        conf_layout.addWidget(self.conf_value)
        sys_layout.addLayout(conf_layout)

        self.conf_bar = QProgressBar()
        self.conf_bar.setMaximum(100)
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(6)
        self.conf_bar.setStyleSheet("""
            QProgressBar { background-color: #0d1b2a; border: 1px solid #1b3a5c; border-radius: 3px; }
            QProgressBar::chunk { background-color: #00ff88; border-radius: 2px; }
        """)
        sys_layout.addWidget(self.conf_bar)

        metrics = QGridLayout()
        metrics.addWidget(QLabel("OBJECT DETECTION"), 0, 0)
        self.obj_det_val = QLabel("--")
        self.obj_det_val.setStyleSheet("color: #8b9bb4; font-weight: bold;")
        metrics.addWidget(self.obj_det_val, 0, 1)

        metrics.addWidget(QLabel("PREDICTION ACCURACY"), 1, 0)
        self.pred_acc_val = QLabel("--")
        self.pred_acc_val.setStyleSheet("color: #8b9bb4; font-weight: bold;")
        metrics.addWidget(self.pred_acc_val, 1, 1)

        metrics.addWidget(QLabel("TRACKED TARGETS"), 2, 0)
        self.tracked_count = QLabel("0")
        self.tracked_count.setStyleSheet("color: #8b9bb4; font-weight: bold;")
        metrics.addWidget(self.tracked_count, 2, 1)
        sys_layout.addLayout(metrics)

        wp_title = QLabel("WEAPON STATUS")
        wp_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none; margin-top: 10px;")
        sys_layout.addWidget(wp_title)

        wp_info = QGridLayout()
        wp_info.addWidget(QLabel("SR-25 Mk1"), 0, 0)
        self.wp_ready = QLabel("READY")
        self.wp_ready.setStyleSheet("color: #00ff88; font-weight: bold;")
        wp_info.addWidget(self.wp_ready, 0, 1)

        wp_info.addWidget(QLabel("AMMUNITION"), 1, 0)
        self.ammo_val = QLabel(f"{self.ammo_count} / {self.max_ammo}")
        self.ammo_val.setStyleSheet("color: #ffaa00; font-weight: bold;")
        wp_info.addWidget(self.ammo_val, 1, 1)

        wp_info.addWidget(QLabel("CALIBER"), 2, 0)
        wp_info.addWidget(QLabel("7.62x51mm"), 2, 1)

        wp_info.addWidget(QLabel("EFFECTIVE RANGE"), 3, 0)
        wp_info.addWidget(QLabel("1200m"), 3, 1)

        wp_info.addWidget(QLabel("WINDAGE"), 4, 0)
        self.windage_val = QLabel("-- MRAD")
        self.windage_val.setStyleSheet("color: #8b9bb4;")
        wp_info.addWidget(self.windage_val, 4, 1)

        wp_info.addWidget(QLabel("ELEVATION"), 5, 0)
        self.elev_val = QLabel("-- MRAD")
        self.elev_val.setStyleSheet("color: #8b9bb4;")
        wp_info.addWidget(self.elev_val, 5, 1)

        wp_info.addWidget(QLabel("SHOTS FIRED"), 6, 0)
        self.shots_val = QLabel("0")
        self.shots_val.setStyleSheet("color: #8b9bb4;")
        wp_info.addWidget(self.shots_val, 6, 1)

        wp_info.addWidget(QLabel("HIT RATIO"), 7, 0)
        self.hit_ratio = QLabel("0%")
        self.hit_ratio.setStyleSheet("color: #8b9bb4;")
        wp_info.addWidget(self.hit_ratio, 7, 1)

        sys_layout.addLayout(wp_info)
        sys_layout.addStretch()

        left_panel.addWidget(sys_status)

        # Environment Panel
        env_frame = QFrame()
        env_frame.setFixedWidth(280)
        env_layout = QVBoxLayout(env_frame)

        env_title = QLabel("ENVIRONMENT")
        env_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        env_layout.addWidget(env_title)

        self.env_labels = {}
        env_items = [
            ("WIND SPEED", "-- m/s", "🌬"), ("WIND GUST", "-- m/s", "💨"),
            ("WIND DIRECTION", "--°", "🧭"), ("TEMPERATURE", "-- °C", "🌡"),
            ("HUMIDITY", "-- %", "💧"), ("PRESSURE", "-- hPa", "📊"),
            ("VISIBILITY", "-- km", "👁"), ("ALTITUDE", "245 m", "🏔")
        ]
        for name, val, icon in env_items:
            row = QHBoxLayout()
            lbl = QLabel(f"{icon} {name}")
            lbl.setStyleSheet("color: #8b9bb4; font-size: 10px;")
            val_lbl = QLabel(val)
            val_lbl.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold;")
            self.env_labels[name] = val_lbl
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val_lbl)
            env_layout.addLayout(row)

        stab_layout = QHBoxLayout()
        stab_lbl = QLabel("WIND STABILITY")
        stab_lbl.setStyleSheet("color: #8b9bb4; font-size: 10px;")
        self.stab_val = QLabel("--")
        self.stab_val.setStyleSheet("color: #00ff88; font-weight: bold;")
        stab_layout.addWidget(stab_lbl)
        stab_layout.addStretch()
        stab_layout.addWidget(self.stab_val)
        env_layout.addLayout(stab_layout)

        left_panel.addWidget(env_frame)

        # ROI Panel
        roi_frame = QFrame()
        roi_frame.setFixedWidth(280)
        roi_layout = QVBoxLayout(roi_frame)

        roi_title = QLabel("REGIONS OF INTEREST")
        roi_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        roi_layout.addWidget(roi_title)

        self.roi_list = ROIListWidget()
        self.roi_list.roi_selected.connect(self.on_roi_selected)
        self.roi_list.roi_activated.connect(self.on_roi_activated)
        self.roi_list.roi_deleted.connect(self.on_roi_deleted)
        roi_layout.addWidget(self.roi_list)

        roi_btn_layout = QHBoxLayout()
        self.roi_rect_btn = QPushButton("▭ Rect")
        self.roi_rect_btn.setCheckable(True)
        self.roi_rect_btn.clicked.connect(lambda: self.set_interaction_mode("roi_rect"))

        self.roi_circle_btn = QPushButton("◯ Circle")
        self.roi_circle_btn.setCheckable(True)
        self.roi_circle_btn.clicked.connect(lambda: self.set_interaction_mode("roi_circle"))

        self.roi_poly_btn = QPushButton("⬡ Poly")
        self.roi_poly_btn.setCheckable(True)
        self.roi_poly_btn.clicked.connect(lambda: self.set_interaction_mode("roi_poly"))

        self.roi_clear_btn = QPushButton("✕ Clear")
        self.roi_clear_btn.clicked.connect(self.clear_all_rois)

        roi_btn_layout.addWidget(self.roi_rect_btn)
        roi_btn_layout.addWidget(self.roi_circle_btn)
        roi_btn_layout.addWidget(self.roi_poly_btn)
        roi_btn_layout.addWidget(self.roi_clear_btn)
        roi_layout.addLayout(roi_btn_layout)

        left_panel.addWidget(roi_frame)
        left_panel.addStretch()

        content.addLayout(left_panel)

        # CENTER PANEL
        center_panel = QVBoxLayout()
        center_panel.setSpacing(5)

        video_frame = QFrame()
        video_frame.setMinimumSize(900, 520)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(2, 2, 2, 2)

        video_header = QHBoxLayout()
        self.feed_title = QLabel("LIVE FEED - OPTICAL ZOOM 1.0x")
        self.feed_title.setStyleSheet("color: #00aaff; font-size: 12px; font-weight: bold; border: none;")
        video_header.addWidget(self.feed_title)
        video_header.addStretch()

        zoom_layout = QHBoxLayout()
        zoom_lbl = QLabel("ZOOM:")
        zoom_lbl.setStyleSheet("color: #8b9bb4; font-size: 10px; border: none;")
        zoom_layout.addWidget(zoom_lbl)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(10)
        self.zoom_slider.setFixedWidth(120)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_display = QLabel("1.0x")
        self.zoom_display.setStyleSheet("color: #00ff88; font-size: 10px; font-weight: bold; border: none;")
        zoom_layout.addWidget(self.zoom_display)
        video_header.addLayout(zoom_layout)

        self.select_mode_btn = QPushButton("🎯 Select")
        self.select_mode_btn.setCheckable(True)
        self.select_mode_btn.clicked.connect(lambda: self.set_interaction_mode("select"))
        video_header.addWidget(self.select_mode_btn)

        self.measure_mode_btn = QPushButton("📏 Measure")
        self.measure_mode_btn.setCheckable(True)
        self.measure_mode_btn.clicked.connect(lambda: self.set_interaction_mode("measure"))
        video_header.addWidget(self.measure_mode_btn)

        self.reset_view_btn = QPushButton("↺ Reset")
        self.reset_view_btn.clicked.connect(self.reset_view)
        video_header.addWidget(self.reset_view_btn)

        self.overlay_info = QLabel("DATE: --    TIME: --    MODE: DAY    ZOOM: 1.0x")
        self.overlay_info.setStyleSheet("color: #8b9bb4; font-size: 10px; border: none;")
        self.overlay_info.setAlignment(Qt.AlignmentFlag.AlignRight)
        video_header.addWidget(self.overlay_info)

        video_layout.addLayout(video_header)

        self.video_widget = InteractiveVideoWidget(self.roi_manager)
        self.video_widget.target_selected.connect(self.on_target_selected)
        self.video_widget.roi_defined.connect(self.on_roi_defined)
        self.video_widget.zoom_changed.connect(self.on_zoom_changed)
        video_layout.addWidget(self.video_widget)

        center_panel.addWidget(video_frame)

        # Target detail panel
        self.target_detail_frame = QFrame()
        self.target_detail_frame.setFixedHeight(100)
        self.target_detail_frame.setVisible(False)
        detail_layout = QHBoxLayout(self.target_detail_frame)

        self.target_detail_label = QLabel("No target selected")
        self.target_detail_label.setStyleSheet("color: #8b9bb4; font-size: 11px;")
        detail_layout.addWidget(self.target_detail_label)
        detail_layout.addStretch()

        self.lock_detail_btn = QPushButton("🔒 LOCK")
        self.lock_detail_btn.setObjectName("active_button")
        self.lock_detail_btn.clicked.connect(self.lock_target)
        detail_layout.addWidget(self.lock_detail_btn)

        self.track_detail_btn = QPushButton("👁 TRACK")
        self.track_detail_btn.clicked.connect(self.track_target)
        detail_layout.addWidget(self.track_detail_btn)

        self.prioritize_btn = QPushButton("⚡ PRIORITY")
        self.prioritize_btn.clicked.connect(self.set_target_priority)
        detail_layout.addWidget(self.prioritize_btn)

        center_panel.addWidget(self.target_detail_frame)

        # Bottom info panels
        bottom_info = QHBoxLayout()
        bottom_info.setSpacing(5)

        traj_frame = QFrame()
        traj_frame.setFixedHeight(200)
        traj_layout = QVBoxLayout(traj_frame)
        traj_title = QLabel("TRAJECTORY PREDICTION")
        traj_title.setStyleSheet("color: #00aaff; font-size: 10px; font-weight: bold; border: none;")
        traj_layout.addWidget(traj_title)
        self.trajectory_graph = TrajectoryGraph()
        traj_layout.addWidget(self.trajectory_graph)
        bottom_info.addWidget(traj_frame)

        wind_frame = QFrame()
        wind_frame.setFixedHeight(200)
        wind_layout = QVBoxLayout(wind_frame)
        wind_title = QLabel("WIND ANALYSIS")
        wind_title.setStyleSheet("color: #00aaff; font-size: 10px; font-weight: bold; border: none;")
        wind_layout.addWidget(wind_title)
        self.wind_compass = WindCompass()
        wind_layout.addWidget(self.wind_compass, alignment=Qt.AlignmentFlag.AlignCenter)
        bottom_info.addWidget(wind_frame)

        vitals_frame = QFrame()
        vitals_frame.setFixedHeight(200)
        vitals_layout = QVBoxLayout(vitals_frame)
        vitals_title = QLabel("TARGET VITALS")
        vitals_title.setStyleSheet("color: #00aaff; font-size: 10px; font-weight: bold; border: none;")
        vitals_layout.addWidget(vitals_title)
        self.target_vitals = TargetVitals()
        vitals_layout.addWidget(self.target_vitals, alignment=Qt.AlignmentFlag.AlignCenter)
        bottom_info.addWidget(vitals_frame)

        center_panel.addLayout(bottom_info)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(5)

        self.scan_btn = QPushButton("◉ SCAN")
        self.scan_btn.setFixedHeight(45)
        self.scan_btn.setObjectName("warning_button")
        self.scan_btn.clicked.connect(self.scan_targets)

        self.track_btn = QPushButton("◎ TRACK")
        self.track_btn.setFixedHeight(45)
        self.track_btn.clicked.connect(self.track_target)
        self.track_btn.setEnabled(False)

        self.lock_btn = QPushButton("⊕ LOCK")
        self.lock_btn.setFixedHeight(45)
        self.lock_btn.setObjectName("active_button")
        self.lock_btn.clicked.connect(self.lock_target)
        self.lock_btn.setEnabled(False)

        self.solve_btn = QPushButton("⚙ SOLVE")
        self.solve_btn.setFixedHeight(45)
        self.solve_btn.clicked.connect(self.solve_ballistics)
        self.solve_btn.setEnabled(False)

        self.fire_btn = QPushButton("◎ FIRE")
        self.fire_btn.setFixedHeight(45)
        self.fire_btn.setObjectName("fire_button")
        self.fire_btn.clicked.connect(self.fire_weapon)
        self.fire_btn.setEnabled(False)

        btn_row.addWidget(self.scan_btn)
        btn_row.addWidget(self.track_btn)
        btn_row.addWidget(self.lock_btn)
        btn_row.addWidget(self.solve_btn)
        btn_row.addWidget(self.fire_btn)

        center_panel.addLayout(btn_row)

        content.addLayout(center_panel)

        # RIGHT PANEL
        right_panel = QVBoxLayout()
        right_panel.setSpacing(5)

        targets_frame = QFrame()
        targets_frame.setFixedWidth(320)
        targets_layout = QVBoxLayout(targets_frame)

        targets_header = QHBoxLayout()
        targets_title = QLabel("TARGETS DETECTED")
        targets_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        targets_header.addWidget(targets_title)
        self.total_targets = QLabel("TOTAL: 0")
        self.total_targets.setStyleSheet("color: #8b9bb4; font-size: 10px; border: none;")
        targets_header.addWidget(self.total_targets)
        targets_layout.addLayout(targets_header)

        self.targets_table = QTableWidget()
        self.targets_table.setColumnCount(2)
        self.targets_table.setHorizontalHeaderLabels(["TARGET", "INFO"])
        self.targets_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.targets_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.targets_table.setColumnWidth(0, 80)
        self.targets_table.setFixedHeight(250)
        self.targets_table.verticalHeader().setVisible(False)
        self.targets_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.targets_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.targets_table.itemSelectionChanged.connect(self.on_table_selection)
        targets_layout.addWidget(self.targets_table)

        target_ctrl = QHBoxLayout()
        self.tbl_lock_btn = QPushButton("🔒 Lock")
        self.tbl_lock_btn.clicked.connect(self.lock_target)
        self.tbl_lock_btn.setEnabled(False)
        target_ctrl.addWidget(self.tbl_lock_btn)

        self.tbl_track_btn = QPushButton("👁 Track")
        self.tbl_track_btn.clicked.connect(self.track_target)
        self.tbl_track_btn.setEnabled(False)
        target_ctrl.addWidget(self.tbl_track_btn)

        self.tbl_remove_btn = QPushButton("✕ Ignore")
        self.tbl_remove_btn.clicked.connect(self.ignore_target)
        self.tbl_remove_btn.setEnabled(False)
        target_ctrl.addWidget(self.tbl_remove_btn)

        targets_layout.addLayout(target_ctrl)

        right_panel.addWidget(targets_frame)

        # Ballistic Solution
        ball_frame = QFrame()
        ball_frame.setFixedWidth(320)
        ball_layout = QVBoxLayout(ball_frame)

        ball_title = QLabel("BALLISTIC SOLUTION")
        ball_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        ball_layout.addWidget(ball_title)

        self.ballistic_labels = {}
        ball_items = [
            ("DISTANCE", "-- m"), ("TIME OF FLIGHT", "-- s"),
            ("DROP", "-- MRAD"), ("WIND DRIFT", "-- MRAD"),
            ("LEAD", "-- MRAD"), ("IMPACT VELOCITY", "-- m/s"),
            ("IMPACT ENERGY", "-- J"), ("SOLUTION CONFIDENCE", "--%")
        ]
        for name, val in ball_items:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setStyleSheet("color: #8b9bb4; font-size: 10px;")
            val_lbl = QLabel(val)
            val_lbl.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold;")
            self.ballistic_labels[name] = val_lbl
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val_lbl)
            ball_layout.addLayout(row)

        adv_ball = QGroupBox("Advanced Parameters")
        adv_layout = QVBoxLayout(adv_ball)

        self.adv_labels = {}
        adv_items = [
            ("AIR DENSITY", "-- kg/m³"), ("HOLDOVER", "-- MRAD"),
            ("WINDAGE HOLD", "-- MRAD"), ("CORIOLIS", "-- m"),
            ("SPIN DRIFT", "-- m")
        ]
        for name, val in adv_items:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setStyleSheet("color: #8b9bb4; font-size: 9px;")
            val_lbl = QLabel(val)
            val_lbl.setStyleSheet("color: #8b9bb4; font-size: 9px; font-weight: bold;")
            self.adv_labels[name] = val_lbl
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val_lbl)
            adv_layout.addLayout(row)

        ball_layout.addWidget(adv_ball)

        self.fire_solution = QLabel("○ AWAITING TARGET LOCK")
        self.fire_solution.setStyleSheet("color: #ffaa00; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")
        self.fire_solution.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ball_layout.addWidget(self.fire_solution)

        right_panel.addWidget(ball_frame)

        # System Log
        log_frame = QFrame()
        log_frame.setFixedWidth(320)
        log_layout = QVBoxLayout(log_frame)

        log_header = QHBoxLayout()
        log_title = QLabel("SYSTEM LOG")
        log_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        log_header.addWidget(log_title)

        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setFixedWidth(60)
        self.clear_log_btn.setStyleSheet("font-size: 9px; padding: 2px;")
        self.clear_log_btn.clicked.connect(self.clear_logs)
        log_header.addWidget(self.clear_log_btn)

        log_layout.addLayout(log_header)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(200)
        log_layout.addWidget(self.log_text)

        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #8b9bb4; font-size: 9px; border: none;")
        stats_layout.addWidget(self.fps_label)

        self.latency_label = QLabel("LAT: --ms")
        self.latency_label.setStyleSheet("color: #8b9bb4; font-size: 9px; border: none;")
        stats_layout.addWidget(self.latency_label)

        self.cpu_label = QLabel("CPU: --%")
        self.cpu_label.setStyleSheet("color: #8b9bb4; font-size: 9px; border: none;")
        stats_layout.addWidget(self.cpu_label)

        log_layout.addLayout(stats_layout)

        right_panel.addWidget(log_frame)
        right_panel.addStretch()

        content.addLayout(right_panel)

        main_layout.addLayout(content)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready | System: ONLINE | Mode: STANDBY | Press 'S' to scan, 'L' to lock, SPACE to fire")
        self.setStatusBar(self.status_bar)

    def init_video(self):
        self.video_thread = VideoCaptureThread(roi_manager=self.roi_manager)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.detections_ready.connect(self.update_detections)
        self.video_thread.start()

    def init_timers(self):
        self.env_timer = QTimer()
        self.env_timer.timeout.connect(self.update_environment)
        self.env_timer.start(2000)

        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.update_ai_metrics)
        self.ai_timer.start(3000)

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
        self.frame_times = deque(maxlen=60)

    def switch_tab(self, tab_name):
        for name, btn in self.tab_buttons.items():
            btn.setChecked(name == tab_name)
            if name == tab_name:
                btn.setStyleSheet("background-color: #1b3a5c; color: #00ff88;")
            else:
                btn.setStyleSheet("")
        self.add_log(f"Switched to {tab_name} tab")

    def set_interaction_mode(self, mode):
        for btn in [self.roi_rect_btn, self.roi_circle_btn, self.roi_poly_btn,
                   self.select_mode_btn, self.measure_mode_btn]:
            btn.setChecked(False)

        mode_map = {
            "roi_rect": self.roi_rect_btn,
            "roi_circle": self.roi_circle_btn,
            "roi_poly": self.roi_poly_btn,
            "select": self.select_mode_btn,
            "measure": self.measure_mode_btn
        }
        if mode in mode_map:
            mode_map[mode].setChecked(True)

        self.video_widget.set_mode(mode)
        self.add_log(f"Interaction mode: {mode.upper()}")

    def reset_view(self):
        self.video_widget.reset_zoom()
        self.zoom_slider.setValue(10)
        self.on_zoom_slider(10)
        self.add_log("View reset to default")

    def on_zoom_slider(self, value):
        zoom = value / 10.0
        self.zoom_level = zoom
        self.video_widget.set_zoom(zoom)
        self.video_thread.set_zoom(zoom)
        self.zoom_display.setText(f"{zoom:.1f}x")
        self.feed_title.setText(f"LIVE FEED - OPTICAL ZOOM {zoom:.1f}x")

    def on_zoom_changed(self, zoom):
        self.zoom_level = zoom
        self.zoom_slider.setValue(int(zoom * 10))
        self.zoom_display.setText(f"{zoom:.1f}x")

    def on_target_selected(self, target):
        self.selected_target = target
        self.video_widget.selected_target = target

        self.target_detail_frame.setVisible(True)
        self.target_detail_label.setText(
            f"TARGET: {target['id']} | CLASS: {target['class'].upper()} | "
            f"DIST: {target['distance']:.1f}m | CONF: {target['confidence']*100:.1f}% | "
            f"PRIORITY: {target.get('priority', TargetPriority.LOW).name}"
        )

        self.lock_btn.setEnabled(True)
        self.track_btn.setEnabled(True)
        self.tbl_lock_btn.setEnabled(True)
        self.tbl_track_btn.setEnabled(True)
        self.tbl_remove_btn.setEnabled(True)

        self.add_log(f"Target {target['id']} selected manually")

    def on_roi_defined(self, name, shape, geometry):
        roi = self.roi_manager.add_roi(name, shape, geometry)
        self.roi_list.update_rois(self.roi_manager.rois)
        self.add_log(f"ROI created: {name} ({shape.name})")

        for btn in [self.roi_rect_btn, self.roi_circle_btn, self.roi_poly_btn]:
            btn.setChecked(False)
        self.video_widget.set_mode("view")

    def on_roi_selected(self, roi_id):
        self.roi_manager.set_active(roi_id)
        self.roi_list.update_rois(self.roi_manager.rois)
        self.add_log(f"ROI activated: {roi_id}")

    def on_roi_activated(self, roi_id, active):
        for roi in self.roi_manager.rois:
            if roi['id'] == roi_id:
                roi['active'] = active
        self.roi_list.update_rois(self.roi_manager.rois)
        self.add_log(f"ROI {roi_id}: {'ACTIVE' if active else 'INACTIVE'}")

    def on_roi_deleted(self, roi_id):
        self.roi_manager.remove_roi(roi_id)
        self.roi_list.update_rois(self.roi_manager.rois)
        self.add_log(f"ROI deleted: {roi_id}")

    def clear_all_rois(self):
        self.roi_manager.clear_all()
        self.roi_list.update_rois([])
        self.add_log("All ROIs cleared")

    def on_table_selection(self):
        selected = self.targets_table.selectedItems()
        if selected:
            row = selected[0].row()
            if row < len(self.detections):
                target = self.detections[row]
                self.selected_target = target
                self.video_widget.selected_target = target
                self.on_target_selected(target)

    def update_frame(self, frame):
        self.current_frame = frame.copy()
        h, w = frame.shape[:2]

        overlay = frame.copy()

        cx, cy = w // 2, h // 2
        cross_size = max(20, int(40 / self.zoom_level))

        cv2.line(overlay, (cx - cross_size, cy), (cx - cross_size//3, cy), (0, 255, 136), 1)
        cv2.line(overlay, (cx + cross_size//3, cy), (cx + cross_size, cy), (0, 255, 136), 1)
        cv2.line(overlay, (cx, cy - cross_size), (cx, cy - cross_size//3), (0, 255, 136), 1)
        cv2.line(overlay, (cx, cy + cross_size//3), (cx, cy + cross_size), (0, 255, 136), 1)
        cv2.circle(overlay, (cx, cy), max(3, int(5/self.zoom_level)), (0, 255, 136), 1)

        compass_y = 40
        cv2.line(overlay, (w//2 - 120, compass_y), (w//2 + 120, compass_y), (50, 80, 110), 1)
        for i, deg in enumerate(['W', '', '300', '', '330', '', 'N', '', '30', '', '60', '', 'E']):
            x = w//2 - 120 + i * 20
            cv2.line(overlay, (x, compass_y - 5), (x, compass_y + 5), (100, 120, 140), 1)
            if deg:
                cv2.putText(overlay, deg, (x-8, compass_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 120, 140), 1)

        for i in range(-20, 21, 10):
            y = h//2 - i * 8
            cv2.line(overlay, (30, y), (40, y), (100, 120, 140), 1)
            cv2.putText(overlay, str(i), (10, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 120, 140), 1)

        for i in range(-20, 21, 10):
            x = w//2 + i * 8
            cv2.line(overlay, (x, h-40), (x, h-30), (100, 120, 140), 1)
            cv2.putText(overlay, str(i), (x-5, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 120, 140), 1)

        if self.system_mode == SystemMode.SCANNING:
            scan_y = int((self.frame_count * 3) % h)
            cv2.line(overlay, (0, scan_y), (w, scan_y), (0, 255, 136, 150), 2)

        rgb_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        scaled_pixmap = pixmap.scaled(self.video_widget.size(), 
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self.video_widget.setPixmap(scaled_pixmap)
        self.video_widget.set_detections(self.detections)

        self.frame_count += 1
        self.frame_times.append(time.time())

    def update_detections(self, detections):
        self.detections = detections

        if self.tracked_target:
            found = False
            for det in detections:
                if det['id'] == self.tracked_target['id']:
                    self.tracked_target = det
                    found = True
                    break
            if not found:
                self.tracked_target = None
                self.status_indicators["TARGET ACQUISITION"].update_status_color("ACTIVE")
                self.add_log("Tracking lost - target out of frame")

        self.update_targets_table()
        self.total_targets.setText(f"TOTAL: {len(detections)}")
        self.tracked_count.setText(str(len([d for d in detections if d.get('tracked')])))

    def update_targets_table(self):
        self.targets_table.setRowCount(len(self.detections))

        for i, det in enumerate(self.detections):
            thumb_label = QLabel()
            thumb = np.zeros((60, 50, 3), dtype=np.uint8)

            pri = det.get('priority', TargetPriority.LOW)
            if pri == TargetPriority.CRITICAL:
                thumb[:, :] = (60, 20, 20)
            elif pri == TargetPriority.HIGH:
                thumb[:, :] = (40, 30, 15)
            elif pri == TargetPriority.MEDIUM:
                thumb[:, :] = (35, 35, 15)
            else:
                thumb[:, :] = (20, 25, 30)

            cv2.circle(thumb, (25, 15), 8, (80, 90, 100), -1)
            cv2.rectangle(thumb, (15, 25), (35, 55), (60, 70, 80), -1)

            status = det.get('status', 'DETECTED')
            if status == 'LOCKED':
                cv2.rectangle(thumb, (0, 0), (49, 59), (0, 255, 136), 2)
            elif status == 'TRACKED':
                cv2.rectangle(thumb, (0, 0), (49, 59), (255, 165, 0), 2)

            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            thumb_img = QImage(thumb_rgb.data, 50, 60, 150, QImage.Format.Format_RGB888)
            thumb_label.setPixmap(QPixmap.fromImage(thumb_img))
            self.targets_table.setCellWidget(i, 0, thumb_label)

            pri = det.get('priority', TargetPriority.LOW)
            status = det.get('status', 'DETECTED')
            status_color = "#00ff88" if status == 'LOCKED' else ("#ffaa00" if status == 'TRACKED' else "#8b9bb4")

            info_text = f"""TARGET ID: {det['id']}
CLASS: {det['class'].upper()}
DISTANCE: {det['distance']:.1f} m
CONFIDENCE: {det['confidence']*100:.1f}%
PRIORITY: <span style='color: {pri.value[1]}; font-weight: bold;'>{pri.name}</span>
STATUS: <span style='color: {status_color}; font-weight: bold;'>{status}</span>"""

            info_label = QLabel(info_text)
            info_label.setStyleSheet("color: #8b9bb4; font-size: 10px;")
            self.targets_table.setCellWidget(i, 1, info_label)
            self.targets_table.setRowHeight(i, 75)

    def update_environment(self):
        self.environment.update()
        data = self.environment.get_data()

        self.env_labels["WIND SPEED"].setText(f"{data['wind_speed']:.1f} m/s")
        self.env_labels["WIND GUST"].setText(f"{data['wind_gust']:.1f} m/s")
        self.env_labels["WIND DIRECTION"].setText(f"NW {data['wind_direction']}°")
        self.env_labels["TEMPERATURE"].setText(f"{data['temperature']:.1f} °C")
        self.env_labels["HUMIDITY"].setText(f"{data['humidity']} %")
        self.env_labels["PRESSURE"].setText(f"{data['pressure']} hPa")
        self.env_labels["VISIBILITY"].setText(f"{data['visibility']:.1f} km")

        self.stab_val.setText(f"{self.environment.get_wind_stability():.1f}%")

        self.wind_compass.set_data(data['wind_speed'], data['wind_gust'], data['wind_direction'])

    def update_clock(self):
        now = datetime.now()
        self.overlay_info.setText(
            f"DATE: {now.strftime('%Y-%m-%d')}    "
            f"TIME: {now.strftime('%H:%M:%S')}    "
            f"MODE: DAY    ZOOM: {self.zoom_level:.1f}x"
        )

    def update_ai_metrics(self):
        if self.detections:
            avg_conf = sum(d['confidence'] for d in self.detections) / len(self.detections)
            self.conf_value.setText(f"{avg_conf*100:.1f}%")
            self.conf_bar.setValue(int(avg_conf * 100))
        else:
            self.conf_value.setText("--")
            self.conf_bar.setValue(0)

        self.obj_det_val.setText(f"{random.randint(8, 18)}ms")
        self.pred_acc_val.setText(f"{random.uniform(93, 98):.1f}%")

    def update_fps(self):
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.latency_label.setText(f"LAT: {random.randint(12, 35)}ms")
        self.cpu_label.setText(f"CPU: {random.randint(15, 45)}%")

    def scan_targets(self):
        if self.system_mode == SystemMode.SCANNING:
            self.system_mode = SystemMode.STANDBY
            self.scan_btn.setText("◉ SCAN")
            self.scan_btn.setObjectName("warning_button")
            self.scan_btn.setStyleSheet("")
            self.status_indicators["TARGET ACQUISITION"].update_status_color("STANDBY")
            self.mode_indicator.setText("STANDBY")
            self.mode_indicator.setStyleSheet("""
                color: #ffaa00; font-size: 12px; font-weight: bold; 
                border: 2px solid #ffaa00; border-radius: 4px; padding: 4px 12px;
            """)
            self.video_thread.set_scan_mode(False)
            self.add_log("Scanning stopped")
        else:
            self.system_mode = SystemMode.SCANNING
            self.scan_btn.setText("⏹ STOP")
            self.scan_btn.setObjectName("")
            self.scan_btn.setStyleSheet("background-color: #3a0a0a; color: #ff4444; border: 2px solid #ff4444;")
            self.status_indicators["TARGET ACQUISITION"].update_status_color("SCANNING")
            self.mode_indicator.setText("SCANNING")
            self.mode_indicator.setStyleSheet("""
                color: #00ffff; font-size: 12px; font-weight: bold; 
                border: 2px solid #00ffff; border-radius: 4px; padding: 4px 12px;
            """)
            self.video_thread.set_scan_mode(True)
            self.add_log("Scanning initiated - sweeping sector")

            QTimer.singleShot(5000, self.scan_complete)

    def scan_complete(self):
        if self.system_mode == SystemMode.SCANNING:
            self.system_mode = SystemMode.STANDBY
            self.scan_btn.setText("◉ SCAN")
            self.scan_btn.setObjectName("warning_button")
            self.scan_btn.setStyleSheet("")
            self.status_indicators["TARGET ACQUISITION"].update_status_color("ACTIVE")
            self.mode_indicator.setText("ACTIVE")
            self.mode_indicator.setStyleSheet("""
                color: #00ff88; font-size: 12px; font-weight: bold; 
                border: 2px solid #00ff88; border-radius: 4px; padding: 4px 12px;
            """)
            self.video_thread.set_scan_mode(False)
            self.add_log(f"Scan complete. {len(self.detections)} targets detected.")

    def track_target(self):
        if self.selected_target:
            self.tracked_target = self.selected_target
            self.selected_target['tracked'] = True
            self.selected_target['status'] = 'TRACKED'

            self.system_mode = SystemMode.TRACKING
            self.status_indicators["TARGET ACQUISITION"].update_status_color("TRACKING")
            self.mode_indicator.setText("TRACKING")
            self.mode_indicator.setStyleSheet("""
                color: #00ff88; font-size: 12px; font-weight: bold; 
                border: 2px solid #00ff88; border-radius: 4px; padding: 4px 12px;
            """)

            self.add_log(f"Tracking target {self.selected_target['id']}")
            self.update_targets_table()

            QTimer.singleShot(2000, self.solve_ballistics)
        else:
            self.add_log("No target selected for tracking")

    def lock_target(self):
        if self.selected_target:
            self.locked_target = self.selected_target
            self.selected_target['status'] = 'LOCKED'

            self.system_mode = SystemMode.LOCKED
            self.status_indicators["TARGET ACQUISITION"].update_status_color("LOCKED")
            self.mode_indicator.setText("LOCKED")
            self.mode_indicator.setStyleSheet("""
                color: #ff4444; font-size: 12px; font-weight: bold; 
                border: 2px solid #ff4444; border-radius: 4px; padding: 4px 12px;
            """)

            self.lock_btn.setText("🔓 UNLOCK")
            self.lock_btn.setObjectName("fire_button")
            self.lock_btn.setStyleSheet("""
                background-color: #3a0a0a; color: #ff4444; border: 2px solid #ff4444; font-size: 14px;
            """)
            self.lock_btn.clicked.disconnect()
            self.lock_btn.clicked.connect(self.unlock_target)

            self.add_log(f"Target {self.locked_target['id']} LOCKED")
            self.update_targets_table()
            self.solve_btn.setEnabled(True)

            QTimer.singleShot(1000, self.solve_ballistics)
        else:
            self.add_log("No target selected for lock")

    def unlock_target(self):
        if self.locked_target:
            self.locked_target['status'] = 'DETECTED'
            self.locked_target = None

        self.system_mode = SystemMode.ACTIVE
        self.status_indicators["TARGET ACQUISITION"].update_status_color("ACTIVE")
        self.mode_indicator.setText("ACTIVE")
        self.mode_indicator.setStyleSheet("""
            color: #00ff88; font-size: 12px; font-weight: bold; 
            border: 2px solid #00ff88; border-radius: 4px; padding: 4px 12px;
        """)

        self.lock_btn.setText("⊕ LOCK")
        self.lock_btn.setObjectName("active_button")
        self.lock_btn.setStyleSheet("")
        self.lock_btn.clicked.disconnect()
        self.lock_btn.clicked.connect(self.lock_target)

        self.solve_btn.setEnabled(False)
        self.fire_btn.setEnabled(False)
        self.fire_solution.setText("○ AWAITING TARGET LOCK")
        self.fire_solution.setStyleSheet("color: #ffaa00; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")

        self.add_log("Target unlocked")
        self.update_targets_table()

    def set_target_priority(self):
        if self.selected_target:
            priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            current = self.selected_target.get('priority', TargetPriority.LOW).name
            idx = priorities.index(current) if current in priorities else 0

            new_pri, ok = QInputDialog.getItem(self, "Set Priority", 
                                               "Select priority level:", priorities, idx, False)
            if ok:
                pri_map = {
                    "LOW": TargetPriority.LOW,
                    "MEDIUM": TargetPriority.MEDIUM,
                    "HIGH": TargetPriority.HIGH,
                    "CRITICAL": TargetPriority.CRITICAL
                }
                self.selected_target['priority'] = pri_map[new_pri]
                self.add_log(f"Target {self.selected_target['id']} priority set to {new_pri}")
                self.update_targets_table()
                self.on_target_selected(self.selected_target)

    def ignore_target(self):
        selected = self.targets_table.selectedItems()
        if selected:
            row = selected[0].row()
            if row < len(self.detections):
                target = self.detections.pop(row)
                self.add_log(f"Target {target['id']} ignored/removed from list")
                self.update_targets_table()
                self.total_targets.setText(f"TOTAL: {len(self.detections)}")

    def solve_ballistics(self):
        if self.locked_target:
            dist = self.locked_target['distance']
            env = self.environment.get_data()

            solution = self.ballistics.calculate_solution(
                dist, env['wind_speed'], env['wind_direction'],
                env['temperature'], env['humidity'], env['pressure'],
                altitude=env.get('altitude', 245),
                target_speed=self.locked_target.get('speed', 0),
                target_angle=self.locked_target.get('direction', 0)
            )

            self.ballistic_labels["DISTANCE"].setText(f"{solution['distance']:.1f} m")
            self.ballistic_labels["TIME OF FLIGHT"].setText(f"{solution['time_of_flight']:.3f} s")
            self.ballistic_labels["DROP"].setText(f"{solution['drop']:.2f} MRAD")
            self.ballistic_labels["WIND DRIFT"].setText(f"{solution['windage']:.2f} MRAD")
            self.ballistic_labels["LEAD"].setText(f"{solution['lead']:.2f} MRAD")
            self.ballistic_labels["IMPACT VELOCITY"].setText(f"{solution['impact_velocity']:.1f} m/s")
            self.ballistic_labels["IMPACT ENERGY"].setText(f"{solution['impact_energy']:.1f} J")
            self.ballistic_labels["SOLUTION CONFIDENCE"].setText(f"{solution['confidence']:.1f}%")

            self.adv_labels["AIR DENSITY"].setText(f"{solution['air_density']:.4f} kg/m³")
            self.adv_labels["HOLDOVER"].setText(f"{solution['holdover']:.2f} MRAD")
            self.adv_labels["WINDAGE HOLD"].setText(f"{solution['windage_hold']:.2f} MRAD")
            self.adv_labels["CORIOLIS"].setText(f"{solution['coriolis']:.4f} m")
            self.adv_labels["SPIN DRIFT"].setText(f"{solution['spin_drift']:.4f} m")

            self.windage_val.setText(f"{solution['windage']:.2f} MRAD")
            self.elev_val.setText(f"{solution['drop']:.2f} MRAD")

            traj_points = self.ballistics.calculate_trajectory_points(solution['distance'])
            self.trajectory_graph.set_data(
                solution['distance'], solution['drop'], 
                solution['windage'], traj_points
            )

            self.target_vitals.set_data(
                solution['confidence'],
                self.locked_target.get('vitals', 'HEART/LUNGS'),
                'LETHAL' if solution['confidence'] > 90 else 'MARGINAL',
                self.locked_target.get('lethality', 0.95)
            )

            self.system_mode = SystemMode.SOLVED
            self.add_log(f"Ballistic solution calculated for target {self.locked_target['id']}")
            self.add_log(f"  Distance: {solution['distance']:.1f}m | TOF: {solution['time_of_flight']:.3f}s")
            self.add_log(f"  Drop: {solution['drop']:.2f} MRAD | Windage: {solution['windage']:.2f} MRAD")

            self.fire_solution.setText("● FIRE SOLUTION READY")
            self.fire_solution.setStyleSheet("color: #00ff88; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")
            self.fire_btn.setEnabled(True)

        else:
            self.add_log("No locked target for ballistics solution")

    def fire_weapon(self):
        if self.locked_target and self.fire_solution.text().find("READY") != -1:
            if self.ammo_count <= 0:
                self.add_log("AMMUNITION DEPLETED - RELOAD REQUIRED")
                QMessageBox.warning(self, "Ammunition Depleted", 
                                  "Magazine empty. Reload required.",
                                  QMessageBox.StandardButton.Ok)
                return

            self.system_mode = SystemMode.FIRING
            self.add_log("FIRING SEQUENCE INITIATED")

            self.video_widget.setStyleSheet("background-color: #ff4444; border: 1px solid #1b3a5c;")
            QTimer.singleShot(80, lambda: self.video_widget.setStyleSheet("background-color: #000000; border: 1px solid #1b3a5c;"))

            self.fire_btn.setStyleSheet("""
                background-color: #ff0000; color: #ffffff; 
                border: 3px solid #ff0000; font-size: 16px; font-weight: bold;
            """)
            self.fire_btn.setText("💥 FIRING")
            self.fire_btn.setEnabled(False)

            QTimer.singleShot(150, self.fire_complete)
        else:
            self.add_log("FIRE SOLUTION NOT READY - LOCK AND SOLVE REQUIRED")

    def fire_complete(self):
        self.ammo_count -= 1
        self.shots_fired += 1

        if self.locked_target:
            conf = float(self.ballistic_labels["SOLUTION CONFIDENCE"].text().replace('%', ''))
            hit = random.random() < (conf / 100)

            if hit:
                self.hits += 1
                self.add_log(f"✓ IMPACT CONFIRMED - Target {self.locked_target['id']} neutralized")
                self.add_log(f"  Impact energy: {self.ballistic_labels['IMPACT ENERGY'].text()}")

                self.detections = [d for d in self.detections if d['id'] != self.locked_target['id']]
                self.locked_target = None
                self.selected_target = None
                self.tracked_target = None

                self.system_mode = SystemMode.STANDBY
                self.status_indicators["TARGET ACQUISITION"].update_status_color("ACTIVE")
                self.mode_indicator.setText("STANDBY")
                self.mode_indicator.setStyleSheet("""
                    color: #ffaa00; font-size: 12px; font-weight: bold; 
                    border: 2px solid #ffaa00; border-radius: 4px; padding: 4px 12px;
                """)

                self.fire_solution.setText("○ AWAITING TARGET LOCK")
                self.fire_solution.setStyleSheet("color: #ffaa00; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")
                self.fire_btn.setEnabled(False)
                self.solve_btn.setEnabled(False)
                self.lock_btn.setEnabled(False)
                self.track_btn.setEnabled(False)

                self.target_detail_frame.setVisible(False)

                self.lock_btn.setText("⊕ LOCK")
                self.lock_btn.setObjectName("active_button")
                self.lock_btn.setStyleSheet("")
                try:
                    self.lock_btn.clicked.disconnect()
                except:
                    pass
                self.lock_btn.clicked.connect(self.lock_target)

            else:
                self.add_log(f"✗ MISS - Target {self.locked_target['id']} still active")
                self.add_log("  Re-solving ballistic solution...")
                QTimer.singleShot(1000, self.solve_ballistics)

        self.ammo_val.setText(f"{self.ammo_count} / {self.max_ammo}")
        if self.ammo_count <= 5:
            self.ammo_val.setStyleSheet("color: #ff4444; font-weight: bold;")
        else:
            self.ammo_val.setStyleSheet("color: #ffaa00; font-weight: bold;")

        self.shots_val.setText(str(self.shots_fired))
        hit_ratio = (self.hits / self.shots_fired * 100) if self.shots_fired > 0 else 0
        self.hit_ratio.setText(f"{hit_ratio:.1f}%")

        self.fire_btn.setStyleSheet("""
            QPushButton#fire_button {
                background-color: #3a0a0a; color: #ff4444; border: 2px solid #ff4444; font-size: 14px;
            }
            QPushButton#fire_button:hover { background-color: #5a0a0a; }
        """)
        self.fire_btn.setText("◎ FIRE")
        self.fire_btn.setObjectName("fire_button")

        self.system_mode = SystemMode.COOLDOWN
        QTimer.singleShot(2000, self.fire_cooldown_complete)

    def fire_cooldown_complete(self):
        if self.system_mode == SystemMode.COOLDOWN:
            self.system_mode = SystemMode.STANDBY
            self.fire_btn.setEnabled(True)
            self.add_log("Weapon ready for next engagement")

    def clear_logs(self):
        self.system_logs.clear()
        self.log_text.clear()
        self.add_log("System log cleared")

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.system_logs.append(log_entry)
        self.log_text.setPlainText("\n".join(self.system_logs))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_S:
            self.scan_targets()
        elif key == Qt.Key.Key_T:
            self.track_target()
        elif key == Qt.Key.Key_L:
            self.lock_target()
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            self.solve_ballistics()
        elif key == Qt.Key.Key_Space:
            self.fire_weapon()
        elif key == Qt.Key.Key_Escape:
            if self.locked_target:
                self.unlock_target()
            else:
                self.video_widget.set_mode("view")
                for btn in [self.roi_rect_btn, self.roi_circle_btn, self.roi_poly_btn,
                           self.select_mode_btn, self.measure_mode_btn]:
                    btn.setChecked(False)
        elif key == Qt.Key.Key_R:
            self.reset_view()
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.zoom_slider.setValue(min(200, self.zoom_slider.value() + 5))
            self.on_zoom_slider(self.zoom_slider.value())
        elif key == Qt.Key.Key_Minus:
            self.zoom_slider.setValue(max(10, self.zoom_slider.value() - 5))
            self.on_zoom_slider(self.zoom_slider.value())
        elif key == Qt.Key.Key_1:
            self.set_interaction_mode("select")
        elif key == Qt.Key.Key_2:
            self.set_interaction_mode("roi_rect")
        elif key == Qt.Key.Key_3:
            self.set_interaction_mode("roi_circle")
        elif key == Qt.Key.Key_4:
            self.set_interaction_mode("measure")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Consolas", 10)
    app.setFont(font)
    app.setStyle("Fusion")

    window = AISniperSystem()
    window.show()
    sys.exit(app.exec())