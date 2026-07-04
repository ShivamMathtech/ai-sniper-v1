
import sys
import random
import math
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QProgressBar,
    QTabWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem,
    QGraphicsEllipseItem, QGraphicsPolygonItem, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QSpacerItem,
    QGroupBox, QSplitter, QTextEdit, QScrollArea, QStackedWidget
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QRect, QPoint, QPointF,
    QPropertyAnimation, QEasingCurve, QSize, QTime
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont,
    QFontDatabase, QLinearGradient, QRadialGradient, QPolygonF,
    QTransform, QIcon, QPalette, QKeyEvent, QMouseEvent
)

# ============================================================================
# YOLO DETECTOR SIMULATION (Replace with actual YOLO model loading)
# ============================================================================
class YOLODetector:
    def __init__(self):
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.classes = ['person', 'vehicle', 'equipment']
        self.colors = {
            'person': (0, 255, 0),
            'vehicle': (255, 165, 0),
            'equipment': (0, 191, 255)
        }

    def detect(self, frame):
        """Simulate YOLO detection - returns bounding boxes"""
        h, w = frame.shape[:2]
        detections = []

        # Simulate 1-4 targets with realistic positioning
        num_targets = random.randint(1, 4)
        for i in range(num_targets):
            # Random position with some clustering
            cx = random.randint(int(w*0.2), int(w*0.8))
            cy = random.randint(int(h*0.2), int(h*0.7))
            bw = random.randint(40, 120)
            bh = random.randint(80, 200)

            confidence = random.uniform(0.65, 0.99)
            class_name = random.choice(self.classes)

            detections.append({
                'bbox': (cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2),
                'confidence': confidence,
                'class': class_name,
                'id': f'T{i+1:02d}'
            })

        return detections

# ============================================================================
# BALLISTICS ENGINE
# ============================================================================
class BallisticsEngine:
    def __init__(self):
        self.gravity = 9.81
        self.air_density = 1.225

    def calculate_solution(self, distance, wind_speed, wind_dir, temp, humidity, pressure):
        """Calculate firing solution parameters"""
        # Simplified ballistics model
        time_of_flight = distance / 850  # ~850 m/s muzzle velocity

        # Bullet drop calculation
        drop = 0.5 * self.gravity * (time_of_flight ** 2)
        drop_mrad = math.atan(drop / distance) * 1000  # Convert to MRAD

        # Wind drift
        wind_rad = math.radians(wind_dir)
        wind_effect = wind_speed * time_of_flight * 0.1
        windage_mrad = math.atan(wind_effect / distance) * 1000

        # Lead calculation (assuming target moving 1.4 m/s)
        lead_mrad = math.atan(1.4 * time_of_flight / distance) * 1000

        # Impact velocity (simplified drag)
        impact_vel = 850 * math.exp(-0.0001 * distance)

        # Solution confidence based on environmental stability
        confidence = min(99.9, 95 + random.uniform(-2, 4))

        return {
            'distance': distance,
            'time_of_flight': round(time_of_flight, 3),
            'drop': round(drop_mrad, 2),
            'windage': round(windage_mrad, 2),
            'lead': round(lead_mrad, 2),
            'impact_velocity': round(impact_vel, 1),
            'confidence': round(confidence, 1)
        }

# ============================================================================
# ENVIRONMENTAL SENSOR SIMULATION
# ============================================================================
class EnvironmentSensor:
    def __init__(self):
        self.wind_speed = 4.2
        self.wind_dir = 315  # NW
        self.temperature = 18.6
        self.humidity = 62
        self.pressure = 1013
        self.visibility = 8.7

    def update(self):
        """Simulate sensor fluctuations"""
        self.wind_speed = max(0, self.wind_speed + random.uniform(-0.3, 0.3))
        self.wind_dir = (self.wind_dir + random.uniform(-5, 5)) % 360
        self.temperature += random.uniform(-0.1, 0.1)
        self.humidity = max(0, min(100, self.humidity + random.uniform(-1, 1)))
        self.pressure += random.uniform(-0.5, 0.5)
        self.visibility = max(0, self.visibility + random.uniform(-0.1, 0.1))

    def get_data(self):
        return {
            'wind_speed': round(self.wind_speed, 1),
            'wind_direction': int(self.wind_dir),
            'temperature': round(self.temperature, 1),
            'humidity': int(self.humidity),
            'pressure': int(self.pressure),
            'visibility': round(self.visibility, 1)
        }

# ============================================================================
# VIDEO CAPTURE THREAD
# ============================================================================
class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    detections_ready = pyqtSignal(list)

    def __init__(self, source=0, parent=None):
        super().__init__(parent)
        self.source = source
        self.running = True
        self.detector = YOLODetector()
        self.frame_count = 0

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            # Create synthetic video feed
            while self.running:
                frame = self.generate_synthetic_frame()
                self.frame_ready.emit(frame)

                if self.frame_count % 5 == 0:
                    detections = self.detector.detect(frame)
                    self.detections_ready.emit(detections)

                self.frame_count += 1
                self.msleep(33)  # ~30 FPS
        else:
            while self.running:
                ret, frame = cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                    if self.frame_count % 5 == 0:
                        detections = self.detector.detect(frame)
                        self.detections_ready.emit(detections)
                    self.frame_count += 1
                self.msleep(33)
        cap.release()

    def generate_synthetic_frame(self):
        """Generate a realistic tactical scene"""
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Create urban/degraded environment
        # Sky gradient
        for y in range(h//2):
            intensity = int(20 + (y / (h//2)) * 30)
            frame[y, :] = [intensity, intensity, intensity + 5]

        # Ground/buildings
        for y in range(h//2, h):
            intensity = int(30 + ((y - h//2) / (h//2)) * 40)
            frame[y, :] = [intensity, intensity, intensity]

        # Add building silhouettes
        buildings = [(100, 300, 200, 400), (400, 200, 300, 500), 
                     (800, 250, 250, 450), (1100, 350, 150, 350)]
        for bx, by, bw, bh in buildings:
            cv2.rectangle(frame, (bx, h//2 + by), (bx + bw, h), (15, 15, 18), -1)
            # Windows
            for wx in range(bx + 10, bx + bw - 20, 40):
                for wy in range(h//2 + by + 20, h - 20, 50):
                    if random.random() > 0.7:
                        cv2.rectangle(frame, (wx, wy), (wx + 25, wy + 30), (40, 45, 35), -1)

        # Add noise/grain
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add scanlines
        frame[::2, :] = (frame[::2, :] * 0.85).astype(np.uint8)

        # Add vignette
        center = (w//2, h//2)
        for y in range(h):
            for x in range(w):
                dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                vignette = max(0, 1 - dist / 900)
                frame[y, x] = (frame[y, x] * vignette).astype(np.uint8)

        return frame

    def stop(self):
        self.running = False
        self.wait()

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
            'DETECTED': '#ffaa00', 'STANDBY': '#ffaa00', 'OFFLINE': '#ff4444'
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
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(140, 140)
        self.setStyleSheet("background-color: transparent;")

    def set_value(self, val):
        self.current_value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = QPointF(70, 70)
        radius = 55

        # Background arc
        pen = QPen(QColor(30, 50, 70))
        pen.setWidth(8)
        painter.setPen(pen)
        painter.drawArc(int(center.x() - radius), int(center.y() - radius), 
                       int(radius*2), int(radius*2), 225*16, -270*16)

        # Value arc
        ratio = (self.current_value - self.min_val) / (self.max_val - self.min_val)
        ratio = max(0, min(1, ratio))
        span = int(-270 * 16 * ratio)

        gradient = QLinearGradient(0, 0, 140, 0)
        gradient.setColorAt(0, QColor(0, 255, 136))
        gradient.setColorAt(1, QColor(0, 200, 255))

        pen = QPen(QBrush(gradient), 8)
        painter.setPen(pen)
        painter.drawArc(int(center.x() - radius), int(center.y() - radius),
                       int(radius*2), int(radius*2), 225*16, span)

        # Text
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
        self.wind_dir = 315
        self.setFixedSize(150, 150)
        self.setStyleSheet("background-color: transparent;")

    def set_data(self, speed, direction):
        self.wind_speed = speed
        self.wind_dir = direction
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = QPointF(75, 75)
        radius = 60

        # Outer circle
        pen = QPen(QColor(50, 80, 110))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(10, 20, 35)))
        painter.drawEllipse(center, radius, radius)

        # Cardinal directions
        painter.setPen(QColor(139, 155, 180))
        font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(font)

        dirs = {'N': (75, 18), 'S': (75, 132), 'E': (132, 78), 'W': (18, 78)}
        for d, (x, y) in dirs.items():
            painter.drawText(x-6, y+4, 12, 12, Qt.AlignmentFlag.AlignCenter, d)

        # Wind arrow
        angle = math.radians(self.wind_dir)
        arrow_len = 45

        end_x = center.x() + arrow_len * math.sin(angle)
        end_y = center.y() - arrow_len * math.cos(angle)

        pen = QPen(QColor(0, 255, 136))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(int(center.x()), int(center.y()), int(end_x), int(end_y))

        # Arrow head
        head_size = 8
        painter.setBrush(QBrush(QColor(0, 255, 136)))
        painter.drawEllipse(QPointF(end_x, end_y), 4, 4)

        # Speed text
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(35, 68, 80, 20, Qt.AlignmentFlag.AlignCenter, f"{self.wind_speed:.1f}")

        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.setPen(QColor(139, 155, 180))
        painter.drawText(35, 85, 80, 15, Qt.AlignmentFlag.AlignCenter, "m/s")

        # Direction text
        painter.drawText(35, 100, 80, 15, Qt.AlignmentFlag.AlignCenter, f"{int(self.wind_dir)}°")

class TrajectoryGraph(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.distance = 612.4
        self.drop = 1.28
        self.setFixedSize(280, 160)
        self.setStyleSheet("background-color: transparent;")

    def set_data(self, distance, drop):
        self.distance = distance
        self.drop = drop
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 30

        # Grid
        pen = QPen(QColor(30, 50, 70))
        pen.setWidth(1)
        painter.setPen(pen)

        for i in range(5):
            x = margin + (w - 2*margin) * i / 4
            painter.drawLine(int(x), margin, int(x), h - margin)

        for i in range(4):
            y = margin + (h - 2*margin) * i / 3
            painter.drawLine(margin, int(y), w - margin, int(y))

        # Trajectory curve
        points = []
        max_dist = max(self.distance * 1.5, 1000)

        for i in range(50):
            t = i / 49
            x = t * max_dist
            # Parabolic trajectory
            y_drop = self.drop * (x / self.distance) ** 2
            px = margin + (w - 2*margin) * t
            py = h - margin - (h - 2*margin) * 0.3 - y_drop * 20
            points.append(QPointF(px, py))

        # Draw curve
        if len(points) > 1:
            pen = QPen(QColor(0, 255, 136))
            pen.setWidth(2)
            painter.setPen(pen)
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i+1])

        # Target point
        target_t = self.distance / max_dist
        target_x = margin + (w - 2*margin) * target_t
        target_y = h - margin - (h - 2*margin) * 0.3 - self.drop * 20

        pen = QPen(QColor(255, 68, 68))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 68, 68, 100)))
        painter.drawEllipse(QPointF(target_x, target_y), 6, 6)

        # Labels
        painter.setPen(QColor(139, 155, 180))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.drawText(int(target_x - 25), int(target_y - 15), 50, 15, 
                        Qt.AlignmentFlag.AlignCenter, f"{self.distance:.1f}m")

        # Axis labels
        painter.drawText(margin, h - 10, 40, 15, Qt.AlignmentFlag.AlignLeft, "0")
        painter.drawText(w - margin - 40, h - 10, 40, 15, Qt.AlignmentFlag.AlignRight, 
                        f"{int(max_dist)}")
        painter.drawText(5, margin + 10, 20, 15, Qt.AlignmentFlag.AlignCenter, "+2")
        painter.drawText(5, h - margin - 5, 20, 15, Qt.AlignmentFlag.AlignCenter, "-2")

class TargetVitals(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.probability = 98.7
        self.vital_area = "HEART / LUNGS"
        self.status = "LETHAL"
        self.setFixedSize(180, 200)
        self.setStyleSheet("background-color: transparent;")

    def set_data(self, probability, vital_area, status):
        self.probability = probability
        self.vital_area = vital_area
        self.status = status
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Draw body silhouette
        center_x = w // 2
        head_y = 30

        # Head
        painter.setPen(QPen(QColor(100, 120, 140), 2))
        painter.setBrush(QBrush(QColor(20, 30, 45)))
        painter.drawEllipse(center_x - 12, head_y, 24, 28)

        # Body
        body_points = [
            QPointF(center_x - 20, head_y + 28),
            QPointF(center_x + 20, head_y + 28),
            QPointF(center_x + 25, head_y + 80),
            QPointF(center_x - 25, head_y + 80)
        ]
        painter.drawPolygon(QPolygonF(body_points))

        # Arms
        painter.drawLine(int(center_x - 20), int(head_y + 35), 
                        int(center_x - 35), int(head_y + 70))
        painter.drawLine(int(center_x + 20), int(head_y + 35), 
                        int(center_x + 35), int(head_y + 70))

        # Legs
        painter.drawLine(int(center_x - 10), int(head_y + 80), 
                        int(center_x - 15), int(head_y + 130))
        painter.drawLine(int(center_x + 10), int(head_y + 80), 
                        int(center_x + 15), int(head_y + 130))

        # Vital zone highlight
        if self.status == "LETHAL":
            gradient = QRadialGradient(center_x, head_y + 45, 20)
            gradient.setColorAt(0, QColor(255, 68, 68, 180))
            gradient.setColorAt(1, QColor(255, 68, 68, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center_x - 20, head_y + 30, 40, 35)

        # Text info
        painter.setPen(QColor(139, 155, 180))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.drawText(0, 0, w, 20, Qt.AlignmentFlag.AlignCenter, "VITAL AREA")

        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 145, w, 20, Qt.AlignmentFlag.AlignCenter, self.vital_area)

        painter.setPen(QColor(0, 255, 136))
        font = QFont("Consolas", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 165, w, 25, Qt.AlignmentFlag.AlignCenter, f"{self.probability}%")

        color = '#00ff88' if self.status == 'LETHAL' else '#ffaa00'
        painter.setPen(QColor(color))
        font = QFont("Consolas", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 190, w, 20, Qt.AlignmentFlag.AlignCenter, self.status)

# ============================================================================
# MAIN WINDOW
# ============================================================================
class AISniperSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI SNIPER SYSTEM v4.2.1")
        self.setGeometry(50, 50, 1600, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #050a10;
            }
            QWidget {
                background-color: #050a10;
                color: #e0e6ed;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QFrame {
                border: 1px solid #1b3a5c;
                background-color: #0a1628;
            }
            QLabel {
                color: #e0e6ed;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0d2137;
                color: #8b9bb4;
                border: 1px solid #1b3a5c;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1b3a5c;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #0d2137;
            }
            QPushButton#fire_button {
                background-color: #3a0a0a;
                color: #ff4444;
                border: 2px solid #ff4444;
                font-size: 14px;
            }
            QPushButton#fire_button:hover {
                background-color: #5a0a0a;
            }
            QPushButton#active_button {
                background-color: #0a3a1a;
                color: #00ff88;
                border: 2px solid #00ff88;
            }
            QTableWidget {
                background-color: #0a1628;
                border: 1px solid #1b3a5c;
                gridline-color: #1b3a5c;
                color: #e0e6ed;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #1b3a5c;
            }
            QHeaderView::section {
                background-color: #0d2137;
                color: #8b9bb4;
                padding: 6px;
                border: 1px solid #1b3a5c;
                font-size: 10px;
                font-weight: bold;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QTextEdit {
                background-color: #0a1628;
                border: 1px solid #1b3a5c;
                color: #8b9bb4;
                font-size: 10px;
                padding: 5px;
            }
        """)

        self.ballistics = BallisticsEngine()
        self.environment = EnvironmentSensor()
        self.detections = []
        self.locked_target = None
        self.system_logs = deque(maxlen=50)
        self.frame_count = 0

        self.init_ui()
        self.init_video()
        self.init_timers()
        self.add_log("System Initialized")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # === TOP BAR ===
        top_bar = QHBoxLayout()

        # Logo
        logo = QLabel("◉ AI SNIPER SYSTEM v4.2.1")
        logo.setStyleSheet("color: #00ff88; font-size: 14px; font-weight: bold; border: none;")
        top_bar.addWidget(logo)

        top_bar.addStretch()

        # Navigation tabs
        tabs = ["DASHBOARD", "TARGETS", "WEAPON", "ENVIRONMENT", "ANALYTICS", "SYSTEM"]
        for tab in tabs:
            btn = QPushButton(tab)
            btn.setFixedWidth(120)
            if tab == "DASHBOARD":
                btn.setStyleSheet("background-color: #1b3a5c; color: #ffffff;")
            top_bar.addWidget(btn)

        top_bar.addStretch()

        # System status
        status = QLabel("● SYSTEM ONLINE")
        status.setStyleSheet("color: #00ff88; font-size: 12px; font-weight: bold; border: none;")
        top_bar.addWidget(status)

        main_layout.addLayout(top_bar)

        # === MAIN CONTENT ===
        content = QHBoxLayout()
        content.setSpacing(5)

        # === LEFT PANEL ===
        left_panel = QVBoxLayout()
        left_panel.setSpacing(5)

        # System Status
        sys_status = QFrame()
        sys_status.setFixedWidth(260)
        sys_layout = QVBoxLayout(sys_status)
        sys_layout.setSpacing(3)

        title = QLabel("SYSTEM STATUS")
        title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        sys_layout.addWidget(title)

        self.status_indicators = {}
        statuses = [
            ("AI MODULE", "ONLINE"), ("TARGET ACQUISITION", "ACTIVE"),
            ("WEAPON SYSTEM", "READY"), ("SENSOR ARRAY", "ONLINE"),
            ("ENVIRONMENT", "STABLE"), ("COMM LINK", "SECURE")
        ]
        for label, status in statuses:
            ind = StatusIndicator(label, status)
            self.status_indicators[label] = ind
            sys_layout.addWidget(ind)

        # AI Processing
        ai_title = QLabel("AI PROCESSING")
        ai_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none; margin-top: 10px;")
        sys_layout.addWidget(ai_title)

        self.neural_status = StatusIndicator("NEURAL NETWORK", "ACTIVE")
        sys_layout.addWidget(self.neural_status)

        conf_layout = QHBoxLayout()
        conf_label = QLabel("CONFIDENCE LEVEL")
        conf_label.setStyleSheet("color: #8b9bb4; font-size: 10px;")
        self.conf_value = QLabel("98.7%")
        self.conf_value.setStyleSheet("color: #00ff88; font-size: 12px; font-weight: bold;")
        conf_layout.addWidget(conf_label)
        conf_layout.addStretch()
        conf_layout.addWidget(self.conf_value)
        sys_layout.addLayout(conf_layout)

        self.conf_bar = QProgressBar()
        self.conf_bar.setMaximum(100)
        self.conf_bar.setValue(98)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(6)
        self.conf_bar.setStyleSheet("""
            QProgressBar {
                background-color: #0d1b2a;
                border: 1px solid #1b3a5c;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #00ff88;
                border-radius: 2px;
            }
        """)
        sys_layout.addWidget(self.conf_bar)

        # Object detection metrics
        metrics = QGridLayout()
        metrics.addWidget(QLabel("OBJECT DETECTION"), 0, 0)
        self.obj_det_val = QLabel("12ms")
        self.obj_det_val.setStyleSheet("color: #00ff88; font-weight: bold;")
        metrics.addWidget(self.obj_det_val, 0, 1)

        metrics.addWidget(QLabel("PREDICTION ACCURACY"), 1, 0)
        self.pred_acc_val = QLabel("96.3%")
        self.pred_acc_val.setStyleSheet("color: #00ff88; font-weight: bold;")
        metrics.addWidget(self.pred_acc_val, 1, 1)
        sys_layout.addLayout(metrics)

        # Weapon Status
        wp_title = QLabel("WEAPON STATUS")
        wp_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none; margin-top: 10px;")
        sys_layout.addWidget(wp_title)

        wp_info = QGridLayout()
        wp_info.addWidget(QLabel("SR-25 Mk1"), 0, 0)
        self.wp_ready = QLabel("READY")
        self.wp_ready.setStyleSheet("color: #00ff88; font-weight: bold;")
        wp_info.addWidget(self.wp_ready, 0, 1)

        wp_info.addWidget(QLabel("AMMUNITION"), 1, 0)
        self.ammo_val = QLabel("7 / 20")
        self.ammo_val.setStyleSheet("color: #ffaa00; font-weight: bold;")
        wp_info.addWidget(self.ammo_val, 1, 1)

        wp_info.addWidget(QLabel("CALIBER"), 2, 0)
        wp_info.addWidget(QLabel("7.62x51mm"), 2, 1)

        wp_info.addWidget(QLabel("EFFECTIVE RANGE"), 3, 0)
        wp_info.addWidget(QLabel("1200m"), 3, 1)

        wp_info.addWidget(QLabel("WINDAGE"), 4, 0)
        self.windage_val = QLabel("-0.42 MRAD")
        self.windage_val.setStyleSheet("color: #ffaa00;")
        wp_info.addWidget(self.windage_val, 4, 1)

        wp_info.addWidget(QLabel("ELEVATION"), 5, 0)
        self.elev_val = QLabel("1.28 MRAD")
        self.elev_val.setStyleSheet("color: #ffaa00;")
        wp_info.addWidget(self.elev_val, 5, 1)

        sys_layout.addLayout(wp_info)
        sys_layout.addStretch()

        left_panel.addWidget(sys_status)

        # Environment Panel
        env_frame = QFrame()
        env_frame.setFixedWidth(260)
        env_layout = QVBoxLayout(env_frame)

        env_title = QLabel("ENVIRONMENT")
        env_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        env_layout.addWidget(env_title)

        self.env_labels = {}
        env_items = [
            ("WIND SPEED", "4.2 m/s", "🌬"),
            ("WIND DIRECTION", "NW 315°", "🧭"),
            ("TEMPERATURE", "18.6 °C", "🌡"),
            ("HUMIDITY", "62 %", "💧"),
            ("PRESSURE", "1013 hPa", "📊"),
            ("VISIBILITY", "8.7 km", "👁")
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

        left_panel.addWidget(env_frame)
        left_panel.addStretch()

        content.addLayout(left_panel)

        # === CENTER PANEL (VIDEO FEED) ===
        center_panel = QVBoxLayout()
        center_panel.setSpacing(5)

        # Video feed frame
        video_frame = QFrame()
        video_frame.setMinimumSize(900, 520)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(2, 2, 2, 2)

        # Header
        video_header = QHBoxLayout()
        self.feed_title = QLabel("LIVE FEED - OPTICAL ZOOM 12.0x")
        self.feed_title.setStyleSheet("color: #00aaff; font-size: 12px; font-weight: bold; border: none;")
        video_header.addWidget(self.feed_title)
        video_header.addStretch()

        # Overlay info
        self.overlay_info = QLabel("DATE: 2024-05-24    TIME: 14:32:17    MODE: DAY    ZOOM: 12.0x")
        self.overlay_info.setStyleSheet("color: #8b9bb4; font-size: 10px; border: none;")
        self.overlay_info.setAlignment(Qt.AlignmentFlag.AlignRight)
        video_header.addWidget(self.overlay_info)

        video_layout.addLayout(video_header)

        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(880, 495)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #1b3a5c;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)

        center_panel.addWidget(video_frame)

        # Bottom info panels
        bottom_info = QHBoxLayout()
        bottom_info.setSpacing(5)

        # Trajectory
        traj_frame = QFrame()
        traj_frame.setFixedHeight(200)
        traj_layout = QVBoxLayout(traj_frame)
        traj_title = QLabel("TRAJECTORY PREDICTION")
        traj_title.setStyleSheet("color: #00aaff; font-size: 10px; font-weight: bold; border: none;")
        traj_layout.addWidget(traj_title)
        self.trajectory_graph = TrajectoryGraph()
        traj_layout.addWidget(self.trajectory_graph)
        bottom_info.addWidget(traj_frame)

        # Wind Analysis
        wind_frame = QFrame()
        wind_frame.setFixedHeight(200)
        wind_layout = QVBoxLayout(wind_frame)
        wind_title = QLabel("WIND ANALYSIS")
        wind_title.setStyleSheet("color: #00aaff; font-size: 10px; font-weight: bold; border: none;")
        wind_layout.addWidget(wind_title)
        self.wind_compass = WindCompass()
        wind_layout.addWidget(self.wind_compass, alignment=Qt.AlignmentFlag.AlignCenter)
        bottom_info.addWidget(wind_frame)

        # Target Vitals
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
        self.scan_btn.clicked.connect(self.scan_targets)

        self.track_btn = QPushButton("◎ TRACK")
        self.track_btn.setFixedHeight(45)
        self.track_btn.clicked.connect(self.track_target)

        self.lock_btn = QPushButton("⊕ LOCK")
        self.lock_btn.setFixedHeight(45)
        self.lock_btn.setObjectName("active_button")
        self.lock_btn.clicked.connect(self.lock_target)

        self.solve_btn = QPushButton("⚙ SOLVE")
        self.solve_btn.setFixedHeight(45)
        self.solve_btn.clicked.connect(self.solve_ballistics)

        self.fire_btn = QPushButton("◎ FIRE")
        self.fire_btn.setFixedHeight(45)
        self.fire_btn.setObjectName("fire_button")
        self.fire_btn.clicked.connect(self.fire_weapon)

        btn_row.addWidget(self.scan_btn)
        btn_row.addWidget(self.track_btn)
        btn_row.addWidget(self.lock_btn)
        btn_row.addWidget(self.solve_btn)
        btn_row.addWidget(self.fire_btn)

        center_panel.addLayout(btn_row)

        content.addLayout(center_panel)

        # === RIGHT PANEL ===
        right_panel = QVBoxLayout()
        right_panel.setSpacing(5)

        # Targets Detected
        targets_frame = QFrame()
        targets_frame.setFixedWidth(300)
        targets_layout = QVBoxLayout(targets_frame)

        targets_header = QHBoxLayout()
        targets_title = QLabel("TARGETS DETECTED")
        targets_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        targets_header.addWidget(targets_title)
        self.total_targets = QLabel("TOTAL: 4")
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
        targets_layout.addWidget(self.targets_table)

        right_panel.addWidget(targets_frame)

        # Ballistic Solution
        ball_frame = QFrame()
        ball_frame.setFixedWidth(300)
        ball_layout = QVBoxLayout(ball_frame)

        ball_title = QLabel("BALLISTIC SOLUTION")
        ball_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        ball_layout.addWidget(ball_title)

        self.ballistic_labels = {}
        ball_items = [
            ("DISTANCE", "612.4 m"),
            ("TIME OF FLIGHT", "0.842 s"),
            ("DROP", "1.28 MRAD"),
            ("WIND DRIFT", "-0.42 MRAD"),
            ("LEAD", "0.18 MRAD"),
            ("IMPACT VELOCITY", "732 m/s"),
            ("SOLUTION CONFIDENCE", "98.7%")
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

        self.fire_solution = QLabel("● FIRE SOLUTION READY")
        self.fire_solution.setStyleSheet("color: #00ff88; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")
        self.fire_solution.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ball_layout.addWidget(self.fire_solution)

        right_panel.addWidget(ball_frame)

        # System Log
        log_frame = QFrame()
        log_frame.setFixedWidth(300)
        log_layout = QVBoxLayout(log_frame)

        log_title = QLabel("SYSTEM LOG")
        log_title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold; border: none;")
        log_layout.addWidget(log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(200)
        log_layout.addWidget(self.log_text)

        right_panel.addWidget(log_frame)
        right_panel.addStretch()

        content.addLayout(right_panel)

        main_layout.addLayout(content)

    def init_video(self):
        self.video_thread = VideoCaptureThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.detections_ready.connect(self.update_detections)
        self.video_thread.start()

    def init_timers(self):
        # Environment update timer
        self.env_timer = QTimer()
        self.env_timer.timeout.connect(self.update_environment)
        self.env_timer.start(2000)

        # Clock timer
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        # AI metrics update
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.update_ai_metrics)
        self.ai_timer.start(3000)

    def update_frame(self, frame):
        self.current_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw HUD overlay
        overlay = frame.copy()

        # Crosshair
        cx, cy = w // 2, h // 2
        cv2.line(overlay, (cx - 30, cy), (cx - 10, cy), (0, 255, 136), 1)
        cv2.line(overlay, (cx + 10, cy), (cx + 30, cy), (0, 255, 136), 1)
        cv2.line(overlay, (cx, cy - 30), (cx, cy - 10), (0, 255, 136), 1)
        cv2.line(overlay, (cx, cy + 10), (cx, cy + 30), (0, 255, 136), 1)
        cv2.circle(overlay, (cx, cy), 5, (0, 255, 136), 1)

        # Compass at top
        cv2.line(overlay, (w//2 - 100, 40), (w//2 + 100, 40), (50, 80, 110), 1)
        for i, deg in enumerate(['W', '', '300', '', '330', '', 'N', '', '30', '', '60', '', 'E']):
            x = w//2 - 100 + i * 16
            cv2.line(overlay, (x, 35), (x, 45), (100, 120, 140), 1)
            if deg:
                cv2.putText(overlay, deg, (x-8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 120, 140), 1)

        # Elevation scale on left
        for i in range(-20, 21, 10):
            y = h//2 - i * 8
            cv2.line(overlay, (30, y), (40, y), (100, 120, 140), 1)
            cv2.putText(overlay, str(i), (10, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 120, 140), 1)

        # Windage scale on bottom
        for i in range(-20, 21, 10):
            x = w//2 + i * 8
            cv2.line(overlay, (x, h-40), (x, h-30), (100, 120, 140), 1)
            cv2.putText(overlay, str(i), (x-5, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 120, 140), 1)

        # Draw detections
        for det in self.detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            tid = det['id']

            # Color based on lock status
            if self.locked_target and self.locked_target['id'] == tid:
                color = (0, 255, 0)  # Green for locked
                thickness = 2
                # Draw target info box
                info_x, info_y = x2 + 10, y1
                cv2.rectangle(overlay, (info_x, info_y), (info_x + 140, info_y + 80), (20, 40, 60), -1)
                cv2.rectangle(overlay, (info_x, info_y), (info_x + 140, info_y + 80), (0, 255, 136), 1)
                cv2.putText(overlay, f"TARGET ID: {tid}", (info_x + 5, info_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
                cv2.putText(overlay, f"DISTANCE: {det.get('distance', 612.4):.1f} m", (info_x + 5, info_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
                cv2.putText(overlay, f"CONFIDENCE: {conf*100:.1f}%", (info_x + 5, info_y + 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
                cv2.putText(overlay, "PRIORITY: HIGH", (info_x + 5, info_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 170, 0), 1)
            else:
                color = (255, 170, 0)  # Orange for detected
                thickness = 1

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # Scanlines effect
        overlay[::3, :] = (overlay[::3, :] * 0.7).astype(np.uint8)

        # Convert to QImage
        rgb_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit
        scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

        self.frame_count += 1

    def update_detections(self, detections):
        self.detections = detections

        # Assign distances
        for det in detections:
            det['distance'] = random.uniform(400, 1100)
            det['status'] = 'DETECTED'

        if self.locked_target:
            # Keep locked target
            found = False
            for det in detections:
                if det['id'] == self.locked_target['id']:
                    det['status'] = 'LOCKED'
                    self.locked_target = det
                    found = True
                    break
            if not found:
                self.locked_target = None
                self.add_log("Target lock lost")
                self.status_indicators["TARGET ACQUISITION"].update_status_color("ACTIVE")

        self.update_targets_table()
        self.total_targets.setText(f"TOTAL: {len(detections)}")

    def update_targets_table(self):
        self.targets_table.setRowCount(len(self.detections))

        for i, det in enumerate(self.detections):
            # Thumbnail
            thumb_label = QLabel()
            thumb = np.zeros((60, 50, 3), dtype=np.uint8)
            thumb[:, :] = [30, 35, 40]
            # Draw simple figure
            cv2.circle(thumb, (25, 15), 8, (80, 90, 100), -1)
            cv2.rectangle(thumb, (15, 25), (35, 55), (60, 70, 80), -1)
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            thumb_img = QImage(thumb_rgb.data, 50, 60, 150, QImage.Format.Format_RGB888)
            thumb_label.setPixmap(QPixmap.fromImage(thumb_img))
            self.targets_table.setCellWidget(i, 0, thumb_label)

            # Info
            status_color = "#00ff88" if det.get('status') == 'LOCKED' else "#ffaa00"
            info_text = f"""TARGET ID: {det['id']}
DISTANCE: {det['distance']:.1f} m
CONFIDENCE: {det['confidence']*100:.1f}%
STATUS: <span style='color: {status_color}; font-weight: bold;'>{det.get('status', 'DETECTED')}</span>"""
            info_label = QLabel(info_text)
            info_label.setStyleSheet("color: #8b9bb4; font-size: 10px;")
            self.targets_table.setCellWidget(i, 1, info_label)
            self.targets_table.setRowHeight(i, 65)

    def update_environment(self):
        self.environment.update()
        data = self.environment.get_data()

        self.env_labels["WIND SPEED"].setText(f"{data['wind_speed']:.1f} m/s")
        self.env_labels["WIND DIRECTION"].setText(f"NW {data['wind_direction']}°")
        self.env_labels["TEMPERATURE"].setText(f"{data['temperature']:.1f} °C")
        self.env_labels["HUMIDITY"].setText(f"{data['humidity']} %")
        self.env_labels["PRESSURE"].setText(f"{data['pressure']} hPa")
        self.env_labels["VISIBILITY"].setText(f"{data['visibility']:.1f} km")

        self.wind_compass.set_data(data['wind_speed'], data['wind_direction'])

    def update_clock(self):
        now = datetime.now()
        self.overlay_info.setText(
            f"DATE: {now.strftime('%Y-%m-%d')}    "
            f"TIME: {now.strftime('%H:%M:%S')}    "
            f"MODE: DAY    ZOOM: 12.0x"
        )

    def update_ai_metrics(self):
        conf = random.uniform(96, 99.5)
        self.conf_value.setText(f"{conf:.1f}%")
        self.conf_bar.setValue(int(conf))
        self.obj_det_val.setText(f"{random.randint(8, 15)}ms")
        self.pred_acc_val.setText(f"{random.uniform(94, 98):.1f}%")

    def scan_targets(self):
        self.add_log("Scanning for targets...")
        self.status_indicators["TARGET ACQUISITION"].update_status_color("ACTIVE")
        QTimer.singleShot(1500, lambda: self.add_log("Scan complete. Targets detected."))

    def track_target(self):
        if self.detections:
            target = self.detections[0]
            self.add_log(f"Tracking target {target['id']}...")
            self.status_indicators["TARGET ACQUISITION"].update_status_color("ACTIVE")
        else:
            self.add_log("No targets to track")

    def lock_target(self):
        if self.detections:
            self.locked_target = self.detections[0]
            self.locked_target['status'] = 'LOCKED'
            self.add_log(f"Target {self.locked_target['id']} LOCKED")
            self.status_indicators["TARGET ACQUISITION"].update_status_color("LOCKED")
            self.update_targets_table()

            # Update ballistics for locked target
            self.solve_ballistics()
        else:
            self.add_log("No targets available for lock")

    def solve_ballistics(self):
        if self.locked_target:
            dist = self.locked_target['distance']
            env = self.environment.get_data()

            solution = self.ballistics.calculate_solution(
                dist, env['wind_speed'], env['wind_direction'],
                env['temperature'], env['humidity'], env['pressure']
            )

            self.ballistic_labels["DISTANCE"].setText(f"{solution['distance']:.1f} m")
            self.ballistic_labels["TIME OF FLIGHT"].setText(f"{solution['time_of_flight']:.3f} s")
            self.ballistic_labels["DROP"].setText(f"{solution['drop']:.2f} MRAD")
            self.ballistic_labels["WIND DRIFT"].setText(f"{solution['windage']:.2f} MRAD")
            self.ballistic_labels["LEAD"].setText(f"{solution['lead']:.2f} MRAD")
            self.ballistic_labels["IMPACT VELOCITY"].setText(f"{solution['impact_velocity']:.1f} m/s")
            self.ballistic_labels["SOLUTION CONFIDENCE"].setText(f"{solution['confidence']:.1f}%")

            self.windage_val.setText(f"{solution['windage']:.2f} MRAD")
            self.elev_val.setText(f"{solution['drop']:.2f} MRAD")

            self.trajectory_graph.set_data(solution['distance'], solution['drop'])

            self.add_log("Ballistic solution calculated")
            self.fire_solution.setText("● FIRE SOLUTION READY")
            self.fire_solution.setStyleSheet("color: #00ff88; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")
        else:
            self.add_log("No locked target for ballistics solution")

    def fire_weapon(self):
        if self.locked_target and self.fire_solution.text().find("READY") != -1:
            self.add_log("FIRING...")
            self.fire_solution.setText("○ FIRING")
            self.fire_solution.setStyleSheet("color: #ff4444; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")

            # Simulate recoil effect
            self.video_label.setStyleSheet("background-color: #ff4444; border: 1px solid #1b3a5c;")
            QTimer.singleShot(100, lambda: self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #1b3a5c;"))

            QTimer.singleShot(500, self.fire_complete)
        else:
            self.add_log("FIRE SOLUTION NOT READY")

    def fire_complete(self):
        self.add_log("Shot fired. Impact confirmed.")
        self.fire_solution.setText("● FIRE SOLUTION READY")
        self.fire_solution.setStyleSheet("color: #00ff88; font-size: 12px; font-weight: bold; border: none; margin-top: 10px;")

        # Update ammo
        current = self.ammo_val.text().split('/')
        if int(current[0].strip()) > 0:
            new_ammo = int(current[0].strip()) - 1
            self.ammo_val.setText(f"{new_ammo} / {current[1].strip()}")

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.system_logs.append(f"[{timestamp}] {message}")
        self.log_text.setText("\n".join(self.system_logs))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application font
    font = QFont("Consolas", 10)
    app.setFont(font)

    window = AISniperSystem()
    window.show()
    sys.exit(app.exec())
