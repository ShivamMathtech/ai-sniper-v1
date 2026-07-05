#!/usr/bin/env python3
"""
AI SNIPER SYSTEM v4.2.1
=======================
Tactical targeting dashboard with PyQt6, OpenCV, and YOLO.
Uses laptop webcam (index 0) for real-time object tracking.

Requirements:
    pip install PyQt6 opencv-python ultralytics numpy

Usage:
    python ai_sniper_system.py
"""

import sys
import os
import random
import math
import time
import threading
from datetime import datetime
from collections import deque

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QProgressBar,
    QScrollArea, QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect,
    QMessageBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QPoint, QRect, QRectF
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QFontDatabase,
    QLinearGradient, QBrush, QPolygon, QIcon, QPalette
)

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Using mock detection.")
    print("Install with: pip install ultralytics")


# =============================================================================
# CONSTANTS & THEME
# =============================================================================

DARK_BG = "#0a0e17"
PANEL_BG = "#0d1117"
BORDER_COLOR = "#1a2332"
ACCENT_BLUE = "#00d4ff"
ACCENT_GREEN = "#00ff88"
ACCENT_RED = "#ff3366"
ACCENT_YELLOW = "#ffcc00"
ACCENT_ORANGE = "#ff6b35"
TEXT_PRIMARY = "#e0e6ed"
TEXT_SECONDARY = "#8b949e"
TEXT_DIM = "#4a5568"
GRID_LINE = "#1a2332"

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"


def randomize_value(base, variance=0.05):
    return base * (1 + random.uniform(-variance, variance))


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class Panel(QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            Panel {{
                background-color: {PANEL_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
            }}
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(4)
        if title:
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet(f"""
                color: {ACCENT_BLUE};
                font-family: {FONT_FAMILY};
                font-size: 10px;
                font-weight: bold;
                letter-spacing: 1px;
                padding-bottom: 4px;
                border-bottom: 1px solid {BORDER_COLOR};
            """)
            self.layout.addWidget(title_lbl)


class StatusIndicator(QFrame):
    def __init__(self, label, status="OFFLINE", parent=None):
        super().__init__(parent)
        self.label_text = label
        self.status_text = status
        self.setFixedHeight(22)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(4, 0, 4, 0)
        self.layout.setSpacing(6)

        self.dot = QLabel("●")
        self.dot.setFixedWidth(12)
        self.dot.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.name_lbl = QLabel(label)
        self.name_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-family: {FONT_FAMILY};")

        self.status_lbl = QLabel(status)
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.status_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-family: {FONT_FAMILY}; font-weight: bold;")

        self.layout.addWidget(self.dot)
        self.layout.addWidget(self.name_lbl, stretch=1)
        self.layout.addWidget(self.status_lbl)

        self.set_status(status)

    def set_status(self, status):
        self.status_text = status
        self.status_lbl.setText(status)
        colors = {
            "ONLINE": ACCENT_GREEN, "ACTIVE": ACCENT_GREEN, "READY": ACCENT_GREEN,
            "STABLE": ACCENT_GREEN, "SECURE": ACCENT_GREEN, "LOCKED": ACCENT_GREEN,
            "DETECTED": ACCENT_YELLOW, "TRACKING": ACCENT_BLUE,
            "OFFLINE": ACCENT_RED, "ERROR": ACCENT_RED, "LETHAL": ACCENT_RED
        }
        color = colors.get(status, TEXT_SECONDARY)
        self.dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.status_lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-family: {FONT_FAMILY}; font-weight: bold;")


class MetricRow(QFrame):
    def __init__(self, label, value, unit="", parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(2, 1, 2, 1)
        self.layout.setSpacing(4)

        self.label_lbl = QLabel(label)
        self.label_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-family: {FONT_FAMILY};")

        self.value_lbl = QLabel(f"{value} {unit}")
        self.value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 10px; font-family: {FONT_MONO}; font-weight: bold;")

        self.layout.addWidget(self.label_lbl, stretch=1)
        self.layout.addWidget(self.value_lbl)

    def set_value(self, value, unit=""):
        self.value_lbl.setText(f"{value} {unit}")


class ProgressBarStyled(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(True)
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                background: {DARK_BG};
                color: {TEXT_PRIMARY};
                font-size: 9px;
                font-family: {FONT_MONO};
                text-align: center;
                height: 14px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_BLUE}, stop:1 {ACCENT_GREEN});
                border-radius: 2px;
            }}
        """)


class ActionButton(QPushButton):
    def __init__(self, text, color, icon_text, parent=None):
        super().__init__(parent)
        self.setText(f"  {icon_text}  {text}")
        self.setFixedHeight(42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.base_style = f"""
            QPushButton {{
                background-color: {PANEL_BG};
                color: {color};
                border: 1px solid {color};
                border-radius: 4px;
                font-family: {FONT_FAMILY};
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                background-color: {color};
                color: {DARK_BG};
            }}
            QPushButton:pressed {{
                background-color: {color};
                color: {DARK_BG};
                padding-top: 2px;
            }}
        """
        self.active_style = f"""
            QPushButton {{
                background-color: {color};
                color: {DARK_BG};
                border: 1px solid {color};
                border-radius: 4px;
                font-family: {FONT_FAMILY};
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                background-color: {color};
                color: {DARK_BG};
            }}
        """
        self.setStyleSheet(self.base_style)
        self.active = False
        self.color = color

    def set_active(self, active):
        self.active = active
        self.setStyleSheet(self.active_style if active else self.base_style)


class TrajectoryWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER_COLOR};")
        self.distance = 612.4
        self.drop = 1.28

    def set_data(self, distance, drop):
        self.distance = distance
        self.drop = drop
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 30
        graph_w = w - margin * 2
        graph_h = h - margin * 2

        # Draw grid
        pen = QPen(QColor(GRID_LINE))
        pen.setWidth(1)
        painter.setPen(pen)
        for i in range(5):
            x = margin + (graph_w / 4) * i
            painter.drawLine(int(x), margin, int(x), h - margin)
        for i in range(5):
            y = margin + (graph_h / 4) * i
            painter.drawLine(margin, int(y), w - margin, int(y))

        # Draw axes
        pen = QPen(QColor(TEXT_SECONDARY))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(margin, h - margin, w - margin, h - margin)
        painter.drawLine(margin, margin, margin, h - margin)

        # Labels
        painter.setFont(QFont(FONT_MONO, 8))
        for i, val in enumerate([0, 200, 400, 600, 800, 1000, 1200]):
            x = margin + (graph_w / 6) * i
            painter.drawText(int(x - 15), h - margin + 15, f"{val}")

        painter.drawText(5, margin - 5, "2.0")
        painter.drawText(5, margin + graph_h // 2 + 5, "0")
        painter.drawText(5, h - margin + 5, "-2.0")

        # Draw trajectory curve
        pen = QPen(QColor(ACCENT_BLUE))
        pen.setWidth(2)
        painter.setPen(pen)

        points = []
        for i in range(100):
            t = i / 100.0
            x = margin + t * graph_w * (self.distance / 1200)
            y_drop = -self.drop * (t ** 2) * (graph_h / 4)
            y = (h / 2) + y_drop
            points.append(QPoint(int(x), int(y)))

        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # Draw target point
        tx = margin + graph_w * (self.distance / 1200)
        ty = (h / 2) + (-self.drop * (graph_h / 4))
        pen = QPen(QColor(ACCENT_GREEN))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(QPoint(int(tx), int(ty)), 4, 4)

        # Distance label
        painter.setPen(QColor(ACCENT_GREEN))
        painter.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        painter.drawText(int(tx - 20), int(ty - 10), f"{self.distance:.1f}m")

        # Axis titles
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont(FONT_FAMILY, 8))
        painter.drawText(w // 2 - 30, h - 8, "DISTANCE (m)")

        painter.end()


class WindWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER_COLOR};")
        self.speed = 4.2
        self.direction = 315

    def set_data(self, speed, direction):
        self.speed = speed
        self.direction = direction
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() // 2, self.height() // 2
        radius = min(cx, cy) - 20

        # Draw compass circle
        pen = QPen(QColor(GRID_LINE))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#111820")))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Cardinal directions
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont(FONT_FAMILY, 8, QFont.Weight.Bold))
        painter.drawText(cx - 4, cy - radius - 2, "N")
        painter.drawText(cx - 4, cy + radius + 12, "S")
        painter.drawText(cx + radius + 4, cy + 4, "E")
        painter.drawText(cx - radius - 12, cy + 4, "W")

        # Inner ticks
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x1 = cx + (radius - 8) * math.sin(rad)
            y1 = cy - (radius - 8) * math.cos(rad)
            x2 = cx + radius * math.sin(rad)
            y2 = cy - radius * math.cos(rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Wind arrow
        rad = math.radians(self.direction)
        dx = math.sin(rad) * (radius - 15)
        dy = -math.cos(rad) * (radius - 15)

        pen = QPen(QColor(ACCENT_BLUE))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(cx, cy, int(cx + dx), int(cy + dy))

        # Arrow head
        arrow_size = 8
        angle = math.atan2(dy, dx)
        ax = cx + dx - arrow_size * math.cos(angle - math.pi / 6)
        ay = cy + dy - arrow_size * math.sin(angle - math.pi / 6)
        bx = cx + dx - arrow_size * math.cos(angle + math.pi / 6)
        by = cy + dy - arrow_size * math.sin(angle + math.pi / 6)

        head = QPolygon([QPoint(int(cx + dx), int(cy + dy)),
                         QPoint(int(ax), int(ay)),
                         QPoint(int(bx), int(by))])
        painter.setBrush(QBrush(QColor(ACCENT_BLUE)))
        painter.drawPolygon(head)

        # Center info
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.setFont(QFont(FONT_MONO, 10, QFont.Weight.Bold))
        painter.drawText(cx - 25, cy - 10, 50, 20, Qt.AlignmentFlag.AlignCenter, f"{self.direction}°")
        painter.setFont(QFont(FONT_MONO, 9))
        painter.drawText(cx - 30, cy + 5, 60, 20, Qt.AlignmentFlag.AlignCenter, f"{self.speed:.2f} m/s")

        painter.setFont(QFont(FONT_FAMILY, 8))
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.drawText(cx - 10, cy + 25, "NW")

        painter.end()


class TargetVitalsWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER_COLOR};")
        self.probability = 98.7

    def set_probability(self, prob):
        self.probability = prob
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx = w // 2 - 30
        cy = h // 2

        # Draw simplified human silhouette
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setBrush(QBrush(QColor("#1a2332")))

        # Head
        painter.drawEllipse(cx - 12, cy - 45, 24, 24)
        # Body
        painter.drawRect(cx - 15, cy - 20, 30, 50)
        # Arms
        painter.drawLine(cx - 15, cy - 10, cx - 30, cy + 10)
        painter.drawLine(cx + 15, cy - 10, cx + 30, cy + 10)
        # Legs
        painter.drawLine(cx - 8, cy + 30, cx - 12, cy + 60)
        painter.drawLine(cx + 8, cy + 30, cx + 12, cy + 60)

        # Heart/lungs highlight area
        if self.probability > 90:
            color = QColor(ACCENT_RED)
            color.setAlpha(120)
        else:
            color = QColor(ACCENT_YELLOW)
            color.setAlpha(80)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(cx - 10, cy - 15, 20, 25)

        # Info text
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont(FONT_FAMILY, 9))
        painter.drawText(cx + 40, cy - 20, "VITAL AREA")
        painter.drawText(cx + 40, cy - 5, "HEART / LUNGS")

        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont(FONT_FAMILY, 8))
        painter.drawText(cx + 40, cy + 15, "PROBABILITY")

        painter.setPen(QColor(ACCENT_GREEN if self.probability > 90 else ACCENT_YELLOW))
        painter.setFont(QFont(FONT_MONO, 14, QFont.Weight.Bold))
        painter.drawText(cx + 40, cy + 35, f"{self.probability:.1f}%")

        painter.setPen(QColor(ACCENT_RED))
        painter.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        painter.drawText(cx + 40, cy + 55, "LETHAL")

        painter.end()


class TargetCard(QFrame):
    def __init__(self, target_id, distance, confidence, status, parent=None):
        super().__init__(parent)
        self.target_id = target_id
        self.distance = distance
        self.confidence = confidence
        self.status = status
        self.setFixedHeight(70)
        self.setStyleSheet(f"""
            TargetCard {{
                background-color: {PANEL_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # Thumbnail placeholder
        self.thumb = QLabel()
        self.thumb.setFixedSize(40, 50)
        self.thumb.setStyleSheet(f"""
            background-color: {DARK_BG};
            border: 1px solid {BORDER_COLOR if status != 'LOCKED' else ACCENT_GREEN};
            color: {TEXT_PRIMARY};
            font-size: 16px;
        """)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setText("🎯")
        layout.addWidget(self.thumb)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)

        id_lbl = QLabel(f"TARGET ID: {target_id:02d}")
        id_lbl.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 9px; font-family: {FONT_FAMILY}; font-weight: bold;")
        info.addWidget(id_lbl)

        dist_lbl = QLabel(f"DISTANCE:  {distance:.1f} m")
        dist_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9px; font-family: {FONT_MONO};")
        info.addWidget(dist_lbl)

        conf_lbl = QLabel(f"CONFIDENCE: {confidence:.1f}%")
        conf_lbl.setStyleSheet(f"color: {ACCENT_GREEN if confidence > 90 else ACCENT_YELLOW}; font-size: 9px; font-family: {FONT_MONO};")
        info.addWidget(conf_lbl)

        status_lbl = QLabel(f"STATUS:  {status}")
        status_lbl.setStyleSheet(f"color: {ACCENT_GREEN if status == 'LOCKED' else ACCENT_YELLOW}; font-size: 9px; font-family: {FONT_FAMILY}; font-weight: bold;")
        info.addWidget(status_lbl)

        layout.addLayout(info, stretch=1)

    def set_locked(self, locked):
        if locked:
            self.setStyleSheet(f"""
                TargetCard {{
                    background-color: {PANEL_BG};
                    border: 1px solid {ACCENT_GREEN};
                    border-radius: 3px;
                }}
            """)
            self.thumb.setStyleSheet(f"""
                background-color: {DARK_BG};
                border: 1px solid {ACCENT_GREEN};
                color: {ACCENT_GREEN};
                font-size: 16px;
            """)
        else:
            self.setStyleSheet(f"""
                TargetCard {{
                    background-color: {PANEL_BG};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 3px;
                }}
            """)
            self.thumb.setStyleSheet(f"""
                background-color: {DARK_BG};
                border: 1px solid {BORDER_COLOR};
                color: {TEXT_PRIMARY};
                font-size: 16px;
            """)


class LiveFeedWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: #000; border: 1px solid {BORDER_COLOR};")
        self.setMinimumSize(640, 480)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        self.current_frame = None
        self.detection_boxes = []
        self.locked_target = None
        self.zoom_level = 12.0
        self.mode = "DAY"
        self.date_str = "2024-05-24"
        self.time_str = "14:32:17"

        self.target_info = {
            "id": "01",
            "distance": 612.4,
            "confidence": 98.7,
            "priority": "HIGH"
        }

    def update_frame(self, frame):
        self.current_frame = frame.copy()
        self.draw_overlays()

    def set_detections(self, boxes):
        self.detection_boxes = boxes

    def set_locked_target(self, target_id):
        self.locked_target = target_id

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.draw_overlays()

    def draw_overlays(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        h, w = frame.shape[:2]

        # Night vision effect
        if self.mode == "NIGHT":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame[:, :, 0] = (frame[:, :, 0] * 0.3).astype(np.uint8)
            frame[:, :, 2] = (frame[:, :, 2] * 0.3).astype(np.uint8)

        # Draw crosshair
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 30, cy), (cx - 10, cy), (0, 212, 255), 1)
        cv2.line(frame, (cx + 10, cy), (cx + 30, cy), (0, 212, 255), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy - 10), (0, 212, 255), 1)
        cv2.line(frame, (cx, cy + 10), (cx, cy + 30), (0, 212, 255), 1)
        cv2.circle(frame, (cx, cy), 5, (0, 212, 255), 1)

        # Draw detection boxes
        for box in self.detection_boxes:
            x1, y1, x2, y2, conf, cls, tid = box
            color = (0, 255, 136) if tid == self.locked_target else (0, 212, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"TARGET ID: {tid:02d}"
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"DISTANCE: {randomize_value(600 + tid * 100, 0.02):.1f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Compass at top
        compass_y = 30
        cv2.line(frame, (w // 2 - 100, compass_y), (w // 2 + 100, compass_y), (100, 100, 100), 1)
        for i, deg in enumerate([300, 330, 0, 30, 60]):
            x = w // 2 - 80 + i * 40
            cv2.line(frame, (x, compass_y - 3), (x, compass_y + 3), (200, 200, 200), 1)
            cv2.putText(frame, str(deg), (x - 8, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(frame, "W", (w // 2 - 100, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "N", (w // 2 - 4, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 212, 255), 1)
        cv2.putText(frame, "E", (w // 2 + 90, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Elevation/windage scales
        for i in range(-20, 21, 10):
            y = cy + i * 8
            cv2.line(frame, (20, y), (30, y), (100, 100, 100), 1)
            cv2.putText(frame, str(i), (5, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        for i in range(-20, 21, 10):
            x = cx + i * 8
            cv2.line(frame, (x, h - 40), (x, h - 30), (100, 100, 100), 1)
            cv2.putText(frame, str(i), (x - 5, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # Info panel top-right
        info_x = w - 180
        cv2.rectangle(frame, (info_x, 10), (w - 10, 90), (10, 15, 20), -1)
        cv2.rectangle(frame, (info_x, 10), (w - 10, 90), (50, 50, 50), 1)
        cv2.putText(frame, f"DATE: {self.date_str}", (info_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"TIME: {self.time_str}", (info_x + 5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"MODE: {self.mode}", (info_x + 5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
        cv2.putText(frame, f"ZOOM: {self.zoom_level:.1f}x", (info_x + 5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Target info overlay
        if self.locked_target is not None:
            cv2.rectangle(frame, (w - 200, h - 120), (w - 10, h - 10), (10, 15, 20), -1)
            cv2.rectangle(frame, (w - 200, h - 120), (w - 10, h - 10), (0, 255, 136), 1)
            cv2.putText(frame, f"TARGET ID: {self.locked_target:02d}", (w - 190, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
            cv2.putText(frame, f"DISTANCE: {self.target_info['distance']:.1f} m", (w - 190, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
            cv2.putText(frame, f"CONFIDENCE: {self.target_info['confidence']:.1f}%", (w - 190, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
            cv2.putText(frame, f"PRIORITY: {self.target_info['priority']}", (w - 190, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 51, 102), 1)

        # Convert to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit widget while maintaining aspect ratio
        scaled = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled)


# =============================================================================
# YOLO DETECTION THREAD
# =============================================================================

class YOLODetector(QThread):
    detections_ready = pyqtSignal(list)

    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.running = True
        self.current_frame = None
        self.frame_ready = False

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8n.pt")
                print("[INFO] YOLOv8 model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load YOLO: {e}")
                self.model = None

    def set_frame(self, frame):
        self.current_frame = frame
        self.frame_ready = True

    def run(self):
        while self.running:
            if self.frame_ready and self.current_frame is not None:
                self.frame_ready = False
                frame = self.current_frame.copy()

                if self.model is not None:
                    try:
                        results = self.model(frame, verbose=False)
                        boxes = []
                        for r in results:
                            for i, box in enumerate(r.boxes):
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                if cls == 0:  # person class
                                    boxes.append((x1, y1, x2, y2, conf, cls, i + 1))
                        self.detections_ready.emit(boxes)
                    except Exception as e:
                        print(f"[YOLO Error] {e}")
                        self.detections_ready.emit([])
                else:
                    # Mock detections if YOLO not available
                    h, w = frame.shape[:2]
                    boxes = [
                        (w//2 - 40, h//2 - 60, w//2 + 40, h//2 + 60, 0.987, 0, 1),
                        (w//2 + 80, h//2 - 50, w//2 + 140, h//2 + 50, 0.873, 0, 2),
                        (w//2 - 140, h//2 - 55, w//2 - 80, h//2 + 55, 0.768, 0, 3),
                        (w//2 + 150, h//2 - 45, w//2 + 200, h//2 + 45, 0.654, 0, 4),
                    ]
                    self.detections_ready.emit(boxes)

            self.msleep(50)

    def stop(self):
        self.running = False
        self.wait()


# =============================================================================
# VIDEO CAPTURE THREAD (MULTI-CAMERA FALLBACK)
# =============================================================================

class VideoCapture(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)

    def __init__(self, source=None):
        super().__init__()
        # Auto-detect camera: try 0 (laptop), then 1, then fallback to synthetic
        self.source = source
        self.running = True
        self.cap = None
        self.camera_found = False
        self.camera_index = 0

    def find_camera(self):
        """Try multiple camera indices to find the laptop webcam."""
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows stability
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    cap.release()
                    print(f"[INFO] Camera found at index {idx}")
                    return idx
                cap.release()
        return None

    def run(self):
        # Determine camera source
        if self.source is not None:
            camera_idx = self.source
        else:
            camera_idx = self.find_camera()

        if camera_idx is not None:
            self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                # Set resolution for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.camera_found = True
                self.camera_index = camera_idx
                print(f"[INFO] Using camera index {camera_idx} at {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        if not self.camera_found:
            print("[WARNING] No camera found. Using synthetic feed.")
            self.camera_error.emit("No camera detected. Using synthetic feed.")

        frame_count = 0
        while self.running:
            if self.camera_found and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame_ready.emit(frame)
                else:
                    # Camera disconnected
                    self.camera_found = False
                    if self.cap:
                        self.cap.release()
            else:
                # Synthetic feed when no camera
                synthetic = self.generate_synthetic_frame(frame_count)
                self.frame_ready.emit(synthetic)
                frame_count += 1

            self.msleep(33)  # ~30 FPS

        if self.cap:
            self.cap.release()

    def generate_synthetic_frame(self, frame_count):
        """Generate a synthetic tactical scene when no camera is available."""
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark background with noise
        frame[:, :] = (15, 20, 25)
        noise = np.random.normal(0, 5, (h, w, 3)).astype(np.uint8)
        frame = cv2.add(frame, noise)

        # Draw building-like structures
        for i in range(5):
            bx = 100 + i * 220
            by = 200 + (i % 3) * 50
            bw = 180
            bh = 300 + (i % 2) * 100
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (30, 35, 40), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 55, 60), 1)
            # Windows
            for wy in range(by + 20, by + bh - 30, 40):
                for wx in range(bx + 15, bx + bw - 15, 35):
                    cv2.rectangle(frame, (wx, wy), (wx + 20, wy + 25), (10, 15, 20), -1)

        # Ground
        cv2.rectangle(frame, (0, 550), (w, h), (20, 25, 30), -1)

        # Add moving "targets" (silhouettes)
        for i in range(4):
            tx = 200 + i * 250 + int(30 * math.sin((frame_count + i * 50) * 0.02))
            ty = 480 + (i % 2) * 30
            # Body
            cv2.rectangle(frame, (tx - 15, ty - 40), (tx + 15, ty + 40), (40, 45, 50), -1)
            # Head
            cv2.circle(frame, (tx, ty - 55), 12, (40, 45, 50), -1)

        return frame

    def stop(self):
        self.running = False
        self.wait()


# =============================================================================
# MAIN WINDOW
# =============================================================================

class AISniperSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI SNIPER SYSTEM v4.2.1")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {DARK_BG};
            }}
            QWidget {{
                font-family: {FONT_FAMILY};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {DARK_BG};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        # State
        self.scanning = False
        self.tracking = False
        self.locked = False
        self.solved = False
        self.fired = False
        self.current_targets = []
        self.locked_target_id = None
        self.log_entries = deque(maxlen=50)
        self.system_online = True

        # Environment data
        self.wind_speed = 4.2
        self.wind_direction = 315
        self.temperature = 18.6
        self.humidity = 62
        self.pressure = 1013
        self.visibility = 8.7

        # Weapon data
        self.weapon = "SR-25 Mk1"
        self.ammo = 7
        self.caliber = "7.62x51mm"
        self.range_eff = 1200
        self.windage = -0.42
        self.elevation = 1.28

        self.init_ui()
        self.init_threads()
        self.init_timers()

        self.log("System Initialized", "info")
        self.log("Sensors Online", "info")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ========== TOP BAR ==========
        top_bar = QHBoxLayout()

        logo = QLabel("⚡ AI SNIPER SYSTEM v4.2.1")
        logo.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        top_bar.addWidget(logo)

        top_bar.addStretch()

        nav_items = [
            ("🏠 DASHBOARD", True),
            ("🎯 TARGETS", False),
            ("🔫 WEAPON", False),
            ("🌡 ENVIRONMENT", False),
            ("📊 ANALYTICS", False),
            ("⚙ SYSTEM", False),
        ]
        for text, active in nav_items:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    color: {'#ffffff' if active else TEXT_SECONDARY};
                    background-color: {'#1a2332' if active else 'transparent'};
                    border: {'1px solid ' + ACCENT_BLUE if active else 'none'};
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: 10px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    color: {TEXT_PRIMARY};
                    background-color: #1a2332;
                }}
            """)
            btn.setFixedHeight(32)
            top_bar.addWidget(btn)

        top_bar.addStretch()

        self.online_btn = QPushButton("🟢 SYSTEM ONLINE")
        self.online_btn.setStyleSheet(f"""
            color: {ACCENT_GREEN};
            background-color: transparent;
            border: 1px solid {ACCENT_GREEN};
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 10px;
            font-weight: bold;
        """)
        self.online_btn.setFixedHeight(32)
        top_bar.addWidget(self.online_btn)

        for sym in ["—", "□", "✕"]:
            btn = QPushButton(sym)
            btn.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; border: none; font-size: 12px;")
            btn.setFixedSize(30, 30)
            top_bar.addWidget(btn)

        main_layout.addLayout(top_bar)

        # ========== MAIN CONTENT ==========
        content = QHBoxLayout()
        content.setSpacing(8)

        # ----- LEFT PANEL -----
        left_panel = QVBoxLayout()
        left_panel.setSpacing(6)

        sys_panel = Panel("SYSTEM STATUS")
        self.ai_module = StatusIndicator("AI MODULE", "ONLINE")
        self.target_acq = StatusIndicator("TARGET ACQUISITION", "ACTIVE")
        self.weapon_sys = StatusIndicator("WEAPON SYSTEM", "READY")
        self.sensor_arr = StatusIndicator("SENSOR ARRAY", "ONLINE")
        self.env_status = StatusIndicator("ENVIRONMENT", "STABLE")
        self.comm_link = StatusIndicator("COMM LINK", "SECURE")
        for w in [self.ai_module, self.target_acq, self.weapon_sys, self.sensor_arr, self.env_status, self.comm_link]:
            sys_panel.layout.addWidget(w)
        left_panel.addWidget(sys_panel)

        ai_panel = Panel("AI PROCESSING")
        self.nn_status = StatusIndicator("NEURAL NETWORK", "ACTIVE")
        ai_panel.layout.addWidget(self.nn_status)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("CONFIDENCE LEVEL"))
        self.conf_bar = ProgressBarStyled()
        self.conf_bar.setValue(98)
        conf_layout.addWidget(self.conf_bar)
        ai_panel.layout.addLayout(conf_layout)

        self.obj_det = MetricRow("OBJECT DETECTION", "12", "ms")
        self.pred_acc = MetricRow("PREDICTION ACCURACY", "96.3", "%")
        ai_panel.layout.addWidget(self.obj_det)
        ai_panel.layout.addWidget(self.pred_acc)
        left_panel.addWidget(ai_panel)

        weap_panel = Panel("WEAPON STATUS")
        weap_img = QLabel("🔫 SR-25 Mk1\n[READY]")
        weap_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        weap_img.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; padding: 10px;")
        weap_panel.layout.addWidget(weap_img)

        self.ammo_row = MetricRow("AMMUNITION", "7", "/ 20")
        self.caliber_row = MetricRow("CALIBER", "7.62x51mm", "")
        self.eff_range_row = MetricRow("EFFECTIVE RANGE", "1200", "m")
        self.windage_row = MetricRow("WINDAGE", "-0.42", "MRAD")
        self.elev_row = MetricRow("ELEVATION", "1.28", "MRAD")
        for w in [self.ammo_row, self.caliber_row, self.eff_range_row, self.windage_row, self.elev_row]:
            weap_panel.layout.addWidget(w)
        left_panel.addWidget(weap_panel)

        env_panel = Panel("ENVIRONMENT")
        self.wind_spd = MetricRow("WIND SPEED", "4.2", "m/s")
        self.wind_dir = MetricRow("WIND DIRECTION", "NW 315°", "")
        self.temp = MetricRow("TEMPERATURE", "18.6", "°C")
        self.humid = MetricRow("HUMIDITY", "62", "%")
        self.press = MetricRow("PRESSURE", "1013", "hPa")
        self.visib = MetricRow("VISIBILITY", "8.7", "km")
        for w in [self.wind_spd, self.wind_dir, self.temp, self.humid, self.press, self.visib]:
            env_panel.layout.addWidget(w)
        left_panel.addWidget(env_panel)

        left_panel.addStretch()
        content.addLayout(left_panel, 1)

        # ----- CENTER PANEL -----
        center_panel = QVBoxLayout()
        center_panel.setSpacing(6)

        feed_header = QHBoxLayout()
        feed_title = QLabel("LIVE FEED - OPTICAL ZOOM 12.0x")
        feed_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        feed_header.addWidget(feed_title)
        feed_header.addStretch()
        close_btn = QLabel("✕")
        close_btn.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        feed_header.addWidget(close_btn)
        center_panel.addLayout(feed_header)

        self.live_feed = LiveFeedWidget()
        center_panel.addWidget(self.live_feed, stretch=1)

        bottom_center = QHBoxLayout()
        bottom_center.setSpacing(6)

        self.traj_widget = TrajectoryWidget()
        traj_panel = Panel("TRAJECTORY PREDICTION")
        traj_panel.layout.addWidget(self.traj_widget)
        bottom_center.addWidget(traj_panel, 1)

        self.wind_widget = WindWidget()
        wind_panel = Panel("WIND ANALYSIS")
        wind_panel.layout.addWidget(self.wind_widget)
        bottom_center.addWidget(wind_panel, 1)

        self.vitals_widget = TargetVitalsWidget()
        vitals_panel = Panel("TARGET VITALS")
        vitals_panel.layout.addWidget(self.vitals_widget)
        bottom_center.addWidget(vitals_panel, 1)

        center_panel.addLayout(bottom_center)
        content.addLayout(center_panel, 4)

        # ----- RIGHT PANEL -----
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)

        targets_header = QHBoxLayout()
        targets_title = QLabel("TARGETS DETECTED")
        targets_title.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        targets_header.addWidget(targets_title)
        self.total_targets = QLabel("TOTAL: 4")
        self.total_targets.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
        targets_header.addWidget(self.total_targets)
        right_panel.addLayout(targets_header)

        self.targets_scroll = QScrollArea()
        self.targets_scroll.setWidgetResizable(True)
        self.targets_scroll.setStyleSheet("background: transparent; border: none;")
        self.targets_container = QWidget()
        self.targets_layout = QVBoxLayout(self.targets_container)
        self.targets_layout.setSpacing(4)
        self.targets_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.targets_scroll.setWidget(self.targets_container)
        right_panel.addWidget(self.targets_scroll)

        self.target_cards = []
        targets_data = [
            (1, 612.4, 98.7, "LOCKED"),
            (2, 743.8, 87.3, "DETECTED"),
            (3, 921.5, 76.8, "DETECTED"),
            (4, 1053.2, 65.4, "DETECTED"),
        ]
        for tid, dist, conf, status in targets_data:
            card = TargetCard(tid, dist, conf, status)
            self.target_cards.append(card)
            self.targets_layout.addWidget(card)

        ball_panel = Panel("BALLISTIC SOLUTION")
        self.ball_rows = {
            "distance": MetricRow("DISTANCE", "612.4", "m"),
            "tof": MetricRow("TIME OF FLIGHT", "0.842", "s"),
            "drop": MetricRow("DROP", "1.28", "MRAD"),
            "wind_drift": MetricRow("WIND DRIFT", "-0.42", "MRAD"),
            "lead": MetricRow("LEAD", "0.18", "MRAD"),
            "impact_vel": MetricRow("IMPACT VELOCITY", "732", "m/s"),
            "solution_conf": MetricRow("SOLUTION CONFIDENCE", "98.7", "%"),
        }
        for row in self.ball_rows.values():
            ball_panel.layout.addWidget(row)
        right_panel.addWidget(ball_panel)

        self.fire_ready = QLabel("● FIRE SOLUTION READY")
        self.fire_ready.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 11px; font-weight: bold; padding: 6px;")
        self.fire_ready.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(self.fire_ready)

        log_panel = Panel("SYSTEM LOG")
        self.log_text = QLabel("System Initialized\nSensors Online")
        self.log_text.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9px; font-family: {FONT_MONO}; padding: 4px;")
        self.log_text.setWordWrap(True)
        self.log_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        log_panel.layout.addWidget(self.log_text)
        right_panel.addWidget(log_panel)

        content.addLayout(right_panel, 1)
        main_layout.addLayout(content, stretch=1)

        # ========== BOTTOM ACTION BAR ==========
        action_bar = QHBoxLayout()
        action_bar.setSpacing(8)

        self.scan_btn = ActionButton("SCAN", ACCENT_BLUE, "🎯")
        self.scan_btn.clicked.connect(self.toggle_scan)

        self.track_btn = ActionButton("TRACK", ACCENT_BLUE, "📡")
        self.track_btn.clicked.connect(self.toggle_track)

        self.lock_btn = ActionButton("LOCK", ACCENT_GREEN, "🔒")
        self.lock_btn.clicked.connect(self.toggle_lock)

        self.solve_btn = ActionButton("SOLVE", ACCENT_BLUE, "📐")
        self.solve_btn.clicked.connect(self.toggle_solve)

        self.fire_btn = ActionButton("FIRE", ACCENT_RED, "💥")
        self.fire_btn.clicked.connect(self.toggle_fire)

        for btn in [self.scan_btn, self.track_btn, self.lock_btn, self.solve_btn, self.fire_btn]:
            action_bar.addWidget(btn)

        main_layout.addLayout(action_bar)

    def init_threads(self):
        self.video_thread = VideoCapture()
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.camera_error.connect(self.on_camera_error)
        self.video_thread.start()

        self.detector = YOLODetector()
        self.detector.detections_ready.connect(self.on_detections)
        self.detector.start()

    def init_timers(self):
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        self.env_timer = QTimer()
        self.env_timer.timeout.connect(self.update_environment)
        self.env_timer.start(2000)

        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.update_ai_metrics)
        self.ai_timer.start(500)

        self.ballistic_timer = QTimer()
        self.ballistic_timer.timeout.connect(self.update_ballistics)
        self.ballistic_timer.start(1000)

    def on_camera_error(self, msg):
        self.log(msg, "warn")

    def on_frame_ready(self, frame):
        self.detector.set_frame(frame)
        self.live_feed.current_frame = frame
        self.live_feed.draw_overlays()

    def on_detections(self, boxes):
        self.current_targets = boxes
        self.live_feed.set_detections(boxes)
        self.total_targets.setText(f"TOTAL: {len(boxes)}")

        for i, card in enumerate(self.target_cards):
            if i < len(boxes):
                card.show()
                x1, y1, x2, y2, conf, cls, tid = boxes[i]
                card.distance = 600 + tid * 100 + random.uniform(-20, 20)
                card.confidence = conf * 100
                card.set_locked(tid == self.locked_target_id)
            else:
                card.hide()

    def update_clock(self):
        now = datetime.now()
        self.live_feed.date_str = now.strftime("%Y-%m-%d")
        self.live_feed.time_str = now.strftime("%H:%M:%S")

    def update_environment(self):
        self.wind_speed = randomize_value(4.2, 0.1)
        self.wind_direction = int(randomize_value(315, 0.02))
        self.temperature = randomize_value(18.6, 0.02)
        self.humidity = int(randomize_value(62, 0.03))
        self.pressure = int(randomize_value(1013, 0.005))
        self.visibility = randomize_value(8.7, 0.05)

        self.wind_spd.set_value(f"{self.wind_speed:.1f}", "m/s")
        self.wind_dir.set_value(f"NW {self.wind_direction}°", "")
        self.temp.set_value(f"{self.temperature:.1f}", "°C")
        self.humid.set_value(f"{self.humidity}", "%")
        self.press.set_value(f"{self.pressure}", "hPa")
        self.visib.set_value(f"{self.visibility:.1f}", "km")

        self.wind_widget.set_data(self.wind_speed, self.wind_direction)

    def update_ai_metrics(self):
        conf = randomize_value(98.7, 0.01)
        self.conf_bar.setValue(int(conf))
        self.obj_det.set_value(f"{random.randint(10, 15)}", "ms")
        self.pred_acc.set_value(f"{randomize_value(96.3, 0.02):.1f}", "%")
        self.vitals_widget.set_probability(conf)

    def update_ballistics(self):
        if self.locked_target_id is not None:
            dist = 600 + self.locked_target_id * 100
            drop = dist / 480.0
            tof = dist / 730.0
            windage = -0.42 + random.uniform(-0.02, 0.02)

            self.ball_rows["distance"].set_value(f"{dist:.1f}", "m")
            self.ball_rows["tof"].set_value(f"{tof:.3f}", "s")
            self.ball_rows["drop"].set_value(f"{drop:.2f}", "MRAD")
            self.ball_rows["wind_drift"].set_value(f"{windage:.2f}", "MRAD")
            self.ball_rows["lead"].set_value(f"{randomize_value(0.18, 0.1):.2f}", "MRAD")
            self.ball_rows["impact_vel"].set_value(f"{int(randomize_value(732, 0.05))}", "m/s")
            self.ball_rows["solution_conf"].set_value(f"{randomize_value(98.7, 0.01):.1f}", "%")

            self.traj_widget.set_data(dist, drop)
            self.windage_row.set_value(f"{windage:.2f}", "MRAD")
            self.elev_row.set_value(f"{drop:.2f}", "MRAD")

    def log(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {"info": TEXT_SECONDARY, "warn": ACCENT_YELLOW, "error": ACCENT_RED, "success": ACCENT_GREEN}.get(level, TEXT_SECONDARY)
        self.log_entries.append(f"<span style='color:{color}'>{timestamp}  {message}</span>")
        self.log_text.setText("<br>".join(self.log_entries))

    def toggle_scan(self):
        self.scanning = not self.scanning
        self.scan_btn.set_active(self.scanning)
        if self.scanning:
            self.log("Scanning initiated...", "info")
            self.target_acq.set_status("SCANNING")
            self.ai_module.set_status("ACTIVE")
        else:
            self.log("Scanning stopped", "info")
            self.target_acq.set_status("ACTIVE")

    def toggle_track(self):
        self.tracking = not self.tracking
        self.track_btn.set_active(self.tracking)
        if self.tracking:
            self.log("Tracking all detected targets", "info")
            self.target_acq.set_status("TRACKING")
        else:
            self.log("Tracking paused", "info")
            self.target_acq.set_status("ACTIVE")

    def toggle_lock(self):
        if not self.current_targets:
            self.log("No targets to lock!", "warn")
            return

        self.locked = not self.locked
        self.lock_btn.set_active(self.locked)

        if self.locked:
            self.locked_target_id = 1
            self.live_feed.set_locked_target(1)
            self.live_feed.target_info["id"] = "01"
            self.live_feed.target_info["distance"] = 612.4
            self.live_feed.target_info["confidence"] = 98.7
            self.log("Target Acquired: ID 01", "success")
            self.log("Target Locked", "success")
            self.target_acq.set_status("LOCKED")

            for card in self.target_cards:
                card.set_locked(card.target_id == 1)
        else:
            self.locked_target_id = None
            self.live_feed.set_locked_target(None)
            self.log("Lock released", "info")
            self.target_acq.set_status("ACTIVE")
            for card in self.target_cards:
                card.set_locked(False)

    def toggle_solve(self):
        if not self.locked:
            self.log("Lock target before solving!", "warn")
            return

        self.solved = not self.solved
        self.solve_btn.set_active(self.solved)

        if self.solved:
            self.log("Ballistic Solution Calculated", "success")
            self.log("Fire Solution Ready", "success")
            self.fire_ready.setText("● FIRE SOLUTION READY")
            self.fire_ready.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 11px; font-weight: bold; padding: 6px;")
        else:
            self.fire_ready.setText("○ FIRE SOLUTION PENDING")
            self.fire_ready.setStyleSheet(f"color: {ACCENT_YELLOW}; font-size: 11px; font-weight: bold; padding: 6px;")

    def toggle_fire(self):
        if not self.solved:
            self.log("Solve ballistic solution before firing!", "warn")
            return

        self.fired = not self.fired
        self.fire_btn.set_active(self.fired)

        if self.fired:
            self.log("FIRING SOLUTION EXECUTED", "error")
            self.log("Target neutralized", "success")
            self.fire_ready.setText("⚠ TARGET NEUTRALIZED")
            self.fire_ready.setStyleSheet(f"color: {ACCENT_RED}; font-size: 11px; font-weight: bold; padding: 6px;")
            self.ammo = max(0, self.ammo - 1)
            self.ammo_row.set_value(f"{self.ammo}", "/ 20")

            QTimer.singleShot(2000, self.reset_fire)

    def reset_fire(self):
        self.fired = False
        self.fire_btn.set_active(False)
        self.fire_ready.setText("● FIRE SOLUTION READY")
        self.fire_ready.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 11px; font-weight: bold; padding: 6px;")

    def closeEvent(self, event):
        self.video_thread.stop()
        self.detector.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(DARK_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(PANEL_BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(DARK_BG))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(PANEL_BG))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    app.setPalette(palette)

    window = AISniperSystem()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()