#!/usr/bin/env python3
"""
AI SNIPER SYSTEM v4.2.1 - FULLY FUNCTIONAL
==========================================
Complete tactical targeting dashboard with functional tabs, buttons,
YOLO object tracking via laptop webcam, and full system controls.

Requirements:
    pip install PyQt6 opencv-python ultralytics numpy matplotlib

Usage:
    python ai_sniper_system_full.py
"""

import sys
import os
import random
import math
import time
import json
from datetime import datetime
from collections import deque

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QProgressBar,
    QScrollArea, QSizePolicy, QSpacerItem, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QTextEdit, QGroupBox, QSplitter, QTabWidget, QFileDialog,
    QMessageBox, QInputDialog, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QPoint, QRect, QRectF
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont,
    QLinearGradient, QBrush, QPolygon, QPalette, QKeyEvent
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
            "DETECTED": ACCENT_YELLOW, "TRACKING": ACCENT_BLUE, "SCANNING": ACCENT_BLUE,
            "OFFLINE": ACCENT_RED, "ERROR": ACCENT_RED, "LETHAL": ACCENT_RED,
            "FIRING": ACCENT_RED, "CALIBRATING": ACCENT_YELLOW, "STANDBY": ACCENT_YELLOW
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


class NavButton(QPushButton):
    def __init__(self, text, active=False, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.active = active
        self.update_style()
        self.setFixedHeight(32)

    def update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                color: {'#ffffff' if self.active else TEXT_SECONDARY};
                background-color: {'#1a2332' if self.active else 'transparent'};
                border: {'1px solid ' + ACCENT_BLUE if self.active else 'none'};
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

    def set_active(self, active):
        self.active = active
        self.update_style()


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

        pen = QPen(QColor(GRID_LINE))
        pen.setWidth(1)
        painter.setPen(pen)
        for i in range(5):
            x = margin + (graph_w / 4) * i
            painter.drawLine(int(x), margin, int(x), h - margin)
        for i in range(5):
            y = margin + (graph_h / 4) * i
            painter.drawLine(margin, int(y), w - margin, int(y))

        pen = QPen(QColor(TEXT_SECONDARY))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(margin, h - margin, w - margin, h - margin)
        painter.drawLine(margin, margin, margin, h - margin)

        painter.setFont(QFont(FONT_MONO, 8))
        for i, val in enumerate([0, 200, 400, 600, 800, 1000, 1200]):
            x = margin + (graph_w / 6) * i
            painter.drawText(int(x - 15), h - margin + 15, f"{val}")
        painter.drawText(5, margin - 5, "2.0")
        painter.drawText(5, margin + graph_h // 2 + 5, "0")
        painter.drawText(5, h - margin + 5, "-2.0")

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

        tx = margin + graph_w * (self.distance / 1200)
        ty = (h / 2) + (-self.drop * (graph_h / 4))
        pen = QPen(QColor(ACCENT_GREEN))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(QPoint(int(tx), int(ty)), 4, 4)

        painter.setPen(QColor(ACCENT_GREEN))
        painter.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        painter.drawText(int(tx - 20), int(ty - 10), f"{self.distance:.1f}m")

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

        pen = QPen(QColor(GRID_LINE))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#111820")))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont(FONT_FAMILY, 8, QFont.Weight.Bold))
        painter.drawText(cx - 4, cy - radius - 2, "N")
        painter.drawText(cx - 4, cy + radius + 12, "S")
        painter.drawText(cx + radius + 4, cy + 4, "E")
        painter.drawText(cx - radius - 12, cy + 4, "W")

        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x1 = cx + (radius - 8) * math.sin(rad)
            y1 = cy - (radius - 8) * math.cos(rad)
            x2 = cx + radius * math.sin(rad)
            y2 = cy - radius * math.cos(rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        rad = math.radians(self.direction)
        dx = math.sin(rad) * (radius - 15)
        dy = -math.cos(rad) * (radius - 15)

        pen = QPen(QColor(ACCENT_BLUE))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(cx, cy, int(cx + dx), int(cy + dy))

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

        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setBrush(QBrush(QColor("#1a2332")))
        painter.drawEllipse(cx - 12, cy - 45, 24, 24)
        painter.drawRect(cx - 15, cy - 20, 30, 50)
        painter.drawLine(cx - 15, cy - 10, cx - 30, cy + 10)
        painter.drawLine(cx + 15, cy - 10, cx + 30, cy + 10)
        painter.drawLine(cx - 8, cy + 30, cx - 12, cy + 60)
        painter.drawLine(cx + 8, cy + 30, cx + 12, cy + 60)

        if self.probability > 90:
            color = QColor(ACCENT_RED)
            color.setAlpha(120)
        else:
            color = QColor(ACCENT_YELLOW)
            color.setAlpha(80)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(cx - 10, cy - 15, 20, 25)

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
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.time_str = datetime.now().strftime("%H:%M:%S")

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

        if self.mode == "NIGHT":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame[:, :, 0] = (frame[:, :, 0] * 0.3).astype(np.uint8)
            frame[:, :, 2] = (frame[:, :, 2] * 0.3).astype(np.uint8)
        elif self.mode == "THERMAL":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 30, cy), (cx - 10, cy), (0, 212, 255), 1)
        cv2.line(frame, (cx + 10, cy), (cx + 30, cy), (0, 212, 255), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy - 10), (0, 212, 255), 1)
        cv2.line(frame, (cx, cy + 10), (cx, cy + 30), (0, 212, 255), 1)
        cv2.circle(frame, (cx, cy), 5, (0, 212, 255), 1)

        for box in self.detection_boxes:
            x1, y1, x2, y2, conf, cls, tid = box
            color = (0, 255, 136) if tid == self.locked_target else (0, 212, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"TARGET ID: {tid:02d}"
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"DISTANCE: {randomize_value(600 + tid * 100, 0.02):.1f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        compass_y = 30
        cv2.line(frame, (w // 2 - 100, compass_y), (w // 2 + 100, compass_y), (100, 100, 100), 1)
        for i, deg in enumerate([300, 330, 0, 30, 60]):
            x = w // 2 - 80 + i * 40
            cv2.line(frame, (x, compass_y - 3), (x, compass_y + 3), (200, 200, 200), 1)
            cv2.putText(frame, str(deg), (x - 8, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(frame, "W", (w // 2 - 100, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "N", (w // 2 - 4, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 212, 255), 1)
        cv2.putText(frame, "E", (w // 2 + 90, compass_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        for i in range(-20, 21, 10):
            y = cy + i * 8
            cv2.line(frame, (20, y), (30, y), (100, 100, 100), 1)
            cv2.putText(frame, str(i), (5, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        for i in range(-20, 21, 10):
            x = cx + i * 8
            cv2.line(frame, (x, h - 40), (x, h - 30), (100, 100, 100), 1)
            cv2.putText(frame, str(i), (x - 5, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        info_x = w - 180
        cv2.rectangle(frame, (info_x, 10), (w - 10, 90), (10, 15, 20), -1)
        cv2.rectangle(frame, (info_x, 10), (w - 10, 90), (50, 50, 50), 1)
        cv2.putText(frame, f"DATE: {self.date_str}", (info_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"TIME: {self.time_str}", (info_x + 5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"MODE: {self.mode}", (info_x + 5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
        cv2.putText(frame, f"ZOOM: {self.zoom_level:.1f}x", (info_x + 5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if self.locked_target is not None:
            cv2.rectangle(frame, (w - 200, h - 120), (w - 10, h - 10), (10, 15, 20), -1)
            cv2.rectangle(frame, (w - 200, h - 120), (w - 10, h - 10), (0, 255, 136), 1)
            cv2.putText(frame, f"TARGET ID: {self.locked_target:02d}", (w - 190, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
            cv2.putText(frame, f"DISTANCE: {self.target_info['distance']:.1f} m", (w - 190, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
            cv2.putText(frame, f"CONFIDENCE: {self.target_info['confidence']:.1f}%", (w - 190, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
            cv2.putText(frame, f"PRIORITY: {self.target_info['priority']}", (w - 190, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 51, 102), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled)


# =============================================================================
# YOLO DETECTOR & VIDEO CAPTURE (same as before)
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
                                if cls == 0:
                                    boxes.append((x1, y1, x2, y2, conf, cls, i + 1))
                        self.detections_ready.emit(boxes)
                    except Exception as e:
                        print(f"[YOLO Error] {e}")
                        self.detections_ready.emit([])
                else:
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


class VideoCapture(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)

    def __init__(self, source=None):
        super().__init__()
        self.source = source
        self.running = True
        self.cap = None
        self.camera_found = False
        self.camera_index = 0
        self.recording = False
        self.video_writer = None

    def find_camera(self):
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    cap.release()
                    print(f"[INFO] Camera found at index {idx}")
                    return idx
                cap.release()
        return None

    def run(self):
        if self.source is not None:
            camera_idx = self.source
        else:
            camera_idx = self.find_camera()

        if camera_idx is not None:
            self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.camera_found = True
                self.camera_index = camera_idx
                print(f"[INFO] Using camera index {camera_idx}")

        if not self.camera_found:
            print("[WARNING] No camera found. Using synthetic feed.")
            self.camera_error.emit("No camera detected. Using synthetic feed.")

        frame_count = 0
        while self.running:
            if self.camera_found and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if self.recording and self.video_writer is not None:
                        self.video_writer.write(frame)
                    self.frame_ready.emit(frame)
                else:
                    self.camera_found = False
                    if self.cap:
                        self.cap.release()
            else:
                synthetic = self.generate_synthetic_frame(frame_count)
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(synthetic)
                self.frame_ready.emit(synthetic)
                frame_count += 1

            self.msleep(33)

        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()

    def generate_synthetic_frame(self, frame_count):
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (15, 20, 25)
        noise = np.random.normal(0, 5, (h, w, 3)).astype(np.uint8)
        frame = cv2.add(frame, noise)

        for i in range(5):
            bx = 100 + i * 220
            by = 200 + (i % 3) * 50
            bw = 180
            bh = 300 + (i % 2) * 100
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (30, 35, 40), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 55, 60), 1)
            for wy in range(by + 20, by + bh - 30, 40):
                for wx in range(bx + 15, bx + bw - 15, 35):
                    cv2.rectangle(frame, (wx, wy), (wx + 20, wy + 25), (10, 15, 20), -1)

        cv2.rectangle(frame, (0, 550), (w, h), (20, 25, 30), -1)

        for i in range(4):
            tx = 200 + i * 250 + int(30 * math.sin((frame_count + i * 50) * 0.02))
            ty = 480 + (i % 2) * 30
            cv2.rectangle(frame, (tx - 15, ty - 40), (tx + 15, ty + 40), (40, 45, 50), -1)
            cv2.circle(frame, (tx, ty - 55), 12, (40, 45, 50), -1)

        return frame

    def start_recording(self, filename="recording.mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (1280, 720))
        self.recording = True
        print(f"[INFO] Recording started: {filename}")

    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print("[INFO] Recording stopped")

    def stop(self):
        self.running = False
        self.wait()


# =============================================================================
# TAB PAGES
# =============================================================================

class TargetsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)

        header = QLabel("TARGET MANAGEMENT")
        header.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        self.layout.addWidget(header)

        # Target table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "TYPE", "DISTANCE", "CONFIDENCE", "STATUS", "PRIORITY", "LAST SEEN", "ACTIONS"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {PANEL_BG};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                gridline-color: {BORDER_COLOR};
                font-size: 10px;
            }}
            QHeaderView::section {{
                background-color: {DARK_BG};
                color: {ACCENT_BLUE};
                padding: 4px;
                border: 1px solid {BORDER_COLOR};
                font-size: 10px;
                font-weight: bold;
            }}
        """)
        self.layout.addWidget(self.table)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 Refresh Targets")
        self.refresh_btn.setStyleSheet(f"color: {ACCENT_BLUE}; background: {PANEL_BG}; border: 1px solid {ACCENT_BLUE}; padding: 8px;")
        btn_layout.addWidget(self.refresh_btn)

        self.clear_btn = QPushButton("🗑 Clear All")
        self.clear_btn.setStyleSheet(f"color: {ACCENT_RED}; background: {PANEL_BG}; border: 1px solid {ACCENT_RED}; padding: 8px;")
        btn_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("📤 Export CSV")
        self.export_btn.setStyleSheet(f"color: {ACCENT_GREEN}; background: {PANEL_BG}; border: 1px solid {ACCENT_GREEN}; padding: 8px;")
        btn_layout.addWidget(self.export_btn)

        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        self.populate_demo()

    def populate_demo(self):
        demo_data = [
            ["01", "PERSON", "612.4 m", "98.7%", "LOCKED", "HIGH", "14:32:17", "VIEW"],
            ["02", "PERSON", "743.8 m", "87.3%", "TRACKING", "MEDIUM", "14:32:15", "VIEW"],
            ["03", "PERSON", "921.5 m", "76.8%", "DETECTED", "LOW", "14:32:10", "VIEW"],
            ["04", "PERSON", "1053.2 m", "65.4%", "DETECTED", "LOW", "14:32:05", "VIEW"],
        ]
        self.table.setRowCount(len(demo_data))
        for i, row in enumerate(demo_data):
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)


class WeaponPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)

        header = QLabel("WEAPON CONFIGURATION")
        header.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        self.layout.addWidget(header)

        # Weapon selector
        selector = QHBoxLayout()
        selector.addWidget(QLabel("SELECT WEAPON:"))
        self.weapon_combo = QComboBox()
        self.weapon_combo.addItems(["SR-25 Mk1", "M24 SWS", "Barrett M82A1", "CheyTac M200", "TAC-50"])
        self.weapon_combo.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG}; border: 1px solid {BORDER_COLOR};")
        selector.addWidget(self.weapon_combo)
        selector.addStretch()
        self.layout.addLayout(selector)

        # Specs grid
        specs = QGridLayout()
        specs.setSpacing(8)

        specs_data = [
            ("Caliber:", "7.62x51mm NATO"),
            ("Effective Range:", "1200 m"),
            ("Muzzle Velocity:", "830 m/s"),
            ("Rate of Fire:", "Semi-auto"),
            ("Magazine:", "20 rounds"),
            ("Weight:", "4.88 kg"),
            ("Barrel Length:", "508 mm"),
            ("Twist Rate:", "1:11"),
        ]
        for i, (label, value) in enumerate(specs_data):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
            val = QLabel(value)
            val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; font-family: {FONT_MONO}; font-weight: bold;")
            specs.addWidget(lbl, i // 2, (i % 2) * 2)
            specs.addWidget(val, i // 2, (i % 2) * 2 + 1)

        self.layout.addLayout(specs)

        # Zeroing controls
        zero_panel = Panel("ZEROING & CALIBRATION")
        zero_grid = QGridLayout()

        zero_grid.addWidget(QLabel("Windage (MRAD):"), 0, 0)
        self.windage_spin = QDoubleSpinBox()
        self.windage_spin.setRange(-5.0, 5.0)
        self.windage_spin.setValue(-0.42)
        self.windage_spin.setSingleStep(0.01)
        self.windage_spin.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG};")
        zero_grid.addWidget(self.windage_spin, 0, 1)

        zero_grid.addWidget(QLabel("Elevation (MRAD):"), 0, 2)
        self.elev_spin = QDoubleSpinBox()
        self.elev_spin.setRange(-5.0, 5.0)
        self.elev_spin.setValue(1.28)
        self.elev_spin.setSingleStep(0.01)
        self.elev_spin.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG};")
        zero_grid.addWidget(self.elev_spin, 0, 3)

        zero_grid.addWidget(QLabel("Zero Distance (m):"), 1, 0)
        self.zero_dist = QSpinBox()
        self.zero_dist.setRange(100, 2000)
        self.zero_dist.setValue(100)
        self.zero_dist.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG};")
        zero_grid.addWidget(self.zero_dist, 1, 1)

        zero_grid.addWidget(QLabel("Ammo Type:"), 1, 2)
        self.ammo_combo = QComboBox()
        self.ammo_combo.addItems(["M118 LR", "M80 Ball", "Mk 316 Mod 0", "M852"])
        self.ammo_combo.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG};")
        zero_grid.addWidget(self.ammo_combo, 1, 3)

        zero_panel.layout.addLayout(zero_grid)
        self.layout.addWidget(zero_panel)

        # Calibrate button
        self.cal_btn = QPushButton("🔧 CALIBRATE WEAPON")
        self.cal_btn.setStyleSheet(f"""
            color: {ACCENT_YELLOW};
            background: {PANEL_BG};
            border: 1px solid {ACCENT_YELLOW};
            padding: 10px;
            font-size: 12px;
            font-weight: bold;
        """)
        self.layout.addWidget(self.cal_btn)
        self.layout.addStretch()


class EnvironmentPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)

        header = QLabel("ENVIRONMENTAL SENSORS")
        header.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        self.layout.addWidget(header)

        grid = QGridLayout()
        grid.setSpacing(10)

        self.sensor_widgets = {}
        sensors = [
            ("Wind Speed", "4.2", "m/s", "🌬"),
            ("Wind Direction", "315", "°", "🧭"),
            ("Temperature", "18.6", "°C", "🌡"),
            ("Humidity", "62", "%", "💧"),
            ("Pressure", "1013", "hPa", "📊"),
            ("Visibility", "8.7", "km", "👁"),
            ("Altitude", "245", "m", "🏔"),
            ("Coriolis", "0.03", "MRAD", "🌍"),
        ]

        for i, (name, val, unit, icon) in enumerate(sensors):
            panel = Panel(f"{icon} {name.upper()}")
            row = MetricRow("CURRENT", val, unit)
            row.setStyleSheet(f"padding: 8px;")
            panel.layout.addWidget(row)
            self.sensor_widgets[name] = row
            grid.addWidget(panel, i // 2, i % 2)

        self.layout.addLayout(grid)

        # Weather API mock
        api_panel = Panel("WEATHER DATA SOURCE")
        api_layout = QHBoxLayout()
        self.api_combo = QComboBox()
        self.api_combo.addItems(["Internal Sensors", "METAR Station", "Weather API", "Manual Input"])
        self.api_combo.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG};")
        api_layout.addWidget(QLabel("Source:"))
        api_layout.addWidget(self.api_combo)
        api_layout.addStretch()
        self.refresh_env_btn = QPushButton("🔄 Refresh")
        self.refresh_env_btn.setStyleSheet(f"color: {ACCENT_BLUE}; background: {PANEL_BG}; border: 1px solid {ACCENT_BLUE};")
        api_layout.addWidget(self.refresh_env_btn)
        api_panel.layout.addLayout(api_layout)
        self.layout.addWidget(api_panel)
        self.layout.addStretch()


class AnalyticsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)

        header = QLabel("MISSION ANALYTICS")
        header.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        self.layout.addWidget(header)

        # Stats grid
        stats = QGridLayout()
        stats.setSpacing(8)

        stat_items = [
            ("TOTAL TARGETS", "4", ACCENT_BLUE),
            ("ENGAGEMENTS", "1", ACCENT_GREEN),
            ("ACCURACY", "96.3%", ACCENT_GREEN),
            ("MISSION TIME", "00:14:32", ACCENT_YELLOW),
        ]
        for i, (label, val, color) in enumerate(stat_items):
            panel = Panel()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
            val_lbl = QLabel(val)
            val_lbl.setStyleSheet(f"color: {color}; font-size: 24px; font-family: {FONT_MONO}; font-weight: bold;")
            panel.layout.addWidget(lbl)
            panel.layout.addWidget(val_lbl)
            stats.addWidget(panel, 0, i)

        self.layout.addLayout(stats)

        # Engagement history
        hist_panel = Panel("ENGAGEMENT HISTORY")
        self.hist_table = QTableWidget()
        self.hist_table.setColumnCount(6)
        self.hist_table.setHorizontalHeaderLabels(["TIME", "TARGET", "DISTANCE", "RESULT", "CONFIDENCE", "NOTES"])
        self.hist_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {PANEL_BG};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                gridline-color: {BORDER_COLOR};
                font-size: 10px;
            }}
            QHeaderView::section {{
                background-color: {DARK_BG};
                color: {ACCENT_BLUE};
                padding: 4px;
                border: 1px solid {BORDER_COLOR};
                font-size: 10px;
                font-weight: bold;
            }}
        """)
        self.hist_table.setRowCount(1)
        self.hist_table.setItem(0, 0, QTableWidgetItem("14:32:17"))
        self.hist_table.setItem(0, 1, QTableWidgetItem("T-01"))
        self.hist_table.setItem(0, 2, QTableWidgetItem("612.4 m"))
        self.hist_table.setItem(0, 3, QTableWidgetItem("NEUTRALIZED"))
        self.hist_table.setItem(0, 4, QTableWidgetItem("98.7%"))
        self.hist_table.setItem(0, 5, QTableWidgetItem("Clean shot"))
        hist_panel.layout.addWidget(self.hist_table)
        self.layout.addWidget(hist_panel)

        # Export
        self.export_analytics_btn = QPushButton("📊 Export Mission Report")
        self.export_analytics_btn.setStyleSheet(f"color: {ACCENT_GREEN}; background: {PANEL_BG}; border: 1px solid {ACCENT_GREEN}; padding: 10px;")
        self.layout.addWidget(self.export_analytics_btn)
        self.layout.addStretch()


class SystemPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)

        header = QLabel("SYSTEM SETTINGS")
        header.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        self.layout.addWidget(header)

        # Camera settings
        cam_panel = Panel("CAMERA & VIDEO")
        cam_grid = QGridLayout()

        cam_grid.addWidget(QLabel("Camera Index:"), 0, 0)
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 5)
        self.cam_index.setValue(0)
        self.cam_index.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG};")
        cam_grid.addWidget(self.cam_index, 0, 1)

        cam_grid.addWidget(QLabel("Resolution:"), 0, 2)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["1280x720", "1920x1080", "640x480"])
        self.res_combo.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG};")
        cam_grid.addWidget(self.res_combo, 0, 3)

        cam_grid.addWidget(QLabel("FPS Limit:"), 1, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG};")
        cam_grid.addWidget(self.fps_spin, 1, 1)

        cam_grid.addWidget(QLabel("Record Path:"), 1, 2)
        self.record_path = QLineEdit("./recordings/")
        self.record_path.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG}; border: 1px solid {BORDER_COLOR};")
        cam_grid.addWidget(self.record_path, 1, 3)

        cam_panel.layout.addLayout(cam_grid)
        self.layout.addWidget(cam_panel)

        # AI settings
        ai_panel = Panel("AI MODEL SETTINGS")
        ai_grid = QGridLayout()

        ai_grid.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self.model_combo.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG};")
        ai_grid.addWidget(self.model_combo, 0, 1)

        ai_grid.addWidget(QLabel("Confidence Threshold:"), 0, 2)
        self.conf_thresh = QDoubleSpinBox()
        self.conf_thresh.setRange(0.1, 1.0)
        self.conf_thresh.setValue(0.5)
        self.conf_thresh.setSingleStep(0.05)
        self.conf_thresh.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG};")
        ai_grid.addWidget(self.conf_thresh, 0, 3)

        ai_grid.addWidget(QLabel("Target Classes:"), 1, 0)
        self.class_check = QCheckBox("Person Only")
        self.class_check.setChecked(True)
        self.class_check.setStyleSheet(f"color: {TEXT_PRIMARY};")
        ai_grid.addWidget(self.class_check, 1, 1)

        ai_grid.addWidget(QLabel("NMS IoU:"), 1, 2)
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.1, 1.0)
        self.nms_spin.setValue(0.45)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {DARK_BG};")
        ai_grid.addWidget(self.nms_spin, 1, 3)

        ai_panel.layout.addLayout(ai_grid)
        self.layout.addWidget(ai_panel)

        # System controls
        ctrl_layout = QHBoxLayout()

        self.save_cfg_btn = QPushButton("💾 Save Config")
        self.save_cfg_btn.setStyleSheet(f"color: {ACCENT_GREEN}; background: {PANEL_BG}; border: 1px solid {ACCENT_GREEN}; padding: 10px;")
        ctrl_layout.addWidget(self.save_cfg_btn)

        self.load_cfg_btn = QPushButton("📂 Load Config")
        self.load_cfg_btn.setStyleSheet(f"color: {ACCENT_BLUE}; background: {PANEL_BG}; border: 1px solid {ACCENT_BLUE}; padding: 10px;")
        ctrl_layout.addWidget(self.load_cfg_btn)

        self.restart_btn = QPushButton("🔄 Restart System")
        self.restart_btn.setStyleSheet(f"color: {ACCENT_YELLOW}; background: {PANEL_BG}; border: 1px solid {ACCENT_YELLOW}; padding: 10px;")
        ctrl_layout.addWidget(self.restart_btn)

        self.shutdown_btn = QPushButton("⏻ Shutdown")
        self.shutdown_btn.setStyleSheet(f"color: {ACCENT_RED}; background: {PANEL_BG}; border: 1px solid {ACCENT_RED}; padding: 10px;")
        ctrl_layout.addWidget(self.shutdown_btn)

        ctrl_layout.addStretch()
        self.layout.addLayout(ctrl_layout)

        # About
        about = QLabel("AI SNIPER SYSTEM v4.2.1\nBuilt with PyQt6 + OpenCV + YOLOv8\nFor simulation and research purposes only.")
        about.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; padding: 20px;")
        about.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(about)
        self.layout.addStretch()


# =============================================================================
# MAIN WINDOW
# =============================================================================

class AISniperSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI SNIPER SYSTEM v4.2.1")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {DARK_BG}; }}
            QWidget {{ font-family: {FONT_FAMILY}; }}
            QScrollArea {{ border: none; background-color: transparent; }}
            QScrollBar:vertical {{ background: {DARK_BG}; width: 8px; border-radius: 4px; }}
            QScrollBar::handle:vertical {{ background: {BORDER_COLOR}; border-radius: 4px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
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
        self.recording = False
        self.mission_start_time = datetime.now()

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
        self.connect_all_signals()

        self.log("System Initialized", "info")
        self.log("Sensors Online", "info")
        self.log("Press F1 for help, F11 for fullscreen", "info")

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

        # Nav buttons
        self.nav_buttons = []
        nav_items = [
            ("🏠 DASHBOARD", True),
            ("🎯 TARGETS", False),
            ("🔫 WEAPON", False),
            ("🌡 ENVIRONMENT", False),
            ("📊 ANALYTICS", False),
            ("⚙ SYSTEM", False),
        ]
        for i, (text, active) in enumerate(nav_items):
            btn = NavButton(text, active)
            btn.clicked.connect(lambda checked, idx=i: self.switch_tab(idx))
            self.nav_buttons.append(btn)
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
        self.online_btn.setCheckable(True)
        self.online_btn.clicked.connect(self.toggle_system_online)
        top_bar.addWidget(self.online_btn)

        for sym in ["—", "□", "✕"]:
            btn = QPushButton(sym)
            btn.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; border: none; font-size: 12px;")
            btn.setFixedSize(30, 30)
            if sym == "✕":
                btn.clicked.connect(self.close)
            elif sym == "□":
                btn.clicked.connect(self.toggle_fullscreen)
            top_bar.addWidget(btn)

        main_layout.addLayout(top_bar)

        # ========== STACKED CONTENT ==========
        self.stack = QStackedWidget()

        # --- DASHBOARD PAGE ---
        self.dashboard_page = QWidget()
        dash_layout = QHBoxLayout(self.dashboard_page)
        dash_layout.setSpacing(8)

        # Left panel (same as before)
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
        dash_layout.addLayout(left_panel, 1)

        # Center panel
        center_panel = QVBoxLayout()
        center_panel.setSpacing(6)

        feed_header = QHBoxLayout()
        feed_title = QLabel("LIVE FEED - OPTICAL ZOOM 12.0x")
        feed_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        feed_header.addWidget(feed_title)

        # Mode toggle
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["DAY", "NIGHT", "THERMAL"])
        self.mode_combo.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG}; border: 1px solid {BORDER_COLOR}; padding: 2px;")
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        feed_header.addWidget(self.mode_combo)

        # Zoom controls
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(24, 24)
        self.zoom_in_btn.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG}; border: 1px solid {BORDER_COLOR};")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        feed_header.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(24, 24)
        self.zoom_out_btn.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG}; border: 1px solid {BORDER_COLOR};")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        feed_header.addWidget(self.zoom_out_btn)

        # Screenshot button
        self.screenshot_btn = QPushButton("📷")
        self.screenshot_btn.setFixedSize(28, 24)
        self.screenshot_btn.setStyleSheet(f"color: {TEXT_PRIMARY}; background: {PANEL_BG}; border: 1px solid {BORDER_COLOR};")
        self.screenshot_btn.setToolTip("Take Screenshot (F2)")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        feed_header.addWidget(self.screenshot_btn)

        # Record button
        self.record_btn = QPushButton("⏺")
        self.record_btn.setFixedSize(28, 24)
        self.record_btn.setStyleSheet(f"color: {ACCENT_RED}; background: {PANEL_BG}; border: 1px solid {ACCENT_RED};")
        self.record_btn.setToolTip("Start/Stop Recording (F3)")
        self.record_btn.clicked.connect(self.toggle_recording)
        feed_header.addWidget(self.record_btn)

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
        dash_layout.addLayout(center_panel, 4)

        # Right panel
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

        dash_layout.addLayout(right_panel, 1)

        self.stack.addWidget(self.dashboard_page)

        # --- OTHER PAGES ---
        self.targets_page = TargetsPage()
        self.stack.addWidget(self.targets_page)

        self.weapon_page = WeaponPage()
        self.stack.addWidget(self.weapon_page)

        self.environment_page = EnvironmentPage()
        self.stack.addWidget(self.environment_page)

        self.analytics_page = AnalyticsPage()
        self.stack.addWidget(self.analytics_page)

        self.system_page = SystemPage()
        self.stack.addWidget(self.system_page)

        main_layout.addWidget(self.stack, stretch=1)

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

    def connect_all_signals(self):
        # Targets page
        self.targets_page.refresh_btn.clicked.connect(self.refresh_targets)
        self.targets_page.clear_btn.clicked.connect(self.clear_targets)
        self.targets_page.export_btn.clicked.connect(self.export_targets_csv)

        # Weapon page
        self.weapon_page.weapon_combo.currentTextChanged.connect(self.change_weapon)
        self.weapon_page.cal_btn.clicked.connect(self.calibrate_weapon)

        # Environment page
        self.environment_page.refresh_env_btn.clicked.connect(self.refresh_environment)

        # Analytics page
        self.analytics_page.export_analytics_btn.clicked.connect(self.export_mission_report)

        # System page
        self.system_page.save_cfg_btn.clicked.connect(self.save_config)
        self.system_page.load_cfg_btn.clicked.connect(self.load_config)
        self.system_page.restart_btn.clicked.connect(self.restart_system)
        self.system_page.shutdown_btn.clicked.connect(self.shutdown_system)

    # ========== TAB SWITCHING ==========
    def switch_tab(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.set_active(i == index)

    # ========== KEYBOARD SHORTCUTS ==========
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F1:
            self.show_help()
        elif event.key() == Qt.Key.Key_F2:
            self.take_screenshot()
        elif event.key() == Qt.Key.Key_F3:
            self.toggle_recording()
        elif event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_fire()
        elif event.key() == Qt.Key.Key_S:
            self.toggle_scan()
        elif event.key() == Qt.Key.Key_T:
            self.toggle_track()
        elif event.key() == Qt.Key.Key_L:
            self.toggle_lock()
        elif event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
        else:
            super().keyPressEvent(event)

    def show_help(self):
        help_text = """AI SNIPER SYSTEM - KEYBOARD SHORTCUTS

F1  - Show this help
F2  - Take screenshot
F3  - Start/Stop recording
F11 - Toggle fullscreen
ESC - Exit fullscreen

Tactical Controls:
S   - Toggle SCAN
T   - Toggle TRACK
L   - Toggle LOCK
SPACE - FIRE (when solved)

Mouse:
Click on target cards to select
Scroll on feed to zoom
"""
        QMessageBox.information(self, "Keyboard Shortcuts", help_text)

    # ========== CAMERA & VIDEO ==========
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

    def change_mode(self, mode):
        self.live_feed.mode = mode
        self.log(f"Switched to {mode} mode", "info")

    def zoom_in(self):
        self.live_feed.zoom_level = min(20.0, self.live_feed.zoom_level + 1.0)
        self.log(f"Zoom: {self.live_feed.zoom_level:.1f}x", "info")

    def zoom_out(self):
        self.live_feed.zoom_level = max(1.0, self.live_feed.zoom_level - 1.0)
        self.log(f"Zoom: {self.live_feed.zoom_level:.1f}x", "info")

    def take_screenshot(self):
        if self.live_feed.current_frame is not None:
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, self.live_feed.current_frame)
            self.log(f"Screenshot saved: {filename}", "success")

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            self.video_thread.start_recording(filename)
            self.record_btn.setStyleSheet(f"color: {ACCENT_RED}; background: {ACCENT_RED}; border: 1px solid {ACCENT_RED};")
            self.log(f"Recording started: {filename}", "info")
        else:
            self.video_thread.stop_recording()
            self.record_btn.setStyleSheet(f"color: {ACCENT_RED}; background: {PANEL_BG}; border: 1px solid {ACCENT_RED};")
            self.log("Recording stopped", "info")

    # ========== SYSTEM ONLINE ==========
    def toggle_system_online(self):
        self.system_online = self.online_btn.isChecked()
        if self.system_online:
            self.online_btn.setText("🟢 SYSTEM ONLINE")
            self.online_btn.setStyleSheet(f"""
                color: {ACCENT_GREEN};
                background-color: transparent;
                border: 1px solid {ACCENT_GREEN};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
            """)
            self.log("System Online", "success")
            for ind in [self.ai_module, self.target_acq, self.weapon_sys, self.sensor_arr, self.env_status, self.comm_link]:
                ind.set_status(ind.status_text.replace("OFFLINE", "ONLINE").replace("ERROR", "ONLINE"))
        else:
            self.online_btn.setText("🔴 SYSTEM OFFLINE")
            self.online_btn.setStyleSheet(f"""
                color: {ACCENT_RED};
                background-color: transparent;
                border: 1px solid {ACCENT_RED};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
            """)
            self.log("System Offline", "error")
            for ind in [self.ai_module, self.target_acq, self.weapon_sys, self.sensor_arr, self.env_status, self.comm_link]:
                ind.set_status("OFFLINE")

    # ========== FULLSCREEN ==========
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ========== TIMERS ==========
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

        # Update environment page sensors
        if hasattr(self, 'environment_page'):
            self.environment_page.sensor_widgets["Wind Speed"].set_value(f"{self.wind_speed:.1f}", "m/s")
            self.environment_page.sensor_widgets["Wind Direction"].set_value(f"{self.wind_direction}", "°")
            self.environment_page.sensor_widgets["Temperature"].set_value(f"{self.temperature:.1f}", "°C")
            self.environment_page.sensor_widgets["Humidity"].set_value(f"{self.humidity}", "%")
            self.environment_page.sensor_widgets["Pressure"].set_value(f"{self.pressure}", "hPa")
            self.environment_page.sensor_widgets["Visibility"].set_value(f"{self.visibility:.1f}", "km")

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

    # ========== BOTTOM BUTTONS ==========
    def toggle_scan(self):
        if not self.system_online:
            self.log("System is offline!", "error")
            return
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
        if not self.system_online:
            self.log("System is offline!", "error")
            return
        self.tracking = not self.tracking
        self.track_btn.set_active(self.tracking)
        if self.tracking:
            self.log("Tracking all detected targets", "info")
            self.target_acq.set_status("TRACKING")
        else:
            self.log("Tracking paused", "info")
            self.target_acq.set_status("ACTIVE")

    def toggle_lock(self):
        if not self.system_online:
            self.log("System is offline!", "error")
            return
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
        if not self.system_online:
            self.log("System is offline!", "error")
            return
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
        if not self.system_online:
            self.log("System is offline!", "error")
            return
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

            # Update analytics
            self.analytics_page.hist_table.setItem(0, 0, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))

            QTimer.singleShot(2000, self.reset_fire)

    def reset_fire(self):
        self.fired = False
        self.fire_btn.set_active(False)
        self.fire_ready.setText("● FIRE SOLUTION READY")
        self.fire_ready.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 11px; font-weight: bold; padding: 6px;")

    # ========== TARGETS PAGE ACTIONS ==========
    def refresh_targets(self):
        self.log("Refreshing target database...", "info")
        self.targets_page.populate_demo()
        self.log("Target database updated", "success")

    def clear_targets(self):
        self.targets_page.table.setRowCount(0)
        self.log("All targets cleared", "warn")

    def export_targets_csv(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Targets", "targets.csv", "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w') as f:
                f.write("ID,TYPE,DISTANCE,CONFIDENCE,STATUS,PRIORITY\n")
                for row in range(self.targets_page.table.rowCount()):
                    vals = [self.targets_page.table.item(row, col).text() for col in range(6)]
                    f.write(",".join(vals) + "\n")
            self.log(f"Targets exported to {filename}", "success")

    # ========== WEAPON PAGE ACTIONS ==========
    def change_weapon(self, weapon_name):
        self.weapon = weapon_name
        self.log(f"Weapon changed to {weapon_name}", "info")
        specs = {
            "SR-25 Mk1": ("7.62x51mm", "1200", "830"),
            "M24 SWS": ("7.62x51mm", "800", "790"),
            "Barrett M82A1": (".50 BMG", "1800", "853"),
            "CheyTac M200": (".408 CheyTac", "2300", "910"),
            "TAC-50": (".50 BMG", "2000", "850"),
        }
        if weapon_name in specs:
            cal, rng, vel = specs[weapon_name]
            self.caliber_row.set_value(cal, "")
            self.eff_range_row.set_value(rng, "m")
            self.weapon_page.windage_spin.setValue(-0.42)
            self.weapon_page.elev_spin.setValue(1.28)

    def calibrate_weapon(self):
        self.weapon_sys.set_status("CALIBRATING")
        self.log("Weapon calibration started...", "info")
        QTimer.singleShot(3000, lambda: (
            self.weapon_sys.set_status("READY"),
            self.log("Weapon calibration complete", "success")
        ))

    # ========== ENVIRONMENT PAGE ACTIONS ==========
    def refresh_environment(self):
        self.log("Refreshing environmental sensors...", "info")
        self.update_environment()
        self.log("Environmental data updated", "success")

    # ========== ANALYTICS PAGE ACTIONS ==========
    def export_mission_report(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Mission Report", "mission_report.txt", "Text Files (*.txt)")
        if filename:
            elapsed = datetime.now() - self.mission_start_time
            with open(filename, 'w') as f:
                f.write(f"AI SNIPER SYSTEM - MISSION REPORT\n")
                f.write(f"=================================\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mission Duration: {elapsed}\n")
                f.write(f"Total Targets: 4\n")
                f.write(f"Engagements: 1\n")
                f.write(f"Accuracy: 96.3%\n")
            self.log(f"Mission report exported to {filename}", "success")

    # ========== SYSTEM PAGE ACTIONS ==========
    def save_config(self):
        config = {
            "camera_index": self.system_page.cam_index.value(),
            "resolution": self.system_page.res_combo.currentText(),
            "fps": self.system_page.fps_spin.value(),
            "model": self.system_page.model_combo.currentText(),
            "confidence": self.system_page.conf_thresh.value(),
            "weapon": self.weapon,
        }
        filename, _ = QFileDialog.getSaveFileName(self, "Save Config", "config.json", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self.log(f"Config saved to {filename}", "success")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if filename and os.path.exists(filename):
            with open(filename, 'r') as f:
                config = json.load(f)
            self.system_page.cam_index.setValue(config.get("camera_index", 0))
            self.system_page.res_combo.setCurrentText(config.get("resolution", "1280x720"))
            self.system_page.fps_spin.setValue(config.get("fps", 30))
            self.system_page.model_combo.setCurrentText(config.get("model", "yolov8n.pt"))
            self.system_page.conf_thresh.setValue(config.get("confidence", 0.5))
            self.log(f"Config loaded from {filename}", "success")

    def restart_system(self):
        self.log("Restarting system...", "warn")
        self.video_thread.stop()
        self.detector.stop()
        QTimer.singleShot(1000, self.init_threads)
        self.log("System restarted", "success")

    def shutdown_system(self):
        reply = QMessageBox.question(self, "Shutdown", "Shutdown AI Sniper System?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.log("Shutting down...", "error")
            self.close()

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