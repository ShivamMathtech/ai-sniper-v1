import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import json

class AISniper:
    """
    AI Sniper System - Head Stability Tracking & ROI Selection

    Features:
    - Person detection & selection
    - Head stability analysis
    - ROI (Region of Interest) selection
    - Coordinate extraction of target person
    """

    def __init__(self, stability_window=30):
        # Initialize MediaPipe Pose & Face Detection
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_detection = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        # Tracking state
        self.detected_persons = []
        self.selected_person_id = None
        self.stability_history = {}  # person_id -> deque of head positions
        self.stability_window = stability_window

        # ROI state
        self.roi_points = []
        self.roi_active = False
        self.drawing_roi = False

        # Display settings
        self.colors = {
            'selected': (0, 255, 0),      # Green
            'unselected': (0, 165, 255),   # Orange
            'unstable': (0, 0, 255),       # Red
            'roi': (255, 0, 255),         # Magenta
            'crosshair': (0, 255, 255)    # Cyan
        }

        self.show_debug = True

    # ==================== PERSON DETECTION ====================

    def detect_persons(self, frame):
        """Detect all persons in frame using pose estimation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        persons = []
        h, w = frame.shape[:2]

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract key body points
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

            # Calculate bounding box
            x_coords = [nose.x, left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x]
            y_coords = [nose.y, left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y]

            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            person = {
                'id': 0,
                'bbox': (x_min, y_min, x_max, y_max),
                'nose': (int(nose.x * w), int(nose.y * h)),
                'center': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                'landmarks': landmarks,
                'confidence': nose.visibility
            }
            persons.append(person)

        self.detected_persons = persons
        return persons

    # ==================== HEAD STABILITY ====================

    def analyze_head_stability(self, person_id, head_pos):
        """
        Analyze head stability using position variance over time.
        Returns stability score (0-100) and movement data.
        """
        if person_id not in self.stability_history:
            self.stability_history[person_id] = deque(maxlen=self.stability_window)

        self.stability_history[person_id].append(head_pos)
        history = self.stability_history[person_id]

        if len(history) < 5:
            return {'score': 0, 'status': 'CALIBRATING', 'variance': 0, 'velocity': 0}

        # Calculate variance
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]

        var_x = np.var(xs)
        var_y = np.var(ys)
        total_variance = var_x + var_y

        # Calculate velocity (movement between frames)
        velocities = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i-1][0]
            dy = history[i][1] - history[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))

        avg_velocity = np.mean(velocities)
        max_velocity = max(velocities) if velocities else 0

        # Stability score: lower variance = higher score
        variance_penalty = min(total_variance / 100, 50)
        velocity_penalty = min(avg_velocity * 2, 30)

        score = max(0, 100 - variance_penalty - velocity_penalty)

        # Determine status
        if score >= 80:
            status = 'STABLE'
        elif score >= 50:
            status = 'MODERATE'
        else:
            status = 'UNSTABLE'

        return {
            'score': round(score, 1),
            'status': status,
            'variance': round(total_variance, 2),
            'velocity': round(avg_velocity, 2),
            'max_velocity': round(max_velocity, 2),
            'samples': len(history)
        }

    # ==================== ROI SELECTION ====================

    def set_roi(self, points):
        """Set Region of Interest polygon."""
        if len(points) >= 3:
            self.roi_points = points
            self.roi_active = True
            return True
        return False

    def clear_roi(self):
        """Clear ROI selection."""
        self.roi_points = []
        self.roi_active = False

    def point_in_roi(self, point):
        """Check if a point is inside the ROI polygon."""
        if not self.roi_active or len(self.roi_points) < 3:
            return True  # If no ROI, everything is valid

        point = np.array(point, dtype=np.float32)
        roi = np.array(self.roi_points, dtype=np.float32)

        return cv2.pointPolygonTest(roi, tuple(point), False) >= 0

    def get_roi_center(self):
        """Get center point of ROI."""
        if not self.roi_points:
            return None
        pts = np.array(self.roi_points)
        return (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))

    # ==================== COORDINATE EXTRACTION ====================

    def get_target_coordinates(self, frame_shape):
        """
        Get precise coordinates of the selected target.
        Returns dict with pixel and normalized coordinates.
        """
        if self.selected_person_id is None or not self.detected_persons:
            return None

        person = self.detected_persons[self.selected_person_id] if self.selected_person_id < len(self.detected_persons) else None
        if person is None:
            return None

        h, w = frame_shape[:2]
        nose = person['nose']
        center = person['center']

        # Check if target is within ROI
        in_roi = self.point_in_roi(nose)

        return {
            'pixel': {
                'nose': nose,
                'center': center,
                'bbox': person['bbox']
            },
            'normalized': {
                'nose': (round(nose[0] / w, 4), round(nose[1] / h, 4)),
                'center': (round(center[0] / w, 4), round(center[1] / h, 4))
            },
            'in_roi': in_roi,
            'stability': self.analyze_head_stability(self.selected_person_id, nose),
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }

    # ==================== VISUALIZATION ====================

    def draw_crosshair(self, frame, center, size=20, color=None):
        """Draw precision crosshair at target center."""
        if color is None:
            color = self.colors['crosshair']

        x, y = center
        cv2.line(frame, (x - size, y), (x + size, y), color, 2)
        cv2.line(frame, (x, y - size), (x, y + size), color, 2)
        cv2.circle(frame, center, size // 2, color, 1)
        cv2.circle(frame, center, 2, (0, 0, 255), -1)

    def draw_stability_bar(self, frame, x, y, score, width=100, height=10):
        """Draw stability indicator bar."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

        # Fill based on score
        fill_width = int((score / 100) * width)
        if score >= 80:
            color = (0, 255, 0)
        elif score >= 50:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)

    def draw_roi(self, frame):
        """Draw ROI polygon on frame."""
        if not self.roi_active or len(self.roi_points) < 3:
            return

        pts = np.array(self.roi_points, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Fill with transparency effect
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 255))
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Draw border
        cv2.polylines(frame, [pts], True, self.colors['roi'], 2)

        # Draw vertices
        for i, pt in enumerate(self.roi_points):
            cv2.circle(frame, pt, 5, self.colors['roi'], -1)
            cv2.putText(frame, str(i+1), (pt[0]+8, pt[1]-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['roi'], 1)

    def draw_info_panel(self, frame, target_data):
        """Draw information panel with target data."""
        h, w = frame.shape[:2]
        panel_w = 320
        panel_x = w - panel_w - 10
        panel_y = 10

        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + 200), (0, 0, 0), -1)
        cv2.addWeighted(frame, 1, overlay, 0.7, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 200), (100, 100, 100), 1)

        if target_data is None:
            cv2.putText(frame, "NO TARGET", (panel_x + 10, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        y_offset = panel_y + 25
        line_height = 22

        # Title
        cv2.putText(frame, "=== TARGET DATA ===", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height + 5

        # Coordinates
        nose_px = target_data['pixel']['nose']
        nose_norm = target_data['normalized']['nose']
        cv2.putText(frame, f"Nose: ({nose_px[0]}, {nose_px[1]})", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        cv2.putText(frame, f"Norm: ({nose_norm[0]}, {nose_norm[1]})", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height + 5

        # Stability
        stab = target_data['stability']
        cv2.putText(frame, f"Stability: {stab['status']}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        self.draw_stability_bar(frame, panel_x + 10, y_offset, stab['score'], width=200)
        y_offset += 18
        cv2.putText(frame, f"Score: {stab['score']}%", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"Var: {stab['variance']} Vel: {stab['velocity']}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height + 5

        # ROI Status
        roi_status = "INSIDE" if target_data['in_roi'] else "OUTSIDE"
        roi_color = (0, 255, 0) if target_data['in_roi'] else (0, 0, 255)
        cv2.putText(frame, f"ROI: {roi_status}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

    def render(self, frame):
        """Main render function - draw everything on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw ROI first (behind everything)
        self.draw_roi(display)

        # Detect and draw persons
        persons = self.detect_persons(frame)

        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person['bbox']
            nose = person['nose']
            center = person['center']

            # Determine color based on selection and stability
            if i == self.selected_person_id:
                stab = self.analyze_head_stability(i, nose)
                if stab['status'] == 'STABLE':
                    color = self.colors['selected']
                elif stab['status'] == 'MODERATE':
                    color = (0, 255, 255)
                else:
                    color = self.colors['unstable']

                # Draw crosshair on selected target
                self.draw_crosshair(display, nose, size=25, color=color)

                # Draw connecting line from ROI center to target
                if self.roi_active:
                    roi_c = self.get_roi_center()
                    if roi_c:
                        cv2.line(display, roi_c, nose, color, 1, cv2.LINE_AA)
            else:
                color = self.colors['unselected']

            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Draw person ID
            label = f"ID:{i}"
            if i == self.selected_person_id:
                label += " [LOCKED]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(display, label, (x1 + 4, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw nose point
            cv2.circle(display, nose, 5, color, -1)

        # Draw info panel
        target_data = self.get_target_coordinates(frame.shape) if self.selected_person_id is not None else None
        self.draw_info_panel(display, target_data)

        # Draw instructions
        instructions = [
            "[Click] Select Target | [R] Reset ROI | [C] Clear Selection | [Q] Quit",
            "[S] Save Coords | [D] Toggle Debug"
        ]
        for idx, text in enumerate(instructions):
            cv2.putText(display, text, (10, h - 20 + idx * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw ROI mode indicator
        if self.drawing_roi:
            cv2.putText(display, "ROI MODE: Click to set points, ENTER to confirm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return display, target_data


# ==================== MAIN APPLICATION ====================

def main():
    """Main application loop."""

    # Initialize sniper
    sniper = AISniper(stability_window=30)

    # Open video source (0 = webcam, or path to video file)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Mouse callback state
    roi_temp_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_temp_points

        if event == cv2.EVENT_LBUTTONDOWN:
            if sniper.drawing_roi:
                # Add ROI point
                roi_temp_points.append((x, y))
                print(f"ROI Point {len(roi_temp_points)}: ({x}, {y})")
            else:
                # Select person by clicking near them
                for i, person in enumerate(sniper.detected_persons):
                    x1, y1, x2, y2 = person['bbox']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        sniper.selected_person_id = i
                        print(f"Selected Person ID: {i}")
                        # Clear stability history for new target
                        sniper.stability_history = {}
                        break

    cv2.namedWindow('AI Sniper System')
    cv2.setMouseCallback('AI Sniper System', mouse_callback)

    print("=" * 50)
    print("AI SNIPER SYSTEM - CONTROLS")
    print("=" * 50)
    print("Click on a person to SELECT target")
    print("R - Start/Complete ROI selection mode")
    print("ENTER - Confirm ROI polygon")
    print("C - Clear target selection")
    print("S - Save coordinates to file")
    print("D - Toggle debug info")
    print("Q - Quit")
    print("=" * 50)

    saved_coordinates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect (optional)
        frame = cv2.flip(frame, 1)

        # Render
        display, target_data = sniper.render(frame)

        # Draw temporary ROI points
        for pt in roi_temp_points:
            cv2.circle(display, pt, 5, (255, 0, 255), -1)
        if len(roi_temp_points) > 1:
            for i in range(len(roi_temp_points) - 1):
                cv2.line(display, roi_temp_points[i], roi_temp_points[i+1], (255, 0, 255), 2)

        cv2.imshow('AI Sniper System', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            if not sniper.drawing_roi:
                sniper.drawing_roi = True
                roi_temp_points = []
                sniper.clear_roi()
                print("ROI Mode: Click points to define region")
            else:
                # Confirm ROI
                if len(roi_temp_points) >= 3:
                    sniper.set_roi(roi_temp_points)
                    print(f"ROI set with {len(roi_temp_points)} points")
                sniper.drawing_roi = False
                roi_temp_points = []

        elif key == 13:  # ENTER key
            if sniper.drawing_roi and len(roi_temp_points) >= 3:
                sniper.set_roi(roi_temp_points)
                print(f"ROI confirmed with {len(roi_temp_points)} points")
                sniper.drawing_roi = False
                roi_temp_points = []

        elif key == ord('c'):
            sniper.selected_person_id = None
            sniper.stability_history = {}
            print("Target cleared")

        elif key == ord('s'):
            if target_data:
                saved_coordinates.append(target_data)
                filename = f"target_coords_{len(saved_coordinates)}.json"
                with open(filename, 'w') as f:
                    json.dump(target_data, f, indent=2, default=str)
                print(f"Coordinates saved to {filename}")
                print(f"Data: {json.dumps(target_data, indent=2, default=str)}")

        elif key == ord('d'):
            sniper.show_debug = not sniper.show_debug
            print(f"Debug: {'ON' if sniper.show_debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    print("\nSession ended.")
    print(f"Total coordinate saves: {len(saved_coordinates)}")


if __name__ == "__main__":
    main()