"""
How detection works:
  1. Capture a frame from the camera.
  2. Convert to HSV colour space.
  3. Threshold to find pixels within the configured colour range.
  4. Find contours, filter by area to remove noise.
  5. Each surviving contour is one box.  Its centre is converted from
     pixel coordinates to robot coordinates using the calibration transform.
  6. Returns a list of Box objects (id, robot_X, robot_Y, pixel bounding rect).

==========================================================================
!!! COLOUR TUNING — DO THIS BEFORE THE FINAL RUN !!!
  Run this file directly:   python vision.py
  A live HSV debug window opens.  Adjust the six sliders until only the
  boxes appear white and everything else is black.  Write the six values
  into HSV_LOWER / HSV_UPPER below, then save.
==========================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import math

# ── camera ─────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 1          # same as calibration.py
TRANSFORM_FILE = "transform.npy"

# ── colour range (HSV) ─────────────────────────────────────────────────────
# Default: brownish cardboard / natural PLA grey.
# Run  `python vision.py`  to open the tuning window and find the right values
# for YOUR boxes under YOUR lighting, then paste them here.
#
# Quick reference — typical starting points:
#   Orange boxes:  lower=(5,100,100)   upper=(20,255,255)
#   Blue boxes:    lower=(100,80,50)   upper=(130,255,255)
#   Green boxes:   lower=(40,60,40)    upper=(80,255,255)
#   Grey/white PLA lower=(0,0,150)     upper=(180,40,255)  (low saturation)
HSV_LOWER =  np.array([18,  128, 108], dtype=np.uint8)
HSV_UPPER = np.array([28,  255, 255], dtype=np.uint8)

# ── detection parameters ───────────────────────────────────────────────────
MIN_CONTOUR_AREA = 500     # px² — ignore blobs smaller than this (noise)
MAX_CONTOUR_AREA = 80000   # px² — ignore blobs larger than this (table edge)
MAX_BOXES = 3              # expected number of boxes


@dataclass
class Box:
    """One detected box."""
    box_id:  int
    robot_x: float   # mm in robot frame
    robot_y: float   # mm in robot frame
    # pixel bounding rectangle (x, y, w, h) for drawing
    pixel_bbox: tuple = field(default_factory=tuple)
    # pixel centre for drawing
    pixel_cx: int = 0
    pixel_cy: int = 0

    def distance_from_origin(self) -> float:
        """Euclidean distance from the robot base origin (0,0) in mm."""
        return math.sqrt(self.robot_x ** 2 + self.robot_y ** 2)


class BoxDetector:
    def __init__(self, source: int = CAMERA_SOURCE,
                 transform_file: str = TRANSFORM_FILE):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera source {source!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Load the calibration transform saved by calibration.py
        try:
            self.transform = np.load(transform_file)
            print(f"[Vision] Loaded transform from '{transform_file}'.")
        except FileNotFoundError:
            raise SystemExit(
                f"[Vision] ERROR: '{transform_file}' not found.\n"
                "Run calibration.py first to create it.")

    def pixel_to_robot(self, u: float, v: float):
        """Convert pixel (u,v) → robot (X,Y) mm using the saved transform."""
        X, Y = self.transform @ np.array([u, v, 1.0], dtype=np.float32)
        return float(X), float(Y)

    def grab_frame(self):
        """Grab one fresh frame from the camera."""
        # Discard a few buffered frames so we get the most recent image
        for _ in range(3):
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("[Vision] Could not read frame from camera.")
        return frame

    def detect(self, frame=None) -> List[Box]:
        """
        Detect boxes in a camera frame.
        If frame is None, a fresh frame is grabbed automatically.
        Returns a list of Box objects sorted by box_id (detection order).
        """
        if frame is None:
            frame = self.grab_frame()

        # 1. Convert to HSV and threshold by colour
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # 2. Clean up the mask — close small holes, remove speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        # 3. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # 4. Filter by area
        valid = [c for c in contours
                 if MIN_CONTOUR_AREA < cv2.contourArea(c) < MAX_CONTOUR_AREA]

        # 5. Sort largest-first so the box IDs are consistent frame-to-frame
        valid.sort(key=cv2.contourArea, reverse=True)
        valid = valid[:MAX_BOXES]   # keep at most MAX_BOXES

        boxes = []
        for idx, cnt in enumerate(valid):
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            robot_x, robot_y = self.pixel_to_robot(cx, cy)
            boxes.append(Box(
                box_id=idx,
                robot_x=robot_x,
                robot_y=robot_y,
                pixel_bbox=(x, y, w, h),
                pixel_cx=cx,
                pixel_cy=cy,
            ))

        return boxes

    def draw(self, frame: np.ndarray, boxes: List[Box],
             active_id: Optional[int] = None,
             storage_pixel: Optional[tuple] = None) -> np.ndarray:
        """
        Draw the required visualisation on top of frame:
          • Red bounding box    = all detected boxes
          • Green bounding box  = the currently active (being picked) box
          • Blue bounding box   = the storage spot target
        Returns the annotated frame (does NOT modify the original).
        """
        out = frame.copy()

        for box in boxes:
            x, y, w, h = box.pixel_bbox
            # choose colour: green if active, red otherwise
            color = (0, 255, 0) if box.box_id == active_id else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            # draw centre cross
            cv2.drawMarker(out, (box.pixel_cx, box.pixel_cy),
                           color, cv2.MARKER_CROSS, 16, 2)
            # label
            label = f"Box {box.box_id}  ({box.robot_x:.0f},{box.robot_y:.0f})"
            cv2.putText(out, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # blue rectangle for the storage spot
        if storage_pixel is not None:
            sx, sy = storage_pixel
            cv2.rectangle(out, (sx - 40, sy - 40), (sx + 40, sy + 40),
                          (255, 80, 0), 2)
            cv2.putText(out, "STORAGE", (sx - 35, sy - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)

        # status text top-left
        status = f"Detected: {len(boxes)} box(es)"
        cv2.putText(out, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def release(self):
        self.cap.release()


# ============================================================================
# STANDALONE COLOUR TUNING MODE
# Run:  python vision.py
# Sliders let you tune HSV_LOWER / HSV_UPPER live.
# When the mask cleanly shows only the boxes, write the values into the
# HSV_LOWER / HSV_UPPER constants above.
# ============================================================================
if __name__ == "__main__":
    WIN_CAM  = "Camera (original)"
    WIN_MASK = "HSV Mask  — tune until only boxes are white"

    try:
        detector = BoxDetector()
    except SystemExit as e:
        # Allow tuning even without a saved calibration
        print(e)
        print("[Tuning] Continuing without calibration (pixel→robot disabled).")
        detector = None
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        cap = detector.cap

    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H low",  WIN_MASK, int(HSV_LOWER[0]), 180, lambda v: None)
    cv2.createTrackbar("S low",  WIN_MASK, int(HSV_LOWER[1]), 255, lambda v: None)
    cv2.createTrackbar("V low",  WIN_MASK, int(HSV_LOWER[2]), 255, lambda v: None)
    cv2.createTrackbar("H high", WIN_MASK, int(HSV_UPPER[0]), 180, lambda v: None)
    cv2.createTrackbar("S high", WIN_MASK, int(HSV_UPPER[1]), 255, lambda v: None)
    cv2.createTrackbar("V high", WIN_MASK, int(HSV_UPPER[2]), 255, lambda v: None)

    print("Adjust sliders until only the boxes appear white.")
    print("Write the six values into HSV_LOWER / HSV_UPPER in vision.py.")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        lo = np.array([cv2.getTrackbarPos("H low",  WIN_MASK),
                       cv2.getTrackbarPos("S low",  WIN_MASK),
                       cv2.getTrackbarPos("V low",  WIN_MASK)], dtype=np.uint8)
        hi = np.array([cv2.getTrackbarPos("H high", WIN_MASK),
                       cv2.getTrackbarPos("S high", WIN_MASK),
                       cv2.getTrackbarPos("V high", WIN_MASK)], dtype=np.uint8)

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow(WIN_CAM,  frame)
        cv2.imshow(WIN_MASK, mask)

        print(f"\rHSV_LOWER=({lo[0]},{lo[1]},{lo[2]})  "
              f"HSV_UPPER=({hi[0]},{hi[1]},{hi[2]})    ", end="")

        if cv2.waitKey(20) & 0xFF in (ord('q'), 27):
            break
        if cv2.getWindowProperty(WIN_MASK, cv2.WND_PROP_VISIBLE) < 1:
            break

    print()
    cap.release()
    cv2.destroyAllWindows()

