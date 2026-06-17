import cv2
import numpy as np
from ultralytics import YOLO

# ── Load model ──────────────────────────────────────────────
model = YOLO("yolov8n.pt")  # general pretrained model for now

# ── Load a sample image from your dataset ───────────────────
image_path = "test/images/"  # we'll point to a specific image next
import os
images = os.listdir(image_path)
img_file = image_path + images[0]  # picks the first test image

frame = cv2.imread(img_file)
H, W = frame.shape[:2]
frame_center_x = W // 2
frame_center_y = H // 2

# ── Run detection ────────────────────────────────────────────
results = model(frame, verbose=False)
boxes = results[0].boxes

# ── Alignment logic ──────────────────────────────────────────
command = "NO PALLET DETECTED"
offset_x = 0
offset_y = 0
box_drawn = None

if len(boxes) > 0:
    # Take the most confident detection
    best = max(boxes, key=lambda b: b.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    cx = (x1 + x2) // 2  # pallet center x
    cy = (y1 + y2) // 2  # pallet center y

    offset_x = cx - frame_center_x
    offset_y = cy - frame_center_y

    # Dead zone: if offset is small, consider aligned
    THRESHOLD = 50  # pixels

    if abs(offset_x) < THRESHOLD and abs(offset_y) < THRESHOLD:
        command = "ALIGNED - INSERT FORKS"
    elif abs(offset_x) >= abs(offset_y):
        if offset_x > 0:
            command = f"MOVE RIGHT  ({offset_x}px)"
        else:
            command = f"MOVE LEFT  ({abs(offset_x)}px)"
    else:
        if offset_y > 0:
            command = f"MOVE FORWARD  ({offset_y}px)"
        else:
            command = f"MOVE BACKWARD  ({abs(offset_y)}px)"

    box_drawn = (x1, y1, x2, y2, cx, cy)

# ── Draw visualization ───────────────────────────────────────
# Bounding box
if box_drawn:
    x1, y1, x2, y2, cx, cy = box_drawn
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
    # Line from pallet center to frame center
    cv2.line(frame, (cx, cy), (frame_center_x, frame_center_y), (0, 165, 255), 2)

# Frame center crosshair
cv2.drawMarker(frame, (frame_center_x, frame_center_y),
               (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

# Command text
cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 0), -1)
cv2.putText(frame, command, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# ── Show result ──────────────────────────────────────────────
cv2.imshow("Pallet Alignment System", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()