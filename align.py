import cv2
import numpy as np
from ultralytics import YOLO

# ── Load your trained model ──────────────────────────────────
model = YOLO("runs/detect/pallet_model/weights/best.pt")

# ── Use webcam (0) or a video file ───────────────────────────
# To use a video file instead: cv2.VideoCapture("your_video.mp4")
cap = cv2.VideoCapture(0)

THRESHOLD = 50  # pixels — dead zone for "aligned"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    frame_cx = W // 2
    frame_cy = H // 2

    # ── Run detection ────────────────────────────────────────
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    command = "NO PALLET DETECTED"
    color = (0, 0, 255)  # red for no detection

    if len(boxes) > 0:
        # Pick most confident detection
        best = max(boxes, key=lambda b: b.conf)
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        conf = float(best.conf)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        offset_x = cx - frame_cx
        offset_y = cy - frame_cy

        # ── Determine command ────────────────────────────────
        if abs(offset_x) < THRESHOLD and abs(offset_y) < THRESHOLD:
            command = "ALIGNED - INSERT FORKS"
            color = (0, 255, 0)  # green
        elif abs(offset_x) >= abs(offset_y):
            if offset_x > 0:
                command = f"MOVE RIGHT  ({offset_x}px)"
            else:
                command = f"MOVE LEFT  ({abs(offset_x)}px)"
            color = (0, 255, 255)  # yellow
        else:
            if offset_y > 0:
                command = f"MOVE FORWARD  ({offset_y}px)"
            else:
                command = f"MOVE BACKWARD  ({abs(offset_y)}px)"
            color = (0, 255, 255)  # yellow

        # ── Draw bounding box and centers ────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.line(frame, (cx, cy), (frame_cx, frame_cy), (0, 165, 255), 2)

        # Confidence label
        cv2.putText(frame, f"pallet {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Frame center crosshair ───────────────────────────────
    cv2.drawMarker(frame, (frame_cx, frame_cy),
                   (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

    # ── Command banner ───────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 0), -1)
    cv2.putText(frame, command, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # ── Show ─────────────────────────────────────────────────
    cv2.imshow("Pallet Alignment System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()