import cv2
import numpy as np
from ultralytics import YOLO

# ── Load model ───────────────────────────────────────────────
model = YOLO("runs/detect/pallet_model/weights/best.pt")
cap = cv2.VideoCapture(0)

THRESHOLD = 50       # pixels — alignment dead zone
FAR_SIZE = 0.10      # bounding box area < 10% of frame = too far
CLOSE_SIZE = 0.50    # bounding box area > 50% of frame = too close

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    frame_cx = W // 2
    frame_cy = H // 2
    frame_area = H * W

    # ── Run detection ────────────────────────────────────────
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    command = "NO PALLET DETECTED"
    cmd_color = (0, 0, 255)
    distance_status = "--"
    distance_color = (180, 180, 180)
    offset_x, offset_y = 0, 0
    conf_val = 0.0

    if len(boxes) > 0:
        best = max(boxes, key=lambda b: b.conf)
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        conf_val = float(best.conf)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        offset_x = cx - frame_cx
        offset_y = cy - frame_cy

        # ── Distance estimation ──────────────────────────────
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area

        if area_ratio < FAR_SIZE:
            distance_status = "TOO FAR"
            distance_color = (0, 0, 255)
        elif area_ratio > CLOSE_SIZE:
            distance_status = "TOO CLOSE"
            distance_color = (0, 0, 255)
        else:
            distance_status = "GOOD DISTANCE"
            distance_color = (0, 255, 0)

        # ── Alignment command ────────────────────────────────
        if abs(offset_x) < THRESHOLD and abs(offset_y) < THRESHOLD:
            command = "ALIGNED - INSERT FORKS"
            cmd_color = (0, 255, 0)
        elif abs(offset_x) >= abs(offset_y):
            direction = "MOVE RIGHT" if offset_x > 0 else "MOVE LEFT"
            command = f"{direction}  ({abs(offset_x)}px)"
            cmd_color = (0, 255, 255)
        else:
            direction = "MOVE FORWARD" if offset_y > 0 else "MOVE BACKWARD"
            command = f"{direction}  ({abs(offset_y)}px)"
            cmd_color = (0, 255, 255)

        # ── Draw bounding box ────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.line(frame, (cx, cy), (frame_cx, frame_cy), (0, 165, 255), 2)
        cv2.putText(frame, f"pallet {conf_val:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Frame center crosshair ───────────────────────────────
    cv2.drawMarker(frame, (frame_cx, frame_cy),
                   (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

    # ── Top command banner ───────────────────────────────────
    cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 0), -1)
    cv2.putText(frame, command, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, cmd_color, 2)

    # ── Side info panel ──────────────────────────────────────
    panel_x = W - 250
    cv2.rectangle(frame, (panel_x, 60), (W, 260), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, 60), (W, 260), (80, 80, 80), 1)

    cv2.putText(frame, "DIAGNOSTICS", (panel_x + 10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.putText(frame, f"Confidence : {conf_val:.2f}", (panel_x + 10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Offset X   : {offset_x}px", (panel_x + 10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Offset Y   : {offset_y}px", (panel_x + 10, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Distance   :", (panel_x + 10, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, distance_status, (panel_x + 10, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, distance_color, 2)

    # ── Show ─────────────────────────────────────────────────
    cv2.imshow("Pallet Alignment System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()