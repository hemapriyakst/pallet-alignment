import cv2
import numpy as np
from ultralytics import YOLO

# ── Load model ───────────────────────────────────────────────
model = YOLO("runs/detect/pallet_model/weights/best.pt")
cap = cv2.VideoCapture(0)

THRESHOLD = 50
FAR_SIZE = 0.10
CLOSE_SIZE = 0.50

PANEL_W = 300  # width of right side panel

def draw_arrow(panel, direction):
    """Draw a direction arrow in the panel."""
    cx, cy = 150, 100
    size = 40
    color = (0, 255, 255)
    thickness = 3

    arrows = {
        "LEFT":     [(cx + size, cy), (cx - size, cy),
                     (cx - size + 20, cy - 20), (cx - size, cy),
                     (cx - size + 20, cy + 20)],
        "RIGHT":    [(cx - size, cy), (cx + size, cy),
                     (cx + size - 20, cy - 20), (cx + size, cy),
                     (cx + size - 20, cy + 20)],
        "FORWARD":  [(cx, cy + size), (cx, cy - size),
                     (cx - 20, cy - size + 20), (cx, cy - size),
                     (cx + 20, cy - size + 20)],
        "BACKWARD": [(cx, cy - size), (cx, cy + size),
                     (cx - 20, cy + size - 20), (cx, cy + size),
                     (cx + 20, cy + size - 20)],
        "ALIGNED":  None,
        "NONE":     None,
    }

    if direction == "ALIGNED":
        cv2.circle(panel, (cx, cy), size, (0, 255, 0), thickness)
        cv2.putText(panel, "✓", (cx - 15, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    elif direction in arrows and arrows[direction]:
        pts = arrows[direction]
        cv2.line(panel, pts[0], pts[1], color, thickness)
        cv2.line(panel, pts[2], pts[3], color, thickness)
        cv2.line(panel, pts[4], pts[3], color, thickness)
    else:
        cv2.putText(panel, "?", (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def draw_panel(H, status, direction, conf, offset_x, offset_y, distance_status, distance_color):
    """Build the right side dashboard panel."""
    panel = np.zeros((H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (15, 15, 15)  # dark background

    # ── Title ────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (PANEL_W, 45), (30, 30, 30), -1)
    cv2.putText(panel, "PALLET ALIGN SYS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ── Status box ───────────────────────────────────────────
    status_colors = {
        "ALIGNED":  (0, 255, 0),
        "MOVE":     (0, 255, 255),
        "NO PALLET":(0, 0, 255),
    }
    key = "ALIGNED" if "ALIGNED" in status else "MOVE" if "MOVE" in status else "NO PALLET"
    s_color = status_colors[key]

    cv2.rectangle(panel, (10, 55), (PANEL_W - 10, 95), (30, 30, 30), -1)
    cv2.rectangle(panel, (10, 55), (PANEL_W - 10, 95), s_color, 1)
    cv2.putText(panel, status[:22], (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1)

    # ── Direction arrow ──────────────────────────────────────
    cv2.rectangle(panel, (10, 105), (PANEL_W - 10, 225), (25, 25, 25), -1)
    arrow_panel = panel[105:225, 10:PANEL_W - 10]
    draw_arrow(arrow_panel, direction)
    panel[105:225, 10:PANEL_W - 10] = arrow_panel

    # ── Divider ──────────────────────────────────────────────
    cv2.line(panel, (20, 235), (PANEL_W - 20, 235), (50, 50, 50), 1)

    # ── Diagnostics ──────────────────────────────────────────
    cv2.putText(panel, "DIAGNOSTICS", (10, 258),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    metrics = [
        (f"Confidence  : {conf:.2f}",   (255, 255, 255), 285),
        (f"Offset X    : {offset_x}px", (255, 255, 255), 310),
        (f"Offset Y    : {offset_y}px", (255, 255, 255), 335),
    ]
    for text, color, y in metrics:
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # ── Distance ─────────────────────────────────────────────
    cv2.line(panel, (20, 350), (PANEL_W - 20, 350), (50, 50, 50), 1)
    cv2.putText(panel, "DISTANCE", (10, 373),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    cv2.putText(panel, distance_status, (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)

    return panel


while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    frame_cx = W // 2
    frame_cy = H // 2
    frame_area = H * W

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    command = "NO PALLET DETECTED"
    direction = "NONE"
    conf_val = 0.0
    offset_x, offset_y = 0, 0
    distance_status = "--"
    distance_color = (180, 180, 180)

    if len(boxes) > 0:
        best = max(boxes, key=lambda b: b.conf)
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        conf_val = float(best.conf)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        offset_x = cx - frame_cx
        offset_y = cy - frame_cy

        # Distance
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

        # Command
        if abs(offset_x) < THRESHOLD and abs(offset_y) < THRESHOLD:
            command = "ALIGNED - INSERT FORKS"
            direction = "ALIGNED"
        elif abs(offset_x) >= abs(offset_y):
            if offset_x > 0:
                command = f"MOVE RIGHT ({abs(offset_x)}px)"
                direction = "RIGHT"
            else:
                command = f"MOVE LEFT ({abs(offset_x)}px)"
                direction = "LEFT"
        else:
            if offset_y > 0:
                command = f"MOVE FORWARD ({abs(offset_y)}px)"
                direction = "FORWARD"
            else:
                command = f"MOVE BACKWARD ({abs(offset_y)}px)"
                direction = "BACKWARD"

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.line(frame, (cx, cy), (frame_cx, frame_cy), (0, 165, 255), 2)
        cv2.putText(frame, f"pallet {conf_val:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Crosshair
    cv2.drawMarker(frame, (frame_cx, frame_cy),
                   (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

    # Top banner
    cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 0), -1)
    cmd_color = (0, 255, 0) if "ALIGNED" in command else \
                (0, 0, 255) if "NO PALLET" in command else (0, 255, 255)
    cv2.putText(frame, command, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cmd_color, 2)

    # Build and attach panel
    panel = draw_panel(H, command, direction, conf_val,
                       offset_x, offset_y, distance_status, distance_color)
    combined = np.hstack([frame, panel])

    cv2.imshow("Pallet Alignment System", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()