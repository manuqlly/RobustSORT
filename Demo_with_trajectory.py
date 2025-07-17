import cv2
from ultralytics import YOLO
from trackertrajectory import RobustBoxTracker  # Make sure this file is in the same dir
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Or your custom model

# Initialize tracker
tracker = RobustBoxTracker(max_disappeared=30, max_distance=80)

# Open video
cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional video output
out = cv2.VideoWriter("RobustSort.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, iou=0.4, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    # Convert detections to list of tuples
    rects = [tuple(map(int, box)) for box in detections]

    # Update tracker
    objects = tracker.update(rects, frame_number=frame_num)
    trajectories = tracker.get_trajectories()

    # Draw trajectories
    for obj_id, points in trajectories.items():
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)
        cv2.circle(frame, points[-1], 4, (0, 0, 255), -1)  # centroid dot


    # Draw tracking output
    for obj_id, center in objects.items():
        # Find box closest to center
        closest_box = min(rects, key=lambda r: np.linalg.norm([(r[0]+r[2])//2 - center[0], (r[1]+r[3])//2 - center[1]]))
        x1, y1, x2, y2 = closest_box

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("RobustSort Demo", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()
