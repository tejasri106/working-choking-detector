import cv2
import yaml
from detectors.choking import detect_choking

# Load configuration file
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cap = cv2.VideoCapture(1)
import time
time.sleep(2)
if not cap.isOpened():
    print("❌ Camera not found.")
    exit()

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break

    # Run choking detection
    event = detect_choking(frame, cfg, frame_id)
    frame_id += 1

    # If choking detected, draw visual alert
    if event:
        print(f"🚨 Detected choking event! conf={event.confidence}")
        cv2.putText(frame, f"Choking! conf={event.confidence}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        x, y = map(int, event.coords)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

    cv2.imshow("Choking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()