import asyncio, cv2, json, websockets, yaml
from detectors.unresponsive import detect_unresponsive

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SEND_INTERVAL = 0.7  # frame delay to stay smooth
last_event_type = None

async def send_events():
    uri = "ws://localhost:8765"
    print("📡 Starting cardiac arrest detection stream...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    frame_id = 0
    global last_event_type

    async with websockets.connect(uri) as ws:
        still_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame capture failed.")
                break

            # Run unresponsive detector only
            evt = detect_unresponsive(frame, cfg, frame_id)

            if evt:
                still_frames += 1
            else:
                still_frames = 0

            # if still for long enough, trigger event
            if still_frames > 30:  # ~20s at 0.7s/frame
                msg = json.dumps({
                    "type": "cardiac_arrest",
                    "confidence": 0.95,
                    "frame_id": frame_id
                })
                await ws.send(msg)
                print(f"📤 Sent cardiac arrest alert! {msg}")
                still_frames = 0

            # show feed
            status = "Monitoring..." if not evt else "Still detected"
            color = (0, 255, 0) if not evt else (0, 0, 255)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("CPR Detection Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(SEND_INTERVAL)
            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(send_events())