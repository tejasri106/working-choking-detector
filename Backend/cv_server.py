import asyncio, cv2, json, websockets, yaml
from detectors.choking import detect_choking

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SEND_INTERVAL = 0.7  # smoother performance
last_event_type = None

async def send_events():
    uri = "ws://localhost:8765"
    print("📡 Starting choking detection stream...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    frame_id = 0
    global last_event_type

    async with websockets.connect(uri) as ws:
        choking_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame capture failed.")
                break

            # Run choking detector
            evt = detect_choking(frame, cfg, frame_id)

            if evt:
                choking_frames += 1
            else:
                choking_frames = 0

            # If choking persists for a few frames, trigger event
            if choking_frames > 5:  # adjust threshold as needed
                msg = json.dumps({
                    "type": "choking",
                    "confidence": evt.confidence if evt else 0.9,
                    "frame_id": frame_id
                })
                await ws.send(msg)
                print(f"📤 Sent choking alert! {msg}")
                choking_frames = 0

            # Display feed
            status = "Monitoring..." if not evt else "Choking detected"
            color = (0, 255, 0) if not evt else (0, 0, 255)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Choking Detection Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(SEND_INTERVAL)
            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(send_events())
