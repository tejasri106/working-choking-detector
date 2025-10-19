import cv2
import mediapipe as mp
import numpy as np
from schemas import DetectionEvent
import yaml
import time
from collections import deque

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- NEW: Keep a short history of confidences ---
CONF_HISTORY = deque(maxlen=5)  # average over last 5 frames

def detect_choking(frame, cfg, frame_id):
    """
    Detects choking based on hand-to-neck proximity using MediaPipe landmarks.
    Stabilized version with rolling average confidence.
    Returns DetectionEvent or None.
    """
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        CONF_HISTORY.append(0)
        return None

    landmarks = results.pose_landmarks.landmark

    # Key body points
    l_hand = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])
    r_hand = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y])
    neck   = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x,
                       landmarks[mp_pose.PoseLandmark.NOSE].y])

    # Measure distance from both hands to neck
    lh_dist = np.linalg.norm(l_hand - neck)
    rh_dist = np.linalg.norm(r_hand - neck)

    # Invert distance to get confidence (closer = higher)
    conf = max(0, 1 - (lh_dist + rh_dist))

    # --- NEW: Rolling average to stabilize ---
    CONF_HISTORY.append(conf)
    avg_conf = np.mean(CONF_HISTORY)

    # --- NEW: Smooth visual marker for confidence ---
    h, w, _ = frame.shape
    cv2.putText(frame, f"Conf(avg)={avg_conf:.2f}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Only trigger if stabilized confidence is high enough
    if avg_conf > cfg["detection"]["choking_conf"]:
        cx, cy = int(neck[0] * w), int(neck[1] * h)
        return DetectionEvent(
            type="choking",
            confidence=round(avg_conf, 2),
            coords=(cx, cy),
            frame_id=frame_id
        )
    return None