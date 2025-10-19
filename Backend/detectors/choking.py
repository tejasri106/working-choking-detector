# import cv2
# import mediapipe as mp
# import numpy as np
# from schemas import DetectionEvent
# import yaml
# import time
# from collections import deque

# # Initialize Mediapipe pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # --- NEW: Keep a short history of confidences ---
# CONF_HISTORY = deque(maxlen=5)  # average over last 5 frames

# def detect_choking(frame, cfg, frame_id):
#     """
#     Detects choking based on hand-to-neck proximity using MediaPipe landmarks.
#     Stabilized version with rolling average confidence.
#     Returns DetectionEvent or None.
#     """
#     results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     if not results.pose_landmarks:
#         CONF_HISTORY.append(0)
#         return None

#     landmarks = results.pose_landmarks.landmark

#     # Key body points
#     l_hand = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
#                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])
#     r_hand = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
#                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y])
#     neck   = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x,
#                        landmarks[mp_pose.PoseLandmark.NOSE].y])

#     # Measure distance from both hands to neck
#     lh_dist = np.linalg.norm(l_hand - neck)
#     rh_dist = np.linalg.norm(r_hand - neck)

#     # Invert distance to get confidence (closer = higher)
#     conf = max(0, 1 - (lh_dist + rh_dist))

#     # --- NEW: Rolling average to stabilize ---
#     CONF_HISTORY.append(conf)
#     avg_conf = np.mean(CONF_HISTORY)

#     # --- NEW: Smooth visual marker for confidence ---
#     h, w, _ = frame.shape
#     cv2.putText(frame, f"Conf(avg)={avg_conf:.2f}", (20, h - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     # Only trigger if stabilized confidence is high enough
#     if avg_conf > cfg["detection"]["choking_conf"]:
#         cx, cy = int(neck[0] * w), int(neck[1] * h)
#         return DetectionEvent(
#             type="choking",
#             confidence=round(avg_conf, 2),
#             coords=(cx, cy),
#             frame_id=frame_id
#         )
#     return None

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from schemas import DetectionEvent

# ---- Initialize Mediapipe solutions ----
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---- Rolling histories for smoothing ----
CONF_HISTORY = deque(maxlen=5)
DIST_HISTORY = deque(maxlen=5)

def detect_choking(frame, cfg, frame_id):
    """
    Detects choking based on hand-to-neck proximity and mouth opening ratio.
    Robust against head tilt. Returns DetectionEvent or None.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    pose_results = pose.process(rgb)
    if not pose_results.pose_landmarks:
        CONF_HISTORY.append(0)
        return None

    lm = pose_results.pose_landmarks.landmark

    try:
        l_hand = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
                           lm[mp_pose.PoseLandmark.LEFT_WRIST].y])
        r_hand = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                           lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
        # Use stable neck midpoint instead of nose
        l_shoulder = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        r_shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        neck = (l_shoulder + r_shoulder) / 2
    except IndexError:
        CONF_HISTORY.append(0)
        return None

    # --- Normalize distances by shoulder width ---
    shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)
    if shoulder_dist < 1e-6:
        CONF_HISTORY.append(0)
        return None

    lh_dist = np.linalg.norm(l_hand - neck) / shoulder_dist
    rh_dist = np.linalg.norm(r_hand - neck) / shoulder_dist
    avg_dist = (lh_dist + rh_dist) / 2
    DIST_HISTORY.append(avg_dist)

    # --- Motion stability ---
    motion_var = np.var(DIST_HISTORY) if len(DIST_HISTORY) > 1 else 0
    motion_penalty = max(0, 1 - 5 * motion_var)

    # --- Mouth open ratio (FaceMesh) ---
    face_results = face_mesh.process(rgb)
    mouth_conf = 0
    h, w, _ = frame.shape
    if face_results.multi_face_landmarks:
        mouth = face_results.multi_face_landmarks[0]
        top = mouth.landmark[13]
        bottom = mouth.landmark[14]
        eye = mouth.landmark[159]
        chin = mouth.landmark[152]
        mouth_open = np.linalg.norm(np.array([top.x, top.y]) - np.array([bottom.x, bottom.y]))
        face_height = np.linalg.norm(np.array([eye.x, eye.y]) - np.array([chin.x, chin.y]))
        if face_height > 1e-6:
            ratio = mouth_open / face_height
            mouth_conf = np.clip((ratio - 0.02) * 20, 0, 1)

            # Visualization: green dot on mouth if open
            mx, my = int((top.x + bottom.x) / 2 * w), int((top.y + bottom.y) / 2 * h)
            if mouth_conf > 0.5:
                cv2.circle(frame, (mx, my), 5, (0, 255, 0), -1)

    # --- Combine hand and mouth confidence ---
    hand_conf = max(0, 1 - avg_dist) * motion_penalty
    combined_conf = 0.7 * hand_conf + 0.3 * mouth_conf
    CONF_HISTORY.append(combined_conf)
    avg_conf = np.mean(CONF_HISTORY)

    # --- Display overlay ---
    cv2.putText(frame, f"Conf(avg)={avg_conf:.2f}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Trigger detection event ---
    if avg_conf > cfg["detection"]["choking_conf"] and motion_penalty > 0.6:
        print(f"🚨 Detected choking! conf={avg_conf:.2f}, motion_penalty={motion_penalty:.2f}, mouth_conf={mouth_conf:.2f}")
        return DetectionEvent(
            type="choking",
            confidence=round(avg_conf, 2),
            coords=(neck[0], neck[1]),  # normalized coords
            frame_id=frame_id
        )

    return None
