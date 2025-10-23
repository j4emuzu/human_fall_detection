import cv2 as cv
import mediapipe as mp
import joblib
import pandas as pd
import math
import time

# === Load trained Random Forest model ===
model_path = "human_fall_model.pkl"
rf = joblib.load(model_path)

# === Setup MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# === Video Capture ===
# video_path = "/home/j4_m3s/Videos/img_processing/humanfall_data_1.MOV"
# cap = cv.VideoCapture(video_path)
cap = cv.VideoCapture(4)
fps = cap.get(cv.CAP_PROP_FPS) if cap.get(cv.CAP_PROP_FPS) > 0 else 60
prev_center_y = None
prev_time = None

# === Helper Functions ===
def get_center(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    cx = (ls.x + rs.x + lh.x + rh.x) / 4
    cy = (ls.y + rs.y + lh.y + rh.y) / 4
    return cx, cy

def get_bbox(landmarks):
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    return min(x_vals), min(y_vals), max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)

def body_tilt_3d(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mx_sh = (ls.x + rs.x) / 2
    my_sh = (ls.y + rs.y) / 2
    mz_sh = (ls.z + rs.z) / 2
    mx_hp = (lh.x + rh.x) / 2
    my_hp = (lh.y + rh.y) / 2
    mz_hp = (lh.z + rh.z) / 2
    dx = mx_sh - mx_hp
    dy = my_sh - my_hp
    dz = mz_sh - mz_hp
    angle = math.degrees(math.atan2(math.sqrt(dx**2 + dz**2), dy))
    return angle

# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        cx, cy = get_center(landmarks)
        tilt = body_tilt_3d(landmarks)
        x, y, w_box, h_box = get_bbox(landmarks)
        aspect_ratio = w_box / h_box if h_box > 0 else 0

        # --- Compute vertical speed ---
        curr_time = time.time()
        if prev_center_y is None or prev_time is None:
            vertical_speed = 0
        else:
            dt = curr_time - prev_time
            dt = max(dt, 1 / fps)
            vertical_speed = abs(cy - prev_center_y) / dt

        prev_center_y = cy
        prev_time = curr_time

        # --- Prepare features for model ---
        feature_names = ["cx", "cy", "width", "height", "aspect_ratio", "tilt_angle", "vertical_speed"]
        feature_df = pd.DataFrame(
            [[cx, cy, w_box, h_box, aspect_ratio, tilt, vertical_speed]], columns=feature_names
        )
        pred = rf.predict(feature_df)[0]

        # --- Label and color ---
        if pred.lower() == "fall":
            text = "Fall"
            color = (0, 0, 255)  # Red
        elif pred.lower() == "lie_down":
            text = "Lie Down"
            color = (0, 165, 255)  # Orange
        else:
            text = "Normal"
            color = (0, 255, 0)  # Green

        # --- Draw detection box and label ---
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + w_box) * w), int((y + h_box) * h)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label box on top of bounding box
        text_scale = 0.7
        text_thickness = 2
        (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        label_y = max(y1 - 10, text_h + 10)

        cv.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        cv.putText(frame, text, (x1 + 2, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)

    else:
        cv.putText(frame, "No person detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.imshow("Real-Time Fall Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()