import cv2 as cv
import mediapipe as mp
import pandas as pd
import math

# === Setup MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

POSE_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
])

EXCLUDED_LANDMARKS = {
    mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT,
    mp_pose.PoseLandmark.LEFT_THUMB, mp_pose.PoseLandmark.RIGHT_THUMB,
    mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
}

# === Video Source ===
video_path = "/home/j4_m3s/Videos/img_processing/humanfall_data_18.MOV"
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)

# --- Desired Processing Limit Time (sec) ---
start_time_sec = 0.0
end_time_sec = 150.0 
frame_count = 0
data = []

# === Helper Functions ===
def get_bbox(landmarks):
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    return min(x_vals), min(y_vals), max(x_vals)-min(x_vals), max(y_vals)-min(y_vals)

def get_center(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    cx = (ls.x + rs.x + lh.x + rh.x)/4
    cy = (ls.y + rs.y + lh.y + rh.y)/4
    return cx, cy

def body_tilt_3d(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mx_sh = (ls.x + rs.x)/2; my_sh = (ls.y + rs.y)/2; mz_sh = (ls.z + rs.z)/2
    mx_hp = (lh.x + rh.x)/2; my_hp = (lh.y + rh.y)/2; mz_hp = (lh.z + rh.z)/2
    dx = mx_sh - mx_hp; dy = my_sh - my_hp; dz = mz_sh - mz_hp
    angle = math.degrees(math.atan2(math.sqrt(dx**2+dz**2), dy))
    return angle

def draw_openpose_style(frame, landmarks):
    h, w, _ = frame.shape
    for start, end in POSE_CONNECTIONS:
        s, e = landmarks[start.value], landmarks[end.value]
        p1 = (int(s.x*w), int(s.y*h)); p2 = (int(e.x*w), int(e.y*h))
        cv.line(frame, p1, p2, (0,255,255), 2)

    # Custom neck + shoulder + nose lines
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    neck_x = (left_shoulder.x + right_shoulder.x) / 2
    neck_y = (left_shoulder.y + right_shoulder.y) / 2
    neck = (int(neck_x * w), int(neck_y * h))
    nose_point = (int(nose.x * w), int(nose.y * h))
    left_shoulder_p = (int(left_shoulder.x * w), int(left_shoulder.y * h))
    right_shoulder_p = (int(right_shoulder.x * w), int(right_shoulder.y * h))

    cv.line(frame, left_shoulder_p, neck, (0, 255, 255), 2)
    cv.line(frame, neck, right_shoulder_p, (0, 255, 255), 2)
    cv.line(frame, nose_point, neck, (0, 255, 255), 2)
    cv.circle(frame, neck, 5, (0, 0, 255), -1)

    for idx, lm in enumerate(landmarks):
        if mp_pose.PoseLandmark(idx) not in EXCLUDED_LANDMARKS:
            cv.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,0,255), -1)

# === Main Loop ===
start_frame_center_y = None
start_frame_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time_sec = frame_count / fps

    # --- Skip frames out of desired time window ---
    if current_time_sec < start_time_sec or current_time_sec > end_time_sec:
        continue

    h, w, _ = frame.shape
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        x, y, w_box, h_box = get_bbox(landmarks)
        aspect_ratio = w_box/h_box if h_box>0 else 0
        cx, cy = get_center(landmarks)
        tilt_angle = body_tilt_3d(landmarks)

# --- Draw center point (normalized) on frame ---
        center_px = int(cx * w)
        center_py = int(cy * h)
        cv.circle(frame, (center_px, center_py), 6, (255, 0, 0), -1)
        cv.putText(frame, f"({cx:.3f}, {cy:.3f})", 
        (center_px + 10, center_py - 10), 
        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Record start point for vertical speed ---
        if start_frame_center_y is None:
            start_frame_center_y = cy
            start_frame_time = current_time_sec

        # --- Vertical speed calculation ---
        dt = current_time_sec - start_frame_time
        if dt <= 0: dt = 1/fps
        vertical_speed = abs(cy - start_frame_center_y)/dt

        # --- Simple fall labeling based on tilt/aspect ratio ---
        if tilt_angle < 140 and aspect_ratio > 1.4:
            label = "fall"
            color = (0, 0, 255)
            text = "Fall Detected"
        else:
            label = "normal"
            color = (0, 255, 0)
            text = "Normal"

        # --- Draw bbox and skeleton ---
        cv.rectangle(frame, (int(x*w), int(y*h)), (int((x+w_box)*w), int((y+h_box)*h)), color,2)
        draw_openpose_style(frame, landmarks)

        cv.putText(frame, f"{text}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.putText(frame, f"Width={w_box:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(frame, f"Height={h_box:.2f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(frame, f"Aspect Ratio={aspect_ratio:.2f}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(frame, f"Tilt={tilt_angle:.2f}", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(frame, f"Vertical Speed={vertical_speed:.5f}", (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Append data ---
        data.append([cx, cy, w_box, h_box, aspect_ratio, tilt_angle, vertical_speed, label])

    else:
        cv.putText(frame, "No person detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.imshow("Fall Detection | Collect Data", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# === Save CSV ===
df = pd.DataFrame(data, columns=["cx", "cy", "width", "height", "aspect_ratio", "tilt_angle", "vertical_speed", "label"])
df.to_csv("/home/j4_m3s/Documents/uni_ws/image_processing/mini_project/csv_folder/fall_data_57.csv", index=False)
print("✅ Saved dataset")