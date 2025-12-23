import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
import os
import pandas as pd
import tensorflow as tf

st.set_page_config(page_title="RQI — YOLO + Lane + Map", layout="wide")

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("potholes.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS pothole_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    class_name TEXT,
    confidence REAL,
    snapshot_path TEXT,
    latitude REAL,
    longitude REAL
)
""")
conn.commit()

# =====================================================
# SNAPSHOTS
# =====================================================
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

CLIP_FOLDER = "clips"
os.makedirs(CLIP_FOLDER, exist_ok=True)

def insert_detection(class_name, confidence, snapshot_path, lat, lon):
    cursor.execute("""
        INSERT INTO pothole_events
        (timestamp, class_name, confidence, snapshot_path, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        class_name,
        confidence,
        snapshot_path,
        lat,
        lon
    ))
    conn.commit()

# =====================================================
# LOAD TFLITE MODEL
# =====================================================
try:
    from tflite_runtime.interpreter import Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = "best_float16.tflite"
CLASS_NAMES = ["Pothole", "Crack", "Faded Lane"]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.success(f"Loaded model: {MODEL_PATH}")

# =====================================================
# LANE DETECTION
# =====================================================
def enhanced_lane_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

# =====================================================
# YOLO TFLITE INFERENCE
# =====================================================
def run_tflite_inference(frame, conf_thres=0.25, iou_thres=0.45):
    ih, iw = frame.shape[:2]
    input_h, input_w = input_details[0]["shape"][1:3]

    # Letterbox resize
    scale = min(input_w / iw, input_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    boxes, scores, classes = [], [], []

    for det in preds:
        conf = det[4]
        if conf < conf_thres:
            continue

        xc, yc, w, h = det[:4]

        # Convert to corner format
        x1 = (xc - w / 2) * input_w
        y1 = (yc - h / 2) * input_h
        x2 = (xc + w / 2) * input_w
        y2 = (yc + h / 2) * input_h

        # Undo letterbox
        x1 = max(0, min((x1) / scale, iw))
        y1 = max(0, min((y1) / scale, ih))
        x2 = max(0, min((x2) / scale, iw))
        y2 = max(0, min((y2) / scale, ih))

        boxes.append([x1 / iw, y1 / ih, x2 / iw, y2 / ih])
        scores.append(float(conf))
        classes.append(int(det[5]))

    return np.array(boxes), np.array(scores), np.array(classes)

COLORS = [(0,255,0),(255,255,0),(255,0,255)]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w, _ = frame.shape
    coords = []

    for i, s in enumerate(scores):
        if s < threshold:
            continue

        x1,y1,x2,y2 = boxes[i]
        x1,x2 = int(x1*w), int(x2*w)
        y1,y2 = int(y1*h), int(y2*h)
        cid = int(classes[i])

        cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS[cid%3],2)

        crop = frame[y1:y2, x1:x2]
        if crop.size:
            path = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(path, crop)
            insert_detection(CLASS_NAMES[cid], float(s), path, lat, lon)
            coords.append((lat, lon))

    return coords

# =====================================================
# UI
# =====================================================
st.title("🚧 RQI (Road Quality Index)")

mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

st.sidebar.subheader("📍 GPS Coordinates (Required for Map)")
lat_txt = st.sidebar.text_input("Latitude")
lon_txt = st.sidebar.text_input("Longitude")

def get_gps():
    try:
        return float(lat_txt), float(lon_txt)
    except:
        return None, None

# =====================================================
# IMAGE MODE
# =====================================================
if mode == "Upload Image":
    img = st.file_uploader("Upload road image", ["jpg","jpeg","png"])

    if img and st.button("Run Detection"):
        frame = np.array(Image.open(img).convert("RGB"))
        overlay = frame.copy()

        boxes, scores, classes = run_tflite_inference(overlay, conf)
        lane_mask = enhanced_lane_detection(overlay)

        lat, lon = get_gps()
        coords = []

        if lat is not None:
            coords = draw_boxes(
                overlay, boxes, scores, classes, lat, lon, conf
            )

        final_img = cv2.addWeighted(
            overlay, 0.7,
            cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR), 0.5, 0
        )

        st.image(final_img, caption="YOLO + Lane Overlay", channels="BGR")

        if coords:
            st.subheader("🗺️ Pothole Location")
            st.map(pd.DataFrame(coords, columns=["lat","lon"]))
        else:
            st.info("No potholes detected or GPS not provided")

# =====================================================
# LIVE CAMERA MODE
# =====================================================
if mode == "Live Webcam / DroidCam":
    cam_url = st.text_input(
        "Camera Stream URL",
        placeholder="http://192.168.1.3:8080/video"
    )

    colA, colB = st.columns(2)
    with colA:
        start = st.checkbox("▶ Start Camera")
    with colB:
        pause = st.checkbox("⏸ Pause Detection")

    frame_box = st.empty()
    map_box = st.empty()

    if start and cam_url:
        cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)

        lat, lon = get_gps()
        detected = []

        recording = False
        clip_frames = []
        clip_start_time = None
        CLIP_DURATION = 5  # seconds

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            if not pause:
                boxes, scores, classes = run_tflite_inference(frame, conf)
                lane_mask = enhanced_lane_detection(frame)

                if lat is not None:
                    coords = draw_boxes(
                        frame, boxes, scores, classes, lat, lon, conf
                    )
                    if coords:
                        detected.extend(coords)
                        if not recording:
                            recording = True
                            clip_frames = []
                            clip_start_time = datetime.datetime.now()

                if recording:
                    clip_frames.append(frame.copy())
                    elapsed = (datetime.datetime.now() - clip_start_time).total_seconds()

                    if elapsed >= CLIP_DURATION:
                        h, w, _ = frame.shape
                        clip_name = f"clip_{clip_start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        clip_path = os.path.join(CLIP_FOLDER, clip_name)

                        out = cv2.VideoWriter(
                            clip_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h)
                        )

                        for f in clip_frames:
                            out.write(f)
                        out.release()

                        recording = False
                        clip_frames = []
                        st.success(f"🎥 Saved clip: {clip_name}")

            blended = cv2.addWeighted(
                frame, 0.7,
                cv2.cvtColor(enhanced_lane_detection(frame), cv2.COLOR_GRAY2BGR), 0.5, 0
            )

            frame_box.image(blended, channels="BGR")

            if detected:
                map_box.map(
                    pd.DataFrame(detected, columns=["lat", "lon"])
                )

        cap.release()

# =====================================================
# DATABASE VIEW
# =====================================================
st.subheader("📋 Detection Log")
st.dataframe(pd.read_sql(
    "SELECT * FROM pothole_events ORDER BY id DESC", conn
))
