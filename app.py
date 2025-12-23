import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
import os
import pandas as pd
import requests
import io
import time

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
def preprocess_for_tflite(frame):
    h, w = input_details[0]["shape"][1:3]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def run_tflite_inference(frame):
    interpreter.set_tensor(input_details[0]["index"], preprocess_for_tflite(frame))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]
    if out.size == 0:
        return [], [], []
    return out[:, :4], out[:, 4], out[:, 5]

COLORS = [(0,255,0),(255,255,0),(255,0,255)]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w, _ = frame.shape
    coords = []

    for i, s in enumerate(scores):
        if s < threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1, x2 = int(x1*w), int(x2*w)
        y1, y2 = int(y1*h), int(y2*h)
        cid = int(classes[i])

        cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS[cid%3],2)
        cv2.putText(frame,f"{CLASS_NAMES[cid]} {s:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[cid%3],2)

        if lat is not None and lon is not None:
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

st.sidebar.subheader("📍 GPS Coordinates")
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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        boxes, scores, classes = run_tflite_inference(frame)
        lane_mask = enhanced_lane_detection(frame)

        lat, lon = get_gps()
        coords = draw_boxes(frame, boxes, scores, classes, lat, lon, conf)

        final_img = cv2.addWeighted(frame, 0.7, cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
        st.image(final_img, channels="BGR", caption="YOLO + Lane Overlay")

        if coords:
            st.map(pd.DataFrame(coords, columns=["lat","lon"]))
        else:
            st.info("No detections or GPS not provided")

# =====================================================
# LIVE CAMERA MODE — MJPEG
# =====================================================
if mode == "Live Webcam / DroidCam":
    cam_url = st.text_input("Camera MJPEG URL", "http://192.168.1.3:8080/video")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Start Camera"):
            st.session_state.camera_running = True
    with col2:
        if st.button("⏸ Pause Detection"):
            st.session_state.camera_pause = True
    with col3:
        if st.button("⏹ Stop Camera"):
            st.session_state.camera_running = False
            st.session_state.camera_pause = False

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "camera_pause" not in st.session_state:
        st.session_state.camera_pause = False

    frame_box = st.empty()
    map_box = st.empty()
    lat, lon = get_gps()
    detected = []

    def mjpeg_stream(url):
        """Generator to yield frames from MJPEG HTTP stream"""
        stream = requests.get(url, stream=True)
        bytes_buffer = b""
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_buffer += chunk
            a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

    if st.session_state.camera_running and cam_url:
        try:
            for frame in mjpeg_stream(cam_url):
                if not st.session_state.camera_running:
                    break
                if not st.session_state.camera_pause:
                    boxes, scores, classes = run_tflite_inference(frame)
                    if lat is not None and lon is not None:
                        coords = draw_boxes(frame, boxes, scores, classes, lat, lon, conf)
                        if coords:
                            detected.extend(coords)

                # Lane detection & overlay
                lane_mask = enhanced_lane_detection(frame)
                blended = cv2.addWeighted(frame, 0.7, cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

                # Show frame & map
                frame_box.image(blended, channels="BGR")
                if detected:
                    map_box.map(pd.DataFrame(detected, columns=["lat","lon"]))

        except Exception as e:
            st.error(f"Camera not accessible: {e}")

# =====================================================
# DATABASE VIEW
# =====================================================
st.subheader("📋 Detection Log")
st.dataframe(pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn))
