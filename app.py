import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
import os
import pandas as pd
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

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
    interpreter.set_tensor(
        input_details[0]["index"],
        preprocess_for_tflite(frame)
    )
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    if output is None or len(output) == 0:
        return [], [], []

    output = output[0]
    return output[:, :4], output[:, 4], output[:, 5]

COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w, _ = frame.shape
    coords = []

    for i, score in enumerate(scores):
        if score < threshold:
            continue

        x1, y1, x2, y2 = boxes[i]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        cid = int(classes[i])

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[cid % 3], 2)
        cv2.putText(
            frame,
            f"{CLASS_NAMES[cid]} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS[cid % 3],
            2
        )

        if lat is not None and lon is not None:
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                path = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(path, crop)
                insert_detection(CLASS_NAMES[cid], float(score), path, lat, lon)
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
    img = st.file_uploader("Upload road image", ["jpg", "jpeg", "png"])

    if img and st.button("Run Detection"):
        frame = np.array(Image.open(img).convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        boxes, scores, classes = run_tflite_inference(frame)
        lane_mask = enhanced_lane_detection(frame)

        lat, lon = get_gps()
        coords = draw_boxes(frame, boxes, scores, classes, lat, lon, conf)

        final_img = cv2.addWeighted(frame, 0.7, lane_mask, 0.5, 0)
        st.image(final_img, caption="YOLO + Lane Overlay", channels="BGR")

        if coords:
            st.map(pd.DataFrame(coords, columns=["lat", "lon"]))
        else:
            st.info("No detections or GPS not provided")

# =====================================================
# LIVE CAMERA MODE
# =====================================================
if mode == "Live Webcam / DroidCam":
    cam_url = st.text_input("Camera URL", "http://192.168.1.3:4747/video")
    start = st.checkbox("Start Camera")

    if start:
        cap = cv2.VideoCapture(cam_url)
        frame_box = st.empty()
        lat, lon = get_gps()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            boxes, scores, classes = run_tflite_inference(frame)
            draw_boxes(frame, boxes, scores, classes, lat, lon, conf)

            lanes = enhanced_lane_detection(frame)
            blended = cv2.addWeighted(frame, 0.7, lanes, 0.5, 0)

            frame_box.image(blended, channels="BGR")
            time.sleep(0.03)

        cap.release()

# =====================================================
# DATABASE VIEW
# =====================================================
st.subheader("📋 Detection Log")
st.dataframe(pd.read_sql(
    "SELECT * FROM pothole_events ORDER BY id DESC", conn
))
