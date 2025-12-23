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

# =====================================================
# LANE COLOR MASK
# =====================================================
def enhanced_lane_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

# =====================================================
# MISSING LANE (EDGE LOGIC)
# =====================================================
def detect_missing_lane_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    roi = edges[int(h * 0.5):, :]

    gap_mask = np.zeros_like(edges)
    missing = False

    for x in range(0, w, 8):
        for y in range(0, roi.shape[0] - 12, 12):
            patch = roi[y:y+12, x:x+8]
            if np.sum(patch > 0) < 8:
                gap_mask[int(h*0.5)+y:int(h*0.5)+y+12, x:x+8] = 255
                missing = True

    overlay = frame.copy()
    overlay[gap_mask == 255] = [0, 0, 255]

    return overlay, missing

# =====================================================
# YOLO
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
mode = st.selectbox("Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3)

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
    img = st.file_uploader("Upload image", ["jpg","png","jpeg"])
    if img:
        frame = np.array(Image.open(img).convert("RGB"))
        overlay = frame.copy()

        boxes, scores, classes = run_tflite_inference(frame)
        lat, lon = get_gps()
        if lat is not None:
            draw_boxes(overlay, boxes, scores, classes, lat, lon, conf)

        missing_view, missing = detect_missing_lane_edges(overlay)

        col1, col2 = st.columns(2)
        col1.image(overlay, channels="BGR", caption="YOLO + Lane")
        col2.image(missing_view, channels="BGR", caption="Missing Lane")

# =====================================================
# LIVE MODE
# =====================================================
if mode == "Live Webcam / DroidCam":
    cam_url = st.text_input("Camera URL")
    start = st.checkbox("▶ Start")
    pause = st.checkbox("⏸ Pause")

    col1, col2 = st.columns(2)
    detected = []

    if start and cam_url:
        cap = cv2.VideoCapture(cam_url)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            yolo_view = frame.copy()
            lat, lon = get_gps()

            if not pause:
                boxes, scores, classes = run_tflite_inference(frame)
                if lat is not None:
                    draw_boxes(yolo_view, boxes, scores, classes, lat, lon, conf)

            missing_view, missing = detect_missing_lane_edges(frame)

            col1.image(yolo_view, channels="BGR", caption="YOLO + Lane")
            col2.image(missing_view, channels="BGR", caption="Missing Lane")

            time.sleep(0.03)

# =====================================================
# DATABASE VIEW
# =====================================================
st.subheader("📋 Detection Log")
st.dataframe(pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn))
